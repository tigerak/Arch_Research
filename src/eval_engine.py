from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from src.classification.methods.arcface import SubCenterArcFace
from src.classification.registry import build_loss, build_model, load_arcface_state_if_present
from src.config_loader import ConfigError
from src.dataset_builder import build_eval_loader


def evaluate_checkpoint(
    cfg: Mapping[str, Any],
    checkpoint_path: str | Path | None = None,
) -> Dict[str, Any]:
    """단독 평가 모드를 실행하고 지표 요약을 반환한다.

    동작 흐름:
    1) cfg 기준으로 모델/손실/평가 데이터로더 구성
    2) 체크포인트 가중치 로드
    3) test_dir(data.test_dir)에서 추론
    4) confusion matrix + 클래스별 precision/recall/f1 + 전체 loss/accuracy 계산

    반환값에는 출력용 문자열뿐 아니라 구조화된 숫자 지표를 함께 담아,
    상위 run_manager에서 리포트(csv/json)로 재사용하기 쉽게 구성한다.
    """
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    criterion = _build_eval_criterion(cfg, model=model)
    loader, idx_to_class = build_eval_loader(cfg)

    # 경로 미지정 시 규약 경로(models/save/<project_name>)에서
    # f1_score_loss_score.pth 형식의 top1 파일을 자동 선택한다.
    if checkpoint_path is None:
        project_name = str(cfg.get("project_name", "experiment"))
        save_dir = Path(str(data_cfg.get("save_dir", "models/save")))
        checkpoint_path = _resolve_default_checkpoint_path(save_dir / project_name)
    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise ConfigError("Checkpoint not found", field="checkpoint_path", file_path=ckpt_path)

    checkpoint_obj = _load_checkpoint(model, ckpt_path)
    if isinstance(criterion, SubCenterArcFace):
        criterion = criterion.to(device)
        load_arcface_state_if_present(criterion, checkpoint_obj)

    eval_loss, eval_acc, conf = _run_eval_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
    )

    class_rows, macro_recall, macro_f1 = _class_metrics_from_confusion(conf, idx_to_class)

    result = {
        "checkpoint_path": str(ckpt_path),
        "num_samples": int(conf.sum().item()),
        "loss": float(eval_loss),
        "accuracy": float(eval_acc),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "confusion_matrix": conf.tolist(),
        "per_class": class_rows,
        "labels": [idx_to_class[i] for i in sorted(idx_to_class.keys())],
        "n_classes": int(train_cfg.get("n_classes", len(idx_to_class))),
    }

    _save_evaluation_outputs(result)

    print_evaluation_report(result)
    return result


def _build_eval_criterion(cfg: Mapping[str, Any], model: nn.Module) -> nn.Module:
    """평가용 손실 함수를 선택한다.

    arcface는 학습 시 feature+label 형태의 metric loss로 동작하므로,
    일반적인 분류 추론(logits) 평가에서는 직접 적용할 수 없다.
    따라서 arcface 설정일 때는 평가 loss를 CrossEntropyLoss로 대체한다.
    """
    return build_loss(cfg, model=model)


def _run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, torch.Tensor]:
    """한 번의 평가 epoch를 수행해 평균 loss/accuracy/confusion matrix를 계산한다."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    conf = None

    iterable = loader if tqdm is None else tqdm(loader, desc="Eval", leave=False)

    with torch.no_grad():
        for images, targets in iterable:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss, score_logits = _compute_eval_loss_and_scores(model, criterion, logits, targets)
            preds = torch.argmax(score_logits, dim=1)

            if conf is None:
                n_classes = int(score_logits.shape[1])
                conf = torch.zeros((n_classes, n_classes), dtype=torch.long)

            for t, p in zip(targets.cpu(), preds.cpu()):
                conf[int(t), int(p)] += 1

            total_loss += float(loss.item()) * images.size(0)
            total_correct += int((preds == targets).sum().item())
            total_samples += int(images.size(0))

    if conf is None:
        raise ConfigError("Evaluation loader is empty", field="data.test_dir")

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc, conf


def _class_metrics_from_confusion(
    conf: torch.Tensor,
    idx_to_class: Dict[int, str],
) -> Tuple[List[Dict[str, Any]], float, float]:
    """confusion matrix에서 클래스별 precision/recall/f1과 macro 지표를 계산한다."""
    rows: List[Dict[str, Any]] = []
    recalls: List[float] = []
    f1s: List[float] = []

    n_classes = int(conf.shape[0])
    for i in range(n_classes):
        tp = float(conf[i, i].item())
        fp = float(conf[:, i].sum().item() - tp)
        fn = float(conf[i, :].sum().item() - tp)
        support = int(conf[i, :].sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        rows.append(
            {
                "class_index": i,
                "class_name": idx_to_class.get(i, str(i)),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        recalls.append(recall)
        f1s.append(f1)

    macro_recall = sum(recalls) / max(1, len(recalls))
    macro_f1 = sum(f1s) / max(1, len(f1s))
    return rows, macro_recall, macro_f1


def print_evaluation_report(result: Mapping[str, Any]) -> None:
    """평가 결과를 콘솔에서 즉시 확인하기 좋은 형태로 출력한다."""
    print("[Eval] ===============================")
    print(f"[Eval] checkpoint: {result['checkpoint_path']}")
    print(f"[Eval] samples:    {result['num_samples']}")
    print(f"[Eval] loss:       {result['loss']:.6f}")
    print(f"[Eval] accuracy:   {result['accuracy']:.6f}")
    print(f"[Eval] macro_recall: {result['macro_recall']:.6f}")
    print(f"[Eval] macro_f1:     {result['macro_f1']:.6f}")

    print("[Eval] confusion_matrix:")
    for row in result["confusion_matrix"]:
        print(f"[Eval]   {row}")

    print("[Eval] per-class:")
    for row in result["per_class"]:
        print(
            f"[Eval]   {row['class_name']:<16} "
            f"P={row['precision']:.6f} "
            f"R={row['recall']:.6f} "
            f"F1={row['f1']:.6f} "
            f"N={row['support']}"
        )
    print("[Eval] ===============================")


def _load_checkpoint(model: nn.Module, checkpoint_path: Path) -> Mapping[str, Any] | Any:
    """체크포인트 포맷 차이를 흡수해 모델 가중치를 로드한다.

    지원 포맷:
    - {"model_state_dict": ...}
    - {"state_dict": ...}
    - raw state_dict
    """
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint_obj

    if isinstance(state, Mapping):
        if "model_state_dict" in state and isinstance(state["model_state_dict"], Mapping):
            state = state["model_state_dict"]
        elif "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = state["state_dict"]

    if not isinstance(state, Mapping):
        raise ConfigError(
            "Unsupported checkpoint format",
            field="checkpoint_path",
            file_path=checkpoint_path,
        )

    cleaned = {str(k).replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    return checkpoint_obj


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """설정 섹션을 안전하게 꺼내는 공통 보조 함수."""
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value


def _resolve_default_checkpoint_path(project_dir: Path) -> Path:
    """project_dir에서 val_f1 상위 체크포인트 파일을 선택한다.

    파일명 규칙: f1_score_loss_score_epoch_n.pth
    예: f1_0.74444_loss_0.43362_epoch_61.pth
    """
    if not project_dir.exists():
        raise ConfigError("Project save directory not found", field="project_dir", file_path=project_dir)

    pattern = re.compile(r"^f1_([0-9]+(?:\.[0-9]+)?)_loss_([0-9]+(?:\.[0-9]+)?)_epoch_([0-9]+)\.pth$")
    candidates: List[Tuple[float, float, int, Path]] = []

    for path in project_dir.glob("*.pth"):
        match = pattern.match(path.name)
        if not match:
            continue
        f1 = float(match.group(1))
        loss = float(match.group(2))
        epoch = int(match.group(3))
        candidates.append((f1, loss, epoch, path))

    if not candidates:
        raise ConfigError(
            "No checkpoint found with expected naming format f1_score_loss_score_epoch_n.pth",
            field="checkpoint_path",
            file_path=project_dir,
        )

    candidates.sort(key=lambda x: (-x[0], x[1], -x[2], str(x[3])))
    return candidates[0][3]


def _save_evaluation_outputs(result: Mapping[str, Any]) -> None:
    """단독 평가 결과의 confusion matrix와 클래스별 지표를 CSV로 저장한다."""
    checkpoint_path = Path(str(result["checkpoint_path"]))
    out_dir = checkpoint_path.parent

    matrix_path = out_dir / "eval_confusion_matrix.csv"
    labels = [str(label) for label in result["labels"]]
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true/pred", *labels])
        for label, row in zip(labels, result["confusion_matrix"]):
            writer.writerow([label, *row])

    per_class_path = out_dir / "eval_per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_index", "class_name", "precision", "recall", "f1", "support"])
        for row in result["per_class"]:
            writer.writerow([
                row["class_index"],
                row["class_name"],
                row["precision"],
                row["recall"],
                row["f1"],
                row["support"],
            ])


def _compute_eval_loss_and_scores(
    model: nn.Module,
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """평가 시 ArcFace와 일반 분류기를 공통 인터페이스로 처리한다."""
    if isinstance(criterion, SubCenterArcFace):
        if not hasattr(model, "get_metric_feature"):
            raise ConfigError("ArcFace model must implement get_metric_feature", field="model")
        features = model.get_metric_feature()
        if features is None:
            raise ConfigError("Metric feature is not available after forward", field="model.get_metric_feature")
        score_logits = criterion.inference_logits(features)
        loss = nn.CrossEntropyLoss()(score_logits, targets)
        return loss, score_logits

    loss = criterion(logits, targets)
    return loss, logits
