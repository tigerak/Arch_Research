from __future__ import annotations

import csv
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

"""classification 학습 엔진의 핵심 구현 모듈.

이 파일은 실제 학습 루프를 구성하는 가장 중요한 위치이다.

역할:
- cfg를 읽어 모델, 손실, 옵티마이저, 스케줄러를 생성한다.
- train/valid 데이터로더를 준비한다.
- epoch 단위 학습 루프를 실행한다.
- 기준 metric 기반 체크포인트 저장을 결정한다.
- confusion matrix, history, checkpoint 경로를 함께 관리한다.

실험 포털에서 classification task의 실제 실행 중심이 이 파일에 모여 있다.
"""

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from src.classification.methods.arcface import SubCenterArcFace
from src.classification.registry import build_arcface_optimizer, build_loss, build_model, build_optimizer, build_scheduler
from src.config_loader import ConfigError
from src.dataset_builder import build_train_val_loaders


def train_one_experiment(cfg: Mapping[str, Any], logger: logging.Logger | None = None) -> Dict[str, Any]:
    """단일 실험 학습을 실행하고 최종 요약 정보를 반환한다.

    저장 규칙(요청사항 반영):
    - train.save_warmup_epochs 이전에는 체크포인트 저장 금지
    - 이후에는 비교 지표를 확인해서 저장
    - 기본 비교 규칙은 train.save_compare_metrics / train.save_compare_modes로 제어

    예시 기본값:
    - metrics: [val_loss, val_f1]
    - modes:   [min, max]
    """
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")

    seed = int(train_cfg.get("seed", 42))
    _set_global_seed(seed, logger=logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    criterion = build_loss(cfg, model=model)
    if isinstance(criterion, SubCenterArcFace):
        criterion = criterion.to(device)

    train_loader, val_loader, idx_to_class = build_train_val_loaders(cfg)
    _log_dataset_distribution(train_loader, "train", idx_to_class, logger)
    _log_dataset_distribution(val_loader, "valid", idx_to_class, logger)

    if isinstance(criterion, SubCenterArcFace):
        optimizer = build_arcface_optimizer(cfg, model, criterion)
    else:
        optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer, steps_per_epoch=len(train_loader))

    max_epoch = int(train_cfg.get("max_epoch", 1))
    accumulation_steps = max(1, int(train_cfg.get("accumulation_steps", 1)))
    use_amp = bool(train_cfg.get("amp", False)) and torch.cuda.is_available()
    show_progress = bool(train_cfg.get("progress_bar", True))
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = _build_grad_scaler(use_amp=use_amp)

    project_name = str(cfg.get("project_name", "experiment"))
    save_root = Path(str(data_cfg.get("save_dir", "models/save"))) / project_name
    save_root.mkdir(parents=True, exist_ok=True)

    # 저장 관련 옵션
    # save_warmup_epochs=15면 1~15 epoch은 저장하지 않고, 16 epoch부터 저장 후보로 판단한다.
    save_warmup_epochs = int(train_cfg.get("save_warmup_epochs", 15))
    compare_metrics = train_cfg.get("save_compare_metrics", ["val_loss", "val_f1"])
    compare_modes = train_cfg.get("save_compare_modes", ["min", "max"])

    if not isinstance(compare_metrics, list) or not isinstance(compare_modes, list):
        raise ConfigError(
            "save_compare_metrics and save_compare_modes must be list",
            field="train.save_compare_metrics/train.save_compare_modes",
        )
    if len(compare_metrics) != len(compare_modes):
        raise ConfigError(
            "save_compare_metrics length must match save_compare_modes length",
            field="train.save_compare_metrics/train.save_compare_modes",
            details={"metrics_len": len(compare_metrics), "modes_len": len(compare_modes)},
        )

    best_values: Dict[str, float] = {}
    for metric_name, mode in zip(compare_metrics, compare_modes):
        mode_l = str(mode).lower()
        if mode_l not in {"min", "max"}:
            raise ConfigError(
                "save_compare_modes item must be min or max",
                field="train.save_compare_modes",
                details={"value": mode},
            )
        best_values[str(metric_name)] = float("inf") if mode_l == "min" else float("-inf")

    history: List[Dict[str, float]] = []
    best_ckpt_path: Path | None = None
    top_k = int(train_cfg.get("save_top_k", 5))
    if top_k <= 0:
        raise ConfigError("train.save_top_k must be >= 1", field="train.save_top_k", details={"value": top_k})
    topk_ckpts: List[Dict[str, Any]] = []

    for epoch in range(1, max_epoch + 1):
        train_loss, train_acc = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
            amp_device_type=amp_device_type,
            scaler=scaler,
            train_mode=True,
            progress_desc=f"Train {epoch:03d}/{max_epoch:03d}",
            show_progress=show_progress,
        )

        val_loss, val_acc, val_recall, val_f1, val_conf = evaluate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            progress_desc=f"Valid {epoch:03d}/{max_epoch:03d}",
            show_progress=show_progress,
        )

        _save_confusion_outputs(
            save_root=save_root,
            file_stem="val_confusion_latest",
            conf=val_conf,
            idx_to_class=idx_to_class,
            epoch=epoch,
        )

        # OneCycle은 iteration마다 step이 더 일반적이지만,
        # 1차 구현에서는 스케줄러 타입별 공통 동작을 위해 epoch 단위 step으로 통일한다.
        scheduler.step()

        pruned_centers = _maybe_prune_arcface_subcenters(criterion, train_cfg, epoch, logger)

        metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
        }
        history.append(metrics)

        should_save = False
        improve_reasons: List[str] = []

        if epoch > save_warmup_epochs:
            for metric_name, mode in zip(compare_metrics, compare_modes):
                key = str(metric_name)
                mode_l = str(mode).lower()
                if key not in metrics:
                    raise ConfigError(
                        "Requested comparison metric is not computed",
                        field="train.save_compare_metrics",
                        details={"metric": key, "available": sorted(metrics.keys())},
                    )

                current = float(metrics[key])
                best_prev = float(best_values[key])
                improved = (current < best_prev) if mode_l == "min" else (current > best_prev)

                if improved:
                    should_save = True
                    improve_reasons.append(f"{key}:{best_prev:.6f}->{current:.6f}")
                    best_values[key] = current

        if epoch > save_warmup_epochs:
            should_save = _is_topk_candidate(topk_ckpts, val_f1=val_f1, val_loss=val_loss, top_k=top_k)

        if should_save:
            ckpt_obj = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                "best_values": best_values,
                "idx_to_class": idx_to_class,
                "compare_metrics": compare_metrics,
                "compare_modes": compare_modes,
                "save_warmup_epochs": save_warmup_epochs,
            }
            if isinstance(criterion, SubCenterArcFace):
                ckpt_obj["arcface_state_dict"] = criterion.state_dict()

            ckpt_name = f"f1_{val_f1:.5f}_loss_{val_loss:.5f}_epoch_{epoch}.pth"
            ckpt_path = save_root / ckpt_name
            torch.save(ckpt_obj, ckpt_path)

            topk_ckpts = _update_topk_ckpts(
                topk_ckpts,
                ckpt_path=ckpt_path,
                val_f1=val_f1,
                val_loss=val_loss,
                top_k=top_k,
            )

            _save_confusion_outputs(
                save_root=save_root,
                file_stem="val_confusion_best",
                conf=val_conf,
                idx_to_class=idx_to_class,
                epoch=epoch,
            )

        msg = (
            f"[Epoch {epoch:03d}/{max_epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_recall={val_recall:.4f} val_f1={val_f1:.4f} "
            f"save={'Y' if should_save else 'N'} prune={pruned_centers}"
        )
        _emit_log(msg, logger=logger)

        per_class_f1 = _per_class_f1_from_confusion(val_conf, idx_to_class)
        class_parts = []
        for class_idx in sorted(per_class_f1.keys()):
            class_name = idx_to_class.get(class_idx, str(class_idx))
            class_f1 = per_class_f1[class_idx]
            class_parts.append(f"{class_name}={class_f1:.4f}")

        class_msg = f"[ValClassF1][Epoch {epoch:03d}] " + " | ".join(class_parts)
        _emit_log(class_msg, logger=logger)

        conf_text = _format_confusion_table(val_conf, idx_to_class)
        _emit_log("\n" + conf_text, logger=logger)

        if should_save:
            rank = next((i + 1 for i, item in enumerate(topk_ckpts) if item["path"] == ckpt_path), -1)
            ckpt_msg = (
                f"[Checkpoint] Saved: {ckpt_path} | "
                f"rank={rank}/{top_k} | reason={', '.join(improve_reasons)}"
            )
            _emit_log(ckpt_msg, logger=logger)

    history_path = save_root / "history.pt"
    torch.save(history, history_path)

    if topk_ckpts:
        best_ckpt_path = topk_ckpts[0]["path"]

    return {
        "project_name": project_name,
        "save_root": str(save_root),
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path is not None else "",
        "history_path": str(history_path),
        "epochs": max_epoch,
        "save_warmup_epochs": save_warmup_epochs,
        "compare_metrics": compare_metrics,
        "compare_modes": compare_modes,
    }


def _is_topk_candidate(topk_ckpts: List[Dict[str, Any]], *, val_f1: float, val_loss: float, top_k: int) -> bool:
    """현재 epoch 결과가 val_f1 기준 top-k 저장 후보인지 판단한다."""
    if len(topk_ckpts) < top_k:
        return True
    worst = topk_ckpts[-1]
    worst_f1 = float(worst["val_f1"])
    worst_loss = float(worst["val_loss"])
    if val_f1 > worst_f1:
        return True
    if val_f1 == worst_f1 and val_loss < worst_loss:
        return True
    return False


def _update_topk_ckpts(
    topk_ckpts: List[Dict[str, Any]],
    *,
    ckpt_path: Path,
    val_f1: float,
    val_loss: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    """top-k 체크포인트 목록을 val_f1 내림차순으로 유지한다."""
    updated = [item for item in topk_ckpts if Path(item["path"]) != ckpt_path]
    updated.append({"path": ckpt_path, "val_f1": float(val_f1), "val_loss": float(val_loss)})
    updated.sort(key=lambda item: (-float(item["val_f1"]), float(item["val_loss"]), str(item["path"])))

    removed = updated[top_k:]
    kept = updated[:top_k]

    for item in removed:
        p = Path(item["path"])
        if p.exists():
            p.unlink()

    return kept


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    accumulation_steps: int,
    use_amp: bool,
    amp_device_type: str,
    scaler: Any,
    train_mode: bool,
    progress_desc: str,
    show_progress: bool,
) -> Tuple[float, float]:
    """학습 또는 검증 모드 한 epoch 실행(정확도/손실).

    train_mode=True일 때만 역전파와 optimizer 업데이트를 수행한다.
    """
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    iterable = _make_progress_iter(loader, desc=progress_desc, enabled=show_progress)

    for step, (images, targets) in enumerate(iterable, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                logits = model(images)
                loss, score_logits = _compute_loss_and_scores(model, criterion, logits, targets)

            if train_mode:
                scaled_loss = loss / accumulation_steps
                scaler.scale(scaled_loss).backward()

                if step % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * images.size(0)
        preds = torch.argmax(score_logits, dim=1)
        total_correct += int((preds == targets).sum().item())
        total_samples += int(images.size(0))

    if train_mode and len(loader) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc


def evaluate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress_desc: str,
    show_progress: bool,
) -> Tuple[float, float, float, float, torch.Tensor]:
    """검증 epoch를 실행하고 loss/accuracy/macro_recall/macro_f1를 계산한다."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    n_classes = None
    conf = None

    iterable = _make_progress_iter(loader, desc=progress_desc, enabled=show_progress)

    with torch.no_grad():
        for images, targets in iterable:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(images)
            loss, score_logits = _compute_loss_and_scores(model, criterion, logits, targets)

            preds = torch.argmax(score_logits, dim=1)

            if n_classes is None:
                n_classes = int(score_logits.shape[1])
                conf = torch.zeros((n_classes, n_classes), dtype=torch.long)

            for t, p in zip(targets.cpu(), preds.cpu()):
                conf[int(t), int(p)] += 1

            total_loss += float(loss.item()) * images.size(0)
            total_correct += int((preds == targets).sum().item())
            total_samples += int(images.size(0))

    if conf is None:
        raise ConfigError("Validation loader is empty", field="data.test_dir")

    recall_macro, f1_macro = _macro_recall_f1_from_confusion(conf)

    avg_loss = total_loss / max(1, total_samples)
    acc = total_correct / max(1, total_samples)
    return avg_loss, acc, recall_macro, f1_macro, conf


def _macro_recall_f1_from_confusion(conf: torch.Tensor) -> Tuple[float, float]:
    """confusion matrix에서 macro recall과 macro f1을 계산한다.

    클래스별 precision/recall/f1를 구한 뒤 단순 평균한다.
    분모 0인 경우는 해당 클래스 지표를 0으로 처리한다.
    """
    n_classes = int(conf.shape[0])

    recalls: List[float] = []
    f1s: List[float] = []

    for i in range(n_classes):
        tp = float(conf[i, i].item())
        fp = float(conf[:, i].sum().item() - tp)
        fn = float(conf[i, :].sum().item() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        recalls.append(recall)
        f1s.append(f1)

    recall_macro = sum(recalls) / max(1, len(recalls))
    f1_macro = sum(f1s) / max(1, len(f1s))
    return recall_macro, f1_macro


def _per_class_f1_from_confusion(conf: torch.Tensor, idx_to_class: Mapping[int, str]) -> Dict[int, float]:
    """confusion matrix에서 클래스별 f1를 계산한다."""
    per_class_f1: Dict[int, float] = {}
    for class_idx in sorted(idx_to_class.keys()):
        tp = float(conf[class_idx, class_idx].item())
        fp = float(conf[:, class_idx].sum().item() - tp)
        fn = float(conf[class_idx, :].sum().item() - tp)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        per_class_f1[class_idx] = f1
    return per_class_f1


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """설정 섹션을 안전하게 꺼내는 공통 보조 함수."""
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value


def _log_dataset_distribution(
    loader: DataLoader,
    split_name: str,
    idx_to_class: Mapping[int, str],
    logger: logging.Logger | None,
) -> None:
    """데이터셋 클래스 분포를 로그로 남긴다."""
    dataset = getattr(loader, "dataset", None)
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return

    counts: Dict[int, int] = {}
    for target in targets:
        key = int(target)
        counts[key] = counts.get(key, 0) + 1

    total = sum(counts.values())
    lines = [f"[Data] {split_name} samples={total}"]
    for class_idx in sorted(idx_to_class.keys()):
        class_name = idx_to_class[class_idx]
        count = counts.get(class_idx, 0)
        ratio = (count / total) if total > 0 else 0.0
        lines.append(f"[Data] {split_name} class={class_name} count={count} ratio={ratio:.4f}")

    for line in lines:
        if logger is not None:
            logger.info(line)
        else:
            print(line)


def _save_confusion_outputs(
    *,
    save_root: Path,
    file_stem: str,
    conf: torch.Tensor,
    idx_to_class: Mapping[int, str],
    epoch: int,
) -> None:
    """confusion matrix와 클래스별 지표를 CSV로 저장한다."""
    labels = [idx_to_class[idx] for idx in sorted(idx_to_class.keys())]

    matrix_path = save_root / f"{file_stem}.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", epoch])
        writer.writerow(["true/pred", *labels])
        for row_idx, label in enumerate(labels):
            writer.writerow([label, *conf[row_idx].tolist()])

    per_class_path = save_root / f"{file_stem}_per_class.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "class_index", "class_name", "precision", "recall", "f1", "support"])
        for class_idx in sorted(idx_to_class.keys()):
            tp = float(conf[class_idx, class_idx].item())
            fp = float(conf[:, class_idx].sum().item() - tp)
            fn = float(conf[class_idx, :].sum().item() - tp)
            support = int(conf[class_idx, :].sum().item())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            writer.writerow([epoch, class_idx, idx_to_class[class_idx], precision, recall, f1, support])


def _format_confusion_table(conf: torch.Tensor, idx_to_class: Mapping[int, str]) -> str:
    """혼동행렬을 로그 친화적인 고정폭 멀티라인 텍스트로 변환한다."""
    class_indices = sorted(idx_to_class.keys())
    labels = [idx_to_class[idx] for idx in class_indices]
    if not labels:
        return "[ValConfusion] <empty>"

    label_width = max(12, max(len(label) for label in labels))
    value_width = max(6, max(len(label) for label in labels))

    lines: List[str] = ["[ValConfusion]"]
    header = " " * (label_width + 2) + " ".join(f"{label:>{value_width}}" for label in labels)
    lines.append(header)

    for row_pos, class_idx in enumerate(class_indices):
        row_label = idx_to_class[class_idx]
        row_vals = conf[row_pos].tolist()
        row_text = f"{row_label:<{label_width}}  " + " ".join(f"{int(v):>{value_width}d}" for v in row_vals)
        lines.append(row_text)

    return "\n".join(lines)


def _emit_log(message: str, *, logger: logging.Logger | None) -> None:
    """logger가 있으면 logger만, 없으면 print만 사용해 중복 출력을 방지한다."""
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def _make_progress_iter(loader: DataLoader, *, desc: str, enabled: bool):
    """tqdm 사용 가능 시 진행률 바를 반환하고, 아니면 원본 iterable을 반환한다."""
    if not enabled or tqdm is None:
        return loader
    return tqdm(loader, desc=desc, leave=False)


def _compute_loss_and_scores(
    model: nn.Module,
    criterion: nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """일반 분류와 ArcFace를 공통 인터페이스로 처리한다."""
    if isinstance(criterion, SubCenterArcFace):
        if not hasattr(model, "get_metric_feature"):
            raise ConfigError("ArcFace model must implement get_metric_feature", field="model")
        features = model.get_metric_feature()
        if features is None:
            raise ConfigError("Metric feature is not available after forward", field="model.get_metric_feature")
        loss = criterion(features, targets)
        score_logits = criterion.inference_logits(features)
        return loss, score_logits

    loss = criterion(logits, targets)
    return loss, logits


def _maybe_prune_arcface_subcenters(
    criterion: nn.Module,
    train_cfg: Mapping[str, Any],
    epoch: int,
    logger: logging.Logger | None,
) -> int:
    """설정된 경우 ArcFace sub-center pruning을 수행한다."""
    if not isinstance(criterion, SubCenterArcFace):
        return 0

    interval = int(train_cfg.get("arcface_prune_interval", 0))
    if interval <= 0 or epoch % interval != 0:
        return 0

    removed = criterion.prune_unused_subcenters(
        warmup_epochs=int(train_cfg.get("arcface_prune_warmup_epochs", 3)),
        current_epoch=epoch,
        min_usage=int(train_cfg.get("arcface_prune_min_usage", 1)),
        max_remove_per_class=int(train_cfg.get("arcface_prune_max_remove_per_class", 1)),
    )
    if removed > 0:
        msg = f"[ArcFace] pruned_sub_centers={removed} at epoch={epoch}"
        print(msg)
        if logger is not None:
            logger.info(msg)
    return removed


def _build_grad_scaler(use_amp: bool) -> Any:
    """PyTorch 버전에 맞춰 GradScaler를 생성한다."""
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        return torch.amp.GradScaler(enabled=use_amp)


def _set_global_seed(seed: int, logger: logging.Logger | None = None) -> None:
    """실험 재현성을 위해 random/numpy/torch seed를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CuDNN 비결정성 최적화를 끄고 결정적 알고리즘을 우선 사용한다.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    msg = f"[Seed] global_seed={seed}"
    print(msg)
    if logger is not None:
        logger.info(msg)
