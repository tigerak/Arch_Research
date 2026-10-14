from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch.onnx import errors as onnx_errors

from src.classification.methods.arcface import SubCenterArcFace
from src.classification.registry import build_loss, build_model
from src.config_loader import ConfigError, load_yaml_file
from src.dataset_builder import build_eval_loader


CONFIG = {
    # None이면 models/save 아래에서 가장 최근 .pth를 자동 선택.
    "checkpoint_path": "D:\\TAPING_Model\\models\\save\\20260901_TAPING_DiNO_Focal_1\\0.79823_0.23821.pth",

    # None이면 checkpoint 폴더에서 config_*.yaml 자동 탐색.
    "merged_config_path": None,

    # None이면 <checkpoint_stem>.onnx
    "output_path": None,

    # 권장 opset. 필요 시 17/19 등으로 변경 가능.
    "opset_version": 18,

    # None이면 dynamic batch, 정수면 고정 batch.
    "fixed_batch_size": 1,

    # ArcFace 체크포인트면 arcface logits 경로로 내보낼지 여부.
    "prefer_arcface_logits": False, # True | False

    # 내보낸 ONNX를 onnx.checker로 검증(onnx 설치 필요).
    "check_onnx": True,

    # ONNX Runtime로 1회 추론 테스트(onnxruntime 설치 필요).
    "test_onnxruntime": True, # True | False

    # ONNX Runtime로 test_dir 전체 추론 후 보고서 생성.
    "run_eval_report": True,
    "eval_report_txt_name": "onnx_inference_report.txt",
    "eval_report_json_name": "onnx_inference_report.json",
    "eval_report_title": "TAPING 추론 성능 리포트",
    "eval_report_subtitle": "ONNX test_dir 전체 평가",
}


class ArcFaceInferenceWrapper(nn.Module):
    """모델 feature + ArcFace state를 결합해 최종 분류 logits를 반환한다."""

    def __init__(self, backbone_model: nn.Module, arcface_head: SubCenterArcFace) -> None:
        super().__init__()
        self.backbone_model = backbone_model
        self.arcface_head = arcface_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = self.backbone_model(x)
        if not hasattr(self.backbone_model, "get_metric_feature"):
            raise RuntimeError("Model does not provide get_metric_feature for ArcFace export")
        features = self.backbone_model.get_metric_feature()
        if features is None:
            raise RuntimeError("Metric feature is not available after forward")
        return self.arcface_head.inference_logits(features)


def main() -> None:
    cfg = dict(CONFIG)

    checkpoint_path = _resolve_checkpoint_path(cfg.get("checkpoint_path"))
    checkpoint_obj = torch.load(checkpoint_path, map_location="cpu")

    merged_config_path = _resolve_config_path(checkpoint_path, cfg.get("merged_config_path"))
    merged_cfg = load_yaml_file(merged_config_path)

    model = build_model(merged_cfg)
    _load_model_state(model, checkpoint_obj, checkpoint_path)
    model.eval()

    export_model: nn.Module = model
    export_kind = "classifier_logits"

    if bool(cfg.get("prefer_arcface_logits", True)):
        maybe = _build_arcface_wrapper_if_possible(merged_cfg, model, checkpoint_obj)
        if maybe is not None:
            export_model = maybe
            export_kind = "arcface_logits"

    batch_size = cfg.get("fixed_batch_size")
    if batch_size is None:
        batch = 1
        dynamic_axes = {
            "input": {0: "batch"},
            "logits": {0: "batch"},
        }
    else:
        batch = int(batch_size)
        dynamic_axes = None

    input_size = _resolve_input_size(merged_cfg)
    dummy = torch.randn(batch, 3, input_size, input_size, dtype=torch.float32)

    output_path = _resolve_output_path(checkpoint_path, cfg.get("output_path"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _export_onnx_with_fallback(
        model=export_model,
        dummy=dummy,
        output_path=output_path,
        dynamic_axes=dynamic_axes,
        opset_version=int(cfg.get("opset_version", 18)),
    )

    if bool(cfg.get("check_onnx", True)):
        _check_onnx(output_path)

    if bool(cfg.get("test_onnxruntime", False)):
        _test_onnxruntime(output_path, dummy)

    if bool(cfg.get("run_eval_report", True)):
        report = _run_onnx_dataset_eval(output_path, merged_cfg)
        txt_path = output_path.with_name(str(cfg.get("eval_report_txt_name", "onnx_inference_report.txt")))
        json_path = output_path.with_name(str(cfg.get("eval_report_json_name", "onnx_inference_report.json")))
        _save_eval_report_txt(
            report,
            txt_path,
            title=str(cfg.get("eval_report_title", "TAPING 추론 성능 리포트")),
            subtitle=str(cfg.get("eval_report_subtitle", "ONNX test_dir 전체 평가")),
        )
        _save_eval_report_json(report, json_path)
        print(f"[ONNX] eval report(txt): {txt_path}")
        print(f"[ONNX] eval report(json): {json_path}")

    print("[ONNX] Export done")
    print(f"[ONNX] checkpoint: {checkpoint_path}")
    print(f"[ONNX] config: {merged_config_path}")
    print(f"[ONNX] export_kind: {export_kind}")
    print(f"[ONNX] input_shape: ({batch}, 3, {input_size}, {input_size})")
    print(f"[ONNX] output: {output_path}")


def _resolve_checkpoint_path(checkpoint_path: Any) -> Path:
    if checkpoint_path is not None:
        path = Path(str(checkpoint_path))
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = list(Path("models/save").glob("**/*.pth"))
    pattern = re.compile(r"^f1_([0-9]+(?:\.[0-9]+)?)_loss_([0-9]+(?:\.[0-9]+)?)_epoch_([0-9]+)\.pth$")
    candidates = [p for p in candidates if pattern.match(p.name)]
    if not candidates:
        raise FileNotFoundError("No .pth checkpoint found under models/save")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_config_path(checkpoint_path: Path, merged_config_path: Any) -> Path:
    if merged_config_path is not None:
        path = Path(str(merged_config_path))
        if not path.exists():
            raise FileNotFoundError(f"Merged config not found: {path}")
        return path

    cfg_candidates = sorted(checkpoint_path.parent.glob("config_*.yaml"))
    if not cfg_candidates:
        raise FileNotFoundError(f"config_*.yaml not found in: {checkpoint_path.parent}")
    return cfg_candidates[-1]


def _load_model_state(model: nn.Module, checkpoint_obj: Any, checkpoint_path: Path) -> None:
    state = checkpoint_obj
    if isinstance(checkpoint_obj, Mapping):
        if isinstance(checkpoint_obj.get("model_state_dict"), Mapping):
            state = checkpoint_obj["model_state_dict"]
        elif isinstance(checkpoint_obj.get("state_dict"), Mapping):
            state = checkpoint_obj["state_dict"]

    if not isinstance(state, Mapping):
        raise ConfigError(
            "Unsupported checkpoint format",
            field="checkpoint_path",
            file_path=checkpoint_path,
        )

    cleaned = {str(k).replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)


def _build_arcface_wrapper_if_possible(
    merged_cfg: Mapping[str, Any],
    model: nn.Module,
    checkpoint_obj: Any,
) -> ArcFaceInferenceWrapper | None:
    train_cfg = merged_cfg.get("train", {}) if isinstance(merged_cfg.get("train", {}), Mapping) else {}
    loss_name = str(train_cfg.get("loss_name", "")).lower()
    if loss_name != "arcface":
        return None

    if not isinstance(checkpoint_obj, Mapping):
        return None
    arcface_state = checkpoint_obj.get("arcface_state_dict")
    if not isinstance(arcface_state, Mapping):
        return None

    criterion = build_loss(merged_cfg, model=model)
    if not isinstance(criterion, SubCenterArcFace):
        return None
    criterion.load_state_dict(arcface_state, strict=False)
    criterion.eval()
    return ArcFaceInferenceWrapper(model, criterion)


def _resolve_output_path(checkpoint_path: Path, output_path: Any) -> Path:
    if output_path is not None:
        return Path(str(output_path))
    return checkpoint_path.with_suffix(".onnx")


def _resolve_input_size(merged_cfg: Mapping[str, Any]) -> int:
    train_cfg = merged_cfg.get("train", {}) if isinstance(merged_cfg.get("train", {}), Mapping) else {}
    img_size = train_cfg.get("img_size", [224, 224])
    if isinstance(img_size, int):
        return int(img_size)
    if isinstance(img_size, list) and len(img_size) == 2:
        return int(img_size[0])
    raise ValueError(f"Invalid train.img_size: {img_size}")


def _export_onnx_with_fallback(
    *,
    model: nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    dynamic_axes: Dict[str, Dict[int, str]] | None,
    opset_version: int,
) -> None:
    """ONNX export를 수행하고 Transformer fastpath 이슈 시 자동 우회한다."""

    def _do_export() -> None:
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                str(output_path),
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )

    try:
        _do_export()
    except onnx_errors.UnsupportedOperatorError as exc:
        message = str(exc)
        if "aten::_transformer_encoder_layer_fwd" not in message:
            raise

        # TransformerEncoderLayer fastpath를 끄면 ONNX 변환 가능한 연산들로 분해된다.
        if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
            torch.backends.mha.set_fastpath_enabled(False)
            print("[ONNX] Fallback: torch.backends.mha fastpath disabled, retry export")
            _do_export()
            return

        raise


def _check_onnx(onnx_path: Path) -> None:
    try:
        import onnx  # type: ignore
    except Exception as exc:
        raise RuntimeError("onnx package is required for check_onnx=True") from exc

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    print("[ONNX] onnx.checker: OK")


def _test_onnxruntime(onnx_path: Path, dummy: torch.Tensor) -> None:
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        raise RuntimeError("onnxruntime package is required for test_onnxruntime=True") from exc

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(["logits"], {"input": dummy.numpy()})
    print(f"[ONNX] onnxruntime test: OK, output_shape={tuple(outputs[0].shape)}")


def _run_onnx_dataset_eval(onnx_path: Path, merged_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """ONNX Runtime로 test_dir 전체 추론을 수행하고 혼동행렬/지표를 계산한다."""
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:
        raise RuntimeError("onnxruntime package is required for run_eval_report=True") from exc

    loader, idx_to_class = build_eval_loader(merged_cfg)
    labels = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
    n_classes = len(labels)

    conf = torch.zeros((n_classes, n_classes), dtype=torch.long)
    total = 0
    correct = 0

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    expected_batch = _resolve_expected_batch_size(input_shape)

    for images, targets in loader:
        image_np = images.cpu().numpy()
        outputs = _run_onnx_with_batch_adaptation(
            sess=sess,
            output_name=output_name,
            input_name=input_name,
            image_np=image_np,
            expected_batch=expected_batch,
        )
        preds = torch.tensor(outputs).argmax(dim=1).tolist()
        target_list = targets.cpu().tolist()

        for t, p in zip(target_list, preds):
            conf[t, p] += 1
            total += 1
            if t == p:
                correct += 1

    per_class = []
    precisions = []
    f1s = []
    recalls = []
    for i, class_name in enumerate(labels):
        tp = float(conf[i, i].item())
        fp = float(conf[:, i].sum().item() - tp)
        fn = float(conf[i, :].sum().item() - tp)
        support = int(conf[i, :].sum().item())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class.append(
            {
                "class_index": i,
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        precisions.append(precision)
        f1s.append(f1)
        recalls.append(recall)

    accuracy = (correct / total) if total > 0 else 0.0
    macro_precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    macro_f1 = float(sum(f1s) / len(f1s)) if f1s else 0.0
    macro_recall = float(sum(recalls) / len(recalls)) if recalls else 0.0

    return {
        "onnx_path": str(onnx_path),
        "num_samples": int(total),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "labels": labels,
        "confusion_matrix": conf.tolist(),
        "per_class": per_class,
    }


def _resolve_expected_batch_size(input_shape: Any) -> int | None:
    """ONNX 입력 shape에서 정적 batch 크기를 읽는다. 동적이면 None 반환."""
    if not isinstance(input_shape, list) or not input_shape:
        return None
    dim0 = input_shape[0]
    if isinstance(dim0, int) and dim0 > 0:
        return dim0
    return None


def _run_onnx_with_batch_adaptation(
    *,
    sess: Any,
    output_name: str,
    input_name: str,
    image_np: Any,
    expected_batch: int | None,
) -> Any:
    """정적 batch ONNX일 때 입력을 batch 단위로 나눠 추론하고 다시 합친다."""
    if expected_batch is None:
        return sess.run([output_name], {input_name: image_np})[0]

    n = int(image_np.shape[0])
    if n == expected_batch:
        return sess.run([output_name], {input_name: image_np})[0]

    out_list = []
    for start in range(0, n, expected_batch):
        end = min(start + expected_batch, n)
        chunk = image_np[start:end]
        if chunk.shape[0] != expected_batch:
            # 마지막 청크가 작으면 첫 샘플 반복으로 패딩 후, 결과는 원래 길이만 사용.
            pad_count = expected_batch - int(chunk.shape[0])
            pad = chunk[:1].repeat(pad_count, axis=0)
            chunk_in = np.concatenate([chunk, pad], axis=0)
            out = sess.run([output_name], {input_name: chunk_in})[0]
            out_list.append(out[: chunk.shape[0]])
        else:
            out = sess.run([output_name], {input_name: chunk})[0]
            out_list.append(out)

    return np.concatenate(out_list, axis=0)


def _save_eval_report_txt(
    report: Mapping[str, Any],
    txt_path: Path,
    *,
    title: str,
    subtitle: str,
) -> None:
    """사람이 읽기 쉬운 ONNX 추론 txt 보고서를 저장한다."""
    width = 72
    line = "=" * width
    subline = "-" * width

    labels = [str(x) for x in report["labels"]]
    conf = report["confusion_matrix"]
    per_class = report["per_class"]

    # Confusion matrix column widths
    row_name_w = max(len("true/pred"), *(len(name) for name in labels), 10)
    col_w = []
    for j, pred_name in enumerate(labels):
        max_num_w = max(len(str(conf[i][j])) for i in range(len(labels)))
        col_w.append(max(len(pred_name), max_num_w, 8))

    def _cm_row(name: str, values: list[int] | None = None) -> str:
        left = name.rjust(row_name_w)
        if values is None:
            cols = [labels[j].rjust(col_w[j]) for j in range(len(labels))]
        else:
            cols = [str(values[j]).rjust(col_w[j]) for j in range(len(values))]
        return f"{left}  " + "  ".join(cols)

    # Per-class table widths
    class_w = max(len("Class"), *(len(str(r["class_name"])) for r in per_class), len("accuracy"), len("macro"))
    p_w = len("Precision")
    r_w = len("Recall")
    f1_w = len("F1")
    s_w = max(len("Support"), len(str(report["num_samples"])))

    def _metric_row(name: str, p: str, r: str, f1: str, s: str) -> str:
        return (
            f"{name.rjust(class_w)}  "
            f"{p.rjust(p_w)}  "
            f"{r.rjust(r_w)}  "
            f"{f1.rjust(f1_w)}  "
            f"{s.rjust(s_w)}"
        )

    lines = []
    lines.append(line)
    lines.append(title.center(width))
    lines.append(line)
    lines.append("")
    lines.append(subline)
    lines.append(f"  {subtitle}    이미지 {report['num_samples']}장")
    lines.append(subline)
    lines.append("")
    lines.append(f"  onnx_path: {report['onnx_path']}")
    lines.append("")

    lines.append("  [Confusion Matrix]  (행=GT, 열=Pred)")
    lines.append("  " + _cm_row("true/pred"))
    for i, gt_name in enumerate(labels):
        row_vals = [int(v) for v in conf[i]]
        lines.append("  " + _cm_row(gt_name, row_vals))
    lines.append("")

    lines.append(
        "  "
        + _metric_row("Class", "Precision", "Recall", "F1", "Support")
    )
    lines.append("  " + _metric_row("-" * class_w, "-" * p_w, "-" * r_w, "-" * f1_w, "-" * s_w))
    for row in per_class:
        lines.append(
            "  "
            + _metric_row(
                str(row["class_name"]),
                f"{float(row['precision']):.4f}",
                f"{float(row['recall']):.4f}",
                f"{float(row['f1']):.4f}",
                str(int(row["support"])),
            )
        )

    lines.append("  " + _metric_row("-" * class_w, "-" * p_w, "-" * r_w, "-" * f1_w, "-" * s_w))
    lines.append(
        "  "
        + _metric_row(
            "macro",
            f"{float(report['macro_precision']):.4f}",
            f"{float(report['macro_recall']):.4f}",
            f"{float(report['macro_f1']):.4f}",
            str(int(report["num_samples"])),
        )
    )
    lines.append(
        "  "
        + _metric_row(
            "accuracy",
            "",
            "",
            f"{float(report['accuracy']):.4f}",
            str(int(report["num_samples"])),
        )
    )

    lines.append("")
    lines.append(line)

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def _save_eval_report_json(report: Mapping[str, Any], json_path: Path) -> None:
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()