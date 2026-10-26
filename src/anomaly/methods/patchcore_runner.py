from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping

import torch

from src.anomaly.artifacts import resolve_save_root, save_image_score_predictions_csv, save_json_artifact
from src.anomaly.config_utils import get_method_config, get_threshold_config
from src.anomaly.image_level_metrics import summarize_image_level_scores
from src.anomaly.image_loaders import build_image_anomaly_loaders
from src.anomaly.methods.patchcore_components import (
    PatchCoreConfig,
    PatchCoreFeatureExtractor,
    build_patchcore_memory_bank,
    score_patchcore_dataset,
)
from src.anomaly.result_schema import build_anomaly_result
from src.config_loader import ConfigError

"""PatchCore anomaly method 구현.

이 파일은 anomaly task의 첫 번째 baseline method이다.

동작 방식:
- 정상 이미지들로부터 feature embedding을 수집한다.
- memory bank를 구성한다.
- 테스트 이미지에 대해 각 patch feature의 거리 기반 anomaly score를 계산한다.
- anomaly threshold를 정하고 AUROC/PR/AUPRC/accuracy를 산출한다.

이 구현은 "사전 학습된 백본 + 정상 샘플 메모리 뱅크" 방식으로 동작하므로,
다른 anomaly method가 추가될 때에도 같은 구조를 따라 확장할 수 있다.
"""


def run_patchcore(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """PatchCore baseline을 학습하고 평가한다.

    학습 단계에서는 정상 이미지 임베딩을 수집하고
    memory bank를 만들고,
    평가 단계에서는 각 테스트 샘플의 anomaly score를 계산한다.
    """
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = _section(cfg, "data")
    model_cfg = _section(cfg, "model")
    method_cfg = get_method_config(cfg, "patchcore")

    patchcore_cfg = PatchCoreConfig(
        model_name=str(model_cfg.get("backbone_name", model_cfg.get("model_name", "vit_small_patch14_dinov2.lvd142m"))),
        pretrained_path=str(Path(str(data_cfg.get("model_dir", "models/pretrained"))) / str(model_cfg.get("pretrained", ""))),
        feature_layers=tuple(method_cfg.get("feature_layers", [])),
        coreset_sampling_ratio=float(method_cfg.get("coreset_sampling_ratio", 0.1)),
        knn_k=int(method_cfg.get("knn_k", 9)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )

    extractor = PatchCoreFeatureExtractor(patchcore_cfg, device=device)
    image_size = extractor.get_input_image_size()
    train_loader, val_loader, test_loader, idx_to_class, normal_class = build_image_anomaly_loaders(
        cfg,
        image_size_override=image_size,
    )
    save_root = resolve_save_root(cfg)
    max_memory_bank_patches = int(method_cfg.get("max_memory_bank_patches", 50000))
    if max_memory_bank_patches <= 0:
        raise ConfigError(
            "method.patchcore.max_memory_bank_patches must be > 0",
            field="method.patchcore.max_memory_bank_patches",
            details={"value": max_memory_bank_patches},
        )
    memory_bank = build_patchcore_memory_bank(
        extractor,
        train_loader,
        sampling_ratio=patchcore_cfg.coreset_sampling_ratio,
        max_memory_bank_patches=max_memory_bank_patches,
    )

    val_scores, val_labels, _, _ = score_patchcore_dataset(
        extractor,
        memory_bank,
        knn_k=patchcore_cfg.knn_k,
        loader=val_loader,
        idx_to_class=idx_to_class,
        normal_class=normal_class,
        collect_paths=False,
    )
    val_threshold_summary = summarize_image_level_scores(val_labels, val_scores, threshold_cfg=get_threshold_config(cfg))

    predictions_path = save_root / "patchcore_predictions.csv" if save_artifacts else None
    scores, labels, _, _ = score_patchcore_dataset(
        extractor,
        memory_bank,
        knn_k=patchcore_cfg.knn_k,
        loader=test_loader,
        idx_to_class=idx_to_class,
        normal_class=normal_class,
        threshold=val_threshold_summary["threshold"],
        prediction_csv_path=str(predictions_path) if predictions_path is not None else None,
        collect_paths=False,
    )

    score_summary = summarize_image_level_scores(
        labels,
        scores,
        threshold_cfg=get_threshold_config(cfg),
        threshold=val_threshold_summary["threshold"],
    )

    artifacts: Dict[str, str] = {}
    if save_artifacts:
        memory_path = save_root / "patchcore_memory_bank.pt"
        torch.save({"memory_bank": memory_bank.detach().cpu(), "config": dict(cfg)}, memory_path)
        artifacts["memory_bank"] = str(memory_path)

        predictions_path = save_root / "patchcore_predictions.csv"
        artifacts["predictions_csv"] = str(predictions_path)

        metrics_path = save_root / "patchcore_metrics.json"
        metrics_obj = {
            "image_auroc": score_summary["image_auroc"],
            "image_auprc": score_summary["image_auprc"],
            "fpr95": score_summary["fpr95"],
            "accuracy": score_summary["accuracy"],
            "threshold": score_summary["threshold"],
            "num_samples": score_summary["num_samples"],
            "normal_class": normal_class,
            "feature_layers": list(patchcore_cfg.feature_layers),
            "knn_k": patchcore_cfg.knn_k,
            "coreset_sampling_ratio": patchcore_cfg.coreset_sampling_ratio,
            "memory_bank_size": int(memory_bank.shape[0]),
            "max_memory_bank_patches": max_memory_bank_patches,
            "threshold_policy": score_summary["threshold_meta"].get("policy", "tpr95"),
        }
        if "target_tpr" in score_summary["threshold_meta"]:
            metrics_obj["target_tpr"] = float(score_summary["threshold_meta"]["target_tpr"])
        save_json_artifact(metrics_path, metrics_obj)
        artifacts["metrics_json"] = str(metrics_path)

    result = build_anomaly_result(
        method="patchcore",
        primary_metric="image_auroc",
        primary_score=score_summary["image_auroc"],
        save_root=str(save_root),
        artifacts=artifacts,
        started=started,
        metrics={
            "image_auroc": score_summary["image_auroc"],
            "image_auprc": score_summary["image_auprc"],
            "fpr95": score_summary["fpr95"],
            "accuracy": score_summary["accuracy"],
            "threshold": score_summary["threshold"],
            "num_samples": score_summary["num_samples"],
        },
    )
    return result


run = run_patchcore

__all__ = ["run_patchcore", "run"]


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value