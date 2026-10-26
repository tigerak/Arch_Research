from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping

import torch

from src.anomaly.artifacts import resolve_save_root, save_image_score_predictions_csv, save_json_artifact
from src.anomaly.config_utils import get_method_config, get_threshold_config
from src.anomaly.image_level_metrics import summarize_image_level_scores
from src.anomaly.image_loaders import build_image_anomaly_loaders
from src.anomaly.methods.dinomaly2_components import Dinomaly2Config, Dinomaly2FeatureModel, score_dinomaly2_dataset, train_dinomaly2_model
from src.anomaly.result_schema import build_anomaly_result
from src.config_loader import ConfigError
from src.data.tensor_augmentations import build_tensor_gpu_augmentation


def run_dinomaly2(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = _section(cfg, "data")
    model_cfg = _section(cfg, "model")
    train_cfg = _section(cfg, "train")
    method_cfg = get_method_config(cfg, "dinomaly2")
    use_gpu_augment = bool(train_cfg.get("gpu_augment", False)) and device.type == "cuda"

    dinomaly2_cfg = Dinomaly2Config(
        model_name=str(model_cfg.get("backbone_name", model_cfg.get("model_name", "vit_small_patch14_dinov2.lvd142m"))),
        pretrained_path=str(Path(str(data_cfg.get("model_dir", "models/pretrained"))) / str(model_cfg.get("pretrained", ""))),
        feature_layers=tuple(method_cfg.get("feature_layers", [])),
        bottleneck_dim=int(method_cfg.get("bottleneck_dim", 256)),
        dropout=float(method_cfg.get("dropout", 0.1)),
        epochs=int(method_cfg.get("epochs", 5)),
        lr=float(method_cfg.get("lr", 1e-4)),
        weight_decay=float(method_cfg.get("weight_decay", 1e-4)),
        score_aggregation=str(method_cfg.get("score_aggregation", "max")).strip().lower(),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )

    if not dinomaly2_cfg.feature_layers:
        raise ConfigError("method.dinomaly2.feature_layers must be a non-empty list", field="method.dinomaly2.feature_layers")

    model = Dinomaly2FeatureModel(dinomaly2_cfg, device=device)
    image_size = model.get_input_image_size()
    train_loader, val_loader, test_loader, idx_to_class, normal_class = build_image_anomaly_loaders(
        cfg,
        image_size_override=image_size,
        use_gpu_augment=use_gpu_augment,
    )
    save_root = resolve_save_root(cfg)
    train_tensor_transform = build_tensor_gpu_augmentation(mode="train") if use_gpu_augment else None
    eval_tensor_transform = build_tensor_gpu_augmentation(mode="eval") if use_gpu_augment else None
    profile_steps = bool(train_cfg.get("profile_steps", False)) or bool(train_cfg.get("profile_transforms", False))
    history = train_dinomaly2_model(
        model,
        train_loader,
        epochs=dinomaly2_cfg.epochs,
        lr=dinomaly2_cfg.lr,
        weight_decay=dinomaly2_cfg.weight_decay,
        tensor_transform=train_tensor_transform,
        profile_steps=profile_steps,
    )

    val_scores, val_labels, _, _ = score_dinomaly2_dataset(
        model,
        val_loader,
        idx_to_class=idx_to_class,
        normal_class=normal_class,
        aggregation=dinomaly2_cfg.score_aggregation,
        tensor_transform=eval_tensor_transform,
        collect_paths=False,
    )
    val_threshold_summary = summarize_image_level_scores(val_labels, val_scores, threshold_cfg=get_threshold_config(cfg))

    predictions_path = save_root / "dinomaly2_predictions.csv" if save_artifacts else None
    scores, labels, _, _ = score_dinomaly2_dataset(
        model,
        test_loader,
        idx_to_class=idx_to_class,
        normal_class=normal_class,
        aggregation=dinomaly2_cfg.score_aggregation,
        tensor_transform=eval_tensor_transform,
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
        checkpoint_path = save_root / "dinomaly2_student_heads.pt"
        torch.save(
            {
                "student_heads": model.student_heads.state_dict(),
                "config": dict(cfg),
                "history": history,
            },
            checkpoint_path,
        )
        artifacts["checkpoint"] = str(checkpoint_path)

        history_path = save_root / "dinomaly2_history.json"
        save_json_artifact(history_path, {"history": history})
        artifacts["history_json"] = str(history_path)

        predictions_path = save_root / "dinomaly2_predictions.csv"
        artifacts["predictions_csv"] = str(predictions_path)

        metrics_path = save_root / "dinomaly2_metrics.json"
        metrics_obj = {
            "image_auroc": score_summary["image_auroc"],
            "image_auprc": score_summary["image_auprc"],
            "fpr95": score_summary["fpr95"],
            "accuracy": score_summary["accuracy"],
            "threshold": score_summary["threshold"],
            "num_samples": score_summary["num_samples"],
            "normal_class": normal_class,
            "feature_layers": list(dinomaly2_cfg.feature_layers),
            "bottleneck_dim": dinomaly2_cfg.bottleneck_dim,
            "dropout": dinomaly2_cfg.dropout,
            "epochs": dinomaly2_cfg.epochs,
            "lr": dinomaly2_cfg.lr,
            "weight_decay": dinomaly2_cfg.weight_decay,
            "score_aggregation": dinomaly2_cfg.score_aggregation,
            "threshold_policy": score_summary["threshold_meta"].get("policy", "tpr95"),
        }
        if "target_tpr" in score_summary["threshold_meta"]:
            metrics_obj["target_tpr"] = float(score_summary["threshold_meta"]["target_tpr"])
        save_json_artifact(metrics_path, metrics_obj)
        artifacts["metrics_json"] = str(metrics_path)

    return build_anomaly_result(
        method="dinomaly2",
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


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value


run = run_dinomaly2

__all__ = ["run_dinomaly2", "run"]