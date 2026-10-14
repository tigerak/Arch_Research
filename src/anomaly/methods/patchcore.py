from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from src.anomaly.dataset import build_patchcore_loaders
from src.anomaly.patchcore import (
    PatchCoreConfig,
    PatchCoreExtractor,
    build_memory_bank,
    compute_patch_scores,
)
from src.config_loader import ConfigError


def run_patchcore(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """PatchCore baseline을 학습하고 평가한다."""
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = _section(cfg, "data")
    model_cfg = _section(cfg, "model")
    anomaly_cfg = _section(cfg, "anomaly")

    train_loader, test_loader, idx_to_class, normal_class = build_patchcore_loaders(cfg)

    patchcore_cfg = PatchCoreConfig(
        model_name=str(model_cfg.get("backbone_name", model_cfg.get("model_name", "vit_small_patch14_dinov2.lvd142m"))),
        pretrained_path=str(Path(str(data_cfg.get("model_dir", "models/pretrained"))) / str(model_cfg.get("pretrained", ""))),
        feature_layers=tuple(anomaly_cfg.get("feature_layers", [])),
        coreset_sampling_ratio=float(anomaly_cfg.get("coreset_sampling_ratio", 0.1)),
        knn_k=int(anomaly_cfg.get("knn_k", 9)),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
    )

    extractor = PatchCoreExtractor(patchcore_cfg, device=device)
    train_embeddings = _collect_train_embeddings(extractor, train_loader)
    memory_bank = build_memory_bank(train_embeddings, patchcore_cfg.coreset_sampling_ratio)

    scores, labels, paths, class_names = _score_test_set(
        extractor,
        memory_bank,
        patchcore_cfg.knn_k,
        test_loader,
        idx_to_class,
        normal_class,
    )

    image_auroc = _binary_roc_auc(labels, scores)
    image_auprc = _average_precision(labels, scores)
    fpr95 = _fpr_at_tpr(labels, scores, target_tpr=0.95)
    threshold = _threshold_at_tpr(labels, scores, target_tpr=0.95)
    predictions = [1 if score >= threshold else 0 for score in scores]
    accuracy = sum(int(p == y) for p, y in zip(predictions, labels)) / max(1, len(labels))

    project_name = str(cfg.get("project_name", "PatchCore_Experiment"))
    save_root = Path(str(data_cfg.get("save_dir", "models/save"))) / project_name
    save_root.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    if save_artifacts:
        memory_path = save_root / "patchcore_memory_bank.pt"
        torch.save({"memory_bank": memory_bank.detach().cpu(), "config": dict(cfg)}, memory_path)
        artifacts["memory_bank"] = str(memory_path)

        predictions_path = save_root / "patchcore_predictions.csv"
        _save_predictions_csv(predictions_path, paths, class_names, labels, scores, predictions, threshold)
        artifacts["predictions_csv"] = str(predictions_path)

        metrics_path = save_root / "patchcore_metrics.json"
        metrics_obj = {
            "image_auroc": image_auroc,
            "image_auprc": image_auprc,
            "fpr95": fpr95,
            "accuracy": accuracy,
            "threshold": threshold,
            "num_samples": len(labels),
            "normal_class": normal_class,
            "feature_layers": list(patchcore_cfg.feature_layers),
            "knn_k": patchcore_cfg.knn_k,
            "coreset_sampling_ratio": patchcore_cfg.coreset_sampling_ratio,
        }
        metrics_path.write_text(json.dumps(metrics_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["metrics_json"] = str(metrics_path)

    result = {
        "status": "SUCCESS",
        "task_type": "anomaly_detection",
        "method": "patchcore",
        "primary_metric": "image_auroc",
        "primary_score": image_auroc,
        "image_auroc": image_auroc,
        "image_auprc": image_auprc,
        "fpr95": fpr95,
        "accuracy": accuracy,
        "threshold": threshold,
        "num_samples": len(labels),
        "save_root": str(save_root),
        "artifacts": artifacts,
        "elapsed_sec": round(perf_counter() - started, 3),
    }
    return result


run = run_patchcore

__all__ = ["run_patchcore", "run"]


@torch.no_grad()
def _collect_train_embeddings(extractor: PatchCoreExtractor, loader: DataLoader) -> torch.Tensor:
    features = []
    for batch in loader:
        images = batch[0]
        embeddings = extractor.embed(images)
        features.append(embeddings.detach().cpu())
    if not features:
        raise ConfigError("Training loader is empty", field="data.train_normal_dir")
    return torch.cat(features, dim=0)


@torch.no_grad()
def _score_test_set(
    extractor: PatchCoreExtractor,
    memory_bank: torch.Tensor,
    knn_k: int,
    loader: DataLoader,
    idx_to_class: Mapping[int, str],
    normal_class: str,
) -> tuple[list[float], list[int], list[str], list[str]]:
    scores: list[float] = []
    labels: list[int] = []
    paths: list[str] = []
    class_names: list[str] = []

    device = next(extractor.parameters()).device
    memory_bank = memory_bank.to(device)

    for batch in loader:
        images, class_idx, batch_paths = batch
        query = extractor.hooked(images.to(device))
        patch_tokens = [extractor._feature_to_patches(feature) for feature in query]
        patch_tokens = [torch.nn.functional.normalize(tokens, dim=-1) for tokens in patch_tokens]
        embedded = torch.cat(patch_tokens, dim=-1)
        batch_scores = compute_patch_scores(embedded, memory_bank, knn_k)

        for score, target, path in zip(batch_scores.detach().cpu().tolist(), class_idx.tolist(), batch_paths):
            class_name = idx_to_class[int(target)]
            labels.append(0 if class_name == normal_class else 1)
            scores.append(float(score))
            paths.append(str(path))
            class_names.append(class_name)

    return scores, labels, paths, class_names


def _save_predictions_csv(
    path: Path,
    image_paths: Sequence[str],
    class_names: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[int],
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "class_name", "binary_label", "score", "prediction", "threshold"],
        )
        writer.writeheader()
        for image_path, class_name, label, score, pred in zip(image_paths, class_names, labels, scores, predictions):
            writer.writerow(
                {
                    "image_path": image_path,
                    "class_name": class_name,
                    "binary_label": int(label),
                    "score": float(score),
                    "prediction": int(pred),
                    "threshold": float(threshold),
                }
            )


def _binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return 0.0

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label == 1:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def _threshold_at_tpr(labels: Sequence[int], scores: Sequence[float], target_tpr: float = 0.95) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    if not positives:
        return max(scores) if scores else 0.0
    positives = sorted(positives, reverse=True)
    index = min(len(positives) - 1, max(0, int((1.0 - target_tpr) * len(positives))))
    return positives[index]


def _fpr_at_tpr(labels: Sequence[int], scores: Sequence[float], target_tpr: float = 0.95) -> float:
    threshold = _threshold_at_tpr(labels, scores, target_tpr=target_tpr)
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not negatives:
        return 0.0
    false_positive = sum(1 for score in negatives if score >= threshold)
    return false_positive / len(negatives)


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
