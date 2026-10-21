from __future__ import annotations

from typing import Any, Dict, Sequence

from src.anomaly.thresholding import compute_threshold, fpr_at_threshold


def summarize_image_level_scores(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    threshold_cfg: Dict[str, Any],
    threshold: float | None = None,
) -> Dict[str, Any]:
    if threshold is None:
        threshold, threshold_meta = compute_threshold(labels, scores, threshold_cfg)
    else:
        threshold_meta = {"policy": "provided", "value": float(threshold)}
    predictions = [1 if score >= threshold else 0 for score in scores]
    accuracy = sum(int(pred == label) for pred, label in zip(predictions, labels)) / max(1, len(labels))
    return {
        "image_auroc": binary_roc_auc(labels, scores),
        "image_auprc": average_precision(labels, scores),
        "fpr95": fpr_at_threshold(labels, scores, threshold),
        "accuracy": accuracy,
        "threshold": threshold,
        "threshold_meta": threshold_meta,
        "predictions": predictions,
        "num_samples": len(labels),
    }


def binary_roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
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


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
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