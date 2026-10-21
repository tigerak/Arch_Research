from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple


def compute_threshold(labels: Sequence[int], scores: Sequence[float], threshold_cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    policy = str(threshold_cfg.get("policy", "tpr95")).strip().lower()
    if policy == "fixed":
        value = float(threshold_cfg["value"])
        return value, {"policy": "fixed", "value": value}

    if policy == "none":
        value = max(scores) if scores else 0.0
        return value, {"policy": "none", "value": value}

    target_tpr = float(threshold_cfg.get("target_tpr", 0.95))
    target_fpr = float(threshold_cfg.get("target_fpr", 0.05))

    if policy in {"normal_percentile", "percentile", "val_percentile"}:
        value = threshold_at_percentile(scores, target_fpr=target_fpr)
        return value, {"policy": policy, "value": value, "target_fpr": target_fpr}

    value = threshold_at_tpr(labels, scores, target_tpr=target_tpr)
    return value, {"policy": "tpr95", "value": value, "target_tpr": target_tpr}


def threshold_at_tpr(labels: Sequence[int], scores: Sequence[float], target_tpr: float = 0.95) -> float:
    positives = [score for score, label in zip(scores, labels) if label == 1]
    if not positives:
        return threshold_at_percentile(scores, target_fpr=max(0.01, 1.0 - target_tpr))
    positives = sorted(positives, reverse=True)
    index = min(len(positives) - 1, max(0, int((1.0 - target_tpr) * len(positives))))
    return positives[index]


def threshold_at_percentile(scores: Sequence[float], target_fpr: float = 0.05) -> float:
    if not scores:
        return 0.0
    ordered = sorted(float(score) for score in scores)
    if len(ordered) == 1:
        return ordered[0]
    q = min(0.9999, max(0.0, 1.0 - target_fpr))
    index = min(len(ordered) - 1, max(0, int(q * len(ordered))))
    return ordered[index]


def fpr_at_threshold(labels: Sequence[int], scores: Sequence[float], threshold: float) -> float:
    negatives = [score for score, label in zip(scores, labels) if label == 0]
    if not negatives:
        return 0.0
    false_positive = sum(1 for score in negatives if score >= threshold)
    return false_positive / len(negatives)