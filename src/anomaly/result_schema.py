from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Mapping

from src.config_loader import ConfigError

REQUIRED_RESULT_KEYS = (
    "status",
    "task_type",
    "method",
    "primary_metric",
    "primary_score",
    "save_root",
    "artifacts",
    "elapsed_sec",
)


def build_anomaly_result(
    *,
    method: str,
    primary_metric: str,
    primary_score: float,
    save_root: str,
    artifacts: Mapping[str, str],
    metrics: Mapping[str, Any] | None = None,
    started: float | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "status": "SUCCESS",
        "task_type": "anomaly_detection",
        "method": method,
        "primary_metric": primary_metric,
        "primary_score": float(primary_score),
        "save_root": str(save_root),
        "artifacts": dict(artifacts),
        "elapsed_sec": round(perf_counter() - started, 3) if started is not None else 0.0,
    }
    if metrics:
        result.update(dict(metrics))
    return result


def validate_anomaly_result(result: Mapping[str, Any]) -> Dict[str, Any]:
    missing = [key for key in REQUIRED_RESULT_KEYS if key not in result]
    if missing:
        raise ConfigError(
            "Anomaly method runner returned incomplete result",
            field="anomaly.result",
            details={"missing_keys": missing},
        )
    normalized = dict(result)
    if normalized.get("task_type") != "anomaly_detection":
        raise ConfigError(
            "Anomaly method runner returned invalid task_type",
            field="anomaly.result.task_type",
            details={"value": normalized.get("task_type")},
        )
    artifacts = normalized.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ConfigError("Anomaly result artifacts must be a mapping", field="anomaly.result.artifacts")
    normalized["artifacts"] = dict(artifacts)
    normalized["elapsed_sec"] = float(normalized.get("elapsed_sec", 0.0))
    return normalized