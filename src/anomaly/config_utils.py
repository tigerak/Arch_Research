from __future__ import annotations

from typing import Any, Dict, Mapping

from src.config_loader import ConfigError


def get_anomaly_section(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    anomaly = cfg.get("anomaly")
    if not isinstance(anomaly, Mapping):
        raise ConfigError("Config section must be a mapping", field="anomaly")
    return anomaly


def get_anomaly_method_name(cfg: Mapping[str, Any]) -> str:
    anomaly_cfg = get_anomaly_section(cfg)
    return str(anomaly_cfg.get("method_name", "patchcore")).strip().lower()


def get_method_config(cfg: Mapping[str, Any], method_name: str | None = None) -> Dict[str, Any]:
    resolved_method_name = method_name or get_anomaly_method_name(cfg)
    method_root = cfg.get("method", {})
    if method_root is None:
        method_root = {}
    if not isinstance(method_root, Mapping):
        raise ConfigError("Config section must be a mapping", field="method")

    method_cfg = method_root.get(resolved_method_name, {})
    if method_cfg is None:
        method_cfg = {}
    if not isinstance(method_cfg, Mapping):
        raise ConfigError(
            "Method-specific config must be a mapping",
            field=f"method.{resolved_method_name}",
        )

    normalized = dict(method_cfg)
    anomaly_cfg = get_anomaly_section(cfg)
    legacy_patchcore_keys = (
        "feature_layers",
        "coreset_sampling_ratio",
        "knn_k",
        "score_aggregation",
        "max_memory_bank_patches",
    )
    if resolved_method_name == "patchcore":
        for key in legacy_patchcore_keys:
            if key not in normalized and key in anomaly_cfg:
                normalized[key] = anomaly_cfg[key]

    return normalized


def get_threshold_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    anomaly_cfg = get_anomaly_section(cfg)
    threshold_cfg = anomaly_cfg.get("threshold", {})
    if threshold_cfg is None:
        threshold_cfg = {}
    if not isinstance(threshold_cfg, Mapping):
        raise ConfigError("anomaly.threshold must be a mapping", field="anomaly.threshold")

    policy = str(threshold_cfg.get("policy", anomaly_cfg.get("threshold_policy", "tpr95"))).strip().lower()
    resolved = {"policy": policy}
    if policy == "tpr95":
        resolved["target_tpr"] = float(threshold_cfg.get("target_tpr", 0.95))
    elif policy == "fixed":
        if "value" not in threshold_cfg:
            raise ConfigError("fixed threshold policy requires value", field="anomaly.threshold.value")
        resolved["value"] = float(threshold_cfg["value"])
    return resolved