from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Dict

from src.config_loader import ConfigError

AnomalyRunner = Callable[..., Dict[str, Any]]

METHOD_REGISTRY: dict[str, str] = {
    "patchcore": "src.anomaly.methods.patchcore:run",
}


def get_anomaly_method(cfg: Mapping[str, Any]) -> AnomalyRunner:
    """설정에서 method_name을 읽어 실행 함수를 반환한다."""
    anomaly_cfg = cfg.get("anomaly")
    if not isinstance(anomaly_cfg, Mapping):
        raise ConfigError("Config section must be a mapping", field="anomaly")

    method_name = str(anomaly_cfg.get("method_name", "patchcore")).strip().lower()
    try:
        target = METHOD_REGISTRY[method_name]
    except KeyError as exc:
        raise ConfigError(
            "Unsupported anomaly method",
            field="anomaly.method_name",
            details={"value": method_name, "supported": sorted(METHOD_REGISTRY.keys())},
        ) from exc

    module_path, function_name = target.split(":", 1)
    module = import_module(module_path)
    runner = getattr(module, function_name, None)
    if runner is None or not callable(runner):
        raise ConfigError(
            "Anomaly method runner is not callable",
            field="anomaly.method_name",
            details={"value": method_name, "target": target},
        )
    return runner


def list_anomaly_methods() -> list[str]:
    return sorted(METHOD_REGISTRY.keys())
