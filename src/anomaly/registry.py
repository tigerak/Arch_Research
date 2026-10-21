from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Mapping

from src.config_loader import ConfigError
from src.anomaly.contracts import AnomalyMethodRunner

"""이상치 탐지 task의 method registry 모듈.

이 파일의 핵심 목적은 "설정 파일의 anomaly.method_name을 받아 실제 구현 함수를 찾는 것"이다.

기존 구조에서는 method 구현이 혼재되어 있었지만, registry를 두면 다음 장점이 있다.
- 새 anomaly 모델을 추가할 때 엔진 로직을 건드리지 않아도 된다.
- method 이름과 구현 경로를 한곳에서 관리한다.
- 설정 파일만 보고 어떤 실험을 돌릴지 파악할 수 있다.
"""

METHOD_REGISTRY: dict[str, str] = {
    "patchcore": "src.anomaly.methods.patchcore_runner:run",
    "dinomaly2": "src.anomaly.methods.dinomaly2_runner:run",
}


def get_anomaly_method(cfg: Mapping[str, Any]) -> AnomalyMethodRunner:
    """설정에서 method_name을 읽어 실제 실행 함수를 반환한다.

    예시:
    anomaly:
      method_name: patchcore

    위 설정이 들어오면 METHOD_REGISTRY["patchcore"] 값을 찾고,
    src.anomaly.methods.patchcore_runner 모듈의 run() 함수를 import해 반환한다.
    """
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
    """등록된 anomaly method 이름 목록을 정렬해 반환한다.

    이 함수는 UI, CLI, 혹은 문서화에서 사용하기 좋다.
    """
    return sorted(METHOD_REGISTRY.keys())
