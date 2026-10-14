from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Mapping

from src.anomaly.registry import get_anomaly_method


def run_anomaly_experiment(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """이상치 탐지 실험을 실행한다.

    실제 구현은 anomaly.method_name으로 선택한 method runner가 담당한다.
    """
    started = perf_counter()
    runner = get_anomaly_method(cfg)
    result = runner(cfg, save_artifacts=save_artifacts)
    result.setdefault("task_type", "anomaly_detection")
    result.setdefault("elapsed_sec", round(perf_counter() - started, 3))
    return result


def run_anomaly_train(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """이상치 탐지 학습 진입점이다.

    현재 PatchCore baseline은 memory bank 구축과 평가를 한 번에 수행한다.
    """
    return run_anomaly_experiment(cfg, save_artifacts=save_artifacts)


def run_anomaly_eval(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """이상치 탐지 평가 진입점이다.

    현재 PatchCore baseline은 memory bank 구축과 평가를 한 번에 수행한다.
    """
    return run_anomaly_experiment(cfg, save_artifacts=save_artifacts)
