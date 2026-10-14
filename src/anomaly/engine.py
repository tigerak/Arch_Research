from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, Mapping

from src.anomaly.registry import get_anomaly_method

"""anomaly task의 공통 엔진 모듈.

이 파일은 방법론 자체를 구현하지 않고, 설정에 따라 선택된 anomaly method runner를 실행하는 역할만 담당한다.

구조상으로는 classification.engine와 매우 유사하지만, anomaly는 일반적인 손실 함수 아키텍처가 아니라
PatchCore 같은 메모리 뱅크 기반 방법을 선택하는 방식이므로 runner 자체가 조금 다르게 동작한다.
"""


def run_anomaly_experiment(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """이상치 탐지 실험을 실행하는 공통 래퍼.

    흐름:
    1. anomaly.method_name을 읽어 registry에서 실행 함수 검색
    2. 해당 runner를 호출
    3. 결과에 task_type과 elapsed_sec을 보강

    이렇게 공통 엔진 레이어를 두면, 이후 다른 anomaly method를 추가하더라도
    train/eval 시 액션 자체는 동일하게 유지할 수 있다.
    """
    started = perf_counter()
    runner = get_anomaly_method(cfg)
    result = runner(cfg, save_artifacts=save_artifacts)
    result.setdefault("task_type", "anomaly_detection")
    result.setdefault("elapsed_sec", round(perf_counter() - started, 3))
    return result


def run_anomaly_train(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """이상치 탐지 학습 진입점.

    현재 PatchCore baseline은 학습 단계에서 정상 이미지 임베딩을 추출하고
    memory bank를 구성하는 흐름으로 동작한다.
    """
    return run_anomaly_experiment(cfg, save_artifacts=save_artifacts)


def run_anomaly_eval(cfg: Mapping[str, Any], *, save_artifacts: bool = True) -> Dict[str, Any]:
    """이상치 탐지 평가 진입점.

    현재 구조에서는 training과 evaluation이 같은 runner를 공유하는 경우가 많다.
    이유는 PatchCore의 경우 메모리 뱅크를 구축한 뒤 바로 점수 계산을 수행하기 때문이다.
    """
    return run_anomaly_experiment(cfg, save_artifacts=save_artifacts)
