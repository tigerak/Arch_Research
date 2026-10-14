from __future__ import annotations

from typing import Any, Dict, Mapping

from src.eval_engine import evaluate_checkpoint
from src.train_engine import train_one_experiment

"""classification task의 공통 엔진 래퍼.

이 파일은 실제 학습/평가 로직을 직접 구현하지 않고,
하위 train_engine/eval_engine으로 업무를 위임하는 스위치 역할만 수행한다.

구조상 의미:
- classification task의 가장 외곽 진입점이다.
- registry나 run_manager가 이 함수를 호출한다.
- 내부적으로는 train_one_experiment(), evaluate_checkpoint()을 호출한다.

이런 구조를 두면 task별 엔진과 method 구현이 분리되며,
새로운 method를 추가하더라도 상위 통합 로직은 거의 바뀌지 않는다.
"""


def run_classification_train(cfg: Mapping[str, Any], logger=None) -> Dict[str, Any]:
    """classification 학습 실험의 공통 진입점.

    실험 포털의 상위 레이어에서 호출되며, 실제 학습 루프는
    src/train_engine.py의 train_one_experiment()에서 수행된다.
    """
    return train_one_experiment(cfg, logger=logger)


def run_classification_eval(cfg: Mapping[str, Any], checkpoint_path=None) -> Dict[str, Any]:
    """classification 평가 실험의 공통 진입점.

    학습이 끝난 체크포인트를 바탕으로 평가를 수행하고,
    결과를 dict 형태로 반환한다.
    """
    return evaluate_checkpoint(cfg, checkpoint_path=checkpoint_path)
