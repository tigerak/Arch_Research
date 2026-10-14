from __future__ import annotations

from typing import Any, Dict, Mapping

from src.eval_engine import evaluate_checkpoint
from src.train_engine import train_one_experiment


def run_classification_train(cfg: Mapping[str, Any], logger=None) -> Dict[str, Any]:
    """classification 학습 실험의 공통 진입점이다."""
    return train_one_experiment(cfg, logger=logger)


def run_classification_eval(cfg: Mapping[str, Any], checkpoint_path=None) -> Dict[str, Any]:
    """classification 평가 실험의 공통 진입점이다."""
    return evaluate_checkpoint(cfg, checkpoint_path=checkpoint_path)
