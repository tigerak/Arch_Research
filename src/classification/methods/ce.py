from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from src.classification.builders import build_optimizer as _build_optimizer

METHOD_NAME = "ce"
SUPPORTED_ALIASES = {"ce", "cross_entropy"}


def build_loss(cfg: Mapping[str, Any], model: nn.Module | None = None) -> nn.Module:
    return nn.CrossEntropyLoss()


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module, criterion: nn.Module | None = None):
    return _build_optimizer(cfg, model)
