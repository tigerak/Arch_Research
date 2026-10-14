from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Mapping

import torch.nn as nn

from src.classification.builders import (
    build_arcface_optimizer as _build_arcface_optimizer,
    build_model as _build_model,
    build_optimizer as _build_optimizer,
    build_scheduler as _build_scheduler,
    load_arcface_state_if_present as _load_arcface_state_if_present,
)
from src.config_loader import ConfigError

METHOD_REGISTRY: dict[str, str] = {
    "ce": "src.classification.methods.ce",
    "cross_entropy": "src.classification.methods.ce",
    "focal": "src.classification.methods.focal",
    "arcface": "src.classification.methods.arcface",
}


def get_classification_method(cfg: Mapping[str, Any]):
    """설정에서 loss_name을 읽어 method 모듈을 반환한다."""
    train_cfg = _section(cfg, "train")
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss", {}), Mapping) else {}
    method_name = str(train_cfg.get("loss_name", loss_cfg.get("name", "ce"))).strip().lower()

    try:
        module_path = METHOD_REGISTRY[method_name]
    except KeyError as exc:
        raise ConfigError(
            "Unsupported classification method",
            field="train.loss_name/loss.name",
            details={"value": method_name, "supported": sorted(METHOD_REGISTRY.keys())},
        ) from exc

    module = import_module(module_path)
    return module


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    return _build_model(cfg)


def build_loss(cfg: Mapping[str, Any], model: nn.Module | None = None) -> nn.Module:
    module = get_classification_method(cfg)
    return module.build_loss(cfg, model=model)


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module, criterion: nn.Module | None = None):
    module = get_classification_method(cfg)
    builder = getattr(module, "build_optimizer", None)
    if callable(builder):
        return builder(cfg, model, criterion=criterion)

    return _build_optimizer(cfg, model)


def build_arcface_optimizer(cfg: Mapping[str, Any], model: nn.Module, arcface_loss: nn.Module):
    module = get_classification_method(cfg)
    builder = getattr(module, "build_optimizer", None)
    if callable(builder):
        return builder(cfg, model, criterion=arcface_loss)

    return _build_arcface_optimizer(cfg, model, arcface_loss)


def load_arcface_state_if_present(arcface_loss: nn.Module, checkpoint_obj: Any) -> None:
    module = import_module(METHOD_REGISTRY["arcface"])
    loader = getattr(module, "load_arcface_state_if_present", None)
    if callable(loader):
        loader(arcface_loss, checkpoint_obj)
        return

    _load_arcface_state_if_present(arcface_loss, checkpoint_obj)


def build_scheduler(cfg: Mapping[str, Any], optimizer, *, steps_per_epoch: int | None = None):
    return _build_scheduler(cfg, optimizer, steps_per_epoch=steps_per_epoch)


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
