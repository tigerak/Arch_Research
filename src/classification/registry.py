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

"""classification task의 method registry와 공통 builder 래퍼.

이 파일은 "설정에서 어떤 loss/method를 사용할지 선택하고,
그에 맞는 method 모듈에서 build_loss/build_optimizer를 호출한다"는 역할을 담당한다.

목적:
- method 구현을 분리한다.
- 현재 실험이 CE, Focal, ArcFace 중 무엇인지 설정상으로만 결정한다.
- 상위 엔진은 registry를 통해 다형적으로 접근할 수 있다.
"""

METHOD_REGISTRY: dict[str, str] = {
    "ce": "src.classification.methods.ce",
    "cross_entropy": "src.classification.methods.ce",
    "focal": "src.classification.methods.focal",
    "arcface": "src.classification.methods.arcface",
}


def get_classification_method(cfg: Mapping[str, Any]):
    """설정에서 loss_name을 읽어 해당 method 모듈을 반환한다.

    train.loss_name 또는 loss.name을 읽어서 어떤 구현을 선택할지 결정한다.
    기본값은 cross entropy(ce)이며, 신설 method를 추가할 때는 METHOD_REGISTRY만 갱신하면 된다.
    """
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
    """모델 생성 로직을 공통 builders로 위임한다."""
    return _build_model(cfg)


def build_loss(cfg: Mapping[str, Any], model: nn.Module | None = None) -> nn.Module:
    """설정에 맞는 loss 객체를 생성한다.

    method별 파일에서 구현된 build_loss()가 호출되며,
    이 파일 자체는 어떤 method를 선택할지 결정하고 실제 구현을 호출하는 역할만 수행한다.
    """
    module = get_classification_method(cfg)
    return module.build_loss(cfg, model=model)


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module, criterion: nn.Module | None = None):
    """설정에 맞는 optimizer를 생성한다.

    각 method 모듈이 build_optimizer를 구현하면 그 구현을 사용하고,
    그렇지 않은 경우 기본 optimizer 생성 로직을 사용한다.
    """
    module = get_classification_method(cfg)
    builder = getattr(module, "build_optimizer", None)
    if callable(builder):
        return builder(cfg, model, criterion=criterion)

    return _build_optimizer(cfg, model)


def build_arcface_optimizer(cfg: Mapping[str, Any], model: nn.Module, arcface_loss: nn.Module):
    """ArcFace 전용 optimizer 생성 함수.

    ArcFace는 일반 모델 파라미터와 loss parameter를 따로 관리하므로,
    별도의 optimizer 생성 로직이 필요하다.
    """
    module = get_classification_method(cfg)
    builder = getattr(module, "build_optimizer", None)
    if callable(builder):
        return builder(cfg, model, criterion=arcface_loss)

    return _build_arcface_optimizer(cfg, model, arcface_loss)


def load_arcface_state_if_present(arcface_loss: nn.Module, checkpoint_obj: Any) -> None:
    """체크포인트에 ArcFace 상태가 포함되어 있으면 복원한다.

    다른 loss method와 달리 ArcFace는 별도의 중심 벡터/활성 마스크 상태를 가지므로,
    일반 모델 로딩만으로는 부족하다.
    """
    module = import_module(METHOD_REGISTRY["arcface"])
    loader = getattr(module, "load_arcface_state_if_present", None)
    if callable(loader):
        loader(arcface_loss, checkpoint_obj)
        return

    _load_arcface_state_if_present(arcface_loss, checkpoint_obj)


def build_scheduler(cfg: Mapping[str, Any], optimizer, *, steps_per_epoch: int | None = None):
    """공통 scheduler 생성 함수.

    이 함수는 loss method와 무관하게 일정한 scheduler를 구성해 준다.
    """
    return _build_scheduler(cfg, optimizer, steps_per_epoch=steps_per_epoch)


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """설정 섹션을 안전하게 읽어오는 helper."""
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
