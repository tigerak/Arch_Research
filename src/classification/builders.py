from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional

import timm
import torch
import torch.nn as nn
import torch.optim as optim

from src.classification.methods.arcface import SubCenterArcFace
from src.classification.methods.focal import FocalLossModule
from src.classification.models import TransformerHeadClassifier
from src.config_loader import ConfigError

"""classification 모델/optimizer/scheduler를 조립하는 빌더 모듈.

이 파일은 method 구현이 아니라, 실제 실험의 도구적 구성 로직을 담당한다.

역할:
- backbone과 head를 조합해 모델을 만든다.
- 사전학습 체크포인트를 로드한다.
- optimizer와 scheduler를 만든다.
- ArcFace 전용 optimizer 파라미터 그룹을 다룬다.

즉, method 파일들은 loss/metric 로직을 정의하고,
이 파일은 그 method가 실험에서 쓸 수 있도록 실제 객체를 생성하는 역할을 한다.
"""


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    model_cfg = _section(cfg, "model")
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")

    model_name = str(model_cfg.get("model_name", "resnet18"))
    head_type = str(model_cfg.get("head_type", "backbone")).lower()
    freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
    n_classes = int(train_cfg.get("n_classes", 2))

    model_dir = Path(str(data_cfg.get("model_dir", "models/pretrained")))
    pretrained_name = str(model_cfg.get("pretrained", "")).strip()

    if head_type == "transformer":
        input_size = _resolve_input_size(train_cfg.get("img_size", [224, 224]))
        backbone = _create_transformer_backbone(model_name=model_name, input_size=input_size)

        use_metric_projection = bool(model_cfg.get("use_metric_projection", train_cfg.get("use_metric_projection", False)))
        detach_metric_projection_input = bool(
            model_cfg.get("detach_metric_projection_input", train_cfg.get("detach_metric_projection_input", False))
        )
        metric_feature_dim = int(
            model_cfg.get(
                "metric_feature_dim",
                train_cfg.get("metric_feature_dim", getattr(backbone, "num_features", 768)),
            )
        )

        args = SimpleNamespace(
            n_classes=n_classes,
            freeze_backbone=freeze_backbone,
            token_fusion=str(model_cfg.get("token_fusion", "cls_only")),
            use_metric_projection=use_metric_projection,
            detach_metric_projection_input=detach_metric_projection_input,
            metric_feature_dim=metric_feature_dim,
        )
        model = TransformerHeadClassifier(backbone=backbone, args=args)
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)

        if freeze_backbone:
            for name, param in model.named_parameters():
                param.requires_grad = False
                if any(key in name for key in ["head", "classifier", "fc"]):
                    param.requires_grad = True

    if pretrained_name:
        ckpt_path = model_dir / pretrained_name
        if not ckpt_path.exists():
            raise ConfigError(
                "Pretrained checkpoint not found",
                field="model.pretrained",
                file_path=ckpt_path,
            )
        _load_checkpoint_strict_false(model, ckpt_path)

    return model


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module) -> optim.Optimizer:
    train_cfg = _section(cfg, "train")
    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer", {}), Mapping) else {}

    name = str(optimizer_cfg.get("name", train_cfg.get("optimizer", "adamw"))).lower()
    lr = float(optimizer_cfg.get("lr", train_cfg.get("lr", 5e-4)))
    weight_decay = float(optimizer_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-4)))

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ConfigError("No trainable parameters found", field="model.parameters")

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    raise ConfigError(
        "Unsupported optimizer",
        field="optimizer.name/train.optimizer",
        details={"value": name, "supported": ["adamw", "adam", "sgd"]},
    )


def build_arcface_optimizer(
    cfg: Mapping[str, Any],
    model: nn.Module,
    arcface_loss: SubCenterArcFace,
) -> optim.Optimizer:
    train_cfg = _section(cfg, "train")
    optimizer_cfg = cfg.get("optimizer", {}) if isinstance(cfg.get("optimizer", {}), Mapping) else {}

    name = str(optimizer_cfg.get("name", train_cfg.get("optimizer", "adamw"))).lower()
    model_lr = float(optimizer_cfg.get("lr", train_cfg.get("lr", 5e-4)))
    arcface_lr = float(train_cfg.get("arcface_lr", model_lr))
    weight_decay = float(optimizer_cfg.get("weight_decay", train_cfg.get("weight_decay", 1e-4)))

    model_params = [p for p in model.parameters() if p.requires_grad]
    arcface_params = [p for p in arcface_loss.parameters() if p.requires_grad]
    if not model_params:
        raise ConfigError("No trainable model parameters found", field="model.parameters")
    if not arcface_params:
        raise ConfigError("No trainable ArcFace parameters found", field="train.arcface")

    param_groups = [
        {"params": model_params, "lr": model_lr, "weight_decay": weight_decay},
        {"params": arcface_params, "lr": arcface_lr, "weight_decay": weight_decay},
    ]

    if name == "adamw":
        return optim.AdamW(param_groups)
    if name == "adam":
        return optim.Adam(param_groups)
    if name == "sgd":
        momentum = float(optimizer_cfg.get("momentum", 0.9))
        return optim.SGD(param_groups, momentum=momentum, nesterov=True)

    raise ConfigError(
        "Unsupported optimizer",
        field="optimizer.name/train.optimizer",
        details={"value": name, "supported": ["adamw", "adam", "sgd"]},
    )


def load_arcface_state_if_present(arcface_loss: SubCenterArcFace, checkpoint_obj: Any) -> None:
    if not isinstance(checkpoint_obj, Mapping):
        return
    state = checkpoint_obj.get("arcface_state_dict")
    if isinstance(state, Mapping):
        arcface_loss.load_state_dict(state, strict=False)


def build_scheduler(
    cfg: Mapping[str, Any],
    optimizer: optim.Optimizer,
    *,
    steps_per_epoch: Optional[int] = None,
) -> optim.lr_scheduler._LRScheduler | optim.lr_scheduler.OneCycleLR:
    train_cfg = _section(cfg, "train")
    scheduler_cfg = cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler", {}), Mapping) else {}

    raw_name_text = str(train_cfg.get("scheduler", scheduler_cfg.get("name", "cosine"))).strip()
    raw_name = raw_name_text.lower()
    scheduler_alias = {
        "cosineannealinglr": "cosine",
        "steplr": "step",
        "onecyclelr": "onecycle",
    }
    name = scheduler_alias.get(raw_name, raw_name)
    max_epoch = int(train_cfg.get("max_epoch", 1))

    if name == "cosine":
        t_max = int(scheduler_cfg.get("t_max", max_epoch))
        eta_min = float(scheduler_cfg.get("eta_min", train_cfg.get("eta_min", 0.0)))
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if name == "step":
        step_size = int(scheduler_cfg.get("step_size", max(1, max_epoch // 3)))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "onecycle":
        if steps_per_epoch is None or steps_per_epoch <= 0:
            raise ConfigError(
                "onecycle scheduler requires positive steps_per_epoch",
                field="scheduler.name",
                details={"steps_per_epoch": steps_per_epoch},
            )

        max_lr = float(scheduler_cfg.get("max_lr", train_cfg.get("lr", 5e-4)))
        total_steps = max_epoch * int(steps_per_epoch)
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
        )

    scheduler_class = _resolve_scheduler_class(raw_name_text)
    if scheduler_class is not None:
        kwargs = _collect_scheduler_kwargs(train_cfg, scheduler_cfg)
        try:
            return scheduler_class(optimizer, **kwargs)
        except TypeError as exc:
            raise ConfigError(
                "Failed to initialize torch scheduler with provided arguments",
                field="train.scheduler/scheduler",
                details={
                    "scheduler": raw_name_text,
                    "kwargs": kwargs,
                    "error": str(exc),
                },
            ) from exc

    raise ConfigError(
        "Unsupported scheduler",
        field="train.scheduler/scheduler.name",
        details={
            "value": raw_name_text,
            "hint": "Use a shorthand (cosine/step/onecycle) or a valid torch.optim.lr_scheduler class name",
        },
    )


def _resolve_scheduler_class(name: str) -> type[optim.lr_scheduler.LRScheduler] | None:
    if hasattr(optim.lr_scheduler, name):
        cls = getattr(optim.lr_scheduler, name)
        if isinstance(cls, type):
            return cls

    lowered = name.lower()
    for attr_name in dir(optim.lr_scheduler):
        if attr_name.lower() != lowered:
            continue
        cls = getattr(optim.lr_scheduler, attr_name)
        if isinstance(cls, type):
            return cls

    return None


def _collect_scheduler_kwargs(
    train_cfg: Mapping[str, Any],
    scheduler_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    ignored_keys = {
        "scheduler",
        "max_epoch",
        "lr",
        "weight_decay",
        "batch_size",
        "num_workers",
        "labels",
        "n_classes",
        "loss_name",
        "focal_gamma",
        "focal_alpha",
        "optimizer",
        "device",
        "save_warmup_epochs",
        "save_compare_metrics",
        "save_compare_modes",
    }

    kwargs: Dict[str, Any] = {}
    for k, v in train_cfg.items():
        if k in ignored_keys:
            continue
        kwargs[k] = v

    for k, v in scheduler_cfg.items():
        if k == "name":
            continue
        kwargs[k] = v

    return kwargs


def _resolve_input_size(img_size_value: Any) -> int:
    if isinstance(img_size_value, int):
        if img_size_value <= 0:
            raise ConfigError("train.img_size must be positive", field="train.img_size", details={"value": img_size_value})
        return int(img_size_value)

    if isinstance(img_size_value, list) and len(img_size_value) == 2:
        h = int(img_size_value[0])
        w = int(img_size_value[1])
        if h <= 0 or w <= 0:
            raise ConfigError("train.img_size entries must be positive", field="train.img_size", details={"value": img_size_value})
        if h != w:
            raise ConfigError(
                "Transformer backbone currently requires square img_size",
                field="train.img_size",
                details={"value": img_size_value},
            )
        return h

    raise ConfigError(
        "train.img_size must be int or [H, W]",
        field="train.img_size",
        details={"value": img_size_value},
    )


def _create_transformer_backbone(model_name: str, input_size: int) -> nn.Module:
    try:
        return timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            img_size=input_size,
            dynamic_img_size=True,
        )
    except TypeError:
        try:
            return timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                img_size=input_size,
            )
        except TypeError:
            return timm.create_model(model_name, pretrained=False, num_classes=0)


def _load_checkpoint_strict_false(model: nn.Module, checkpoint_path: Path) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(state, Mapping):
        if "state_dict" in state and isinstance(state["state_dict"], Mapping):
            state = state["state_dict"]
        elif "model_state_dict" in state and isinstance(state["model_state_dict"], Mapping):
            state = state["model_state_dict"]

    if not isinstance(state, Mapping):
        raise ConfigError(
            "Unsupported checkpoint format",
            field="model.pretrained",
            file_path=checkpoint_path,
        )

    cleaned = {str(k).replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
