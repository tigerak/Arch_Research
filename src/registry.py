from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional

import timm
import torch
import torch.nn as nn
import torch.optim as optim

# Local Modules
from src.config_loader import ConfigError
from src.losses import SubCenterArcFace, focal_loss
from utils.setup_model import TransformerHeadClassifier


class FocalLossModule(nn.Module):
    """focal loss를 nn.Module 인터페이스로 감싼 래퍼.

    학습 엔진에서 CrossEntropyLoss처럼 동일한 방식으로 호출할 수 있도록
    (logits, targets) -> scalar loss 형태를 맞추기 위해 사용한다.
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )


def build_model(cfg: Mapping[str, Any]) -> nn.Module:
    """설정(dict) 기반으로 모델을 생성한다.

    현재 지원 전략:
    - head_type=transformer: timm backbone + TransformerHeadClassifier
    - 그 외(head_type=backbone/linear/...): timm의 분류 헤드를 직접 사용

    참고:
    - 로컬 프리트레인 체크포인트(model.pretrained)가 있으면 strict=False로 로드한다.
    - 프리트레인 파일이 지정되어 있는데 파일이 없으면 즉시 ConfigError를 발생시킨다.
    """
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
        # TransformerHeadClassifier는 토큰 시퀀스를 반환하는 백본을 기대한다.
        # num_classes=0으로 생성하면 백본 분류 헤드를 제거한 feature extractor로 사용 가능하다.
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

        # 기존 setup_model의 클래스를 재사용하기 위해 필요한 최소 인자만 네임스페이스로 전달한다.
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
        # 일반 분류 헤드를 쓰는 간단한 경로.
        # linear/general_gem/effnet_gem 등의 이름이 들어오더라도 1차 버전에서는
        # timm 기본 분류 헤드 경로로 통일해 빠르게 실험을 돌릴 수 있게 한다.
        model = timm.create_model(model_name, pretrained=False, num_classes=n_classes)

        if freeze_backbone:
            # head 외 파라미터를 고정해 "헤드만 학습" 실험을 쉽게 재현한다.
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


def build_loss(cfg: Mapping[str, Any], model: nn.Module | None = None) -> nn.Module:
    """설정(dict) 기반으로 손실 함수를 생성한다.

    지원 loss:
    - ce / cross_entropy
    - focal
    - arcface

    arcface는 일반 CE처럼 logits를 바로 넣는 방식이 아니라
    "특징 벡터 + 라벨" 입력이 필요한 metric loss다.
    따라서 학습 엔진에서 arcface 분기를 별도 처리해야 한다.
    """
    train_cfg = _section(cfg, "train")

    # 두 가지 스키마를 동시에 허용한다.
    # 1) train.loss_name
    # 2) loss.name
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss", {}), Mapping) else {}
    loss_name = str(train_cfg.get("loss_name", loss_cfg.get("name", "ce"))).lower()

    if loss_name in {"ce", "cross_entropy"}:
        return nn.CrossEntropyLoss()

    if loss_name == "focal":
        gamma = float(train_cfg.get("focal_gamma", loss_cfg.get("focal_gamma", 2.0)))
        alpha = float(train_cfg.get("focal_alpha", loss_cfg.get("focal_alpha", 1.0)))
        return FocalLossModule(alpha=alpha, gamma=gamma)

    if loss_name == "arcface":
        n_classes = int(train_cfg.get("n_classes", 2))

        # ArcFace 입력 차원은 모델의 metric feature 출력과 반드시 일치해야 한다.
        # 설정값이 다르더라도 모델에서 노출한 차원을 우선 사용해 einsum shape mismatch를 방지한다.
        if model is not None and hasattr(model, "metric_feature_dim"):
            feature_dim = int(getattr(model, "metric_feature_dim"))
        else:
            model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), Mapping) else {}
            feature_dim = int(train_cfg.get("metric_feature_dim", model_cfg.get("metric_feature_dim", 768)))

        num_sub_centers = int(train_cfg.get("arcface_num_sub_centers", train_cfg.get("num_sub_centers", 4)))
        margin = float(train_cfg.get("arcface_margin", 0.3))
        scale = float(train_cfg.get("arcface_scale", 64.0))
        return SubCenterArcFace(
            num_classes=n_classes,
            feature_dim=feature_dim,
            num_sub_centers=num_sub_centers,
            margin=margin,
            scale=scale,
        )

    raise ConfigError(
        "Unsupported loss_name",
        field="train.loss_name/loss.name",
        details={"value": loss_name, "supported": ["ce", "focal", "arcface"]},
    )


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module) -> optim.Optimizer:
    """설정(dict) 기반으로 옵티마이저를 생성한다.

    지원 optimizer:
    - adamw(기본)
    - adam
    - sgd

    requires_grad=True 파라미터만 학습 대상으로 선택한다.
    """
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
    """ArcFace 실험용으로 모델과 loss 파라미터를 함께 학습한다."""
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
    """체크포인트에 ArcFace 상태가 있으면 로드한다."""
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
    """설정(dict) 기반으로 러닝레이트 스케줄러를 생성한다.

    지원 scheduler:
    - cosine: CosineAnnealingLR
    - step: StepLR
    - onecycle: OneCycleLR

    구현 포인트:
    - cosine의 t_max를 별도 지정하지 않으면 max_epoch를 기본으로 사용한다.
    - onecycle은 total_steps 계산을 위해 steps_per_epoch가 필요하다.
    """
    train_cfg = _section(cfg, "train")
    scheduler_cfg = cfg.get("scheduler", {}) if isinstance(cfg.get("scheduler", {}), Mapping) else {}

    # 사용자가 입력한 scheduler 문자열을 가능한 그대로 해석한다.
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
    """문자열 이름으로 torch 스케줄러 클래스를 찾는다.

    1) exact match
    2) case-insensitive match
    """
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
    """스케줄러 생성에 사용할 kwargs를 수집한다.

    우선순위:
    - scheduler 섹션 값이 train 섹션의 동명 키를 덮어쓴다.
    """
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
    """train.img_size 설정값에서 모델 입력 크기 한 변을 추출한다."""
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
    """입력 크기를 반영해 timm transformer backbone을 생성한다.

    dynamic_img_size를 우선 시도하고, 모델이 해당 인자를 지원하지 않으면
    img_size만 적용해 가능한 경로로 생성한다.
    """
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
    """체크포인트를 strict=False로 로드해 키 불일치에 유연하게 대응한다.

    다양한 저장 포맷을 흡수하기 위해 아래 키를 우선 탐색한다.
    - state_dict
    - model_state_dict
    - raw dict(가중치 키/텐서 직접 포함)
    """
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
    """설정 섹션이 dict 형태인지 확인해 반환한다.

    레지스트리 함수들이 공통으로 사용하는 기본 방어 로직이다.
    """
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
