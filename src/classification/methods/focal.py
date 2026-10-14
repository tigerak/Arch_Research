from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

METHOD_NAME = "focal"
SUPPORTED_ALIASES = {"focal"}


def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)

    if alpha is None:
        alpha_factor = torch.ones_like(ce_loss)
    elif isinstance(alpha, (list, tuple)):
        alpha_t = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
        alpha_factor = alpha_t[targets]
    elif torch.is_tensor(alpha):
        alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)
        if alpha_t.ndim == 0:
            alpha_factor = torch.full_like(ce_loss, float(alpha_t.item()))
        else:
            alpha_factor = alpha_t[targets]
    else:
        alpha_factor = torch.full_like(ce_loss, float(alpha))

    loss = alpha_factor * (1.0 - pt).pow(gamma) * ce_loss

    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


class FocalLossModule(nn.Module):
    """focal loss를 nn.Module 인터페이스로 감싼 래퍼."""

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


def build_loss(cfg: Mapping[str, Any], model: nn.Module | None = None) -> nn.Module:
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), Mapping) else {}
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg.get("loss", {}), Mapping) else {}

    gamma = float(train_cfg.get("focal_gamma", loss_cfg.get("focal_gamma", 2.0)))
    alpha = float(train_cfg.get("focal_alpha", loss_cfg.get("focal_alpha", 1.0)))
    return FocalLossModule(alpha=alpha, gamma=gamma)


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module, criterion: nn.Module | None = None):
    from src.classification.builders import build_optimizer as _build_optimizer
    return _build_optimizer(cfg, model)
