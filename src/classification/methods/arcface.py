from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

METHOD_NAME = "arcface"
SUPPORTED_ALIASES = {"arcface"}


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace loss.

    각 클래스마다 여러 sub-center를 두고, 클래스 점수는 그중 가장 높은 center로 계산한다.
    pruning은 warmup 이후에만 수행하며, 한 번에 클래스당 최대 1개만 비활성화한다.
    """

    def __init__(self, num_classes, feature_dim, num_sub_centers=4, margin=0.3, scale=64.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_sub_centers = int(num_sub_centers)
        self.margin = float(margin)
        self.scale = float(scale)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.num_sub_centers, self.feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer('active_mask', torch.ones(self.num_classes, self.num_sub_centers, dtype=torch.bool))
        self.register_buffer('sub_center_usage', torch.zeros(self.num_classes, self.num_sub_centers))

    def _cosine_logits(self, features):
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=2)
        return torch.einsum('bd,ckd->bck', features, weight)

    def inference_logits(self, features):
        cosine = self._cosine_logits(features)
        cosine = cosine.masked_fill(~self.active_mask.unsqueeze(0), -1e4)
        class_logits, _ = cosine.max(dim=2)
        return class_logits * self.scale

    def forward(self, features, labels):
        batch_size = features.size(0)
        device = features.device

        cosine = self._cosine_logits(features)
        cosine = cosine.masked_fill(~self.active_mask.unsqueeze(0), -1e4)

        class_logits, class_best_idx = cosine.max(dim=2)

        batch_indices = torch.arange(batch_size, device=device)
        target_cos = class_logits[batch_indices, labels].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(target_cos)
        class_logits[batch_indices, labels] = torch.cos(theta + self.margin)

        target_center_idx = class_best_idx[batch_indices, labels]
        self.sub_center_usage[labels, target_center_idx] += 1

        return F.cross_entropy(class_logits * self.scale, labels)

    def prune_unused_subcenters(self, warmup_epochs=3, current_epoch=0, min_usage=1, max_remove_per_class=1):
        if current_epoch < warmup_epochs:
            self.sub_center_usage.zero_()
            return 0

        removed_total = 0
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                active_idx = torch.nonzero(self.active_mask[class_idx], as_tuple=False).flatten()
                if active_idx.numel() <= 1:
                    continue

                usage = self.sub_center_usage[class_idx, active_idx]
                low_usage_order = torch.argsort(usage)

                removed_for_class = 0
                for order_idx in low_usage_order.tolist():
                    center_idx = active_idx[order_idx].item()
                    if usage[order_idx].item() > min_usage:
                        break
                    if self.active_mask[class_idx].sum().item() <= 1:
                        break
                    self.active_mask[class_idx, center_idx] = False
                    removed_total += 1
                    removed_for_class += 1
                    if removed_for_class >= max_remove_per_class:
                        break

            self.sub_center_usage.zero_()

        return removed_total


def build_loss(cfg: Mapping[str, Any], model: nn.Module | None = None) -> nn.Module:
    train_cfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), Mapping) else {}

    n_classes = int(train_cfg.get("n_classes", 2))
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


def build_optimizer(cfg: Mapping[str, Any], model: nn.Module, criterion: nn.Module | None = None):
    if criterion is None or not isinstance(criterion, SubCenterArcFace):
        raise ValueError("arcface optimizer requires SubCenterArcFace criterion")
    from src.classification.builders import build_arcface_optimizer as _build_arcface_optimizer
    return _build_arcface_optimizer(cfg, model, criterion)


def load_arcface_state_if_present(arcface_loss: SubCenterArcFace, checkpoint_obj: Any) -> None:
    from src.classification.builders import load_arcface_state_if_present as _load_arcface_state_if_present
    _load_arcface_state_if_present(arcface_loss, checkpoint_obj)
