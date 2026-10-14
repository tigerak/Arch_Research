from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Numerically stable supervised contrastive loss.

    Expected input:
    - features: [2B, D] (two augmented views concatenated)
    - labels: [B]
    """

    def __init__(self, temperature=0.07, base_temperature=0.07, class_weights=None, eps=1e-12):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if base_temperature <= 0:
            raise ValueError(f"base_temperature must be > 0, got {base_temperature}")

        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        self.eps = float(eps)

        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', weight_tensor)
        else:
            self.class_weights = None

    def forward(self, features, labels):
        if features.ndim != 2:
            raise ValueError(f"features must be 2D [2B, D], got shape={tuple(features.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D [B], got shape={tuple(labels.shape)}")

        device = features.device
        batch = labels.shape[0]
        if features.shape[0] != 2 * batch:
            raise ValueError("features should be 2*B in first dim when labels length is B")

        features = F.normalize(features, dim=1)
        labels = labels.repeat(2)
        n = labels.shape[0]

        logits = torch.matmul(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        positive_count = positive_mask.sum(dim=1)
        positive_count_f = positive_count.clamp_min(1).float()
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count_f

        valid = positive_count > 0
        if not torch.any(valid):
            return features.sum() * 0.0

        per_anchor_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid]

        if self.class_weights is not None:
            max_class = int(labels.max().item()) if labels.numel() > 0 else -1
            if max_class >= self.class_weights.numel():
                raise ValueError(
                    f"class_weights length ({self.class_weights.numel()}) is smaller than max label index ({max_class})"
                )

            anchor_weights = self.class_weights[labels[valid]].to(device=device, dtype=per_anchor_loss.dtype)
            denom = anchor_weights.sum().clamp_min(self.eps)
            return (per_anchor_loss * anchor_weights).sum() / denom

        return per_anchor_loss.mean()
