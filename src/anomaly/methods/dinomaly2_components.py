from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from src.anomaly.backbone_features import LayerFeatureExtractor
from src.config_loader import ConfigError
from src.profiling.transform_timing import StepTimer


@dataclass
class Dinomaly2Config:
    model_name: str
    pretrained_path: str
    feature_layers: tuple[str, ...]
    bottleneck_dim: int = 256
    dropout: float = 0.1
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-4
    score_aggregation: str = "max"
    freeze_backbone: bool = True


class TokenReconstructionHead(torch.nn.Module):
    def __init__(self, token_dim: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = max(8, min(token_dim, int(bottleneck_dim)))
        self.layers = torch.nn.Sequential(
            torch.nn.LayerNorm(token_dim),
            torch.nn.Linear(token_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.layers(tokens)


class Dinomaly2FeatureModel(torch.nn.Module):
    """DINOv2 feature distillation 기반 anomaly 모델."""

    def __init__(self, cfg: Dinomaly2Config, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.teacher = LayerFeatureExtractor(
            model_name=cfg.model_name,
            pretrained_path=cfg.pretrained_path,
            feature_layers=cfg.feature_layers,
            device=device,
            freeze_backbone=cfg.freeze_backbone,
        )
        self.student_heads = torch.nn.ModuleList()
        self.to(device)
        self._initialize_student_heads()

    def _initialize_student_heads(self) -> None:
        image_size = self.get_input_image_size() or 224
        dummy = torch.zeros((1, 3, image_size, image_size), device=self.device)
        with torch.no_grad():
            teacher_tokens = self.teacher.forward_layer_tokens(dummy, normalize=False)

        self.student_heads = torch.nn.ModuleList(
            [TokenReconstructionHead(tokens.shape[-1], self.cfg.bottleneck_dim, self.cfg.dropout) for tokens in teacher_tokens]
        )
        self.student_heads = self.student_heads.to(self.device)

    def get_input_image_size(self) -> int | None:
        return self.teacher.get_input_image_size()

    def trainable_parameters(self):
        return self.student_heads.parameters()

    def compute_loss(self, images: torch.Tensor) -> torch.Tensor:
        teacher_tokens = self.teacher.forward_layer_tokens(images, normalize=False)
        losses: list[torch.Tensor] = []
        for teacher_tokens_for_layer, head in zip(teacher_tokens, self.student_heads):
            teacher_tokens_for_layer = teacher_tokens_for_layer.detach()
            predicted = head(teacher_tokens_for_layer)
            predicted_norm = F.normalize(predicted, dim=-1)
            teacher_norm = F.normalize(teacher_tokens_for_layer, dim=-1)
            mse = F.mse_loss(predicted_norm, teacher_norm)
            cosine = 1.0 - F.cosine_similarity(predicted, teacher_tokens_for_layer, dim=-1).mean()
            losses.append(mse + cosine)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def score_images(self, images: torch.Tensor, *, aggregation: str = "max") -> torch.Tensor:
        teacher_tokens = self.teacher.forward_layer_tokens(images, normalize=False)
        layer_errors = []
        for teacher_tokens_for_layer, head in zip(teacher_tokens, self.student_heads):
            predicted = head(teacher_tokens_for_layer)
            teacher_norm = F.normalize(teacher_tokens_for_layer, dim=-1)
            predicted_norm = F.normalize(predicted, dim=-1)
            mse = (predicted_norm - teacher_norm).pow(2).mean(dim=-1)
            cosine = 1.0 - F.cosine_similarity(predicted, teacher_tokens_for_layer, dim=-1)
            layer_errors.append(mse + cosine)

        stacked_errors = torch.stack(layer_errors, dim=0).mean(dim=0)
        if aggregation == "mean":
            return stacked_errors.mean(dim=1)
        return stacked_errors.max(dim=1).values


def train_dinomaly2_model(
    model: Dinomaly2FeatureModel,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    tensor_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    profile_steps: bool = False,
) -> list[dict[str, float]]:
    if epochs <= 0:
        raise ConfigError("method.dinomaly2.epochs must be > 0", field="method.dinomaly2.epochs", details={"value": epochs})
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=lr, weight_decay=weight_decay)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.student_heads.train()
        total_loss = 0.0
        total_samples = 0
        iterable = loader if tqdm is None else tqdm(loader, desc=f"Dinomaly2 Train {epoch}/{epochs}", leave=False)
        for images, _targets, _paths in iterable:
            images = images.to(model.device, non_blocking=True)
            if tensor_transform is not None:
                with StepTimer("tensor_transform", enabled=profile_steps):
                    images = tensor_transform(images)
            optimizer.zero_grad(set_to_none=True)
            with StepTimer("forward", enabled=profile_steps):
                loss = model.compute_loss(images)
            with StepTimer("backward", enabled=profile_steps):
                loss.backward()
            with StepTimer("optimizer_step", enabled=profile_steps):
                optimizer.step()

            batch_size = int(images.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            if tqdm is not None and hasattr(iterable, "set_postfix"):
                iterable.set_postfix(loss=float(loss.item()))

        if total_samples == 0:
            raise ConfigError("Training loader is empty", field="data.train_normal_dir")
        history.append({"epoch": float(epoch), "train_loss": total_loss / total_samples})

    return history


@torch.no_grad()
def score_dinomaly2_dataset(
    model: Dinomaly2FeatureModel,
    loader: DataLoader,
    *,
    idx_to_class: Mapping[int, str],
    normal_class: str,
    aggregation: str,
    tensor_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[list[float], list[int], list[str], list[str]]:
    if aggregation not in {"max", "mean"}:
        raise ConfigError(
            "method.dinomaly2.score_aggregation must be 'max' or 'mean'",
            field="method.dinomaly2.score_aggregation",
            details={"value": aggregation},
        )

    model.student_heads.eval()
    scores: list[float] = []
    labels: list[int] = []
    paths: list[str] = []
    class_names: list[str] = []
    iterable = loader if tqdm is None else tqdm(loader, desc="Dinomaly2 Eval", leave=False)
    for images, class_idx, batch_paths in iterable:
        images = images.to(model.device, non_blocking=True)
        if tensor_transform is not None:
            images = tensor_transform(images)
        batch_scores = model.score_images(images, aggregation=aggregation)
        for score, target, path in zip(batch_scores.detach().cpu().tolist(), class_idx.tolist(), batch_paths):
            class_name = idx_to_class[int(target)]
            labels.append(0 if class_name == normal_class else 1)
            scores.append(float(score))
            paths.append(str(path))
            class_names.append(class_name)
    return scores, labels, paths, class_names