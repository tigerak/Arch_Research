from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F
import timm


@dataclass
class PatchCoreConfig:
    model_name: str
    pretrained_path: str
    feature_layers: Sequence[str]
    coreset_sampling_ratio: float = 0.1
    knn_k: int = 9
    freeze_backbone: bool = True


class HookedBackbone(torch.nn.Module):
    """지정한 레이어의 출력을 수집하는 timm 백본 래퍼."""

    def __init__(self, backbone: torch.nn.Module, layer_names: Sequence[str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_names = list(layer_names)
        self._features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        named_modules = dict(self.backbone.named_modules())
        missing = [name for name in self.layer_names if name not in named_modules]
        if missing:
            raise ValueError(f"Missing feature layers in backbone: {missing}")

        for name in self.layer_names:
            module = named_modules[name]
            self._hooks.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self._features[name] = output

        return hook

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._features = {}
        _ = self.backbone.forward_features(x)
        return [self._features[name] for name in self.layer_names]

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


class PatchCoreExtractor(torch.nn.Module):
    """PatchCore용 패치 임베딩 추출기."""

    def __init__(self, cfg: PatchCoreConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.backbone = timm.create_model(cfg.model_name, pretrained=False, num_classes=0)
        self.backbone = self.backbone.to(device)
        self.backbone.eval()

        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._load_pretrained(cfg.pretrained_path)
        self.hooked = HookedBackbone(self.backbone, cfg.feature_layers)

    def _load_pretrained(self, pretrained_path: str) -> None:
        if not pretrained_path:
            return
        from pathlib import Path

        ckpt_path = Path(pretrained_path)
        if not ckpt_path.exists():
            return
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        else:
            state_dict = checkpoint
        if isinstance(state_dict, dict):
            self.backbone.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.hooked(images.to(self.device))
        patch_tokens = [self._feature_to_patches(feature) for feature in features]
        patch_tokens = [F.normalize(tokens, dim=-1) for tokens in patch_tokens]
        embedded = torch.cat(patch_tokens, dim=-1)
        return embedded.flatten(0, 1)

    @staticmethod
    def _feature_to_patches(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            if feature.shape[1] > 1:
                feature = feature[:, 1:, :]
            return feature
        if feature.ndim == 4:
            batch, channels, height, width = feature.shape
            return feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        raise ValueError(f"Unsupported feature shape: {tuple(feature.shape)}")


def build_memory_bank(embeddings: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")
    num_samples = embeddings.shape[0]
    if num_samples == 0:
        raise ValueError("No embeddings collected for memory bank")

    num_keep = max(1, int(num_samples * float(sampling_ratio)))
    if num_keep >= num_samples:
        return embeddings.contiguous()

    indices = torch.randperm(num_samples, device=embeddings.device)[:num_keep]
    return embeddings.index_select(0, indices).contiguous()


@torch.no_grad()
def compute_patch_scores(query_embeddings: torch.Tensor, memory_bank: torch.Tensor, knn_k: int) -> torch.Tensor:
    """각 이미지의 anomaly score를 계산한다.

    query_embeddings: [B, P, D]
    memory_bank: [M, D]
    return: [B]
    """
    if query_embeddings.ndim != 3:
        raise ValueError(f"query_embeddings must be 3D, got {tuple(query_embeddings.shape)}")
    if memory_bank.ndim != 2:
        raise ValueError(f"memory_bank must be 2D, got {tuple(memory_bank.shape)}")

    batch_size, patch_count, _ = query_embeddings.shape
    flat_queries = query_embeddings.reshape(batch_size * patch_count, -1)
    distances = torch.cdist(flat_queries, memory_bank)
    topk = min(int(knn_k), memory_bank.shape[0])
    nearest = distances.topk(topk, largest=False).values
    patch_scores = nearest.mean(dim=1)
    patch_scores = patch_scores.reshape(batch_size, patch_count)
    return patch_scores.max(dim=1).values
