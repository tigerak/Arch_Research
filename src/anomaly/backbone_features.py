from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import timm


class HookedFeatureBackbone(torch.nn.Module):
    """지정한 레이어 출력을 수집하는 범용 timm 백본 래퍼."""

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

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._features = {}
        _ = self.backbone.forward_features(x)
        return [self._features[name] for name in self.layer_names]

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks = []


class LayerFeatureExtractor(torch.nn.Module):
    """선택한 레이어의 patch token을 추출하는 범용 특징 추출기."""

    def __init__(
        self,
        *,
        model_name: str,
        pretrained_path: str,
        feature_layers: Sequence[str],
        device: torch.device,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        self.backbone = self.backbone.to(device)
        self.backbone.eval()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self._load_pretrained(pretrained_path)
        self.hooked = HookedFeatureBackbone(self.backbone, feature_layers)

    def _load_pretrained(self, pretrained_path: str) -> None:
        if not pretrained_path:
            return

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

    def get_input_image_size(self) -> int | None:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        img_size = getattr(patch_embed, "img_size", None)
        if isinstance(img_size, tuple) and img_size:
            return int(img_size[0])
        if isinstance(img_size, list) and img_size:
            return int(img_size[0])
        if isinstance(img_size, int):
            return int(img_size)
        return None

    @torch.no_grad()
    def forward_layer_tokens(self, images: torch.Tensor, *, normalize: bool = False) -> list[torch.Tensor]:
        features = self.hooked(images.to(self.device))
        patch_tokens = [self.feature_to_patch_tokens(feature) for feature in features]
        if normalize:
            patch_tokens = [F.normalize(tokens, dim=-1) for tokens in patch_tokens]
        return patch_tokens

    @staticmethod
    def feature_to_patch_tokens(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            if feature.shape[1] > 1:
                feature = feature[:, 1:, :]
            return feature
        if feature.ndim == 4:
            batch, channels, height, width = feature.shape
            return feature.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        raise ValueError(f"Unsupported feature shape: {tuple(feature.shape)}")