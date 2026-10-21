from __future__ import annotations

import torch
from torchvision.transforms import v2


class AddGaussianNoiseTensor(torch.nn.Module):
    def __init__(self, p: float = 0.08, mean: float = 0.0, std: float = 0.006) -> None:
        super().__init__()
        self.p = float(p)
        self.mean = float(mean)
        self.std = float(std)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=tensor.device).item() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def build_tensor_gpu_augmentation(*, mode: str) -> v2.Compose:
    if mode == "train":
        return v2.Compose(
            [
                v2.RandomHorizontalFlip(p=0.3),
                v2.RandomVerticalFlip(p=0.3),
                v2.RandomApply([v2.ColorJitter(brightness=0.10, contrast=0.12)], p=0.3),
                v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))], p=0.08),
                AddGaussianNoiseTensor(p=0.08, mean=0.0, std=0.006),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    if mode == "eval":
        return v2.Compose(
            [
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    raise ValueError(f"Unsupported augmentation mode: {mode}")


__all__ = ["AddGaussianNoiseTensor", "build_tensor_gpu_augmentation"]