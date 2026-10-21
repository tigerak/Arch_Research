from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.config_loader import ConfigError
from src.dataset_builder import build_base_image_transform, img_transform


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder 변형으로 이미지 경로까지 함께 반환한다."""

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path


class SubsetWithPaths(torch.utils.data.Dataset):
    """ImageFolder subset에서 경로를 함께 보존한다."""

    def __init__(self, dataset: datasets.ImageFolder, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        sample_index = self.indices[index]
        sample, target = self.dataset[sample_index]
        path, _ = self.dataset.samples[sample_index]
        return sample, target, path


def build_image_anomaly_loaders(
    cfg: Mapping[str, Any],
    *,
    image_size_override: int | None = None,
    use_gpu_augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str], str]:
    """이미지 기반 anomaly method용 train/val/test DataLoader를 생성한다."""
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")
    anomaly_cfg = _section(cfg, "anomaly")

    train_normal_dir = Path(str(data_cfg.get("train_normal_dir", "")))
    test_dir = Path(str(data_cfg.get("test_dir", "")))
    if not train_normal_dir.exists():
        raise ConfigError("Normal training directory not found", field="data.train_normal_dir", file_path=train_normal_dir)
    if not test_dir.exists():
        raise ConfigError("Test directory not found", field="data.test_dir", file_path=test_dir)

    if image_size_override is not None:
        size = int(image_size_override)
    else:
        img_size = train_cfg.get("img_size", [224, 224])
        if isinstance(img_size, list) and len(img_size) == 2:
            size = int(img_size[0])
        elif isinstance(img_size, int):
            size = int(img_size)
        else:
            raise ConfigError("train.img_size must be int or [H,W]", field="train.img_size", details={"value": img_size})

    profile_transforms = bool(train_cfg.get("profile_transforms", False))

    if use_gpu_augment:
        train_transform = build_base_image_transform(size=size, mode="train", profile=profile_transforms)
        test_transform = build_base_image_transform(size=size, mode="eval", profile=profile_transforms)
    else:
        train_transform = img_transform(size=size, mode="train", profile=profile_transforms)
        test_transform = img_transform(size=size, mode="eval", profile=profile_transforms)

    full_train_ds = datasets.ImageFolder(str(train_normal_dir), transform=train_transform)
    test_ds = ImageFolderWithPaths(str(test_dir), transform=test_transform)

    val_ratio = float(train_cfg.get("val_ratio", train_cfg.get("validation_ratio", 0.1)))
    if not 0.0 < val_ratio < 1.0:
        raise ConfigError("train.val_ratio must be between 0 and 1", field="train.val_ratio", details={"value": val_ratio})

    if len(full_train_ds) == 0:
        raise ConfigError("Normal training directory is empty", field="data.train_normal_dir")

    generator = torch.Generator().manual_seed(int(train_cfg.get("seed", 42)))
    train_indices = torch.randperm(len(full_train_ds), generator=generator).tolist()
    val_count = max(1, int(round(len(train_indices) * val_ratio))) if len(train_indices) > 1 else 0
    val_count = min(val_count, len(train_indices) - 1)
    val_indices = train_indices[:val_count]
    train_indices = train_indices[val_count:]

    train_ds = SubsetWithPaths(full_train_ds, train_indices)
    val_ds = SubsetWithPaths(full_train_ds, val_indices)

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    idx_to_class = {idx: name for name, idx in test_ds.class_to_idx.items()}
    normal_class = str(anomaly_cfg.get("normal_class", "OK"))
    if normal_class not in test_ds.class_to_idx:
        raise ConfigError(
            "normal_class not found in test_dir class folders",
            field="anomaly.normal_class",
            details={"normal_class": normal_class, "classes": sorted(test_ds.class_to_idx.keys())},
        )

    return train_loader, val_loader, test_loader, idx_to_class, normal_class


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value