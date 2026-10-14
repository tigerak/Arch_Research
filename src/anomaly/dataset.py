from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from src.config_loader import ConfigError
from src.dataset_builder import img_transform


class ImageFolderWithPaths(datasets.ImageFolder):
    """ImageFolder 변형으로 이미지 경로까지 함께 반환한다."""

    def __getitem__(self, index: int):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return sample, target, path


def build_patchcore_loaders(cfg: Mapping[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[int, str], str]:
    """PatchCore용 train/test DataLoader를 생성한다.

    - train_normal_dir: 정상 샘플만 포함한 학습 루트
    - test_dir: 정상 + 이상 샘플을 함께 포함한 평가 루트
    - normal_class: 평가 시 정상으로 간주할 클래스 이름
    """
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")
    anomaly_cfg = _section(cfg, "anomaly")

    train_normal_dir = Path(str(data_cfg.get("train_normal_dir", "")))
    test_dir = Path(str(data_cfg.get("test_dir", "")))
    if not train_normal_dir.exists():
        raise ConfigError("Normal training directory not found", field="data.train_normal_dir", file_path=train_normal_dir)
    if not test_dir.exists():
        raise ConfigError("Test directory not found", field="data.test_dir", file_path=test_dir)

    img_size = train_cfg.get("img_size", anomaly_cfg.get("image_size", [224, 224]))
    if isinstance(img_size, list) and len(img_size) == 2:
        size = int(img_size[0])
    elif isinstance(img_size, int):
        size = int(img_size)
    else:
        raise ConfigError("train.img_size must be int or [H,W]", field="train.img_size", details={"value": img_size})

    train_ds = datasets.ImageFolder(str(train_normal_dir), transform=img_transform(size=size, mode="train"))
    test_ds = ImageFolderWithPaths(str(test_dir), transform=img_transform(size=size, mode="eval"))

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

    return train_loader, test_loader, idx_to_class, normal_class


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
