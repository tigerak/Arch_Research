from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from src.config_loader import ConfigError

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


class StackGrayToRGB:
    """입력 이미지를 항상 3채널 RGB 형태로 정규화한다.

    공정 이미지가 8bit 1채널일 때는 채널을 3번 복제해 RGB로 맞추고,
    이미 RGB인 경우에는 그대로 통과시킨다.
    """

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.mode == "RGB":
            return img
        if img.mode in {"L", "I;16", "I", "F"}:
            gray = img.convert("L")
            arr = np.array(gray, dtype=np.uint8)
            rgb = np.stack([arr, arr, arr], axis=-1)
            return Image.fromarray(rgb, mode="RGB")
        return img.convert("RGB")


class RandomDownscaleRestore:
    """크롭 없이 축소 후 복원하는 스케일 변형 증강.

    작은 결함이 위치한 좌표를 크게 바꾸지 않으면서,
    해상도/리샘플링 품질 변화에 대한 강건성을 높이기 위한 목적이다.
    """

    def __init__(self, p: float = 0.5, scale: Tuple[float, float] = (0.85, 1.0)) -> None:
        self.p = float(p)
        self.scale = scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        w, h = img.size
        low, high = self.scale
        s = random.uniform(low, high)
        new_w = max(1, int(w * s))
        new_h = max(1, int(h * s))

        small = TF.resize(img, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        restored = TF.resize(small, [h, w], interpolation=InterpolationMode.BILINEAR)
        return restored


class RandomCLAHE:
    """CLAHE(적응형 히스토그램 평활화) 기반 지역 대비 향상 증강.

    OpenCV(cv2)가 있으면 실제 CLAHE를 수행하고,
    없는 환경에서는 PIL Equalize로 안전하게 대체한다.
    """

    def __init__(
        self,
        p: float = 0.25,
        clip_limit: Tuple[float, float] = (1.5, 3.0),
        tile_grid_size: Tuple[int, int] = (8, 8),
    ) -> None:
        self.p = float(p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        # cv2가 없는 경우에도 파이프라인이 깨지지 않도록 대체 경로를 제공한다.
        if cv2 is None:
            return ImageOps.equalize(img)

        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)

        clip = random.uniform(self.clip_limit[0], self.clip_limit[1])
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=self.tile_grid_size)
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])

        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        return Image.fromarray(out)


class RandomJPEGCompression:
    """JPEG 재인코딩으로 저장 품질 편차를 시뮬레이션한다."""

    def __init__(self, p: float = 0.2, quality: Tuple[int, int] = (60, 95)) -> None:
        self.p = float(p)
        self.quality = quality

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img

        q = int(random.uniform(self.quality[0], self.quality[1]))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        out = Image.open(buf).convert("RGB")
        return out


class AddGaussianNoise:
    """텐서 입력에 가우시안 노이즈를 추가한다."""

    def __init__(self, p: float = 0.15, mean: float = 0.0, std: float = 0.01) -> None:
        self.p = float(p)
        self.mean = float(mean)
        self.std = float(std)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        noise = torch.randn_like(tensor) * self.std + self.mean
        return tensor + noise


def img_transform(size: int, mode: str = "train") -> transforms.Compose:
    """SURFACE 데이터셋용 transform.

    - 8bit 1ch 이미지는 3ch로 스태킹해 RGB 파이프라인으로 통일
    - mode=train에서는 형태/대비/압축/노이즈 계열 증강을 적용
    - mode=eval에서는 resize + normalize만 적용
    """
    if mode == "train":
        transform = transforms.Compose(
            [
                StackGrayToRGB(),
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                RandomDownscaleRestore(p=0.25, scale=(0.9, 1.0)),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.10, contrast=0.12)],
                    p=0.3,
                ),
                transforms.RandomApply(
                    [transforms.RandomAdjustSharpness(sharpness_factor=1.3, p=1.0)],
                    p=0.12,
                ),
                transforms.RandomApply(
                    [
                        transforms.RandomChoice(
                            [
                                transforms.RandomAutocontrast(p=1.0),
                                transforms.RandomEqualize(p=1.0),
                            ]
                        )
                    ],
                    p=0.2,
                ),
                RandomCLAHE(p=0.15, clip_limit=(1.2, 2.2), tile_grid_size=(8, 8)),
                RandomJPEGCompression(p=0.1, quality=(75, 95)),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.6))],
                    p=0.08,
                ),
                transforms.ToTensor(),
                AddGaussianNoise(p=0.08, mean=0.0, std=0.006),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                StackGrayToRGB(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transform


def build_train_val_loaders(cfg: Mapping[str, Any]) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    """학습/검증 데이터로더를 생성한다.

    규칙:
    - train_dir, test_dir 바로 아래 폴더명을 클래스명으로 사용
    - 각 클래스 폴더의 하위 폴더까지 재귀적으로 이미지 수집
      (torchvision.datasets.ImageFolder의 기본 동작)
    """
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")

    train_dir = Path(str(data_cfg.get("train_dir", "")))
    test_dir = Path(str(data_cfg.get("test_dir", "")))
    if not train_dir.exists():
        raise ConfigError("Train directory not found", field="data.train_dir", file_path=train_dir)
    if not test_dir.exists():
        raise ConfigError("Test directory not found", field="data.test_dir", file_path=test_dir)

    img_size = train_cfg.get("img_size", [224, 224])
    if isinstance(img_size, list) and len(img_size) == 2:
        size = int(img_size[0])
    elif isinstance(img_size, int):
        size = int(img_size)
    else:
        raise ConfigError("train.img_size must be int or [H,W]", field="train.img_size", details={"value": img_size})

    train_ds = datasets.ImageFolder(str(train_dir), transform=img_transform(size=size, mode="train"))
    val_ds = datasets.ImageFolder(str(test_dir), transform=img_transform(size=size, mode="eval"))

    # 클래스 이름 불일치가 있으면 지표 해석이 틀어지므로 여기서 즉시 차단한다.
    if train_ds.classes != val_ds.classes:
        raise ConfigError(
            "Train/Test class names are different",
            field="data.train_dir/data.test_dir",
            details={"train_classes": train_ds.classes, "test_classes": val_ds.classes},
        )

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

    idx_to_class = {idx: name for name, idx in train_ds.class_to_idx.items()}
    return train_loader, val_loader, idx_to_class


def build_eval_loader(cfg: Mapping[str, Any]) -> Tuple[DataLoader, Dict[int, str]]:
    """단독 평가용 데이터로더를 생성한다."""
    data_cfg = _section(cfg, "data")
    train_cfg = _section(cfg, "train")

    test_dir = Path(str(data_cfg.get("test_dir", "")))
    if not test_dir.exists():
        raise ConfigError("Evaluation directory not found", field="data.test_dir", file_path=test_dir)

    img_size = train_cfg.get("img_size", [224, 224])
    if isinstance(img_size, list) and len(img_size) == 2:
        size = int(img_size[0])
    elif isinstance(img_size, int):
        size = int(img_size)
    else:
        raise ConfigError("train.img_size must be int or [H,W]", field="train.img_size", details={"value": img_size})

    ds = datasets.ImageFolder(str(test_dir), transform=img_transform(size=size, mode="eval"))

    batch_size = int(train_cfg.get("batch_size", 32))
    num_workers = int(train_cfg.get("num_workers", 4))
    persistent_workers = bool(train_cfg.get("persistent_workers", True)) and num_workers > 0
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    idx_to_class = {idx: name for name, idx in ds.class_to_idx.items()}
    return loader, idx_to_class


def _section(cfg: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = cfg.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError("Config section must be a mapping", field=name)
    return value
