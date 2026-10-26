from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from src.anomaly.backbone_features import LayerFeatureExtractor
from src.config_loader import ConfigError


@dataclass
class PatchCoreConfig:
    model_name: str
    pretrained_path: str
    feature_layers: tuple[str, ...]
    coreset_sampling_ratio: float = 0.1
    knn_k: int = 9
    freeze_backbone: bool = True


class PatchCoreFeatureExtractor(torch.nn.Module):
    """PatchCore용 패치 임베딩 추출기."""

    def __init__(self, cfg: PatchCoreConfig, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = LayerFeatureExtractor(
            model_name=cfg.model_name,
            pretrained_path=cfg.pretrained_path,
            feature_layers=cfg.feature_layers,
            device=device,
            freeze_backbone=cfg.freeze_backbone,
        )

    def get_input_image_size(self) -> int | None:
        return self.feature_extractor.get_input_image_size()

    @property
    def device(self) -> torch.device:
        return self.feature_extractor.device

    @torch.no_grad()
    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.feature_extractor.forward_layer_tokens(images, normalize=True)
        return torch.cat(patch_tokens, dim=-1)

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        embedded = self.forward_tokens(images)
        return embedded.flatten(0, 1)


@torch.no_grad()
def build_patchcore_memory_bank(
    extractor: PatchCoreFeatureExtractor,
    loader: DataLoader,
    *,
    sampling_ratio: float,
    max_memory_bank_patches: int,
) -> torch.Tensor:
    reservoir: torch.Tensor | None = None
    seen_embeddings = 0
    filled_embeddings = 0
    iterable = loader if tqdm is None else tqdm(loader, desc="PatchCore Train", leave=False)
    for batch in iterable:
        images = batch[0]
        embeddings = extractor.embed(images).detach().cpu()
        embeddings = _sample_embeddings(embeddings, sampling_ratio)
        if embeddings.numel() == 0:
            continue
        reservoir, seen_embeddings, filled_embeddings = _update_reservoir(
            reservoir,
            seen_embeddings,
            filled_embeddings,
            embeddings,
            max_memory_bank_patches,
        )
        if tqdm is not None and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(kept=int(filled_embeddings), seen=int(seen_embeddings))
    if reservoir is None or filled_embeddings == 0:
        raise ConfigError("Training loader is empty", field="data.train_normal_dir")
    return reservoir[:filled_embeddings].contiguous()


@torch.no_grad()
def score_patchcore_dataset(
    extractor: PatchCoreFeatureExtractor,
    memory_bank: torch.Tensor,
    *,
    knn_k: int,
    loader: DataLoader,
    idx_to_class: Mapping[int, str],
    normal_class: str,
    threshold: float | None = None,
    prediction_csv_path: str | None = None,
    collect_paths: bool = True,
) -> tuple[list[float], list[int], list[str], list[str]]:
    scores: list[float] = []
    labels: list[int] = []
    paths: list[str] = []
    class_names: list[str] = []

    csv_writer = None
    csv_file = None
    if prediction_csv_path is not None:
        import csv
        csv_file = open(prediction_csv_path, "w", encoding="utf-8", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["image_path", "class_name", "binary_label", "score", "prediction", "threshold"])
        csv_writer.writeheader()

    memory_bank = memory_bank.to(extractor.device)
    try:
        iterable = loader if tqdm is None else tqdm(loader, desc="PatchCore Eval", leave=False)
        for batch in iterable:
            images, class_idx, batch_paths = batch
            embedded = extractor.forward_tokens(images)
            batch_scores = compute_patch_scores(embedded, memory_bank, knn_k)

            for score, target, path in zip(batch_scores.detach().cpu().tolist(), class_idx.tolist(), batch_paths):
                class_name = idx_to_class[int(target)]
                label = 0 if class_name == normal_class else 1
                score_value = float(score)
                labels.append(label)
                scores.append(score_value)
                if collect_paths:
                    paths.append(str(path))
                    class_names.append(class_name)
                if csv_writer is not None:
                    prediction = 1 if threshold is not None and score_value >= threshold else 0
                    csv_writer.writerow({
                        "image_path": str(path),
                        "class_name": class_name,
                        "binary_label": int(label),
                        "score": score_value,
                        "prediction": int(prediction),
                        "threshold": float(threshold) if threshold is not None else 0.0,
                    })
    finally:
        if csv_file is not None:
            csv_file.close()

    return scores, labels, paths, class_names


@torch.no_grad()
def compute_patch_scores(query_embeddings: torch.Tensor, memory_bank: torch.Tensor, knn_k: int) -> torch.Tensor:
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


def _sample_embeddings(embeddings: torch.Tensor, sampling_ratio: float) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got {tuple(embeddings.shape)}")
    if embeddings.shape[0] == 0:
        return embeddings

    num_keep = max(1, int(embeddings.shape[0] * float(sampling_ratio)))
    if num_keep >= embeddings.shape[0]:
        return embeddings

    indices = torch.randperm(embeddings.shape[0])[:num_keep]
    return embeddings.index_select(0, indices)


def _update_reservoir(
    reservoir: torch.Tensor | None,
    seen_embeddings: int,
    filled_embeddings: int,
    embeddings: torch.Tensor,
    max_memory_bank_patches: int,
) -> tuple[torch.Tensor, int, int]:
    current = reservoir
    seen = int(seen_embeddings)
    filled = int(filled_embeddings)

    if current is None:
        current = torch.empty((max_memory_bank_patches, embeddings.shape[1]), dtype=embeddings.dtype)

    if filled < max_memory_bank_patches:
        num_to_copy = min(max_memory_bank_patches - filled, embeddings.shape[0])
        current[filled:filled + num_to_copy] = embeddings[:num_to_copy]
        filled += num_to_copy
        seen += num_to_copy
        embeddings = embeddings[num_to_copy:]

    for row in embeddings:
        seen += 1
        replacement_index = int(torch.randint(0, seen, (1,)).item())
        if replacement_index < max_memory_bank_patches:
            current[replacement_index] = row

    return current, seen, filled