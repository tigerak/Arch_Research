from __future__ import annotations

from src.classification.methods.arcface import build_loss as build_arcface_loss
from src.classification.methods.ce import build_loss as build_ce_loss
from src.classification.methods.focal import build_loss as build_focal_loss

__all__ = ["build_arcface_loss", "build_ce_loss", "build_focal_loss"]
