import torch
import torch.nn as nn

from src.classification.methods.arcface import SubCenterArcFace
from src.classification.methods.focal import FocalLossModule, focal_loss
from src.classification.methods.supcon import SupConLoss
from src.common.checkpoint_utils import extract_model_state_dict

__all__ = [
    "cross_entropy_loss",
    "focal_loss",
    "FocalLossModule",
    "SubCenterArcFace",
    "SupConLoss",
    "extract_model_state_dict",
]


def cross_entropy_loss(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)
