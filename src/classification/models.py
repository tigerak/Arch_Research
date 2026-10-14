from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

"""classification task용 커스텀 모델 정의 모듈.

이 파일에는 실제 backbone을 감싸는 classification head 모델들이 정의된다.

주요 구성:
- TransformerHeadClassifier: ViT/DINO 계열 토큰 특성을 후처리하는 head
- SupConModel: contrastive supervision용 부가 모델

이 구조는 단순히 timm 모델 하나를 쓰는 것이 아니라,
토큰 융합, metric projection, classifier head까지 구성하기 위해 별도 모델 클래스로 분리한 것에 의미가 있다.
"""


class TransformerHeadClassifier(nn.Module):
    def __init__(self, backbone, args, num_layers: int = 2):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.num_features
        self.num_classes = args.n_classes
        self.freeze_backbone = bool(getattr(args, "freeze_backbone", True))
        self.token_fusion = getattr(args, "token_fusion", None)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()
            print("[INFO] Backbone frozen. Only transformer_head and classifier will be trained.")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.2,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))

        self.use_metric_projection = bool(getattr(args, "use_metric_projection", False))
        self.detach_metric_projection_input = bool(getattr(args, "detach_metric_projection_input", False))
        self.metric_feature_dim = int(getattr(args, "metric_feature_dim", self.embed_dim))
        if self.use_metric_projection:
            self.metric_projection = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.GELU(),
                nn.Linear(self.embed_dim, self.metric_feature_dim),
            )
        else:
            self.metric_projection = None
            self.metric_feature_dim = self.embed_dim

        if self.token_fusion in ["concat", "wt_concat"]:
            self.concat_fusion = nn.Sequential(
                nn.LayerNorm(self.embed_dim * 2),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
            )

        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes),
        )

    def forward(self, x):
        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                features = self.backbone.forward_features(x)
        else:
            features = self.backbone.forward_features(x)

        if features.ndim != 3:
            raise ValueError(
                f"Expected token features with shape [B, N, D], got {tuple(features.shape)}. "
                "This head requires a backbone that returns token sequences with a CLS token."
            )

        refined_features = self.transformer_head(features)
        cls_token = refined_features[:, 0, :]
        patch_tokens = refined_features[:, 1:, :]

        if self.token_fusion == "attn":
            query = self.q_proj(cls_token)
            keys = self.k_proj(patch_tokens)
            attn_weights = torch.einsum("bd, bnd -> bn", query, keys)
            attn_weights = torch.softmax(attn_weights / (self.embed_dim ** 0.5), dim=1)
            self._last_attn_weights = attn_weights
            patch_avg = torch.einsum("bn, bnd -> bd", attn_weights, patch_tokens)
            alpha = torch.sigmoid(self.fusion_logit)
            combined = alpha * cls_token + (1.0 - alpha) * patch_avg
        elif self.token_fusion == "concat":
            patch_avg = patch_tokens.mean(dim=1)
            combined = torch.cat([cls_token, patch_avg], dim=1)
            combined = self.concat_fusion(combined)
        elif self.token_fusion == "wt_concat":
            query = self.q_proj(cls_token)
            keys = self.k_proj(patch_tokens)
            attn_weights = torch.einsum("bd, bnd -> bn", query, keys)
            attn_weights = torch.softmax(attn_weights / (self.embed_dim ** 0.5), dim=1)
            self._last_attn_weights = attn_weights
            patch_summary = torch.einsum("bn, bnd -> bd", attn_weights, patch_tokens)
            combined = torch.cat([cls_token, patch_summary], dim=1)
            combined = self.concat_fusion(combined)
        elif self.token_fusion == "cls_only":
            combined = cls_token
        elif self.token_fusion == "sum05":
            patch_avg = patch_tokens.mean(dim=1)
            combined = 0.5 * cls_token + 0.5 * patch_avg
        else:
            patch_avg = patch_tokens.mean(dim=1)
            combined = cls_token + patch_avg

        self._last_feature = combined
        metric_input = combined.detach() if self.detach_metric_projection_input else combined
        if self.metric_projection is not None:
            self._last_metric_feature = self.metric_projection(metric_input)
        else:
            self._last_metric_feature = metric_input

        return self.classifier(combined)

    def get_metric_feature(self):
        if hasattr(self, "_last_metric_feature") and self._last_metric_feature is not None:
            return self._last_metric_feature
        if hasattr(self, "_last_feature") and self._last_feature is not None:
            return self._last_feature
        return None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


class SupConModel(nn.Module):
    def __init__(self, backbone, feat_dim: int = 512, proj_dim: int = 128, num_classes: int = 3):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat_map = self.backbone.forward_features(x)
        feat = feat_map.mean(dim=[2, 3])
        proj = self.proj(feat)
        proj = F.normalize(proj, dim=1)
        logits = self.classifier(feat)
        return feat, proj, logits
