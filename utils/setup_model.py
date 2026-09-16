import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

# from models import timm
import timm


class TransformerHeadClassifier(nn.Module):
    def __init__(self, backbone, args, num_layers=2):
        super().__init__()
        self.backbone = backbone
        self.embed_dim = backbone.num_features # 768
        self.num_classes = args.n_classes
        self.freeze_backbone = bool(getattr(args, 'freeze_backbone', True))
        self.token_fusion = getattr(args, 'token_fusion', None)  # 'attn', 'concat', 'wt_concat', or None
        
        # Freeze backbone parameters if requested.
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # 백본 평가 모드로 설정 추가
            self.backbone.eval()
            print(f"[INFO] Backbone frozen. Only transformer_head and classifier will be trained.")
        
        # 새롭게 학습할 추가 Transformer 레이어 정의
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=8, # Multi-head Attention 개수
            dim_feedforward=2048,
            dropout=0.2,
            activation='gelu',
            batch_first=True, # 입력 형태가 (Batch, Seq, Dim) 이므로 True
            norm_first=True, # LayerNorm을 나중에 적용해서 테스트 해볼 것
        )
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)

        # 토큰 단위 어텐션 풀링을 위한 경량 학습 투영층
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # cls_token과 patch_avg 비중을 조절하는 학습 게이트
        self.fusion_logit = nn.Parameter(torch.tensor(0.0))

        # Metric 분기 projection 옵션 (ArcFace / SupCon 공용)
        # - focal classifier 경로는 그대로 유지
        # - use_metric_projection=True 일 때만 metric 분기 입력을 변환
        self.use_metric_projection = bool(getattr(args, 'use_metric_projection', False))
        self.detach_metric_projection_input = bool(getattr(args, 'detach_metric_projection_input', False))
        self.metric_feature_dim = int(getattr(args, 'metric_feature_dim', self.embed_dim))
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

        # Concat 계열 fusion 시 분류기 입력 차원 조정
        if self.token_fusion in ['concat', 'wt_concat']:
            self.concat_fusion = nn.Sequential(
                nn.LayerNorm(self.embed_dim * 2),
                nn.Linear(self.embed_dim * 2, self.embed_dim),
                nn.GELU(),
            )
        
        # 최종 분류기
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.num_classes)
        )

    def forward(self, x):
        # 백본에서 전체 토큰 가져오기
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
        
        # 새 Transformer 레이어를 통과시켜 공정 도메인에 맞게 특징 재조합
        refined_features = self.transformer_head(features)
        
        # 재조합된 [CLS] 토큰(0번 인덱스) 사용
        cls_token = refined_features[:, 0, :]
        patch_tokens = refined_features[:, 1:, :]

        # # patch tokens 없는 경우 처리 (ONNX export 등에서 경고 발생 가능)
        # if not torch.jit.is_tracing() and patch_tokens.shape[1] == 0:
        #     return self.classifier(cls_token)

        if self.token_fusion == 'attn':
            # 어텐션 가중치를 이용한 패치 토큰 가중 평균
            query = self.q_proj(cls_token)  # [B, D]
            keys = self.k_proj(patch_tokens)  # [B, N-1, D]
            attn_weights = torch.einsum('bd, bnd -> bn', query, keys)  # [B, N-1]
            attn_weights = torch.softmax(attn_weights / (self.embed_dim ** 0.5), dim=1)  # [B, N-1]
            self._last_attn_weights = attn_weights

            patch_avg = torch.einsum('bn, bnd -> bd', attn_weights, patch_tokens)  # [B, D]

            alpha = torch.sigmoid(self.fusion_logit)
            combined = alpha * cls_token + (1.0 - alpha) * patch_avg
        elif self.token_fusion == 'concat':
            # 단순 평균 patch summary + concat
            patch_avg = patch_tokens.mean(dim=1)  # [B, D]
            combined = torch.cat([cls_token, patch_avg], dim=1)  # [B, 2*D]
            combined = self.concat_fusion(combined)  # [B, D]
        elif self.token_fusion == 'wt_concat':
            # attn의 gated blend 대신, weighted patch summary를 concat으로 결합
            query = self.q_proj(cls_token)  # [B, D]
            keys = self.k_proj(patch_tokens)  # [B, N-1, D]
            attn_weights = torch.einsum('bd, bnd -> bn', query, keys)  # [B, N-1]
            attn_weights = torch.softmax(attn_weights / (self.embed_dim ** 0.5), dim=1)  # [B, N-1]
            self._last_attn_weights = attn_weights

            patch_summary = torch.einsum('bn, bnd -> bd', attn_weights, patch_tokens)  # [B, D]
            combined = torch.cat([cls_token, patch_summary], dim=1)  # [B, 2*D]
            combined = self.concat_fusion(combined)  # [B, D]
        elif self.token_fusion == 'cls_only':
            combined = cls_token
        elif self.token_fusion == 'sum05':
            # cls_token과 patch_avg를 단순 평균
            patch_avg = patch_tokens.mean(dim=1)  # [B, D]
            combined = 0.5 * cls_token + 0.5 * patch_avg
        else:
            # 단순 평균 풀링 (*0.5도 안 함 <- 초기 버전)
            patch_avg = patch_tokens.mean(dim=1)  # [B, D] 
            combined = cls_token + patch_avg

        # Feature extraction mode: Sub-center ArcFace 등에서 사용
        self._last_feature = combined
        metric_input = combined.detach() if self.detach_metric_projection_input else combined
        if self.metric_projection is not None:
            self._last_metric_feature = self.metric_projection(metric_input)
        else:
            self._last_metric_feature = metric_input
        
        return self.classifier(combined)

    def get_metric_feature(self):
        if hasattr(self, '_last_metric_feature') and self._last_metric_feature is not None:
            return self._last_metric_feature
        if hasattr(self, '_last_feature') and self._last_feature is not None:
            return self._last_feature
        return None
        

    #
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self


## FTF 탭접힘에서 contrastive learning을 활용 .. class2, 3을 잘 나누기 위함 
class SupConModel(nn.Module):

    def __init__(self, backbone, feat_dim=512, proj_dim=128, num_classes=3):
        super().__init__()
        self.backbone = backbone  # returns feature vector of size feat_dim

        # projection head for contrastive
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, proj_dim)
        )
        # classification head for supervised CE (optional, can help)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        # Step 1: Backbone에서 feature 추출
        feat_map  = self.backbone.forward_features(x)  # [B, feat_dim]

        # Step 2: Global average pooling → (B, 576)
        feat = feat_map.mean(dim=[2, 3])

        # Step 3: Projection head (normalized for contrastive loss)
        proj = self.proj(feat)
        proj = F.normalize(proj, dim=1)

        # Step 4: Classification head
        logits = self.classifier(feat)

        return feat, proj, logits