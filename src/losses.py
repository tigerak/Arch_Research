import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
import math


def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets)

def focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='mean'):
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # 확률 변환
    
    if alpha is None:
        alpha_factor = torch.ones_like(ce_loss)
    
    elif isinstance(alpha, (list, tuple)):
        alpha_t = torch.tensor(alpha, dtype=logits.dtype, device=logits.device)
        alpha_factor = alpha_t[targets]
    elif torch.is_tensor(alpha):
        alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)
        if alpha_t.ndim == 0:
            alpha_factor = torch.full_like(ce_loss, float(alpha_t.item()))
        else:
            alpha_factor = alpha_t[targets]
    else:
        alpha_factor = torch.full_like(ce_loss, float(alpha))

    loss = alpha_factor * (1.0 - pt).pow(gamma) * ce_loss

    if reduction == 'sum':
        return loss.sum()
    if reduction == 'none':
        return loss
    return loss.mean()


class SubCenterArcFace(nn.Module):
    """Sub-center ArcFace loss.

    각 클래스마다 여러 sub-center를 두고, 클래스 점수는 그중 가장 높은 center로 계산한다.
    pruning은 warmup 이후에만 수행하며, 한 번에 클래스당 최대 1개만 비활성화한다.
    """

    def __init__(self, num_classes, feature_dim, num_sub_centers=4, margin=0.3, scale=64.0):
        super().__init__()
        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_sub_centers = int(num_sub_centers)
        self.margin = float(margin)
        self.scale = float(scale)

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.num_sub_centers, self.feature_dim))
        nn.init.xavier_uniform_(self.weight)

        self.register_buffer('active_mask', torch.ones(self.num_classes, self.num_sub_centers, dtype=torch.bool))
        self.register_buffer('sub_center_usage', torch.zeros(self.num_classes, self.num_sub_centers))

    def _cosine_logits(self, features):
        # Feature / center를 모두 L2 normalize해서 cosine similarity를 직접 로짓으로 사용한다.
        features = F.normalize(features, dim=1)
        weight = F.normalize(self.weight, dim=2)
        return torch.einsum('bd,ckd->bck', features, weight)

    def inference_logits(self, features):
        """Margin 없이 ArcFace 분류 logits를 반환합니다.

        검증/평가에서는 ground-truth label 기반 margin 조작을 하지 않고,
        각 클래스에서 가장 유사한 sub-center의 cosine 값(max over K)을
        scale만 곱한 logits로 사용합니다.
        """
        cosine = self._cosine_logits(features)
        cosine = cosine.masked_fill(~self.active_mask.unsqueeze(0), -1e4)
        class_logits, _ = cosine.max(dim=2)
        return class_logits * self.scale

    def forward(self, features, labels):
        batch_size = features.size(0)
        device = features.device

        cosine = self._cosine_logits(features)
        cosine = cosine.masked_fill(~self.active_mask.unsqueeze(0), -1e4)

        class_logits, class_best_idx = cosine.max(dim=2)

        batch_indices = torch.arange(batch_size, device=device)
        target_cos = class_logits[batch_indices, labels].clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(target_cos)
        class_logits[batch_indices, labels] = torch.cos(theta + self.margin)

        target_center_idx = class_best_idx[batch_indices, labels]
        self.sub_center_usage[labels, target_center_idx] += 1

        return F.cross_entropy(class_logits * self.scale, labels)

    def prune_unused_subcenters(self, warmup_epochs=3, current_epoch=0, min_usage=1, max_remove_per_class=1):
        """사용량이 낮은 sub-center를 보수적으로 비활성화한다.

        - warmup 이전에는 pruning을 수행하지 않고 usage 통계만 초기화
        - 클래스당 최소 1개 center는 반드시 유지
        - 한 번의 호출에서 클래스당 최대 max_remove_per_class만 제거
        """
        if current_epoch < warmup_epochs:
            self.sub_center_usage.zero_()
            return 0

        removed_total = 0
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                active_idx = torch.nonzero(self.active_mask[class_idx], as_tuple=False).flatten()
                if active_idx.numel() <= 1:
                    continue

                usage = self.sub_center_usage[class_idx, active_idx]
                low_usage_order = torch.argsort(usage)

                removed_for_class = 0
                for order_idx in low_usage_order.tolist():
                    center_idx = active_idx[order_idx].item()
                    if usage[order_idx].item() > min_usage:
                        break
                    if self.active_mask[class_idx].sum().item() <= 1:
                        break
                    self.active_mask[class_idx, center_idx] = False
                    removed_total += 1
                    removed_for_class += 1
                    if removed_for_class >= max_remove_per_class:
                        break

            # 다음 pruning 주기 전까지 usage를 다시 누적하기 위해 초기화.
            self.sub_center_usage.zero_()

        return removed_total


def extract_model_state_dict(weight_obj):
    """체크포인트 호환 로더.

    새 포맷(dict: model_state_dict/arcface_state_dict/...)과
    구 포맷(raw state_dict) 둘 다 지원한다.
    """
    if isinstance(weight_obj, dict) and 'model_state_dict' in weight_obj:
        return weight_obj['model_state_dict']
    return weight_obj


class SupConLoss(nn.Module):
    """Numerically stable supervised contrastive loss.

    Expected input:
    - features: [2B, D] (two augmented views concatenated)
    - labels: [B]
    """

    def __init__(self, temperature=0.07, base_temperature=0.07, class_weights=None, eps=1e-12):
        super().__init__()
        # temperature가 너무 작거나 0이면 수치적으로 불안정해지고,
        # division by zero 문제가 생기므로 생성 시점에 방어한다.
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if base_temperature <= 0:
            raise ValueError(f"base_temperature must be > 0, got {base_temperature}")

        # temperature:
        #   유사도(logit) 스케일. 작을수록 분포가 뾰족해져 hard negative 압력이 커진다.
        # base_temperature:
        #   최종 loss scaling 계수. 논문/레퍼런스 구현 호환을 위해 분리.
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)
        # log(0) 방지용 최소값
        self.eps = float(eps)

        # class_weights를 주면 애매한 클래스(anchor)에 더 큰 contrastive 가중치를 줄 수 있다.
        # 예: 11개 클래스 중 혼동이 심한 클래스에 1.2~1.8 배 가중치
        if class_weights is not None:
            weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', weight_tensor)
        else:
            self.class_weights = None

    def forward(self, features, labels):
        # 입력 형태 가정:
        # features: [2B, D] (두 view를 concat)
        # labels  : [B]
        if features.ndim != 2:
            raise ValueError(f"features must be 2D [2B, D], got shape={tuple(features.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D [B], got shape={tuple(labels.shape)}")

        device = features.device
        batch = labels.shape[0]
        if features.shape[0] != 2 * batch:
            raise ValueError("features should be 2*B in first dim when labels length is B")

        # cosine similarity를 쓰기 위해 feature를 정규화한다.
        # (upstream에서 normalize 했더라도 안전하게 한 번 더 수행)
        features = F.normalize(features, dim=1)
        # 두 view를 concat했으므로 label도 동일한 순서로 복제한다.
        labels = labels.repeat(2)
        n = labels.shape[0]

        # 1) pairwise similarity matrix (N x N)
        #    logits[i, j] = sim(z_i, z_j) / temperature
        logits = torch.matmul(features, features.T) / self.temperature

        # 2) 수치 안정화:
        #    exp 전에 row-wise max를 빼면 overflow 위험이 크게 줄어든다.
        #    detach를 써서 이 상수 이동이 gradient를 왜곡하지 않도록 한다.
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        # 3) 마스크 구성
        #    - self_mask: 자기 자신(i==j)은 분모/분자에서 제외
        #    - positive_mask: 같은 클래스이면서 자기 자신이 아닌 쌍
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & (~self_mask)

        # 4) log_prob 계산
        #    log_prob(i,j) = logits(i,j) - log(sum_{k!=i} exp(logits(i,k)))
        #    즉 anchor i 기준의 소프트맥스 로그확률.
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps))

        # 5) anchor별 positive 평균 로그확률
        #    positive가 0개인 anchor(배치 내 해당 클래스가 1개뿐인 경우)는
        #    일단 분모 clamp로 안전 계산 후, 아래 valid 마스크에서 제외한다.
        positive_count = positive_mask.sum(dim=1)
        positive_count_f = positive_count.clamp_min(1).float()
        mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_count_f

        # valid anchor: positive pair가 1개 이상 있는 anchor만 loss에 반영
        valid = positive_count > 0
        if not torch.any(valid):
            # 중요한 안정화 포인트:
            # 배치 전체에 valid positive가 없더라도 그래프를 끊지 않기 위해
            # 상수 0이 아닌 "features에 연결된 0"을 반환한다.
            # (torch.tensor(0.) 반환 대비, autograd/분산학습/로깅 연동에 안전)
            return features.sum() * 0.0

        # SupCon 기본식:
        # L = -(T / T_base) * mean_{i in valid}( mean_{p in P(i)} log_prob(i,p) )
        per_anchor_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid]

        if self.class_weights is not None:
            # 클래스 가중치 길이 검증: 최대 label 인덱스를 커버해야 한다.
            max_class = int(labels.max().item()) if labels.numel() > 0 else -1
            if max_class >= self.class_weights.numel():
                raise ValueError(
                    f"class_weights length ({self.class_weights.numel()}) is smaller than max label index ({max_class})"
                )

            # anchor 라벨 기준 가중평균.
            # 혼동이 심한 클래스의 representation 분리를 더 강하게 유도한다.
            anchor_weights = self.class_weights[labels[valid]].to(device=device, dtype=per_anchor_loss.dtype)
            denom = anchor_weights.sum().clamp_min(self.eps)
            return (per_anchor_loss * anchor_weights).sum() / denom

        # 기본은 valid anchor 평균
        return per_anchor_loss.mean()