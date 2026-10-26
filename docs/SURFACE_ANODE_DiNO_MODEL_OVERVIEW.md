# SURFACE_ANODE_DiNO 모델 구조 및 추론 흐름 설명

## 1. 문서 목적

이 문서는 SURFACE_ANODE_DiNO 분류 모델의 구조, 학습 objective, 추론 경로, 그리고 기존 Focal Loss 모델과의 차이를 backend 개발자에게 전달하기 위한 설명서입니다.

현재 실험에서 사용되는 주요 설정은 다음과 같습니다.

- 모델: DINOv2 기반 backbone + transformer head
- loss: ArcFace
- head 설정: `token_fusion: "cls_only"`
- metric projection: 활성화 (`use_metric_projection: true`)
- metric feature dimension: 384
- 학습 시 backbone freeze: true

핵심 설정 파일:
- [configs/classification/SURFACE_ANODE_DiNO/Arc_CLS_S32_M02_Proj.yaml](../configs/classification/SURFACE_ANODE_DiNO/Arc_CLS_S32_M02_Proj.yaml)

---

## 2. 모델 요약

이 모델은 단순히 DINOv2 backbone만 사용하는 것이 아니라, 다음 단계들을 거칩니다.

1. DINOv2 backbone으로 이미지에서 patch token과 CLS token 추출
2. CLS token을 transformer head로 다시 정제
3. 선택적으로 metric projection을 적용해 ArcFace에 적합한 feature space 생성
4. ArcFace sub-center loss로 최종 분류 margin 학습

즉, 단순 분류기보다 “feature embedding이 잘 분리되도록 학습하는 구조”에 가깝습니다.

이 구조는 표면 결함 분류처럼 클래스를 구분하기 어렵고 유사도가 높을 때 특히 유리합니다.

---

## 3. 현재 모델의 실제 구성

### 3.1 Backbone

모델 생성 로직은 [src/classification/builders.py](../src/classification/builders.py) 에서 구성됩니다.

- 기본 backbone: `timm` 기반 DINOv2 small
- `freeze_backbone: true` 이므로 pretrained backbone은 고정
- 학습 가능한 레이어는 주로 transformer head와 classifier 그리고 ArcFace 파라미터

DINOv2의 핵심 아이디어는 이미지를 patch token sequence로 변환하고, CLS token을 중심으로 전역 의미를 유지하는 것입니다. 이 프로젝트에서는 그 token sequence를 다시 classification head에서 다듬는 구조를 사용합니다.

### 3.2 TransformerHeadClassifier

핵심 구현은 [src/classification/models.py](../src/classification/models.py) 의 `TransformerHeadClassifier` 입니다.

이 클래스는 다음을 수행합니다.

- backbone.forward_features(x) 호출
- token sequence 형태의 feature 확보
- transformer encoder로 token 관계을 재학습
- CLS token을 최종 pooled representation으로 사용
- 선택적으로 metric projection 수행
- 최종 classifier로 클래스 logit 계산

핵심 구조는 다음과 같습니다.

```python
self.backbone
self.transformer_head = nn.TransformerEncoder(...)
self.q_proj = nn.Linear(embed_dim, embed_dim)
self.k_proj = nn.Linear(embed_dim, embed_dim)
self.metric_projection = nn.Sequential(
    nn.LayerNorm(embed_dim),
    nn.Linear(embed_dim, embed_dim),
    nn.GELU(),
    nn.Linear(embed_dim, metric_feature_dim),
)
self.classifier = nn.Sequential(
    nn.LayerNorm(embed_dim),
    nn.Linear(embed_dim, num_classes),
)
```

### 3.3 token_fusion = "cls_only"

현재 설정 파일에서 `token_fusion: "cls_only"`입니다. 즉, 최종 representation은 CLS token만 사용합니다.

- patch tokens는 transformer에서 처리되지만
- 최종 결합에는 CLS token만 사용
- 모델은 전역적 의미를 더 강하게 반영
- 표면 결함처럼 부분적 defect가 섞인 이미지에서 클래스 간 경계를 더 명확하게 만들 수 있음

다른 옵션들(`attn`, `concat`, `wt_concat`, `sum05`)도 있지만 현재 실험에서는 `cls_only`를 사용하고 있습니다.

### 3.4 metric projection

`use_metric_projection: true` 이므로 모델은 class logits 외에 metric feature를 별도로 생성합니다.

```python
self._last_metric_feature = self.metric_projection(metric_input)
```

이 feature는 ArcFace loss에서 입력 feature로 사용됩니다.

즉, 모델이 단순히 출력 logit만 만드는 것이 아니라, 다음과 같은 두 가지 feature를 유지합니다.

- `combined`: classification용 feature
- `metric_feature`: ArcFace용 feature

ArcFace는 단순 logits cross-entropy가 아니라, embedding space에서 클래스 중심과의 angular margin을 강화하는 방식이므로, 전용 metric feature가 매우 중요합니다.

---

## 4. ArcFace loss의 의미

### 4.1 기본 개념

현재 모델은 Focal Loss가 아니라 ArcFace를 사용합니다. ArcFace 구현은 [src/classification/methods/arcface.py](../src/classification/methods/arcface.py) 에서 확인할 수 있습니다.

ArcFace의 핵심 아이디어는 다음과 같습니다.

- feature를 L2 정규화
- class center weight도 L2 정규화
- cosine similarity를 계산
- target class의 angular margin을 추가
- 가장 비슷한 class center와의 각도를 키워 class separation 강화

수식적으로는 보통 다음과 같은 형태를 갖습니다.

- $cos(\theta)$를 계산
- target class에 대해 $\theta + m$ 만큼 마진 부여
- 최종 logits = $s \cdot cos(\theta + m)$

여기서:

- $m$: margin
- $s$: scale

현재 설정에서는 다음 값을 사용합니다.

- `arcface_margin: 0.2`
- `arcface_scale: 32.0`
- `arcface_num_sub_centers: 4`

### 4.2 Sub-center ArcFace

이 구현은 일반 ArcFace가 아니라 Sub-center ArcFace 입니다.

각 클래스에 대해 여러 sub-center를 두고, 해당 클래스의 best matching center만 사용합니다. 구현 핵심은 다음입니다.

```python
self.weight = nn.Parameter(torch.empty(self.num_classes, self.num_sub_centers, self.feature_dim))
self.active_mask = torch.ones(self.num_classes, self.num_sub_centers, dtype=torch.bool)
```

즉, 같은 class 내부에도 여러 중심점을 두고, 실제 각 샘플은 가장 가까운 sub-center를 선택합니다. 이를 통해 다음 효과를 기대할 수 있습니다.

- 클래스 내부의 다중 패턴 대응
- noisy sample에 대한 robustness 증가
- class 내부 variation이 큰 결함 종류에서 더 안정적인 분리

특히 표면 결함들은 서로 비슷하면서도 세부 패턴이 다양한 편이므로, sub-center 기반 분리가 매우 유리합니다.

### 4.3 pruning

이 구현은 일정 epoch 이후 sub-center 중 사용률이 낮은 center를 prune합니다.

설정:

- `arcface_prune_warmup_epochs: 20`
- `arcface_prune_interval: 5`
- `arcface_prune_min_usage: 2`
- `arcface_prune_max_remove_per_class: 1`

따라서 초기 안정 학습 후, 단순히 클래스마다 하나의 center만 고정하지 않고 불필요한 sub-center를 줄여 학습 안정성을 높입니다.

---

## 5. Focal Loss와의 차이

### 5.1 Focal Loss

Focal Loss는 [src/classification/methods/focal.py](../src/classification/methods/focal.py) 에 구현되어 있습니다.

기본 형태는 다음과 같습니다.

- CE loss 계산
- 예측 확신이 높은 sample에 모듈레이션 factor $ (1-p_t)^\gamma $ 적용
- hard example에 더 큰 손실 부여

즉,

- 쉬운 sample은 덜 학습
- 어려운 sample은 더 많이 학습

이 방식은 불균형 데이터에서 매우 유용합니다.

### 5.2 ArcFace와의 핵심 차이

| 항목 | Focal Loss | ArcFace |
|---|---|---|
| 목적 | hard example 강조 | class margin 강화 |
| feature space | 보통 logits 기반 | embedding space 기반 |
| 중심 개념 | 확률 분포 | class prototype / angular margin |
| 좋은 상황 | 클래스 불균형 | 유사 클래스 분리 |
| 현재 적용 | 일반 분류 | 미세 결함 분류 |

Focal Loss는 “얼마나 어려운 샘플인가?”를 강조합니다.

ArcFace는 “어떤 클래스 중심과 얼마나 멀리/가깝게 위치하는가?”를 강조합니다.

표면 결함 데이터는 클래스 간 시각적 유사도가 높기 때문에, 단순 CE/Focal보다 feature margin이 분리 성능을 크게 좌우할 수 있습니다.

### 5.3 현재 모델에서 ArcFace가 의미 있는 이유

표면 결함은 다음 특성을 보입니다.

- CRACK, PROTRUSION, CRATER 같은 클래스가 유사한 형태를 가진다
- 파생 클래스들(CRACK_1, CRACK_2, PROTRUSION_1, PROTRUSION_2 등)이 형태상 비슷하다
- 단순 cross-entropy만으로는 class boundary가 약하게 잡히기 쉽다

ArcFace는 이런 “비슷하지만 다른 class”를 더 잘 분리하도록 도와줍니다.

그래서 현재 실험에서는 Focal Loss 대신 ArcFace 구조를 택한 것으로 보입니다.

---

## 6. 실제 추론 흐름

### 6.1 모델 빌드

[src/classification/builders.py](../src/classification/builders.py) 의 `build_model()` 함수는 다음 순서로 모델을 구성합니다.

1. DINOv2 backbone 생성
2. `head_type == "transformer"` 이면 `TransformerHeadClassifier` 생성
3. `freeze_backbone` 옵션 적용
4. pretrained checkpoint 로드
5. 최종 모델 반환

### 6.2 전처리

입력은 보통 이미지 크기 224x224이며, evaluation/train 모두 동일한 기본 이미지 크기를 사용합니다.

- `train.img_size = [224, 224]`
- `build_eval_loader`에서 테스트 이미지 로딩

### 6.3 forward path

모델 forward 동작은 [src/classification/models.py](../src/classification/models.py) 에서 다음처럼 이루어집니다.

1. `features = self.backbone.forward_features(x)`
2. `refined_features = self.transformer_head(features)`
3. `cls_token = refined_features[:, 0, :]`
4. `combined = cls_token` (현재 `cls_only` 설정)
5. `self.metric_projection(combined)` 로 metric feature 생성
6. `self.classifier(combined)` 로 class logits 생성

즉, 최종 모델 출력은 보통 다음 형태입니다.

- logits: [B, num_classes]
- metric feature: [B, metric_feature_dim]

### 6.4 ArcFace 훈련 시

학습 단계에서는 다음이 수행됩니다.

- metric feature를 ArcFace loss에 입력
- class별 center weight와 cosine similarity 계산
- margin 추가
- cross-entropy on angular logits 계산
- loss backprop

### 6.5 평가 시

평가 코드 [src/eval_engine.py](../src/eval_engine.py) 에서는 다음 흐름으로 동작합니다.

- 이미지 로드
- model.eval()
- logits = model(images)
- ArcFace일 경우 `criterion`은 평가 loss 계산용으로 CrossEntropy로 대체되며,
- `model.get_metric_feature()` 또는 ArcFace inference path를 통해 의미 있는 feature를 처리
- `argmax`로 예측 클래스 결정
- confusion matrix 계산

즉, 훈련과 추론 시 ArcFace는 “로짓 단순 분류기”처럼 보이지만 실제로는 feature space와 angular margin을 함께 쓰는 구조입니다.

---

## 7. 현재 실험의 구조적 의미

이번 실험의 핵심은 단순히 “좋은 backbone + 더 많은 epoch”이 아니라, 다음 조합입니다.

- DINOv2 feature extractor
- transformer head refinement
- metric projection
- ArcFace sub-center margin learning
- class-level neighborhood separation

이 조합은 기존 Focal Loss 기준 모델보다 세밀한 defect class를 구분하는 데 유리합니다.

특히 다음과 같은 상황에서 유리합니다.

- visually similar defects
- class 내부 variation이 큰 사례
- 같은 defect family 내 여러 하위 타입이 존재하는 경우

---

## 8. 백엔드 관점에서 이해해야 할 포인트

Backend 입장에서 이 모델을 이해할 때 가장 중요한 점은 다음입니다.

### 8.1 입력과 출력

- 입력: 이미지 tensor [B, 3, H, W]
- 출력: logits [B, num_classes]
- 최종 예측: argmax(logits)

### 8.2 실제 분류는 “logit”보다 “feature margin”에 의존

이 모델은 단순 확률 분류가 아니라, feature space의 angular margin이 분류 품질을 결정합니다.

그래서 backend에서는 다음을 인지하는 것이 중요합니다.

- output prob 자체보다 class boundary가 얼마나 분리되어 있는지 중요
- 유사 class 간 오분류가 출력으로 나타날 수 있음
- 모델이 같은 family 안의 하위 타입 차이를 구분하는 목적을 가짐

### 8.3 라벨 그룹핑이 의미 있음

실제 데이터에는 같은 family의 하위 클래스로 번호가 붙는 경우가 많습니다.

예:

- CRACK_1, CRACK_2 → CRACK
- PROTRUSION_1, PROTRUSION_2, PROTRUSION_3, PROTRUSION_4 → PROTRUSION
- CRATER_1, CRATER_2 → CRATER

이 경우 backend 리포트에서 individual 클래스나 collapsed 클래스 둘 다 확인하는 것이 유용합니다.

---

## 9. 정리

이 모델의 핵심은 “DINOv2 feature + transformer refinement + ArcFace metric learning” 조합입니다.

- Focal Loss는 hard example를 강조하는 분류 손실
- ArcFace는 class center 간 거리를 강조하는 metric 손실
- 현재의 표면 결함 분류는 유사 defect가 많아 ArcFace가 더 적합한 구조
- 최종 추론은 backbone → transformer head → metric projection → classifier로 진행

즉, 현재 모델은 단순 실험용 분류 모델이 아니라, **유사 클래스 분리와 세밀한 defect 차별을 위해 설계된 metric-learning 기반 분류 모델**입니다.

---

## 10. 참고 파일

- [src/classification/models.py](../src/classification/models.py)
- [src/classification/builders.py](../src/classification/builders.py)
- [src/classification/methods/focal.py](../src/classification/methods/focal.py)
- [src/classification/methods/arcface.py](../src/classification/methods/arcface.py)
- [src/eval_engine.py](../src/eval_engine.py)
- [configs/classification/SURFACE_ANODE_DiNO/Arc_CLS_S32_M02_Proj.yaml](../configs/classification/SURFACE_ANODE_DiNO/Arc_CLS_S32_M02_Proj.yaml)
