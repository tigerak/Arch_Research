# Anomaly torchvision v2 GPU Augmentation Draft

## Goal

이 문서는 anomaly track, 특히 Dinomaly2 학습에서 CPU 중심 이미지 전처리를 일부 GPU/tensor 기반 파이프라인으로 옮기기 위한 리팩터링 초안이다.

목표는 다음과 같다.

- CPU worker 병목 완화
- GPU idle 시간 감소
- 기존 classification/anomaly 공통 구조를 깨지 않고 점진 이관

---

## Current State

현재 전처리 핵심은 `src/dataset_builder.py`의 `img_transform(size, mode)`에 있다.

train transform에는 다음이 포함되어 있다.

- grayscale to RGB 변환
- resize
- horizontal/vertical flip
- downscale-restore
- color jitter
- sharpness
- autocontrast/equalize
- CLAHE
- JPEG recompression
- blur
- noise
- normalize

문제점:

- PIL, numpy, OpenCV, tensor 연산이 한 파이프라인에 섞여 있다.
- 고해상도 `518 x 518`에서 CPU worker 비용이 크다.
- Dinomaly2처럼 GPU가 놀고 CPU worker만 바쁜 경우가 생긴다.

---

## Design Principle

리팩터링 원칙은 다음과 같다.

1. 디코딩 직후 최소 전처리만 CPU에서 수행한다.
2. batch 단위로 이득이 있는 변환은 tensor/GPU 단계로 옮긴다.
3. 무거운 비-tensor 변환은 가능한 한 줄이거나 offline preprocessing 대상으로 분리한다.
4. classification과 anomaly 모두 재사용 가능한 공통 인터페이스를 우선 설계한다.

---

## Recommended Split

전처리를 두 단계로 나눈다.

### Stage 1: CPU Decode / Minimal Normalize Input

역할:

- 파일 로드
- grayscale to RGB
- resize
- tensor 변환

이 단계는 DataLoader worker에서 수행한다.

남길 항목:

- `StackGrayToRGB`
- `Resize`
- `ToImage` 또는 `ToTensor` 전환

제거 후보:

- CLAHE
- JPEG recompression
- blur
- noise
- color jitter
- autocontrast/equalize

### Stage 2: Tensor/GPU Augment

역할:

- flip
- blur
- color jitter
- normalize
- noise
- 가능한 tensor 기반 contrast/photometric 변환

이 단계는 training step 직전 또는 model forward 직전에 적용한다.

---

## Transform Classification

현 전처리를 세 그룹으로 나눈다.

### A. GPU/tensor 이관 권장

- horizontal flip
- vertical flip
- resize if tensor-first pipeline is used consistently
- color jitter
- gaussian blur
- normalize
- gaussian noise

추천 구현:

- `torchvision.transforms.v2`
- 필요 시 `kornia` 보완 사용

### B. CPU 유지 또는 빈도 대폭 축소

- CLAHE
- JPEG recompression
- PIL/OpenCV 기반 equalize/autocontrast 대체가 필요한 경우

이유:

- tensor/GPU 이관 난이도 대비 체감 이득이 불확실하거나 구현 복잡도가 높음
- 학습마다 무조건 필요하지 않을 수 있음

### C. Offline preprocessing 후보

- CLAHE
- JPEG 품질 변형 사전 생성본
- downscale-restore의 deterministic variant

---

## Proposed File Layout

기존 파일명을 무리하게 재사용하지 않고 역할에 맞춰 분리한다.

### Common transform builders

- `src/data/transform_presets.py`
- `src/data/tensor_augmentations.py`
- `src/data/image_decoding.py`

### Anomaly-specific bridge

- `src/anomaly/image_loaders.py`
- `src/anomaly/augmentation_profiles.py`

### Optional future

- `src/common/augmentation_runtime.py`

현재 repo 구조를 크게 흔들고 싶지 않다면 1차 단계에서는 아래처럼 간다.

- keep: `src/dataset_builder.py`
- add: `src/data/tensor_augmentations.py`
- add: `src/anomaly/augmentation_profiles.py`

---

## Proposed API

### 1. CPU-side base transform

`src/dataset_builder.py`

```python
def build_base_image_transform(size: int, *, mode: str) -> v2.Transform:
    ...
```

역할:

- grayscale to RGB
- resize
- convert to tensor image

### 2. Tensor augmentation profile

`src/data/tensor_augmentations.py`

```python
def build_tensor_gpu_augmentation(*, mode: str):
    ...
```

### 3. Loader contract

`src/anomaly/image_loaders.py`

```python
def build_image_anomaly_loaders(
    cfg,
    *,
    image_size_override=None,
):
    ...
```

### 4. Runtime application

runner 안에서:

```python
tensor_augment = build_tensor_gpu_augmentation(mode="train")
images = tensor_augment(images)
```

GPU 적용 시:

```python
images = images.to(device, non_blocking=True)
images = tensor_augment(images)
```

---

## torchvision v2 Mapping Draft

### CPU base transform draft

```python
from torchvision.transforms import v2

base_train = v2.Compose([
    StackGrayToRGB(),
    v2.Resize((size, size), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])

base_eval = v2.Compose([
    StackGrayToRGB(),
    v2.Resize((size, size), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
])
```

### Tensor/GPU train profile draft

```python
from torchvision.transforms import v2

train_gpu_aug = v2.Compose([
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomVerticalFlip(p=0.3),
    v2.RandomApply([v2.ColorJitter(brightness=0.08, contrast=0.10)], p=0.2),
    v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.05),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Eval GPU transform draft

```python
eval_gpu_transform = v2.Compose([
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

### Noise helper draft

`v2`에 가우시안 노이즈가 직접 충분하지 않으면 custom tensor transform 유지:

```python
class AddGaussianNoiseTensor:
    def __init__(self, p=0.08, std=0.006):
        self.p = p
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        return x + torch.randn_like(x) * self.std
```

---

## Recommended First GPU Scope

초기 이관 범위는 다음 정도가 적절하다.

- horizontal flip
- vertical flip
- color jitter
- gaussian blur
- normalize
- gaussian noise

이유:

- batch tensor에 바로 적용 가능하다.
- `torchvision v2`로 무리 없이 옮길 수 있다.
- CPU worker 병목을 줄일 가능성이 크다.
- 모델 의미를 크게 바꾸지 않는다.

반대로 아래는 1차 이관 대상에서 제외한다.

- CLAHE
- JPEG recompression
- downscale-restore

이 셋은 CPU에 남기거나 offline preprocessing으로 분리하는 편이 안전하다.

---

## Migration Plan

### Phase 1: Interface introduction

- `img_transform()`는 유지한다.
- 새 함수 `build_base_image_transform()` 추가
- 새 함수 `build_tensor_gpu_augmentation(mode)` 추가
- CPU transform과 GPU transform을 동시에 반환하는 경로 추가

목표:

- 기존 코드와 완전 호환
- GPU augmentation 호출 지점을 명시적으로 분리

### Phase 2: Dinomaly2 opt-in

- Dinomaly2 runner에서만 GPU augmentation 호출
- 성능과 GPU utilization 비교

### Phase 3: PatchCore opt-in

- PatchCore도 동일 인터페이스 사용
- memory bank 생성/평가 경로에서 normalize 위치만 주의해서 적용

### Phase 4: Deprecated path cleanup

- 사용하지 않는 PIL/OpenCV augmentation 축소
- `img_transform()`를 thin wrapper로 축소

---

## Config Draft

이 초안의 초점은 profile이 아니라 runtime 분리다. 따라서 설정도 단순하게 가져간다.

```yaml
train:
    gpu_augment: true
```

필요하면 augmentation 적용 자체와 runtime만 분리한다.

```yaml
train:
    gpu_augment: true
    cpu_only_heavy_aug: false
```

핵심은 method별 이름을 늘리는 것이 아니라, 현재 있는 augmentation 중 어떤 연산을 GPU 단계로 넘길지 제어하는 것이다.

---

## Risks

1. PIL 기반 변환과 tensor 기반 변환의 수치 차이로 재현성이 달라질 수 있다.
2. CLAHE/JPEG recompression 제거 시 성능이 변할 수 있다.
3. GPU augmentation은 forward 직전 적용 위치를 잘못 잡으면 profiling이 헷갈릴 수 있다.
4. 너무 많은 GPU augmentation은 오히려 model compute와 경합할 수 있다.

---

## Success Criteria

- GPU utilization 유의미 상승
- DataLoader worker crash 빈도 감소
- epoch time 단축
- Dinomaly2 smoke run 안정화
- PatchCore와 Dinomaly2 모두 같은 CPU/GPU transform contract 사용 가능

---

## First Implementation Slice

가장 작은 첫 구현 단위는 다음이다.

1. `src/dataset_builder.py`에 tensor-first base transform 추가
2. `src/data/tensor_augmentations.py` 추가
3. `src/anomaly/image_loaders.py`가 CPU base transform만 담당하도록 정리
4. `src/anomaly/methods/dinomaly2_runner.py`에서 GPU augmentation 호출 추가

이 단계에서는 PatchCore는 건드리지 않아도 된다.

---

## Recommendation

지금 상황에서는 전면 교체보다 Dinomaly2 학습 경로에서만 먼저 GPU augmentation을 붙여 보는 것이 가장 안전하다.

그 후:

- throughput 개선 확인
- GPU utilization 비교
- worker 안정성 확인

이 통과하면 PatchCore까지 같은 CPU/GPU transform contract로 확장한다.