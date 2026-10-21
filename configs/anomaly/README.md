# 이상치 탐지 설정 규약

이 폴더는 이 저장소의 이상치 탐지 실험 트랙을 정의한다.

## 데이터 규약

학습에는 정상 데이터만 사용하고, 평가에는 정상/이상 데이터를 함께 넣는 구성을 권장한다.

권장 디렉터리 구조는 다음과 같다.

```text
D:\TAPING_Model\IMG_DATA_AD\PATCHCORE\
  TRAIN_NORMAL\
    OK\
      *.png / *.jpg
  TEST\
    OK\
      *.png / *.jpg
    BUBBLE\
      *.png / *.jpg
    CONTAMINATION\
      *.png / *.jpg
    PARTICLE\
      *.png / *.jpg
  MASK\
    OK\
    BUBBLE\
    CONTAMINATION\
    PARTICLE\
```

규칙:
- `TRAIN_NORMAL`에는 정상 샘플만 둔다.
- `TEST`에는 정상과 이상 클래스를 함께 둔다.
- 픽셀 단위 마스크가 있으면 `MASK` 아래에 두고, 테스트 이미지와 동일한 파일 stem을 사용한다.
- `ImageFolder` 방식으로 읽을 경우 학습/평가 간 클래스 이름을 일관되게 유지한다.

## 설정 규칙

- `task.type`은 반드시 `anomaly_detection`이어야 한다.
- `data.train_normal_dir`는 정상 데이터만 있는 학습 루트다.
- `data.test_dir`는 평가 루트다.
- `train.val_ratio`는 정상 학습 데이터에서 검증용 비율을 정한다. 기본값은 `0.1`이며, 이 비율을 통해 train/val을 분리한다.
- 검증 데이터는 임계치 조정용으로 사용하고, 최종 평가는 별도의 test_dir를 사용한다.
- `model.backbone_name` 또는 `model.model_name`으로 feature extractor를 지정해야 한다.
- `anomaly.normal_class`는 정상으로 간주할 클래스 이름이다. 기본값은 `OK`다.
- `anomaly.method_name`으로 AD 방법을 선택한다. 예: `patchcore`
- 공통 threshold 정책은 `anomaly.threshold` 아래에 둔다. 예: `policy: tpr95`
- method별 세부 설정은 `method.<name>` 아래에 둔다.
- PatchCore를 사용할 경우 `method.patchcore.feature_layers`, `method.patchcore.coreset_sampling_ratio`, `method.patchcore.knn_k`가 필요하다.

## 첫 베이스라인

처음에는 PatchCore부터 붙이고, 그 다음에 다른 방법을 추가하는 순서를 권장한다.

추천 순서:
1. PatchCore
2. PaDiM
3. DRAEM 또는 Reverse Distillation 같은 학습형 방법 1개
4. Dinomaly 계열 방법
