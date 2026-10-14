# Arch Research Experiment Portal

이 프로젝트는 여러 task와 여러 method를 공통 포털 구조로 운영하는 실험 코드베이스입니다.

핵심 목표는 다음과 같습니다.

- classification과 anomaly를 같은 실행 구조 안에서 관리
- task별 engine과 method registry 분리
- base config + custom config 병합을 통한 실험 자동화
- 여러 실험을 schedule 기반으로 일괄 실행
- 결과 저장, 로그, 요약 CSV를 통일된 방식으로 관리

---

## 1. 프로젝트 개요

현재 구조는 다음과 같은 계층으로 나뉩니다.

- src: 실제 실험 로직
- src/common: 여러 task에서 재사용하는 helper
- utils: 로깅, 환경 설정, 운영 보조 도구
- configs: base/custom 실험 설정 파일
- root: 실행 진입점(main_*.py), 스케줄러, 포털 launcher

실험은 보통 아래 흐름으로 동작합니다.

1. 루트의 main_*.py 실행
2. src/entrypoints.py의 공통 래퍼 호출
3. src/run_manager.py가 schedule과 config 병합 처리
4. task type 판별
5. classification 또는 anomaly engine 호출
6. registry 기반 method 선택
7. 학습 또는 평가 수행
8. 로그/체크포인트/요약 결과 저장

---

## 2. 디렉터리 구조

```text
Arch_Research/
├── README.md
├── run_schedule.py
├── main_classification_train.py
├── main_classification_eval.py
├── main_anomaly_train.py
├── main_anomaly_eval.py
├── configs/
│   ├── classification/
│   │   └── TAPING_DiNO/
│   │       ├── base_config.yaml
│   │       └── *.yaml
│   └── anomaly/
│       └── WA_DiNO/
│           ├── base_config.yaml
│           └── *.yaml
├── src/
│   ├── config_loader.py
│   ├── run_manager.py
│   ├── entrypoints.py
│   ├── experiment_resolver.py
│   ├── train_engine.py
│   ├── eval_engine.py
│   ├── dataset_builder.py
│   ├── classification/
│   │   ├── engine.py
│   │   ├── builders.py
│   │   ├── models.py
│   │   ├── registry.py
│   │   └── methods/
│   │       ├── ce.py
│   │       ├── focal.py
│   │       ├── arcface.py
│   │       └── supcon.py
│   ├── anomaly/
│   │   ├── engine.py
│   │   ├── registry.py
│   │   ├── dataset.py
│   │   └── methods/
│   │       └── patchcore.py
│   └── common/
│       └── checkpoint_utils.py
├── utils/
│   ├── setup_logger.py
│   └── setup_model.py
├── models/
│   ├── pretrained/
│   └── save/
├── IMG_DATA/
└── docs/
    ├── PORTAL_ARCHITECTURE.md
    ├── ENTRYPOINTS_GUIDE.md
    └── REFACTOR_CHECKLIST.md
```

---

## 3. 실행 방법

### 3.1 Classification 학습

```bash
python main_classification_train.py
```

기본 경로는 다음과 같은 구조를 사용합니다.

- [main_classification_train.py](main_classification_train.py)
- [src/entrypoints.py](src/entrypoints.py)
- [src/run_manager.py](src/run_manager.py)

### 3.2 Classification 평가

```bash
python main_classification_eval.py
```

### 3.3 Anomaly 학습

```bash
python main_anomaly_train.py
```

### 3.4 Anomaly 평가

```bash
python main_anomaly_eval.py
```

### 3.5 여러 실험 일괄 실행

```bash
python run_schedule.py
```

`run_schedule.py`는 schedule key를 읽고, base config와 custom config를 병합해서 여러 실험을 순차 실행합니다.

---

## 4. 설정 방식

가장 중요한 설정 파일은 [src/config_loader.py](src/config_loader.py) 안에서 로드/검증됩니다.

핵심 동작:

- base YAML을 읽음
- custom YAML을 재귀 병합
- task type 확인
- 필수 키 검사
- loss, scheduler, anomaly 조건 검증

예시 구조:

```text
configs/
├── classification/
│   └── TAPING_DiNO/
│       ├── base_config.yaml
│       ├── Arc_CLS_S32_M02.yaml
│       └── Focal_CLS_FA1.yaml
└── anomaly/
    └── WA_DiNO/
        ├── base_config.yaml
        └── DINOv2_PatchCore_01.yaml
```

실험은 보통 다음 형태로 실행됩니다.

- base config: 공통 기본값
- custom config: 실험별 오버라이드 값
- merged config: 최종 실행 설정

---

## 5. classification 작업

classification task는 `task.type=classification`을 기준으로 동작합니다.

주요 파일:

- [src/classification/engine.py](src/classification/engine.py)
- [src/classification/builders.py](src/classification/builders.py)
- [src/classification/models.py](src/classification/models.py)
- [src/classification/registry.py](src/classification/registry.py)

지원 방식 예시:

- CE loss
- Focal loss
- ArcFace loss

---

## 6. anomaly 작업

anomaly task는 `task.type=anomaly_detection`을 기준으로 동작합니다.

주요 파일:

- [src/anomaly/engine.py](src/anomaly/engine.py)
- [src/anomaly/registry.py](src/anomaly/registry.py)
- [src/anomaly/dataset.py](src/anomaly/dataset.py)
- [src/anomaly/methods/patchcore.py](src/anomaly/methods/patchcore.py)

현재 baseline method는 PatchCore이고, 구조가 registry 기반으로 확장되도록 설계되어 있습니다.

---

## 7. 데이터 폴더 기대 구조

anomaly 코드는 다음 형태를 기대합니다.

```text
IMG_DATA/
└── WA_DiNO/
    ├── TRAIN/
    │   └── OK/
    │       ├── ok_001.png
    │       ├── ok_002.png
    │       └── ...
    └── TEST/
        ├── OK/
        │   ├── ok_001.png
        │   └── ...
        ├── BUBBLE/
        │   ├── bubble_001.png
        │   └── ...
        ├── CONTAMINATION/
        │   └── ...
        └── PARTICLE/
            └── ...
```

이 구조와 다르면 [src/anomaly/dataset.py](src/anomaly/dataset.py) 에서 `ImageFolder` 기반 데이터셋 로딩이 실패할 수 있습니다.

---

## 8. 검증 상태

아래 항목은 실제로 확인했습니다.

- config 파일 로딩 및 병합 검증: 통과
- classification base config smoke run: 통과
- anomaly base config smoke run: 데이터셋 구조 불일치로 실패

즉, classification은 현재 실행 경로가 정상적으로 구동되며, anomaly는 실제 데이터 구조를 코드가 기대하는 구조로 맞춰야 학습이 시작됩니다.

---

## 9. 참고 문서

추가 설명 문서:

- [docs/PORTAL_ARCHITECTURE.md](docs/PORTAL_ARCHITECTURE.md)
- [docs/ENTRYPOINTS_GUIDE.md](docs/ENTRYPOINTS_GUIDE.md)
- [docs/REFACTOR_CHECKLIST.md](docs/REFACTOR_CHECKLIST.md)

---

## 10. 요약

이 저장소는 단일 실험이 아니라 여러 실험을 비교/운영하는 실험 포털에 가깝습니다.

- 공통 실행기
- task별 engine
- registry 기반 method 확장
- config 기반 자동화
- 결과 저장 구조

이 구조를 기준으로 실험을 추가하거나 새로운 method를 붙이는 방식으로 확장할 수 있습니다.
