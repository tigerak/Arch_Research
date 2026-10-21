# 최종 실험 포털 구조 리팩터 체크리스트

## 목표

이 프로젝트는 단일 모델 전용 실험 코드가 아니라, 여러 실험 트랙을 공통 인터페이스로 관리하는 실험 포털 구조를 유지해야 한다.

기본 책임 분리는 다음과 같다.

- src = 실제 실험 로직
- src/common = 여러 task에서 재사용되는 공통 helper
- utils = 로거, 환경 셋업, 호환성 지원 등 운영 보조 코드
- root = 엔트리포인트 및 스케줄링 코드

---

## 1. 책임 분리 원칙

### src에 유지해야 하는 것

- 모델 아키텍처 코드
- 학습 로직
- 평가 로직
- 데이터셋/데이터 파이프라인
- method registry
- method별 loss/optimizer 로직
- task별 engine
- config validation 및 merge 로직

### src/common에 유지해야 하는 것

- checkpoint/state 정규화 helper
- 공통 tensor/device 변환 유틸
- task를 넘나드는 수치 보정/중립화 함수
- 여러 실험에서 재사용되는 도메인 공통 로직

### utils에 유지해야 하는 것

- logger 생성
- 결과 출력 서식
- compatibility wrapper
- 운영 환경 보조 코드
- CLI/runner glue

즉, 실험 자체를 결정하는 로직은 src에 두고, 실제 실행 환경을 돕는 코드는 utils로 분리한다.

---

## 2. 현재 구조 분류

### src

- src/config_loader.py
- src/dataset_builder.py
- src/train_engine.py
- src/eval_engine.py
- src/run_manager.py
- src/experiment_resolver.py
- src/classification/**
- src/anomaly/**

### src/common

- src/common/__init__.py
- src/common/checkpoint_utils.py

### utils

- utils/setup_logger.py
- utils/setup_model.py

### root

- run_schedule.py
- main_classification_train.py
- main_classification_eval.py
- main_anomaly_train.py
- main_anomaly_eval.py
- export_onnx.py

---

## 3. src 안에서 유지해야 하는 핵심 파일

### classification track

- src/classification/builders.py
- src/classification/models.py
- src/classification/engine.py
- src/classification/registry.py
- src/classification/methods/ce.py
- src/classification/methods/focal.py
- src/classification/methods/arcface.py

### anomaly track

- src/anomaly/engine.py
- src/anomaly/registry.py
- src/anomaly/backbone_features.py
- src/anomaly/image_loaders.py
- src/anomaly/image_level_metrics.py
- src/anomaly/methods/patchcore_runner.py
- src/anomaly/methods/patchcore_components.py
- src/anomaly/methods/dinomaly2_runner.py
- src/anomaly/methods/dinomaly2_components.py

이 파일들은 실제 알고리즘과 실험 실행 흐름을 담당하므로 절대로 utils로 밀어 넣어서는 안 된다.

---

## 4. utils에 두어도 되는 파일의 기준

다음 파일은 실험 로직이 아니라 보조 도구로 남아야 한다.

- utils/setup_logger.py
- utils/setup_model.py

이들은 실제 모델 계산을 수행하지 않고, 실행 환경을 보조하는 역할만 한다.

---

## 5. 보관 전 반드시 검사할 기준

각 파일을 볼 때는 아래 질문을 기준으로 판단해야 한다.

- 이 파일이 실제 모델/손실/평가 로직을 정의하는가?
- 이 파일이 특정 method 구현에 속하는가?
- 이 파일이 여러 task 엔진에서 공통으로 쓰이는가?
- 이 파일이 단순한 logger, bridge, compatibility 코드인가?

판단 기준은 다음과 같다.

- 실제 알고리즘/실험 제어: src
- 실행 환경 보조: utils
- 재사용 가능한 도메인 helper: src/common

---

## 6. 현재 리팩터 상태

### 완료된 항목

- classification method 분리 완료
- anomaly method 분리 완료
- task별 엔진 구조 정립 완료
- 공통 checkpoint helper 존재
- logger 지원은 utils로 유지
- 실험 포털 구조가 본격적으로 갖춰짐

### 남은 점검

- utils로 실험 로직 누수 여부 확인
- src.losses 의존성 잔재 여부 확인
- src/common이 너무 커지지 않았는지 점검
- 실제 classification/anomaly 실험 end-to-end 검증

---

## 7. 다음 실행 순서

1. import 경로와 naming 통일 여부 확인
2. classification 설정 1개로 실제 실험 실행
3. anomaly 설정 1개로 실제 실험 실행
4. 로그와 summary CSV 확인
5. 새 helper가 나오면 src/common 또는 utils로 재분류

---

## 8. 최종 원칙

상황이 애매할 때는 아래 규칙을 따른다.

- 실험 동작을 계산하거나 제어하는 코드 -> src
- 실행 환경을 보조하는 코드 -> utils
- 여러 task에서 재사용되는 도메인 helper -> src/common

---

## 9. 결론

이 프로젝트는 이제 단순한 모델 실험 코드가 아니라, 여러 알고리즘을 비교하고 확장하는 실험 포털 구조로 정리되어야 한다.

이 구조가 유지되면, 새로운 method를 추가할 때마다 전체 프로젝트를 재작성하지 않고도 같은 포털 안에서 실험을 추가하고 비교할 수 있다.
