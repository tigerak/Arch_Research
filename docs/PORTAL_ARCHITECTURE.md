# Experiment Portal Architecture

## 1. 목적

이 저장소는 단일 분류 실험용 코드베이스가 아니라, 여러 실험 트랙을 공통 인터페이스로 실행하는 실험 포털로 설계되었다.

핵심 목표는 다음과 같다.

- 하나의 repo 안에서 여러 task를 관리한다.
- task별 엔진과 method registry를 분리한다.
- 실험 실행, 평가, 결과 요약을 공통 runner로 통합한다.
- 새 method를 추가할 때 기존 포털 구조를 깨지 않고 등록만 하면 된다.

즉, 본 프로젝트는 "하나의 모델만 실험하는 코드"가 아니라 "여러 방법을 비교하는 실험실" 성격을 갖는다.

## 2. 현재 구조의 핵심 원칙

### 2.1 src = 실험 로직

아래 코드는 실제 알고리즘, 모델, loss, dataset, training/eval 로직에 속한다.

- src/config_loader.py
- src/dataset_builder.py
- src/train_engine.py
- src/eval_engine.py
- src/run_manager.py
- src/classification/**
- src/anomaly/**

### 2.2 src/common = 공통 helper

실험 로직을 짜되, 여러 task에서 재사용되는 공통 지원 함수는 여기로 둔다.

- checkpoint state normalization
- tensor/device conversion helper
- 공통 수치 안정화 유틸

### 2.3 utils = 운영/도구 코드

utils는 실험 코어가 아니라 실험을 돌리는 환경을 보조하는 코드만 둔다.

- logger setup
- 파일/디렉터리 생성 보조
- CLI/runner glue
- 결과 표 출력
- legacy compatibility shim

## 3. track 구조

### 3.1 Classification track

classification은 method registry 기반으로 동작한다.

- src/classification/registry.py
- src/classification/engine.py
- src/classification/builders.py
- src/classification/models.py
- src/classification/methods/
  - ce.py
  - focal.py
  - arcface.py

설정의 task type과 loss_name에 따라 특정 method를 선택하고, 해당 method의 builder가 model/loss/optimizer를 만든다.

### 3.2 Anomaly track

anomaly도 동일한 패턴으로 구성한다.

- src/anomaly/engine.py
- src/anomaly/registry.py
- src/anomaly/image_loaders.py
- src/anomaly/image_level_metrics.py
- src/anomaly/methods/patchcore_runner.py
- src/anomaly/methods/dinomaly2_runner.py

method_name을 registry에서 읽어서 해당 runner를 선택한다.

## 4. 공통 portal runner

공통 실행 루프는 src/run_manager.py에서 담당한다.

역할은 다음과 같다.

- schedule에 있는 실험 key를 순회한다.
- base config + custom config를 병합한다.
- task type을 결정한다.
- classification 또는 anomaly 엔진을 호출한다.
- 실행 결과를 요약 CSV와 로그로 저장한다.

즉, 포털의 핵심은 task별 엔진을 공통으로 호출하는 구조에 있다.

## 5. 왜 이 구조가 좋은가

### 장점

- 실험 중간에 다른 task를 추가해도 기존 엔진을 유지한다.
- method는 registry로 등록되므로 import 경로를 크게 바꾸지 않아도 된다.
- 동일한 portal runner로 여러 실험을 자동화할 수 있다.
- 파일의 책임이 명확해져 유지보수성이 좋다.

### 분리 기준

- 실험 코드: src
- 공통 helper: src/common
- 실행 보조: utils

이 경계가 유지되면, 코드가 혼재되지 않고 확장 가능하다.

## 6. 작업 원칙

새 method를 추가할 때는 다음 규칙을 따른다.

1. task별 method는 각 track의 methods 폴더에 둔다.
2. method는 registry key와 연결한다.
3. experiment runner는 공통 portal에서 호출한다.
4. 핵심 알고리즘은 src 쪽에 둔다.
5. logging, summary, compatibility는 utils 쪽으로 둔다.

## 7. 현재 상태 판단

현재 프로젝트는 구조적으로 이미 portal 중심으로 정리된 상태에 가깝다.

다음 단계는 보통 다음과 같다.

- 실제 classification 실행 검증
- 실제 anomaly run 검증
- 추가 method 확장
- 마지막 도구/utility 경계 정리

## 8. 최종 목표

이 저장소는 결국 다음 형태를 목표로 한다.

- 하나의 실험 포털
- 여러 task / 여러 method
- 공통 스케줄러
- 공통 config merge + validation
- task별 별도 엔진
- method registry 기반 확장

이 구조가 갖추어지면, 새로운 실험을 추가할 때마다 새 repo를 만들지 않고도 같은 포털 안에서 비교 실험을 운영할 수 있다.
