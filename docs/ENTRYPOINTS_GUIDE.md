# 실행 엔트리포인트 가이드

## 1. 목적

이 문서는 프로젝트를 실제로 실행할 때 어디에서 어떤 스크립트를 돌려야 하는지 정리한 안내서다.

핵심은 다음과 같다.

- 실제 실행은 루트의 `main_*.py` 스크립트에서 시작한다.
- 각 스크립트는 `src/entrypoints.py`의 공통 래퍼를 호출한다.
- 공통 래퍼는 `src/run_manager.py`를 통해 schedule 기반으로 실험을 실행한다.
- 최종적으로 classification/anomaly task의 engine과 registry가 선택된다.

즉, 사용자가 보게 되는 가장 바깥쪽 실행점은 단순히 launcher이며,
실제 로직은 모두 `src/` 하위에서 처리된다.

---

## 2. 실행 경로 흐름

실제 실행의 호출 흐름은 다음과 같다.

1. 루트의 `main_classification_train.py` 또는 `main_anomaly_train.py` 실행
2. 해당 파일 내부에서 `run_scheduled_entrypoint(...)` 호출
3. `src/entrypoints.py`의 `run_scheduled_entrypoint()` 실행
4. `src/run_manager.py`의 `run_experiments()` 실행
5. base config + custom config 병합
6. task가 classification인지 anomaly인지 판별
7. 각 task의 engine/registry가 최종 method 선택
8. 학습 또는 평가 수행
9. 결과 요약 CSV/로그 저장

이 구조를 두면, 루트 스크립트는 "무엇을 돌릴지"만 정하고,
실제 실험 로직은 공통 runner가 책임진다.

---

## 3. 루트 진입점 스크립트

### 3.1 Classification 학습

파일:
- `main_classification_train.py`

실행 예시:

```bash
python main_classification_train.py
```

동작:

```python
BASE_CONFIG_PATH = Path("configs/classifier/TAPING_DiNO/base_config.yaml")
CONFIGS_DIR = Path("configs/classifier/TAPING_DiNO")
SCHEDULE_KEYS = ["Arc_CLS_S32_M02_Proj"]
```

이 값은 학습할 설정 그룹과 schedule key를 의미한다.

### 3.2 Classification 평가

파일:
- `main_classification_eval.py`

실행 예시:

```bash
python main_classification_eval.py
```

동작:

- 저장된 체크포인트를 읽어 평가한다.
- `mode="eval"`로 전달되어 `evaluate_checkpoint()` 경로로 이동한다.

### 3.3 Anomaly 학습

파일:
- `main_anomaly_train.py`

실행 예시:

```bash
python main_anomaly_train.py
```

이 경우에는 anomaly config 경로를 사용한다.

### 3.4 Anomaly 평가

파일:
- `main_anomaly_eval.py`

실행 예시:

```bash
python main_anomaly_eval.py
```

이 스크립트는 anomaly 평가 전용 엔트리포인트다.

---

## 4. 공통 실행 래퍼

공통 진입점은 다음 파일에서 정의된다.

- `src/entrypoints.py`

핵심 함수:

```python
def run_scheduled_entrypoint(
    *,
    entrypoint_name: str,
    base_config_path: str | Path,
    configs_dir: str | Path,
    schedule_keys: Iterable[str],
    mode: str,
) -> None:
```

### 역할

- schedule에 포함된 실험 key를 수집한다.
- `run_experiments()`를 호출한다.
- `mode` 값에 따라 train/eval 경로를 분기한다.
- 최종 실행 결과를 콘솔에 출력한다.

즉, 각 루트 main_* 파일은 복잡한 로직을 쓰지 않고,
이 공통 래퍼를 호출하는 단순 launcher만 갖고 있다.

---

## 5. 실제 실험 오케스트레이션

`src/run_manager.py`가 실험 실행의 핵심이다.

이 파일은 다음 일을 수행한다.

- schedule key를 순회한다.
- base config와 custom yaml을 병합한다.
- `task_type`을 읽는다.
- classification 또는 anomaly task를 선택한다.
- 알맞은 method를 registry에서 찾는다.
- 결과를 로그와 summary csv로 저장한다.

중요한 점은, 루트 스크립트는 `run_manager`를 직접 호출하는 대신,
`src/entrypoints.py`를 거친다는 것이다.

---

## 6. schedule 방식

### 6.1 단일 실험 실행

예를 들어 classification train을 단일 실험으로 돌릴 때는 아래처럼 설정한다.

```python
SCHEDULE_KEYS = ["Arc_CLS_S32_M02_Proj"]
```

### 6.2 여러 실험 자동 실행

`run_schedule.py`는 폴더 단위로 schedule을 자동으로 생성할 수 있다.

실행 예시:

```bash
python run_schedule.py
```

이 파일은 다음을 처리한다.

- base config 경로 해석
- configs 폴더에서 yaml 파일 목록 스캔
- 없을 경우 schedule 내용에서 자동 구성
- `run_experiments()`로 일괄 실행

---

## 7. config 경로 해석 규칙

`run_schedule.py`는 아래 우선순위를 가진다.

1. 환경변수 `TAPING_BASE_CONFIG`
2. 코드 상수 `BASE_CONFIG_PATH`
3. 환경변수 `TAPING_CONFIG_GROUP`
4. 코드 상수 `CONFIG_GROUP`
5. 환경변수 `TAPING_TASK_NAME`
6. 코드 상수 `TASK_NAME`

즉, 실행 시 환경변수로 설정을 덮어쓸 수 있다.

예시:

```bash
$env:TAPING_BASE_CONFIG = "configs/classifier/TAPING_DiNO/base_config.yaml"
python run_schedule.py
```

---

## 8. 실제 사용 시나리오

### 시나리오 A: classification 실험 1개만 돌릴 때

```bash
python main_classification_train.py
```

### 시나리오 B: evaluation만 다시 돌릴 때

```bash
python main_classification_eval.py
```

### 시나리오 C: anomaly 실험 실행

```bash
python main_anomaly_train.py
```

### 시나리오 D: 여러 yaml을 자동으로 경로 스캔하며 실행

```bash
python run_schedule.py
```

---

## 9. 엔트리포인트를 이해하는 핵심

실제 엔트리포인트는 아래처럼 계층화되어 있다.

```text
main_classification_train.py
    -> src.entrypoints.run_scheduled_entrypoint()
        -> src.run_manager.run_experiments()
            -> config_loader / task parser
            -> classification engine
            -> registry / method selection
            -> train/eval logic
```

```text
main_anomaly_train.py
    -> src.entrypoints.run_scheduled_entrypoint()
        -> src.run_manager.run_experiments()
            -> config_loader / task parser
            -> anomaly engine
            -> registry / method selection
            -> PatchCore or anomaly runner
```

이 구조를 기준으로 보면, 실험을 추가할 때는 루트 launcher만 정리하면 되고,
실제 로직은 task engine과 method registry에서 확장하면 된다.

---

## 10. 운영 팁

- 단일 실험을 빠르게 검증할 때는 `main_*_train.py`를 사용한다.
- 여러 실험을 일괄 수행할 때는 `run_schedule.py`를 사용한다.
- config 경로가 꼬였을 때는 환경변수 override로 빠르게 테스트한다.
- 로깅/결과 저장 경로는 `run_manager`에서 일괄 관리되므로, 루트 스크립트는 단순화되어 있다.

---

## 11. 정리

이 프로젝트의 실행 모델은 다음과 같이 이해하면 가장 쉽다.

- `main_*.py`: 사용자 진입점
- `src/entrypoints.py`: 공통 실행 래퍼
- `src/run_manager.py`: 실제 실행 엔진
- `src/classification` / `src/anomaly`: task별 구현
- `registry`: method 선택의 핵심

이 구조를 따르면, 하나의 코드베이스에서 classification과 anomaly를 모두 관리하면서도
실행 인터페이스는 일관되게 유지할 수 있다.
