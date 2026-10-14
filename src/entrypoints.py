from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.run_manager import run_experiments

"""실험 포털의 공통 진입점 모듈.

이 파일은 모든 main_* 스크립트가 공통으로 호출하는 마지막 경로이다.

핵심 역할:
- schedule key를 받아 실험을 실행한다.
- run_manager를 통해 base/custom config를 병합한다.
- mode(train/eval)를 전달해 classification/anomaly를 각각 실행한다.
- 최종 결과를 표준 형식으로 출력한다.

이 구조를 두면 main_classification_train.py, main_anomaly_train.py 등은
실제 실험 로직 대신 "어떤 설정을 누구에게 넘길지"만 판단하는 단순 launcher 역할만 수행한다.
"""


def run_scheduled_entrypoint(
    *,
    entrypoint_name: str,
    base_config_path: str | Path,
    configs_dir: str | Path,
    schedule_keys: Iterable[str],
    mode: str,
) -> None:
    """공통 main_* 진입점이 공통으로 사용하는 실행 래퍼.

    이 함수는 다음 흐름을 담당한다.
    1. schedule_keys 목록을 읽어 실험들을 순서대로 배치한다.
    2. run_experiments()를 호출해 실제 train/eval을 수행한다.
    3. 결과를 콘솔에 정형화된 문자열로 출력한다.

    이렇게 분리해 두면 각 main_* 파일은 실행 단순화와 실험 의도 표현만 신경 쓰고,
    복잡한 로직은 run_manager에 모아둘 수 있다.
    """
    summary = run_experiments(
        schedule_keys=schedule_keys,
        base_config_path=base_config_path,
        configs_dir=configs_dir,
        mode=mode,
        continue_on_error=True,
        save_summary=True,
    )

    print(f"[{entrypoint_name}] done")
    print(
        f"[{entrypoint_name}] total={summary['total']} "
        f"success={summary['success']} failed={summary['failed']}"
    )
