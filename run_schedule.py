from __future__ import annotations

import os
from pathlib import Path

from src.run_manager import run_experiments


# 실험 키 목록:
SCHEDULE = [
    # "Arc_CLS_S32_M02",
    # "Focal_CLS_FA1",
]

# 파일 실행 전 확인.
RUN_MODE = "train"  # "train" | "eval"

# 방법 1) config group + task 이름만 지정하면 자동으로 아래 경로를 사용한다.
# - base:   configs/<CONFIG_GROUP>/<TASK_NAME>/base_config.yaml
# - custom: configs/<CONFIG_GROUP>/<TASK_NAME>/*.yaml
CONFIG_GROUP = "anomaly"  # "classifier" | "anomaly"
TASK_NAME = "WA_DiNO"     # "WA_DiNO" | "TAPING_DiNO"
# 방법 2) base config를 직접 지정한다. (지정 시 TASK_NAME보다 우선)
BASE_CONFIG_PATH = None  # Path("configs/classifier/TAPING_DiNO/base_config.yaml")

# custom 설정 디렉터리 강제 지정(보통 None 권장)
# None이면 BASE_CONFIG_PATH의 부모 폴더를 사용한다.
CONFIGS_DIR = None

# 학습 중 오류 발생 시 즉시 중단 여부 (False면 다음 실험으로 넘어감)
STOP_ON_ERROR = False


def _resolve_paths() -> tuple[Path, Path]:
    """실험 경로를 해석한다.

    우선순위:
    1) 환경변수 TAPING_BASE_CONFIG
    2) 코드 상수 BASE_CONFIG_PATH
    3) 환경변수 TAPING_CONFIG_GROUP
    4) 코드 상수 CONFIG_GROUP
    5) 환경변수 TAPING_TASK_NAME
    6) 코드 상수 TASK_NAME
    """
    env_base = os.getenv("TAPING_BASE_CONFIG", "").strip()
    env_group = os.getenv("TAPING_CONFIG_GROUP", "").strip()
    env_task = os.getenv("TAPING_TASK_NAME", "").strip()

    if env_base:
        base = Path(env_base)
    elif BASE_CONFIG_PATH is not None:
        base = Path(BASE_CONFIG_PATH)
    else:
        config_group = env_group or str(CONFIG_GROUP)
        task = env_task or str(TASK_NAME)
        base = Path("configs") / config_group / task / "base_config.yaml"

    if CONFIGS_DIR is not None:
        cfg_dir = Path(CONFIGS_DIR)
    else:
        cfg_dir = base.parent

    return base, cfg_dir


def main() -> None:
    base_config_path, configs_dir = _resolve_paths()

    if not base_config_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not configs_dir.exists():
        raise FileNotFoundError(f"Configs dir not found: {configs_dir}")

    schedule = SCHEDULE
    if not schedule:
        # SCHEDULE이 비어 있으면 task 폴더 내 모든 실험 yaml을 자동 실행한다.
        schedule = sorted(
            p.stem
            for p in configs_dir.glob("*.yaml")
            if p.name != "base_config.yaml"
        )
        if not schedule:
            raise ValueError(
                f"SCHEDULE is empty and no experiment yaml found in: {configs_dir}"
            )
        print(f"[run_schedule] auto schedule from folder: {schedule}")

    summary = run_experiments(
        schedule_keys=schedule,
        base_config_path=base_config_path,
        configs_dir=configs_dir,
        mode=RUN_MODE,
        continue_on_error=not STOP_ON_ERROR,
        save_summary=True,
    )

    print("[run_schedule] done")
    print(f"[run_schedule] base_config={base_config_path}")
    print(f"[run_schedule] configs_dir={configs_dir}")
    print(f"[run_schedule] total={summary['total']} success={summary['success']} failed={summary['failed']}")
    if "summary_csv" in summary:
        print(f"[run_schedule] summary_csv={summary['summary_csv']}")


if __name__ == "__main__":
    main()