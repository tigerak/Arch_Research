from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.run_manager import run_experiments


def run_scheduled_entrypoint(
    *,
    entrypoint_name: str,
    base_config_path: str | Path,
    configs_dir: str | Path,
    schedule_keys: Iterable[str],
    mode: str,
) -> None:
    """공통 main_* 진입점 출력 형식을 통일한다."""
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
