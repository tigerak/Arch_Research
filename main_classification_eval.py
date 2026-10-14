from __future__ import annotations

from pathlib import Path

from src.entrypoints import run_scheduled_entrypoint


BASE_CONFIG_PATH = Path("configs/classifier/TAPING_DiNO/base_config.yaml")
CONFIGS_DIR = Path("configs/classifier/TAPING_DiNO")
SCHEDULE_KEYS = ["Arc_CLS_S32_M02_Proj"]


def main() -> None:
    """classification 단일/다중 실험 평가 진입점."""
    run_scheduled_entrypoint(
        entrypoint_name="main_classification_eval",
        base_config_path=BASE_CONFIG_PATH,
        configs_dir=CONFIGS_DIR,
        schedule_keys=SCHEDULE_KEYS,
        mode="eval",
    )


if __name__ == "__main__":
    main()
