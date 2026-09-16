from __future__ import annotations

from src.run_manager import run_experiments


def main() -> None:
	"""단일/다중 실험 학습 진입점.

	기본적으로 configs/custom_config.yaml을 실행한다.
	다른 실험 파일을 쓰고 싶으면 run_schedule.py의 SCHEDULE을 사용하거나,
	여기의 schedule_keys를 교체하면 된다.
	"""
	summary = run_experiments(
		schedule_keys=["custom_config"],
		base_config_path="configs/base_config.yaml",
		configs_dir="configs",
		mode="train",
		continue_on_error=True,
		save_summary=True,
	)

	print("[main_train] done")
	print(f"[main_train] total={summary['total']} success={summary['success']} failed={summary['failed']}")


if __name__ == "__main__":
	main()

