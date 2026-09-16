from __future__ import annotations

from src.run_manager import run_experiments


def main() -> None:
	"""단일/다중 실험 평가 진입점.

	기본적으로 configs/custom_config.yaml을 대상으로,
	규약 경로의 best_model.pth를 찾아 평가를 수행한다.
	"""
	summary = run_experiments(
		schedule_keys=["custom_config"],
		base_config_path="configs/base_config.yaml",
		configs_dir="configs",
		mode="eval",
		continue_on_error=True,
		save_summary=True,
	)

	print("[main_eval] done")
	print(f"[main_eval] total={summary['total']} success={summary['success']} failed={summary['failed']}")


if __name__ == "__main__":
	main()

