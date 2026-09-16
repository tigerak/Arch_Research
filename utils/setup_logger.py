from __future__ import annotations

import logging
from pathlib import Path


def get_file_logger(logger_name: str, log_file: str | Path, level: int = logging.INFO) -> logging.Logger:
	"""파일 로그를 남기는 로거를 생성한다.

	동일한 logger_name으로 여러 번 호출되어도 핸들러가 중복 추가되지 않도록
	기존 핸들러를 정리한 뒤 다시 연결한다.
	"""
	path = Path(log_file)
	path.parent.mkdir(parents=True, exist_ok=True)

	logger = logging.getLogger(logger_name)
	logger.setLevel(level)
	logger.propagate = False

	# 중복 출력 방지를 위해 기존 핸들러를 제거한다.
	for h in list(logger.handlers):
		logger.removeHandler(h)

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	file_handler = logging.FileHandler(path, encoding="utf-8")
	file_handler.setLevel(level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	return logger

