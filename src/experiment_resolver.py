from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from src.config_loader import ConfigError, resolve_custom_config_path

"""실험 schedule key를 실제 config 경로로 변환하는 resolver 모듈.

이 파일은 실험 키를 문자열로 넘겨받았을 때,
어느 YAML 파일을 읽어야 하는지 결정하는 경로 해석기 역할을 수행한다.

핵심 목적:
- schedule key를 파일 경로로 직렬화한다.
- 존재하지 않는 config를 조기에 걸러낸다.
- 자동 스케줄링을 위해 파일 리스트를 일관되게 생성한다.

실험 포털의 최상위 실행 루프가 이 resolver를 거쳐서 config를 찾기 때문에,
이름 규칙과 경로 규칙이 정확해야 전체 실험이 안정적으로 동작한다.
"""


def resolve_experiment_config_path(
    schedule_key: str,
    configs_dir: str | Path = "configs",
) -> Path:
    """SCHEDULE의 실험 키를 실제 커스텀 설정 파일 경로로 변환하고 존재 여부를 검증한다.

    동작 규칙:
    - custom_config         -> configs/custom_config.yaml
    - custom_config.yaml    -> configs/custom_config.yaml
    - configs/x.yaml        -> configs/x.yaml
    - 절대 경로             -> 그대로 사용

    존재하지 않는 경로면 즉시 ConfigError를 발생시켜,
    학습이 시작되기 전에 설정 오타를 빠르게 잡아낸다.
    """
    config_path = resolve_custom_config_path(schedule_key, configs_dir=configs_dir)

    if not config_path.exists():
        raise ConfigError(
            "Custom config file not found",
            field="schedule_key",
            file_path=config_path,
            details={"schedule_key": schedule_key},
        )

    return config_path


def resolve_schedule_keys(
    schedule_keys: Iterable[str],
    configs_dir: str | Path = "configs",
) -> List[Path]:
    """여러 SCHEDULE 키를 한 번에 해석하여 파일 경로 목록으로 반환한다.

    반환 순서는 입력 순서를 그대로 유지한다.
    중간에 하나라도 파일이 없으면 즉시 ConfigError를 발생시킨다.
    """
    resolved_paths: List[Path] = []
    for key in schedule_keys:
        resolved_paths.append(resolve_experiment_config_path(key, configs_dir=configs_dir))
    return resolved_paths
