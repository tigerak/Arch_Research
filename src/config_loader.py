from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml

"""실험 설정 로딩, 병합, 검증을 담당하는 공통 설정 모듈.

이 파일은 실험 포털의 핵심 입력 경로이다.

역할:
- YAML 파일을 읽어 dict로 변환한다.
- base config와 custom config를 재귀적으로 병합한다.
- task type(classification/anomaly_detection)을 판별한다.
- 필수 키와 값 범위를 검증해 런타임 오류를 조기에 차단한다.

모든 실험은 이 모듈에서 만들어진 cfg를 기준으로 동작하므로,
설정 검증이 약하면 어떤 실험이든 예상치 못한 실패를 일으킬 수 있다.
"""


class ConfigError(ValueError):
    """설정 파일 처리 단계에서 발생하는 예외를 구조적으로 표현하는 클래스.

    단순 문자열 메시지뿐 아니라 아래 컨텍스트를 함께 보관한다.
    - field: 문제가 난 설정 키(예: train.labels)
    - file_path: 문제가 난 설정 파일 경로
    - details: 추가 설명(누락 키 목록 등)

    이렇게 구성하면 상위 실행기에서 로그를 남길 때 원인을 더 빠르게 파악할 수 있다.
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        file_path: str | Path | None = None,
        details: Any | None = None,
    ) -> None:
        self.message = message
        self.field = field
        self.file_path = str(file_path) if file_path is not None else None
        self.details = details
        super().__init__(self.__str__())

    def __str__(self) -> str:
        parts = [self.message]
        if self.field:
            parts.append(f"field={self.field}")
        if self.file_path:
            parts.append(f"file={self.file_path}")
        if self.details is not None:
            parts.append(f"details={self.details}")
        return " | ".join(parts)


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    """YAML 파일을 읽어 dict 형태로 반환한다.

    이 함수는 단순 로딩 외에 "초기 방어 로직"을 포함한다.
    1) 파일 존재 여부 확인
    2) 비어 있는 YAML(null) 처리: 빈 dict로 정규화
    3) 최상위 타입 검증: 반드시 mapping(dict)이어야 함

    초기 단계에서 강하게 검증해 두면,
    이후 병합/검증 단계의 에러 원인을 더 빠르게 추적할 수 있다.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigError("Config file not found", file_path=p)

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML must be a mapping", file_path=p)

    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """base 설정 위에 override 설정을 재귀적으로 덮어쓴다.

    병합 규칙:
    - 두 값이 모두 mapping이면 하위 키까지 재귀 병합
    - 그 외 타입(숫자, 문자열, bool, 리스트 등)은 override로 완전 교체

    설계 의도:
    - 실험 설정에서 예측 가능성을 높이기 위해 리스트는 부분 병합하지 않는다.
      즉, labels 같은 리스트는 override 값으로 전체 교체된다.
    - 입력 원본은 변경하지 않고 새로운 dict를 반환해 부작용을 방지한다.
    """
    merged: Dict[str, Any] = dict(base)

    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def resolve_custom_config_path(
    custom_name_or_path: str,
    configs_dir: str | Path = "configs",
) -> Path:
    """커스텀 설정 식별자를 실제 경로로 변환한다.

    지원 입력 패턴:
    - custom_config
      -> configs/custom_config.yaml
    - custom_config.yaml
      -> configs/custom_config.yaml
    - D:/repo/configs/custom_config.yaml
      -> 절대 경로이므로 그대로 사용

    이 함수를 통해 SCHEDULE에는 짧은 이름만 넣고,
    실제 파일 해석은 일관된 규칙으로 자동 처리할 수 있다.
    """
    raw = Path(custom_name_or_path)

    # 사용자가 절대경로나 하위 폴더를 명시한 경우에는 의도를 그대로 따른다.
    if raw.is_absolute() or raw.parent != Path("."):
        candidate = raw
    else:
        # 파일명만 들어온 경우 기본 configs 디렉터리 아래에서 찾는다.
        candidate = Path(configs_dir) / raw

    # 확장자가 없으면 기본적으로 .yaml을 보정한다.
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        candidate = candidate.with_suffix(".yaml")

    return candidate


def merge_base_and_custom(
    base_config_path: str | Path,
    custom_name_or_path: Optional[str] = None,
    custom_config_path: Optional[str | Path] = None,
    configs_dir: str | Path = "configs",
) -> Dict[str, Any]:
    """base와 custom 설정을 로드해 최종 실험 설정을 생성한다.

    우선순위:
    1) base 설정이 기본값
    2) custom 설정이 동일 키를 덮어씀

    입력 제약:
    - custom_name_or_path와 custom_config_path는 동시에 받지 않는다.
      두 값을 동시에 허용하면 어느 쪽을 우선할지 모호해지고,
      호출부 실수를 늦게 발견하게 되므로 초기에 차단한다.

    출력 보강:
    - custom이 적용된 경우 _meta에 설정 출처를 기록한다.
      실험 재현 시 "어떤 base와 custom으로 실행했는지"를 바로 추적할 수 있다.
    """
    if custom_name_or_path and custom_config_path:
        raise ConfigError(
            "Use only one of custom_name_or_path or custom_config_path",
            field="custom_name_or_path/custom_config_path",
        )

    base_cfg = load_yaml_file(base_config_path)

    if custom_config_path is not None:
        custom_path = Path(custom_config_path)
    elif custom_name_or_path:
        custom_path = resolve_custom_config_path(custom_name_or_path, configs_dir=configs_dir)
    else:
        custom_path = None

    if custom_path is None:
        # base만 사용하는 경우에도 동일 검증을 거쳐 일관성을 보장한다.
        merged = dict(base_cfg)
        validate_merged_config(merged)
        return merged

    custom_cfg = load_yaml_file(custom_path)
    merged = deep_merge(base_cfg, custom_cfg)
    validate_merged_config(merged)

    merged["_meta"] = {
        "base_config": str(Path(base_config_path)),
        "custom_config": str(custom_path),
    }

    return merged


def validate_merged_config(cfg: Mapping[str, Any]) -> None:
    """최종 병합 설정의 구조와 핵심 값 제약을 검증한다.

    검증 목적:
    - 학습 시작 전 설정 오류를 조기 차단
    - 런타임 중간 실패 대신, 원인 명확한 메시지 제공

    현재 검증 항목:
    - 최상위 필수 섹션 존재 여부
    - data/model/train 섹션 필수 키 존재 여부
    - focal loss 선택 시 focal 파라미터 유효성
    - labels 길이와 n_classes 정합성
    - scheduler 허용 값 검증
    """
    required_top = ("project_name", "data", "model")
    _require_keys(cfg, required_top, where="root")

    task_type = get_task_type(cfg)
    data = _ensure_mapping(cfg["data"], "data")
    model = _ensure_mapping(cfg["model"], "model")
    train = _ensure_mapping(cfg["train"], "train") if "train" in cfg else {}

    if task_type == "classification":
        if "train" not in cfg:
            raise ConfigError("classification config requires train section", field="train")

        _require_keys(data, ("train_dir", "test_dir", "save_dir", "model_dir"), where="data")
        _require_keys(model, ("model_name", "pretrained", "freeze_backbone"), where="model")
        _require_keys(
            train,
            ("max_epoch", "lr", "scheduler", "weight_decay", "n_classes", "labels"),
            where="train",
        )

        _validate_loss_config(train)

        n_classes = int(train["n_classes"])
        labels = train["labels"]
        if not isinstance(labels, list):
            raise ConfigError("train.labels must be a list", field="train.labels")
        if len(labels) != n_classes:
            raise ConfigError(
                "train.labels length must match train.n_classes",
                field="train.labels/train.n_classes",
                details={"labels_len": len(labels), "n_classes": n_classes},
            )

        scheduler = str(train["scheduler"]).strip()
        if not scheduler:
            raise ConfigError(
                "train.scheduler must not be empty",
                field="train.scheduler",
            )
        return

    if task_type == "anomaly_detection":
        _validate_anomaly_config(cfg, data=data, model=model, train=train)
        return

    raise ConfigError(
        "Unsupported task type",
        field="task.type",
        details={"value": task_type, "supported": ["classification", "anomaly_detection"]},
    )


def get_task_type(cfg: Mapping[str, Any]) -> str:
    task = cfg.get("task")
    if task is None:
        return "classification"
    task_map = _ensure_mapping(task, "task")
    task_type = str(task_map.get("type", task_map.get("task_type", "classification"))).strip().lower()
    if task_type in {"classification", "cls", "classifier"}:
        return "classification"
    if task_type in {"anomaly_detection", "anomaly", "ad"}:
        return "anomaly_detection"
    return task_type


def _validate_anomaly_config(
    cfg: Mapping[str, Any],
    *,
    data: Mapping[str, Any],
    model: Mapping[str, Any],
    train: Mapping[str, Any],
) -> None:
    """이상치 탐지 설정의 핵심 키를 검증한다.

    1단계 목적은 PatchCore 기반 baseline을 안정적으로 돌릴 수 있는 최소 계약을
    정의하는 것이다. classification 전용 필드(n_classes, labels, scheduler 등)는
    여기서 강제하지 않는다.
    """
    _require_keys(data, ("train_normal_dir", "test_dir", "save_dir", "model_dir"), where="data")

    if not any(key in model for key in ("backbone_name", "model_name")):
        raise ConfigError(
            "anomaly config requires backbone_name or model_name",
            field="model.backbone_name/model.model_name",
        )
    _require_keys(model, ("pretrained", "freeze_backbone"), where="model")

    _require_keys(cfg, ("anomaly",), where="root")
    anomaly = _ensure_mapping(cfg["anomaly"], "anomaly")
    _require_keys(anomaly, ("method_name",), where="anomaly")

    method_root = cfg.get("method", {})
    if method_root is None:
        method_root = {}
    if not isinstance(method_root, Mapping):
        raise ConfigError("Config section must be a mapping", field="method")

    method_name = str(anomaly["method_name"]).strip().lower()
    if method_name == "patchcore":
        patchcore_cfg = method_root.get("patchcore", {})
        if patchcore_cfg is None:
            patchcore_cfg = {}
        if not isinstance(patchcore_cfg, Mapping):
            raise ConfigError("method.patchcore must be a mapping", field="method.patchcore")

        normalized_patchcore_cfg = dict(patchcore_cfg)
        for legacy_key in ("feature_layers", "coreset_sampling_ratio", "knn_k", "score_aggregation", "max_memory_bank_patches"):
            if legacy_key not in normalized_patchcore_cfg and legacy_key in anomaly:
                normalized_patchcore_cfg[legacy_key] = anomaly[legacy_key]

        _require_keys(
            normalized_patchcore_cfg,
            ("feature_layers", "coreset_sampling_ratio", "knn_k"),
            where="method.patchcore",
        )

        feature_layers = normalized_patchcore_cfg["feature_layers"]
        if not isinstance(feature_layers, list) or not feature_layers:
            raise ConfigError(
                "method.patchcore.feature_layers must be a non-empty list",
                field="method.patchcore.feature_layers",
            )

        coreset_sampling_ratio = float(normalized_patchcore_cfg["coreset_sampling_ratio"])
        knn_k = int(normalized_patchcore_cfg["knn_k"])
        if not (0.0 < coreset_sampling_ratio <= 1.0):
            raise ConfigError(
                "method.patchcore.coreset_sampling_ratio must be in (0, 1]",
                field="method.patchcore.coreset_sampling_ratio",
                details={"value": coreset_sampling_ratio},
            )
        if knn_k <= 0:
            raise ConfigError(
                "method.patchcore.knn_k must be > 0",
                field="method.patchcore.knn_k",
                details={"value": knn_k},
            )

    if method_name == "dinomaly2":
        dinomaly2_cfg = method_root.get("dinomaly2", {})
        if dinomaly2_cfg is None:
            dinomaly2_cfg = {}
        if not isinstance(dinomaly2_cfg, Mapping):
            raise ConfigError("method.dinomaly2 must be a mapping", field="method.dinomaly2")
        _require_keys(
            dinomaly2_cfg,
            ("feature_layers", "bottleneck_dim", "epochs", "lr", "weight_decay", "score_aggregation"),
            where="method.dinomaly2",
        )
        feature_layers = dinomaly2_cfg["feature_layers"]
        if not isinstance(feature_layers, list) or not feature_layers:
            raise ConfigError(
                "method.dinomaly2.feature_layers must be a non-empty list",
                field="method.dinomaly2.feature_layers",
            )
        if int(dinomaly2_cfg["bottleneck_dim"]) <= 0:
            raise ConfigError(
                "method.dinomaly2.bottleneck_dim must be > 0",
                field="method.dinomaly2.bottleneck_dim",
                details={"value": dinomaly2_cfg["bottleneck_dim"]},
            )
        if int(dinomaly2_cfg["epochs"]) <= 0:
            raise ConfigError(
                "method.dinomaly2.epochs must be > 0",
                field="method.dinomaly2.epochs",
                details={"value": dinomaly2_cfg["epochs"]},
            )
        if float(dinomaly2_cfg["lr"]) <= 0:
            raise ConfigError(
                "method.dinomaly2.lr must be > 0",
                field="method.dinomaly2.lr",
                details={"value": dinomaly2_cfg["lr"]},
            )
        if float(dinomaly2_cfg["weight_decay"]) < 0:
            raise ConfigError(
                "method.dinomaly2.weight_decay must be >= 0",
                field="method.dinomaly2.weight_decay",
                details={"value": dinomaly2_cfg["weight_decay"]},
            )
        if str(dinomaly2_cfg["score_aggregation"]).strip().lower() not in {"max", "mean"}:
            raise ConfigError(
                "method.dinomaly2.score_aggregation must be 'max' or 'mean'",
                field="method.dinomaly2.score_aggregation",
                details={"value": dinomaly2_cfg["score_aggregation"]},
            )

    if train:
        for key in ("img_size", "batch_size", "num_workers", "seed"):
            if key not in train:
                raise ConfigError(
                    "anomaly config requires runtime settings in train section",
                    field=f"train.{key}",
                    details={"required_keys": ["img_size", "batch_size", "num_workers", "seed"]},
                )


def _validate_loss_config(train: Mapping[str, Any]) -> None:
    """손실 함수 관련 설정을 검증한다.

    규칙:
    - loss_name 미지정 시 ce로 간주
    - loss_name=focal이면 focal_gamma, focal_alpha 필수

    수치 제약:
    - focal_gamma >= 0
    - focal_alpha > 0
    """
    loss_name = str(train.get("loss_name", "ce")).lower()

    if loss_name == "focal":
        if "focal_gamma" not in train or "focal_alpha" not in train:
            raise ConfigError(
                "train.loss_name=focal requires focal_gamma and focal_alpha",
                field="train.focal_gamma/train.focal_alpha",
            )

        gamma = float(train["focal_gamma"])
        alpha = float(train["focal_alpha"])

        if gamma < 0:
            raise ConfigError("train.focal_gamma must be >= 0", field="train.focal_gamma", details={"value": gamma})
        if alpha <= 0:
            raise ConfigError("train.focal_alpha must be > 0", field="train.focal_alpha", details={"value": alpha})

    if loss_name == "arcface":
        required_any = ["arcface_num_sub_centers", "num_sub_centers"]
        if not any(key in train for key in required_any):
            raise ConfigError(
                "train.loss_name=arcface requires arcface_num_sub_centers or num_sub_centers",
                field="train.arcface_num_sub_centers/train.num_sub_centers",
            )
        if "arcface_margin" not in train or "arcface_scale" not in train:
            raise ConfigError(
                "train.loss_name=arcface requires arcface_margin and arcface_scale",
                field="train.arcface_margin/train.arcface_scale",
            )

        num_sub_centers = int(train.get("arcface_num_sub_centers", train.get("num_sub_centers", 0)))
        margin = float(train["arcface_margin"])
        scale = float(train["arcface_scale"])

        if num_sub_centers <= 0:
            raise ConfigError(
                "arcface_num_sub_centers/num_sub_centers must be > 0",
                field="train.arcface_num_sub_centers/train.num_sub_centers",
                details={"value": num_sub_centers},
            )
        if margin < 0:
            raise ConfigError("train.arcface_margin must be >= 0", field="train.arcface_margin", details={"value": margin})
        if scale <= 0:
            raise ConfigError("train.arcface_scale must be > 0", field="train.arcface_scale", details={"value": scale})


def _ensure_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """섹션 값이 mapping 타입인지 확인한다.

    data/model/train처럼 dict 구조를 기대하는 위치에서
    문자열/리스트 등이 잘못 들어온 경우를 즉시 식별하기 위한 보조 함수다.
    """
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} must be a mapping", field=field_name)
    return value


def _require_keys(mapping: Mapping[str, Any], keys: Iterable[str], where: str) -> None:
    """필수 키 목록의 존재 여부를 일괄 검사한다.

    누락 항목을 한 번에 수집해 메시지로 전달하므로,
    사용자가 여러 번 실행하며 한 개씩 고치는 비효율을 줄일 수 있다.
    """
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise ConfigError(
            f"Missing required keys in {where}",
            field=where,
            details={"missing": missing},
        )
