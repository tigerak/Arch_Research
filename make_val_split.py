from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path
from typing import Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# -----------------------------------------------------------------------------
# 1) 설정 파일 로딩
# -----------------------------------------------------------------------------
# YAML 설정 파일을 읽어서 기본 경로와 비율을 자동으로 가져온다.
# 예를 들어 base_config.yaml 안에 아래 값이 있으면 그대로 사용한다.
#
# data:
#   train_normal_dir: ".../TRAIN"
#   test_dir: ".../TEST"
#
# anomaly:
#   normal_class: "OK"
#
# train:
#   val_ratio: 0.1
#
# 이렇게 설정해두면 CLI 인자 없이도 바로 스크립트를 실행할 수 있다.
def _read_yaml_config(config_path: Path) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required when using --config. Install it with: pip install pyyaml")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


# -----------------------------------------------------------------------------
# 2) 기본 경로 및 비율 계산
# -----------------------------------------------------------------------------
# 이 함수는 설정 파일에서 train/test 경로와 검증 비율을 읽어서 리턴한다.
# config.yaml이 없거나 비어 있으면 기본 프로젝트 설정 파일을 찾는다.
def _resolve_default_paths(config_path: Path | None) -> tuple[Path, Path, str, float, float]:
    if config_path is not None:
        cfg = _read_yaml_config(config_path)
        data_cfg = (cfg or {}).get("data", {})

        # data.train_normal_dir: 정상 학습 데이터 루트
        # data.test_dir: 테스트 데이터 루트
        train_dir = Path(str(data_cfg.get("train_normal_dir", ""))).expanduser()
        test_dir = Path(str(data_cfg.get("test_dir", ""))).expanduser()

        # anomaly.normal_class: 정상 클래스 이름, 보통 OK
        normal_class = str((cfg or {}).get("anomaly", {}).get("normal_class", "OK"))

        # train.val_ratio: 정상 데이터 중 validation으로 뽑을 비율
        train_ratio = float(((cfg or {}).get("train", {}) or {}).get("val_ratio", 0.1))

        # anomaly.val_ratio: 비정상 클래스에서 validation으로 뽑을 비율
        # 보통 이 값은 0.2 처럼 비정상 쪽 비율을 조금 더 크게 잡는다.
        test_ratio_raw = (cfg or {}).get("anomaly", {}).get("val_ratio", 0.2)
        test_ratio = float(test_ratio_raw) if isinstance(test_ratio_raw, (int, float)) else 0.2

        return train_dir, test_dir, normal_class, train_ratio, test_ratio

    # config 인자가 없으면 기본 프로젝트 설정 파일을 찾는다.
    default_cfg = Path(__file__).resolve().parent / "configs" / "anomaly" / "WA_DiNO" / "base_config.yaml"
    if default_cfg.exists():
        return _resolve_default_paths(default_cfg)

    raise FileNotFoundError("Default config not found, pass --config or provide --train-normal-dir and --test-dir explicitly.")


# -----------------------------------------------------------------------------
# 3) 이미지 파일 찾기
# -----------------------------------------------------------------------------
# 특정 폴더 안에 있는 이미지 파일만 골라서 리스트로 만든다.
# 확장자를 제한해 JPG/PNG 등 실제 데이터 파일만 처리한다.
# 재귀 탐색을 추가해 TRAIN/OK/OK_1_1/... 같은 중첩 구조에서도 동작한다.
def _list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []

    files: list[Path] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return files


def _resolve_normal_train_dir(train_dir: str | Path, normal_class: str) -> Path:
    train_path = Path(train_dir).expanduser().resolve()
    if train_path.name == normal_class:
        return train_path
    return train_path / normal_class


def _resolve_dataset_root(train_dir: str | Path, normal_class: str) -> Path:
    train_path = Path(train_dir).expanduser().resolve()
    if train_path.name == normal_class:
        train_path = train_path.parent
    if train_path.name.lower() == "train" or train_path.name.lower().endswith("train"):
        return train_path.parent
    return train_path


# -----------------------------------------------------------------------------
# 4) 비율 기반 개수 계산
# -----------------------------------------------------------------------------
# 전체 이미지 개수에서 비율을 적용해 이동할 개수를 계산한다.
# 예: 100장의 이미지, train_ratio=0.1 -> 10장 이동
# 최소 1개는 보장한다. 데이터가 비어 있으면 0개를 반환한다.
def _sample_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if ratio <= 0:
        return 0
    return max(1, math.ceil(total * ratio))


# -----------------------------------------------------------------------------
# 5) 실제 validation split 생성
# -----------------------------------------------------------------------------
# 핵심 함수.
# - 정상 학습 데이터의 일부를 VAL/OK로 이동
# - 테스트 데이터의 비정상 클래스 일부를 VAL/<클래스명>으로 이동
# - 파일을 복사하는 대신 이동(shutil.move)한다.
# - dry_run 옵션을 켜면 실제 이동이 아니라 로그만 출력한다.
def create_val_split(
    train_normal_dir: str | Path,
    test_dir: str | Path,
    val_root: str | Path | None = None,
    *,
    normal_class: str = "OK",
    train_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    dry_run: bool = False,
) -> dict[str, int]:
    # 경로 정규화
    train_dir = Path(train_normal_dir).expanduser().resolve()
    test_root = Path(test_dir).expanduser().resolve()

    # 필수 디렉터리 검사
    if not train_dir.exists():
        raise FileNotFoundError(f"Train normal directory not found: {train_dir}")
    if not test_root.exists():
        raise FileNotFoundError(f"Test directory not found: {test_root}")

    # 검증 결과를 저장할 루트 결정
    # 기본 값: train_root와 test_root가 같은 부모면 그 밑의 VAL 폴더를 사용
    if val_root is None:
        parent = train_dir.parent if train_dir.parent == test_root.parent else train_dir.parent
        val_root = parent / "VAL"
    val_root = Path(val_root).expanduser().resolve()

    # 정상 클래스 폴더가 실제로 있는지 확인
    # 예시: TRAIN/OK 또는 TRAIN/OK/OK_1_1/... 와 같이 중첩된 구조를 모두 처리한다.
    normal_train_dir = _resolve_normal_train_dir(train_dir, normal_class)
    if not normal_train_dir.exists():
        raise FileNotFoundError(f"Normal class folder not found: {normal_train_dir}")

    # 랜덤 선택을 동일하게 유지하기 위해 시드 고정
    rng = random.Random(seed)

    # 결과 보고용 딕셔너리
    # 예: {"OK": 15, "BUBBLE": 10, "PARTICLE": 8}
    moved_summary: dict[str, int] = {normal_class: 0}

    # ---------------------------------------------------------------------
    # A. 정상 학습 데이터에서 일부를 validation으로 이동
    # ---------------------------------------------------------------------
    # 정상 데이터는 보통 train_normal_dir 아래에 class별 폴더가 있다고 가정한다.
    # 예: TRAIN/OK/*.png
    ok_files = _list_image_files(normal_train_dir)
    ok_to_move_count = _sample_count(len(ok_files), train_ratio)

    # 무작위로 선택된 파일만 뽑는다.
    ok_to_move = rng.sample(ok_files, k=min(ok_to_move_count, len(ok_files)))

    # VAL/OK 폴더 생성
    ok_destination = val_root / normal_class
    if not dry_run:
        ok_destination.mkdir(parents=True, exist_ok=True)

    # 실제 이동
    for src in ok_to_move:
        dst = ok_destination / src.name
        if dry_run:
            print(f"[DRY-RUN] {src} -> {dst}")
        else:
            shutil.move(str(src), str(dst))
        moved_summary[normal_class] += 1

    # ---------------------------------------------------------------------
    # B. 테스트 데이터의 비정상 클래스에서 일부를 validation으로 이동
    # ---------------------------------------------------------------------
    # test_dir 구조는 보통 다음처럼 되어 있다.
    #
    # TEST/
    #   OK/
    #   BUBBLE/
    #   CONTAMINATION/
    #   PARTICLE/
    #
    # 정상 클래스 OK는 제외하고, 나머지 클래스만 validation으로 추출한다.
    for class_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        class_name = class_dir.name

        # 정상 클래스는 이미 TRAIN/OK에서 뽑았으므로 여기서는 건너뛴다.
        if class_name == normal_class:
            continue

        files = _list_image_files(class_dir)
        count = _sample_count(len(files), test_ratio)
        if count == 0:
            continue

        selected = rng.sample(files, k=min(count, len(files)))
        destination = val_root / class_name
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)

        for src in selected:
            dst = destination / src.name
            if dry_run:
                print(f"[DRY-RUN] {src} -> {dst}")
            else:
                shutil.move(str(src), str(dst))

        moved_summary[class_name] = len(selected)

    # 최종 결과 출력
    if not dry_run:
        print(f"Created validation set at: {val_root}")
    print(f"Moved summary: {moved_summary}")
    return moved_summary


# -----------------------------------------------------------------------------
# 6) CLI 인자 파싱
# -----------------------------------------------------------------------------
# 이 스크립트는 터미널에서 직접 실행될 수 있도록 argparse를 사용한다.
# 예:
#   python make_val_split.py --train-normal-dir D:\IMG_DATA\WA_DiNO\TRAIN --test-dir D:\IMG_DATA\WA_DiNO\TEST
#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validation split by moving a random subset of normal and anomaly images into a VAL folder."
    )

    # YAML 설정 파일 경로. 있으면 기본 값으로 읽어온다.
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config path. If omitted, uses configs/anomaly/WA_DiNO/base_config.yaml")

    # 트레이닝 루트와 테스트 루트. 직접 지정 가능
    parser.add_argument("--train-normal-dir", type=Path, default=None, help="Path to the normal training directory, e.g. .../TRAIN/OK or parent TRAIN folder")
    parser.add_argument("--test-dir", type=Path, default=None, help="Path to the test directory, e.g. .../TEST")

    # 검증 결과를 최종적으로 저장할 VAL 디렉터리
    parser.add_argument("--val-root", type=Path, default=None, help="Path to output VAL directory. Defaults to sibling VAL folder")

    # 정상 클래스 명
    parser.add_argument("--normal-class", default="OK", help="Normal class name in train/test folders")

    # 비율 설정
    parser.add_argument("--train-ratio", type=float, default=0.1, help="Fraction of normal training images to move to validation (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of anomaly images in each test class to move to validation (default: 0.2)")

    # 랜덤 시드 고정 (재현 가능한 결과)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")

    # 실제 이동 대신 로그만 출력
    parser.add_argument("--dry-run", action="store_true", help="Print moves without actually moving files")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 7) 메인 실행부
# -----------------------------------------------------------------------------
# 이 함수는 실제로 스크립트가 실행될 때 제일 먼저 호출된다.
#
# 동작 순서:
# 1. config 또는 CLI 인자에서 경로를 읽음
# 2. train/test 경로를 정리함
# 3. create_val_split() 을 호출해 파일 이동
# 4. 종료 코드 0 반환
#
def main() -> int:
    args = parse_args()

    # config 파일이 있으면 기본 경로값을 읽고, CLI 값이 있으면 덮어쓴다.
    if args.config is not None:
        train_dir, test_dir, normal_class, train_ratio, test_ratio = _resolve_default_paths(args.config)

        if args.train_normal_dir is not None:
            train_dir = args.train_normal_dir
        if args.test_dir is not None:
            test_dir = args.test_dir
        if args.normal_class != "OK":
            normal_class = args.normal_class
        if args.train_ratio != 0.1:
            train_ratio = args.train_ratio
        if args.test_ratio != 0.2:
            test_ratio = args.test_ratio
    else:
        # config 없이 CLI 인자만 쓰는 경우
        train_dir = args.train_normal_dir
        test_dir = args.test_dir
        normal_class = args.normal_class
        train_ratio = args.train_ratio
        test_ratio = args.test_ratio

        # CLI에서 트레이닝 루트, 테스트 루트가 빠지면 기본 config를 사용
        if train_dir is None or test_dir is None:
            train_dir, test_dir, normal_class, train_ratio, test_ratio = _resolve_default_paths(None)

    # 최종적으로 경로가 없으면 에러
    if train_dir is None or test_dir is None:
        raise ValueError("Both --train-normal-dir and --test-dir are required unless a config file is provided.")

    train_root = Path(train_dir).expanduser().resolve()

    # VAL 루트 결정.
    # --val-root가 없으면 데이터 루트의 sibling "VAL" 폴더를 만든다.
    if args.val_root is not None:
        val_root = args.val_root
    else:
        dataset_root = _resolve_dataset_root(train_root, normal_class)
        val_root = dataset_root / "VAL"

    # 실제 분할 실행
    create_val_split(
        train_normal_dir=train_root,
        test_dir=test_dir,
        val_root=val_root,
        normal_class=normal_class,
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    # 스크립트가 셀 실행이 아니라 파일 실행으로 호출되면 main()을 돌린다.
    raise SystemExit(main())
