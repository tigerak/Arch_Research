from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch
import yaml

"""실험 포털의 공통 스케줄러 및 요약 관리자.

이 파일은 "실제로 무엇을 실행할지"를 결정하는 상위 오케스트레이터 역할을 한다.

핵심 기능:
- schedule key 목록을 읽어 하나씩 실행한다.
- base config와 custom config를 병합한다.
- task type(classification/anomaly)을 확인한다.
- 각 task 엔진을 호출한다.
- 최종 성공/실패 결과를 요약 CSV와 콘솔 테이블로 저장한다.

간단히 말하면, 이 파일은 실험실의 관리자이며
하위 모듈인 train_engine/eval_engine/anomaly.engine을 조율하는 중앙 제어장치이다.
"""

from src.anomaly.engine import run_anomaly_eval, run_anomaly_train
from src.classification.engine import run_classification_eval, run_classification_train
from src.config_loader import ConfigError, get_task_type, load_yaml_file, merge_base_and_custom
from src.experiment_resolver import resolve_experiment_config_path
from utils.setup_logger import get_file_logger


def run_experiments(
    schedule_keys: Iterable[str],
    *,
    base_config_path: str | Path = "configs/base_config.yaml",
    configs_dir: str | Path = "configs",
    mode: str = "train",
    continue_on_error: bool = True,
    save_summary: bool = True,
) -> Dict[str, Any]:
    """여러 실험을 순차 실행하고 통합 요약 결과를 반환한다.

    핵심 목적:
    - SCHEDULE에 적힌 키(custom_config 등)를 순서대로 실행
    - 각 실험의 성공/실패, best score, 저장 경로를 구조화해 수집
    - 실행이 끝나면 콘솔 요약 테이블과 CSV 요약 파일을 생성

    파라미터:
    - schedule_keys: 실험 키 목록(예: ["custom_config", "ablation_lr1e4"])
    - base_config_path: 공통 기본 설정 파일
    - configs_dir: 커스텀 설정 파일 기본 디렉터리
    - mode: "train" 또는 "eval"
    - continue_on_error: 실패 실험이 있어도 다음 실험을 계속할지 여부
    - save_summary: 실행 결과를 CSV로 저장할지 여부
    """
    mode_l = str(mode).lower()
    if mode_l not in {"train", "eval"}:
        raise ConfigError("mode must be train or eval", field="mode", details={"value": mode})

    today = datetime.now().strftime("%y%m%d")
    run_root_for_summary: Path | None = None
    today_project_name_for_summary = ""

    records: List[Dict[str, Any]] = []

    for order, key in enumerate(schedule_keys, start=1):
        started = perf_counter()
        key_str = str(key)
        logger = None

        try:
            # 1) SCHEDULE 키를 커스텀 설정 파일 경로로 해석하고 존재 여부를 확인한다.
            custom_path = resolve_experiment_config_path(key_str, configs_dir=configs_dir)

            # 2) base + custom을 병합하고 스키마 검증까지 한 번에 수행한다.
            cfg = merge_base_and_custom(
                base_config_path=base_config_path,
                custom_config_path=custom_path,
                configs_dir=configs_dir,
            )

            data_cfg = cfg.get("data")
            if not isinstance(data_cfg, dict):
                raise ConfigError("data section must be mapping", field="data")

            task_name = custom_path.parent.name
            base_save_dir = Path(str(data_cfg.get("save_dir", "models/save")))
            task_save_dir = base_save_dir / task_name
            task_save_dir.mkdir(parents=True, exist_ok=True)
            data_cfg["save_dir"] = str(task_save_dir)

            config_project_name = str(cfg.get("project_name", "experiment"))
            today_project_name = f"{today}_{config_project_name}"
            cfg["project_name"] = today_project_name
            run_root = Path(str(cfg["data"]["save_dir"])) / today_project_name
            run_root.mkdir(parents=True, exist_ok=True)
            run_root_for_summary = run_root
            today_project_name_for_summary = today_project_name
            task_type = get_task_type(cfg)

            # run_manager 로그는 각 실험 폴더 안에 저장한다.
            logger = get_file_logger(
                logger_name=f"run_manager_{today}_{key_str}_{order}",
                log_file=run_root / f"run_manager_{today}.log",
            )
            logger.info("Run started")
            logger.info("mode=%s", mode_l)
            logger.info("task_type=%s", task_type)

            logger.info("[%s] custom=%s", key_str, custom_path)
            logger.info("[%s] task_name=%s", key_str, task_name)
            logger.info("[%s] task_save_dir=%s", key_str, task_save_dir)
            logger.info("[%s] today_project_name=%s", key_str, today_project_name)

            # 3) 재현성을 위해 "실행 당시 실제 병합 결과"를 실험 폴더에 저장한다.
            merged_cfg_path = save_merged_config(cfg, config_project_name=config_project_name)
            logger.info("[%s] Config=%s", key_str, merged_cfg_path)

            run_result = _dispatch_task_run(cfg, mode=mode_l, logger=logger)
            record = _build_success_record(
                order=order,
                schedule_key=key_str,
                mode=mode_l,
                task_type=task_type,
                cfg=cfg,
                custom_path=custom_path,
                merged_cfg_path=merged_cfg_path,
                run_result=run_result,
                started=started,
            )
            logger.info(
                "[%s] SUCCESS | task_type=%s | best_metric=%s | best_score=%s",
                key_str,
                task_type,
                record["best_metric"],
                record["best_score"],
            )

            records.append(record)

        except Exception as exc:
            # 실패도 요약에 남겨야 후속 분석(어느 설정이 깨졌는지)이 가능하다.
            record = {
                "order": order,
                "schedule_key": key_str,
                "status": "FAILED",
                "task_type": "",
                "mode": mode_l,
                "project_name": "",
                "custom_config_path": "",
                "merged_config_path": "",
                "best_metric": "",
                "best_score": "",
                "checkpoint_path": "",
                "save_root": "",
                "error": str(exc),
                "duration_sec": round(perf_counter() - started, 3),
            }
            records.append(record)
            if logger is not None:
                logger.exception("[%s] FAILED: %s", key_str, exc)
            else:
                print(f"[run_manager][{key_str}] FAILED: {exc}")

            if not continue_on_error:
                break

    summary = {
        "mode": mode_l,
        "total": len(records),
        "success": sum(1 for r in records if r["status"] == "SUCCESS"),
        "failed": sum(1 for r in records if r["status"] == "FAILED"),
        "records": records,
    }

    print_summary_table(records)

    summary_csv_path = ""
    if save_summary and run_root_for_summary is not None:
        summary_csv_path = str(
            save_summary_csv(
                records,
                run_root=run_root_for_summary,
                today_project_name=today_project_name_for_summary,
            )
        )
        summary["summary_csv"] = summary_csv_path
        print(f"[RunManager] Summary CSV saved: {summary_csv_path}")

    return summary


def save_merged_config(cfg: Mapping[str, Any], *, config_project_name: str) -> Path:
    """병합된 최종 설정을 실험 폴더에 저장한다.

    파일명에 schedule_key와 timestamp를 포함해,
    같은 프로젝트를 여러 번 돌려도 설정 파일이 덮어쓰이지 않게 한다.
    """
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, Mapping):
        raise ConfigError("data section must be mapping", field="data")

    save_dir = Path(str(data_cfg.get("save_dir", "models/save")))
    project_name = str(cfg.get("project_name", "experiment"))
    out_dir = save_dir / project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_config_project = config_project_name.replace("/", "_").replace("\\", "_").replace(":", "_")
    out_path = out_dir / f"config_{safe_config_project}.yaml"

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, allow_unicode=True, sort_keys=False)

    return out_path


def save_summary_csv(records: List[Dict[str, Any]], *, run_root: Path, today_project_name: str) -> Path:
    """실험 요약 레코드를 CSV로 저장한다."""
    run_root.mkdir(parents=True, exist_ok=True)
    out_path = run_root / f"{today_project_name}_run_summary.csv"

    columns = [
        "order",
        "schedule_key",
        "status",
        "task_type",
        "mode",
        "project_name",
        "custom_config_path",
        "merged_config_path",
        "best_metric",
        "best_score",
        "checkpoint_path",
        "save_root",
        "duration_sec",
        "error",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    return out_path


def build_today_project_root(base_config_path: str | Path) -> tuple[str, Path]:
    """base_config의 project_name을 이용해 today_project_name과 저장 루트를 만든다.

    형식:
    - today_project_name: YYMMDD_<base_project_name>
    - run_root: models/save/today_project_name
    """
    base_cfg = load_yaml_file(base_config_path)
    base_project_name = str(base_cfg.get("project_name", "experiment"))
    today = datetime.now().strftime("%y%m%d")
    today_project_name = f"{today}_{base_project_name}"
    run_root = Path("models") / "save" / today_project_name
    run_root.mkdir(parents=True, exist_ok=True)
    return today_project_name, run_root


def print_summary_table(records: List[Dict[str, Any]]) -> None:
    """콘솔에서 빠르게 결과를 훑을 수 있도록 요약 테이블을 출력한다."""
    if not records:
        print("[RunManager] No records")
        return

    headers = ["#", "key", "task", "status", "best(metric)", "best(score)", "save_root", "sec"]
    rows: List[List[str]] = []

    for r in records:
        best_metric = str(r.get("best_metric", ""))
        best_score = r.get("best_score", "")
        if isinstance(best_score, float):
            best_score_text = f"{best_score:.6f}"
        else:
            best_score_text = str(best_score)

        rows.append(
            [
                str(r.get("order", "")),
                str(r.get("schedule_key", "")),
                str(r.get("task_type", "")),
                str(r.get("status", "")),
                best_metric,
                best_score_text,
                str(r.get("save_root", "")),
                str(r.get("duration_sec", "")),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _fmt(row: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)

    print("[RunManager] Summary")
    print(_fmt(headers))
    print(sep)
    for row in rows:
        print(_fmt(row))

    failed = [r for r in records if r.get("status") == "FAILED"]
    if failed:
        print("[RunManager] Failed details:")
        for r in failed:
            print(f"  - key={r.get('schedule_key')} error={r.get('error')}")


def _extract_best_score_from_checkpoint(checkpoint_path: Optional[str]) -> tuple[str, float]:
    """체크포인트에서 best score를 추출한다.

    기준:
    - compare_metrics의 첫 번째 항목을 대표 best metric으로 사용
    - best_values에 해당 키가 있으면 그 값을 우선 사용
    - 없으면 metrics의 값을 사용
    - 없거나 체크포인트가 없으면 NaN 반환
    """
    if not checkpoint_path:
        return "", float("nan")

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        return "", float("nan")

    obj = torch.load(ckpt, map_location="cpu")
    if not isinstance(obj, Mapping):
        return "", float("nan")

    compare_metrics = obj.get("compare_metrics", [])
    if isinstance(compare_metrics, list) and compare_metrics:
        metric_name = str(compare_metrics[0])
    else:
        metric_name = "val_loss"

    best_values = obj.get("best_values", {})
    if isinstance(best_values, Mapping) and metric_name in best_values:
        try:
            return metric_name, float(best_values[metric_name])
        except Exception:
            return metric_name, float("nan")

    metrics = obj.get("metrics", {})
    if isinstance(metrics, Mapping) and metric_name in metrics:
        try:
            return metric_name, float(metrics[metric_name])
        except Exception:
            return metric_name, float("nan")

    return metric_name, float("nan")


def _dispatch_task_run(cfg: Mapping[str, Any], *, mode: str, logger) -> Dict[str, Any]:
    task_type = get_task_type(cfg)

    if task_type == "classification":
        if mode == "train":
            return run_classification_train(cfg, logger=logger)
        if mode == "eval":
            return run_classification_eval(cfg)

    if task_type == "anomaly_detection":
        if mode == "train":
            return run_anomaly_train(cfg)
        if mode == "eval":
            return run_anomaly_eval(cfg)

    raise ConfigError(
        "Unsupported task/mode combination",
        field="task.type/mode",
        details={"task_type": task_type, "mode": mode},
    )


def _build_success_record(
    *,
    order: int,
    schedule_key: str,
    mode: str,
    task_type: str,
    cfg: Mapping[str, Any],
    custom_path: Path,
    merged_cfg_path: Path,
    run_result: Mapping[str, Any],
    started: float,
) -> Dict[str, Any]:
    best_metric, best_score, checkpoint_path, save_root = _extract_result_summary(
        task_type=task_type,
        mode=mode,
        cfg=cfg,
        run_result=run_result,
    )

    return {
        "order": order,
        "schedule_key": schedule_key,
        "status": "SUCCESS",
        "task_type": task_type,
        "mode": mode,
        "project_name": str(cfg.get("project_name", "")),
        "custom_config_path": str(custom_path),
        "merged_config_path": str(merged_cfg_path),
        "best_metric": best_metric,
        "best_score": best_score,
        "checkpoint_path": checkpoint_path,
        "save_root": save_root,
        "error": "",
        "duration_sec": round(perf_counter() - started, 3),
    }


def _extract_result_summary(
    *,
    task_type: str,
    mode: str,
    cfg: Mapping[str, Any],
    run_result: Mapping[str, Any],
) -> tuple[str, float, str, str]:
    if task_type == "classification":
        if mode == "train":
            checkpoint_path = str(run_result.get("best_checkpoint", ""))
            best_metric_name, best_metric_value = _extract_best_score_from_checkpoint(checkpoint_path)
            save_root = str(run_result.get("save_root", ""))
            return best_metric_name, best_metric_value, checkpoint_path, save_root

        checkpoint_path = str(run_result.get("checkpoint_path", ""))
        save_root = str(Path(str(cfg["data"]["save_dir"])) / str(cfg.get("project_name", "")))
        return "macro_f1", float(run_result.get("macro_f1", 0.0)), checkpoint_path, save_root

    if task_type == "anomaly_detection":
        best_metric = str(run_result.get("primary_metric", "image_auroc"))
        best_score = float(run_result.get("primary_score", 0.0))
        save_root = str(run_result.get("save_root", Path(str(cfg["data"]["save_dir"])) / str(cfg.get("project_name", ""))))
        return best_metric, best_score, "", save_root

    raise ConfigError(
        "Unsupported task type in result summary",
        field="task.type",
        details={"task_type": task_type},
    )
