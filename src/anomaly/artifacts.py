from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def resolve_save_root(cfg: Mapping[str, Any]) -> Path:
    data_cfg = cfg["data"]
    project_name = str(cfg.get("project_name", "Anomaly_Experiment"))
    save_root = Path(str(data_cfg.get("save_dir", "models/save"))) / project_name
    save_root.mkdir(parents=True, exist_ok=True)
    return save_root


def save_json_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def save_image_score_predictions_csv(
    path: Path,
    image_paths: Sequence[str],
    class_names: Sequence[str],
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[int],
    threshold: float,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "class_name", "binary_label", "score", "prediction", "threshold"],
        )
        writer.writeheader()
        for image_path, class_name, label, score, pred in zip(image_paths, class_names, labels, scores, predictions):
            writer.writerow(
                {
                    "image_path": image_path,
                    "class_name": class_name,
                    "binary_label": int(label),
                    "score": float(score),
                    "prediction": int(pred),
                    "threshold": float(threshold),
                }
            )