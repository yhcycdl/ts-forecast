from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import csv
import json
import math
import numpy as np

from src.labels.margin_score import (
    FeatureSpec,
    build_fixed_rule_margin_config,
    default_feature_specs,
    fit_margin_config,
    score_rows,
)

EPS = 1e-6
BINARY_LABEL_NAMES = {0: "safe", 1: "risky"}
THREE_CLASS_LABEL_NAMES = {0: "safe", 1: "pre_instability", 2: "unstable"}


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _quantile(values: list[float], q: float, fallback: float) -> float:
    if not values:
        return float(fallback)
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, float(q)))


def _label_name_map(kind: str) -> dict[int, str]:
    return BINARY_LABEL_NAMES if kind == "binary" else THREE_CLASS_LABEL_NAMES


def _distribution_rows(
    rows: list[dict[str, object]],
    label_key: str,
    kind: str,
) -> list[dict[str, object]]:
    name_map = _label_name_map(kind)
    grouped: dict[tuple[str, str, int], int] = {}
    totals: dict[tuple[str, str], int] = {}
    for row in rows:
        split = str(row["split"])
        condition_id = str(row["condition_id"])
        label_value = int(row[label_key])
        grouped[(split, condition_id, label_value)] = grouped.get((split, condition_id, label_value), 0) + 1
        totals[(split, condition_id)] = totals.get((split, condition_id), 0) + 1

    dist_rows: list[dict[str, object]] = []
    for (split, condition_id, label_value), count in sorted(grouped.items()):
        total = totals[(split, condition_id)]
        dist_rows.append(
            {
                "label_kind": kind,
                "split": split,
                "condition_id": condition_id,
                "label_value": label_value,
                "label_name": name_map.get(label_value, str(label_value)),
                "count": count,
                "proportion": count / max(total, 1),
            }
        )
    return dist_rows


def _threshold_sensitivity_rows(train_scores: list[float], scored_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not train_scores:
        return []
    quantiles = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    rows: list[dict[str, object]] = []
    for q in quantiles:
        threshold = float(np.quantile(np.asarray(train_scores, dtype=np.float64), q))
        for split in ["train", "val", "test"]:
            split_rows = [row for row in scored_rows if str(row["split"]) == split]
            positives = sum(1 for row in split_rows if float(row["margin_score"]) >= threshold)
            total = len(split_rows)
            rows.append(
                {
                    "quantile": q,
                    "threshold": threshold,
                    "split": split,
                    "positive_count": positives,
                    "negative_count": total - positives,
                    "positive_rate": positives / max(total, 1),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(slots=True)
class IndicatorLabelGenerator:
    margin_mode: str = "train-fitted"
    weight_mode: str = "physics_rule"
    threshold_mode: str = "train-fitted"
    binary_quantile: float = 0.80
    low_quantile: float = 0.50
    high_quantile: float = 0.85
    fixed_binary_threshold: float = 0.65
    fixed_low_threshold: float = 0.45
    fixed_high_threshold: float = 0.70
    feature_specs: list[FeatureSpec] | None = None
    margin_config: dict[str, object] | None = field(default=None, init=False)
    label_config: dict[str, object] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.margin_mode = str(self.margin_mode).lower()
        self.weight_mode = str(self.weight_mode).lower()
        self.threshold_mode = str(self.threshold_mode).lower()
        self.feature_specs = self.feature_specs or default_feature_specs()

    def fit(self, rows: Iterable[dict[str, object]]) -> "IndicatorLabelGenerator":
        rows = list(rows)
        if self.margin_mode == "train-fitted":
            self.margin_config = fit_margin_config(
                rows,
                feature_specs=self.feature_specs,
                weight_mode=self.weight_mode,
            )
        elif self.margin_mode == "fixed-rule":
            self.margin_config = build_fixed_rule_margin_config(
                feature_specs=self.feature_specs,
                weight_mode=self.weight_mode,
            )
        else:
            raise ValueError(f"Unsupported margin_mode: {self.margin_mode}")

        scored_rows = score_rows(rows, self.margin_config)
        train_scores = [float(row["margin_score"]) for row in scored_rows if str(row.get("split")) == "train"]
        if not train_scores:
            raise ValueError("No train rows found while fitting label thresholds.")

        if self.threshold_mode == "train-fitted":
            binary_threshold = _quantile(train_scores, self.binary_quantile, fallback=self.fixed_binary_threshold)
            low_threshold = _quantile(train_scores, self.low_quantile, fallback=self.fixed_low_threshold)
            high_threshold = _quantile(train_scores, self.high_quantile, fallback=self.fixed_high_threshold)
        elif self.threshold_mode == "fixed-rule":
            binary_threshold = float(self.fixed_binary_threshold)
            low_threshold = float(self.fixed_low_threshold)
            high_threshold = float(self.fixed_high_threshold)
        else:
            raise ValueError(f"Unsupported threshold_mode: {self.threshold_mode}")

        if low_threshold > high_threshold:
            low_threshold, high_threshold = high_threshold, low_threshold

        self.label_config = {
            "margin_mode": self.margin_mode,
            "weight_mode": self.weight_mode,
            "threshold_mode": self.threshold_mode,
            "binary_threshold": float(binary_threshold),
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
            "binary_quantile": float(self.binary_quantile),
            "low_quantile": float(self.low_quantile),
            "high_quantile": float(self.high_quantile),
        }
        return self

    def transform(self, rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
        if self.margin_config is None or self.label_config is None:
            raise RuntimeError("IndicatorLabelGenerator must be fit() before transform().")

        scored_rows = score_rows(rows, self.margin_config)
        binary_threshold = float(self.label_config["binary_threshold"])
        low_threshold = float(self.label_config["low_threshold"])
        high_threshold = float(self.label_config["high_threshold"])

        labeled_rows: list[dict[str, object]] = []
        for row in scored_rows:
            score = float(row["margin_score"])
            binary_label = 1 if score >= binary_threshold else 0
            if score < low_threshold:
                three_class_label = 0
            elif score >= high_threshold:
                three_class_label = 2
            else:
                three_class_label = 1

            enriched = dict(row)
            enriched["label_binary"] = binary_label
            enriched["label_binary_name"] = BINARY_LABEL_NAMES[binary_label]
            enriched["label_3class"] = three_class_label
            enriched["label_3class_name"] = THREE_CLASS_LABEL_NAMES[three_class_label]
            labeled_rows.append(enriched)
        return labeled_rows

    def fit_transform(self, rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
        rows = list(rows)
        self.fit(rows)
        return self.transform(rows)


def load_indicator_rows(csv_path: str | Path) -> list[dict[str, object]]:
    path = Path(csv_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def export_label_artifacts(
    rows: list[dict[str, object]],
    generator: IndicatorLabelGenerator,
    output_dir: str | Path,
    source_indicator_csv: str | Path,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled_rows = generator.fit_transform(rows)
    train_scores = [float(row["margin_score"]) for row in labeled_rows if str(row.get("split")) == "train"]
    label_distribution_rows = (
        _distribution_rows(labeled_rows, label_key="label_binary", kind="binary")
        + _distribution_rows(labeled_rows, label_key="label_3class", kind="three_class")
    )
    sensitivity_rows = _threshold_sensitivity_rows(train_scores, labeled_rows)

    labeled_table_path = output_dir / "labeled_indicator_table.csv"
    config_path = output_dir / "label_config.json"
    distribution_path = output_dir / "label_distribution_by_condition.csv"
    sensitivity_path = output_dir / "threshold_sensitivity.csv"
    report_path = output_dir / "label_report.md"

    _write_csv(labeled_table_path, labeled_rows)
    _write_csv(distribution_path, label_distribution_rows)
    _write_csv(sensitivity_path, sensitivity_rows)

    payload = {
        "source_indicator_csv": str(Path(source_indicator_csv).resolve()),
        "margin_config": generator.margin_config,
        "label_config": generator.label_config,
    }
    _write_json(config_path, payload)

    split_counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
        split_rows = [row for row in labeled_rows if str(row.get("split")) == split]
        split_counts[split] = {
            "count": len(split_rows),
            "binary_safe": sum(1 for row in split_rows if int(row["label_binary"]) == 0),
            "binary_risky": sum(1 for row in split_rows if int(row["label_binary"]) == 1),
            "safe": sum(1 for row in split_rows if int(row["label_3class"]) == 0),
            "pre_instability": sum(1 for row in split_rows if int(row["label_3class"]) == 1),
            "unstable": sum(1 for row in split_rows if int(row["label_3class"]) == 2),
        }

    report_lines = [
        "# Label Report",
        "",
        "## Source",
        f"- indicator_table: `{Path(source_indicator_csv).resolve()}`",
        f"- margin_mode: `{generator.margin_mode}`",
        f"- weight_mode: `{generator.weight_mode}`",
        f"- threshold_mode: `{generator.threshold_mode}`",
        "",
        "## Thresholds",
        f"- binary_threshold: `{generator.label_config['binary_threshold']:.6f}`",
        f"- low_threshold: `{generator.label_config['low_threshold']:.6f}`",
        f"- high_threshold: `{generator.label_config['high_threshold']:.6f}`",
        "",
        "## Split Counts",
    ]
    for split, counts in split_counts.items():
        report_lines.append(
            f"- {split}: total={counts['count']}, "
            f"binary_safe={counts['binary_safe']}, binary_risky={counts['binary_risky']}, "
            f"safe={counts['safe']}, pre_instability={counts['pre_instability']}, unstable={counts['unstable']}"
        )
    report_lines.append("")
    report_lines.append("## Notes")
    report_lines.append("- 标签完全由 future-window 指标构造，不使用任何外部真值标签。")
    report_lines.append("- `train-fitted` 模式下，margin calibrator 和阈值仅使用 `train` split 拟合。")
    report_lines.append("- `CECP` 当前仍为 TODO，占位值为 `NaN`，不会进入当前默认 margin_score。")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "labeled_indicator_table": str(labeled_table_path),
        "label_config": str(config_path),
        "label_distribution_by_condition": str(distribution_path),
        "threshold_sensitivity": str(sensitivity_path),
        "label_report": str(report_path),
    }
