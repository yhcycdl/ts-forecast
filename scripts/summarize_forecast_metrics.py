#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PREFERRED_COLUMNS = [
    "checkpoint_dir",
    "model",
    "model_id",
    "data_path",
    "input_cols",
    "output_cols",
    "seq_len",
    "pred_len",
    "loss",
    "mse_raw",
    "mae_raw",
    "pearson_raw",
    "mse_norm",
    "mae_norm",
    "pearson_norm",
    "dominant_period_true_samples",
    "dominant_period_pred_samples",
    "dominant_period_error_samples",
    "dominant_period_relative_error",
    "spectral_energy_l1",
    "envelope_relative_mae",
    "peak_count_true",
    "peak_count_pred",
    "peak_count_error",
    "peak_time_mae_samples",
    "peak_hit_rate",
    "metrics_json",
    "values_csv",
]


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect(root: Path) -> list[dict]:
    rows: list[dict] = []
    for metrics_path in sorted(root.rglob("point_metrics.json")):
        metrics = _read_json(metrics_path)
        args = _read_json(metrics_path.with_name("run_args.json"))
        row: dict[str, object] = {
            "checkpoint_dir": str(metrics_path.parent),
            "metrics_json": str(metrics_path),
        }
        for key in [
            "model",
            "model_id",
            "data_path",
            "input_cols",
            "output_cols",
            "seq_len",
            "pred_len",
            "loss",
            "run_tag",
            "run_stamp",
        ]:
            if key in args:
                row[key] = args[key]
        for key, value in metrics.items():
            row[key] = value
        rows.append(row)
    return rows


def _sort_key(row: dict):
    data = str(row.get("data_path", ""))
    model = str(row.get("model", ""))
    pred_len = row.get("pred_len", "")
    model_id = str(row.get("model_id", ""))
    return data, model, str(pred_len), model_id, str(row.get("checkpoint_dir", ""))


def _write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extra_keys = sorted({key for row in rows for key in row.keys()} - set(PREFERRED_COLUMNS))
    fieldnames = [key for key in PREFERRED_COLUMNS if any(key in row for row in rows)] + extra_keys
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize forecast point/structure metrics from checkpoint folders.")
    parser.add_argument("--root", default="./checkpoints", help="Checkpoint root to scan recursively.")
    parser.add_argument("--output", default="./outputs/forecast_metrics_summary.csv", help="Output CSV path.")
    parser.add_argument("--min-pearson", type=float, default=None, help="Optional filter: keep rows with pearson_raw >= value.")
    parser.add_argument("--max-mse", type=float, default=None, help="Optional filter: keep rows with mse_raw <= value.")
    args = parser.parse_args()

    rows = _collect(Path(args.root))
    if args.min_pearson is not None:
        rows = [row for row in rows if float(row.get("pearson_raw", float("-inf"))) >= float(args.min_pearson)]
    if args.max_mse is not None:
        rows = [row for row in rows if float(row.get("mse_raw", float("inf"))) <= float(args.max_mse)]
    rows = sorted(rows, key=_sort_key)

    output_path = Path(args.output)
    _write_csv(rows, output_path)
    print(f"Saved summary: {output_path}")
    print(f"Rows: {len(rows)}")
    if rows:
        best = sorted(rows, key=lambda row: float(row.get("mse_raw", float("inf"))))[0]
        print(
            "Best mse_raw: "
            f"{best.get('mse_raw')} | model={best.get('model')} | "
            f"data={best.get('data_path')} | dir={best.get('checkpoint_dir')}"
        )


if __name__ == "__main__":
    main()
