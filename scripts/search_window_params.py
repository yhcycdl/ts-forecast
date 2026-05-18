#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.future_indicator import WindowParams, assign_window_split, iter_window_indices, ms_to_samples


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_float_list(raw: str) -> list[float]:
    values = [float(piece.strip()) for piece in str(raw).split(",") if piece.strip()]
    if not values:
        raise ValueError("At least one candidate value is required.")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search candidate window parameter settings on the Zenodo CSV manifest.")
    parser.add_argument("--manifest-json", type=str, default="./data/zenodo_timeseries_csv/manifest.json")
    parser.add_argument("--output-dir", type=str, default="./outputs/window_search")
    parser.add_argument("--history-ms-list", type=str, default="100,300,500")
    parser.add_argument("--delta-ms-list", type=str, default="10,30,50")
    parser.add_argument("--future-ms-list", type=str, default="30,100,300")
    parser.add_argument(
        "--stride-mode",
        type=str,
        default="equal_future",
        choices=["equal_future", "half_future", "quarter_future"],
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, default="total")
    return parser


def _stride_ms(future_ms: float, stride_mode: str) -> float:
    if stride_mode == "equal_future":
        return float(future_ms)
    if stride_mode == "half_future":
        return float(future_ms) / 2.0
    return float(future_ms) / 4.0


def main() -> None:
    args = build_parser().parse_args()
    manifest = _load_json(args.manifest_json)
    history_ms_list = _parse_float_list(args.history_ms_list)
    delta_ms_list = _parse_float_list(args.delta_ms_list)
    future_ms_list = _parse_float_list(args.future_ms_list)

    rows: list[dict[str, Any]] = []
    for history_ms in history_ms_list:
        for delta_ms in delta_ms_list:
            for future_ms in future_ms_list:
                per_run_counts: list[int] = []
                total_counts = {"train": 0, "val": 0, "test": 0, "cross_boundary": 0}

                for op in manifest.get("operating_points", []):
                    sample_rate = float(op["sampling_rate_hz"])
                    params = WindowParams(
                        history_length=ms_to_samples(history_ms, sample_rate),
                        lead_gap=ms_to_samples(delta_ms, sample_rate),
                        future_length=ms_to_samples(future_ms, sample_rate),
                        stride=ms_to_samples(_stride_ms(future_ms, args.stride_mode), sample_rate),
                    )
                    run_total = 0
                    total_length = int(op["signal_length_samples"])
                    for window in iter_window_indices(total_length=total_length, params=params):
                        split = assign_window_split(
                            history_start=window["history_start"],
                            future_end=window["future_end"],
                            total_length=total_length,
                            train_ratio=float(args.train_ratio),
                            val_ratio=float(args.val_ratio),
                            split_mode=args.split_mode,
                        )
                        total_counts[split] = total_counts.get(split, 0) + 1
                        if split != "cross_boundary":
                            run_total += 1
                    per_run_counts.append(run_total)

                rows.append(
                    {
                        "history_ms": float(history_ms),
                        "delta_ms": float(delta_ms),
                        "future_ms": float(future_ms),
                        "stride_ms": float(_stride_ms(future_ms, args.stride_mode)),
                        "history_len": params.history_length,
                        "lead_gap": params.lead_gap,
                        "future_len": params.future_length,
                        "stride": params.stride,
                        "num_runs": len(per_run_counts),
                        "min_windows_per_run": min(per_run_counts),
                        "mean_windows_per_run": sum(per_run_counts) / max(len(per_run_counts), 1),
                        "max_windows_per_run": max(per_run_counts),
                        "total_train_windows": total_counts["train"],
                        "total_val_windows": total_counts["val"],
                        "total_test_windows": total_counts["test"],
                        "total_cross_boundary_windows": total_counts["cross_boundary"],
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "window_param_grid.csv"
    json_path = output_dir / "window_param_grid_config.json"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "manifest_json": str(Path(args.manifest_json).resolve()),
                "history_ms_list": history_ms_list,
                "delta_ms_list": delta_ms_list,
                "future_ms_list": future_ms_list,
                "stride_mode": args.stride_mode,
                "train_ratio": float(args.train_ratio),
                "val_ratio": float(args.val_ratio),
                "split_mode": args.split_mode,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("Window search grid exported.")
    print(f"CSV: {csv_path}")
    print(f"Config: {json_path}")


if __name__ == "__main__":
    main()
