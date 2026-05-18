#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _parse_cols(value: str | None) -> list[str] | None:
    if value is None:
        return None
    cols = [c.strip() for c in value.split(",") if c.strip()]
    return cols or None


def _moving_average(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1:
        return values.copy()
    window = int(window)
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))

    if mode == "causal":
        ends = np.arange(1, values.size + 1, dtype=np.int64)
        starts = np.maximum(0, ends - window)
    elif mode == "centered":
        half_left = window // 2
        half_right = window - half_left - 1
        centers = np.arange(values.size, dtype=np.int64)
        starts = np.maximum(0, centers - half_left)
        ends = np.minimum(values.size, centers + half_right + 1)
    else:
        raise ValueError(f"Unsupported moving-average mode: {mode}")

    counts = ends - starts
    return (cumsum[ends] - cumsum[starts]) / counts


def _load_csv_columns(csv_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        buffers: dict[str, list[float]] = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                buffers[name].append(float(row[name]))
    return fieldnames, {name: np.asarray(values, dtype=np.float64) for name, values in buffers.items()}


def _crop_mask(time: np.ndarray, time_start: float | None, time_end: float | None) -> np.ndarray:
    mask = np.ones(time.size, dtype=bool)
    if time_start is not None:
        mask &= time >= float(time_start)
    if time_end is not None:
        mask &= time <= float(time_end)
    if not np.any(mask):
        raise ValueError(f"No samples remain after crop: time_start={time_start}, time_end={time_end}")
    return mask


def _load_window_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    path = Path(config_path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--window-config must be a JSON object.")
    windows = raw.get("windows", raw)
    if not isinstance(windows, dict):
        raise ValueError("--window-config JSON must be a mapping or contain a 'windows' mapping.")
    return windows


def _resolve_time_window(
    csv_path: Path,
    default_start: float | None,
    default_end: float | None,
    window_config: dict,
) -> tuple[float | None, float | None]:
    keys = (
        str(csv_path),
        str(csv_path.resolve()),
        csv_path.name,
        csv_path.stem,
    )
    spec = None
    for key in keys:
        if key in window_config:
            spec = window_config[key]
            break
    if spec is None:
        return default_start, default_end
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ValueError(f"Window list for {csv_path} must contain [time_start, time_end].")
        return spec[0], spec[1]
    if not isinstance(spec, dict):
        raise ValueError(f"Window spec for {csv_path} must be a dict or [time_start, time_end].")
    return spec.get("time_start", default_start), spec.get("time_end", default_end)


def _split_labels(length: int, train_ratio: float, val_ratio: float) -> np.ndarray:
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    if train_end <= 0 or val_end <= train_end or val_end >= length:
        raise ValueError("Invalid train/val ratios for the available segment length.")
    split = np.full(length, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    return split


def _train_val_labels(length: int, train_ratio: float) -> np.ndarray:
    train_end = int(length * train_ratio)
    if train_end <= 0 or train_end >= length:
        raise ValueError("Invalid train_ratio for train/val-only source.")
    split = np.full(length, "val", dtype=object)
    split[:train_end] = "train"
    return split


def _adapt_labels(length: int, train_ratio: float, val_ratio: float, test_start_ratio: float | None) -> np.ndarray:
    if test_start_ratio is None:
        return _split_labels(length, train_ratio, val_ratio)

    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    test_start = int(length * test_start_ratio)
    if train_end < 0 or val_end < train_end or test_start <= val_end or test_start >= length:
        raise ValueError(
            "Invalid adapt split ratios: require 0 <= train <= train+val < test_start < 1."
        )

    split = np.full(length, "ignore", dtype=object)
    if train_end > 0:
        split[:train_end] = "train"
    if val_end > train_end:
        split[train_end:val_end] = "val"
    split[test_start:] = "test"
    return split


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert pressure channels into segment-wise waveform forecasting rows.")
    parser.add_argument("--csvs", nargs="*", default=[], help="Pressure/qdot CSV files. With --test-csvs/--adapt-csvs, these are source train/val files.")
    parser.add_argument(
        "--test-csvs",
        nargs="*",
        default=None,
        help="Optional held-out condition CSV files. When set, all rows from these files are labeled test.",
    )
    parser.add_argument(
        "--adapt-csvs",
        nargs="*",
        default=None,
        help="Optional target-condition CSV files split chronologically into adapt train/val/test rows.",
    )
    parser.add_argument("--pressure-cols", default=None, help="Comma-separated pressure columns. Defaults to all p* columns.")
    parser.add_argument("--time-start", type=float, default=0.16)
    parser.add_argument("--time-end", type=float, default=0.50)
    parser.add_argument(
        "--window-config",
        default=None,
        help="Optional JSON mapping per CSV/stem/name to {time_start,time_end}; overrides global time-start/time-end.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--adapt-train-ratio", type=float, default=0.25)
    parser.add_argument("--adapt-val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--adapt-test-start-ratio",
        type=float,
        default=None,
        help="If set, adapt CSV rows from this ratio onward are test; rows between adapt val and test are ignored.",
    )
    parser.add_argument("--input-smooth-window", type=int, default=2048)
    parser.add_argument("--input-smooth-mode", choices=["causal", "centered"], default="causal")
    parser.add_argument("--target-smooth-window", type=int, default=2048)
    parser.add_argument("--target-smooth-mode", choices=["causal", "centered"], default="centered")
    parser.add_argument("--output", required=True, help="Output long-format CSV path.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not args.csvs and not args.test_csvs and not args.adapt_csvs:
        raise ValueError("Provide at least one of --csvs, --adapt-csvs, or --test-csvs.")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    requested_cols = _parse_cols(args.pressure_cols)
    window_config = _load_window_config(args.window_config)
    fieldnames = ["time", "p_raw", f"p_input_ma{args.input_smooth_window}", f"p_target_cma{args.target_smooth_window}", "split", "segment_id"]
    config = {
        "input_csvs": [str(Path(path).resolve()) for path in args.csvs],
        "test_csvs": [] if args.test_csvs is None else [str(Path(path).resolve()) for path in args.test_csvs],
        "adapt_csvs": [] if args.adapt_csvs is None else [str(Path(path).resolve()) for path in args.adapt_csvs],
        "split_policy": (
            "trainval_csvs_plus_adapt_csvs_and_test_csvs"
            if args.adapt_csvs or args.test_csvs
            else "within_each_csv"
        ),
        "pressure_cols": requested_cols,
        "time_start": args.time_start,
        "time_end": args.time_end,
        "window_config": None if args.window_config is None else str(Path(args.window_config).resolve()),
        "resolved_windows": {},
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "adapt_train_ratio": args.adapt_train_ratio,
        "adapt_val_ratio": args.adapt_val_ratio,
        "adapt_test_start_ratio": args.adapt_test_start_ratio,
        "input_smooth_window": args.input_smooth_window,
        "input_smooth_mode": args.input_smooth_mode,
        "target_smooth_window": args.target_smooth_window,
        "target_smooth_mode": args.target_smooth_mode,
        "output_csv": str(output_path.resolve()),
        "segments": [],
        "rows_written": 0,
    }

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        sources = [(csv_arg, "within") for csv_arg in args.csvs]
        if args.test_csvs or args.adapt_csvs:
            sources = [(csv_arg, "trainval") for csv_arg in args.csvs]
            sources += [(csv_arg, "adapt") for csv_arg in (args.adapt_csvs or [])]
            sources += [(csv_arg, "test") for csv_arg in (args.test_csvs or [])]

        for csv_arg, split_role in sources:
            csv_path = Path(csv_arg)
            header, columns = _load_csv_columns(csv_path)
            if "time" not in columns:
                raise ValueError(f"Input CSV must contain a time column: {csv_path}")
            pressure_cols = requested_cols or [name for name in header if name.startswith("p")]
            missing = [name for name in pressure_cols if name not in columns]
            if missing:
                raise ValueError(f"Missing pressure columns in {csv_path}: {missing}")

            time_start, time_end = _resolve_time_window(csv_path, args.time_start, args.time_end, window_config)
            config["resolved_windows"][str(csv_path)] = {"time_start": time_start, "time_end": time_end}

            mask = _crop_mask(columns["time"], time_start, time_end)
            time = columns["time"][mask]
            if split_role == "within":
                split = _split_labels(time.size, args.train_ratio, args.val_ratio)
            elif split_role == "trainval":
                split = _train_val_labels(time.size, args.train_ratio)
            elif split_role == "adapt":
                split = _adapt_labels(time.size, args.adapt_train_ratio, args.adapt_val_ratio, args.adapt_test_start_ratio)
            elif split_role == "test":
                split = np.full(time.size, "test", dtype=object)
            else:
                raise ValueError(f"Unknown split role: {split_role}")

            for col in pressure_cols:
                raw = columns[col][mask]
                p_input = _moving_average(raw, args.input_smooth_window, args.input_smooth_mode)
                p_target = _moving_average(raw, args.target_smooth_window, args.target_smooth_mode)
                segment_id = f"{csv_path.stem}_{col}"
                for i in range(time.size):
                    writer.writerow(
                        {
                            "time": float(time[i]),
                            "p_raw": float(raw[i]),
                            f"p_input_ma{args.input_smooth_window}": float(p_input[i]),
                            f"p_target_cma{args.target_smooth_window}": float(p_target[i]),
                            "split": str(split[i]),
                            "segment_id": segment_id,
                        }
                    )
                config["segments"].append(
                    {
                        "source_csv": str(csv_path),
                        "pressure_col": col,
                        "segment_id": segment_id,
                        "split_role": split_role,
                        "time_start": None if time_start is None else float(time_start),
                        "time_end": None if time_end is None else float(time_end),
                        "first_time": float(time[0]),
                        "last_time": float(time[-1]),
                        "rows": int(time.size),
                        "train_rows": int(np.sum(split == "train")),
                        "val_rows": int(np.sum(split == "val")),
                        "test_rows": int(np.sum(split == "test")),
                    }
                )
                config["rows_written"] += int(time.size)

    config["num_segments"] = len(config["segments"])
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared channel waveform CSV: {output_path}")
    print(f"Config JSON: {config_path}")
    print(f"Segments: {config['num_segments']}")
    print(f"Rows written: {config['rows_written']}")
    print(f"Input column: p_input_ma{args.input_smooth_window}")
    print(f"Target column: p_target_cma{args.target_smooth_window}")


if __name__ == "__main__":
    main()
