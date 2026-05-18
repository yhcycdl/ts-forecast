#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _moving_average(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1:
        return values.copy()
    window = int(window)
    if mode == "causal":
        cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
        ends = np.arange(1, values.size + 1, dtype=np.int64)
        starts = np.maximum(0, ends - window)
        counts = ends - starts
        return (cumsum[ends] - cumsum[starts]) / counts

    half_left = window // 2
    half_right = window - half_left - 1
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    centers = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, centers - half_left)
    ends = np.minimum(values.size, centers + half_right + 1)
    counts = ends - starts
    return (cumsum[ends] - cumsum[starts]) / counts


def _load_columns(csv_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
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


def _crop(columns: dict[str, np.ndarray], time_start: float | None, time_end: float | None) -> dict[str, np.ndarray]:
    if "time" not in columns:
        raise ValueError("Input CSV must contain a time column.")
    time = columns["time"]
    mask = np.ones(time.size, dtype=bool)
    if time_start is not None:
        mask &= time >= float(time_start)
    if time_end is not None:
        mask &= time <= float(time_end)
    if not np.any(mask):
        raise ValueError(f"No samples remain after crop: {time_start=} {time_end=}")
    return {name: values[mask] for name, values in columns.items()}


def _local_peak_candidates(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.asarray([], dtype=np.int64)
    left = values[1:-1] > values[:-2]
    right = values[1:-1] >= values[2:]
    return np.flatnonzero(left & right).astype(np.int64) + 1


def _passes_prominence(values: np.ndarray, idx: int, distance: int, prominence: float) -> bool:
    if prominence <= 0:
        return True
    half = max(1, int(distance) // 2)
    start = max(0, idx - half)
    end = min(values.size, idx + half + 1)
    left_min = float(np.min(values[start : idx + 1]))
    right_min = float(np.min(values[idx:end]))
    local_prom = float(values[idx] - max(left_min, right_min))
    return local_prom >= float(prominence)


def _find_peaks(values: np.ndarray, distance: int, prominence: float) -> np.ndarray:
    candidates = _local_peak_candidates(values)
    candidates = np.asarray([i for i in candidates if _passes_prominence(values, int(i), distance, prominence)], dtype=np.int64)
    if candidates.size == 0:
        return candidates

    # Keep the strongest peak in each minimum-distance neighborhood.
    order = candidates[np.argsort(values[candidates])[::-1]]
    keep: list[int] = []
    blocked = np.zeros(values.size, dtype=bool)
    distance = max(1, int(distance))
    for idx in order:
        idx = int(idx)
        if blocked[idx]:
            continue
        keep.append(idx)
        start = max(0, idx - distance + 1)
        end = min(values.size, idx + distance)
        blocked[start:end] = True
    return np.asarray(sorted(keep), dtype=np.int64)


def _split_labels(length: int, train_ratio: float, val_ratio: float) -> np.ndarray:
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    split = np.full(length, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    return split


def _build_samples(
    peak_time: np.ndarray,
    peak_height: np.ndarray,
    input_peaks: int,
    output_peaks: int,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    start_min = 1
    start_max = peak_time.size - input_peaks - output_peaks
    for start in range(start_min, start_max + 1):
        row: dict[str, float] = {}
        in_idx = np.arange(start, start + input_peaks)
        out_idx = np.arange(start + input_peaks, start + input_peaks + output_peaks)
        for j, idx in enumerate(in_idx):
            row[f"in_h_{j}"] = float(peak_height[idx])
            row[f"in_dt_{j}"] = float(peak_time[idx] - peak_time[idx - 1])
        prev_idx = int(in_idx[-1])
        for j, idx in enumerate(out_idx):
            prev = prev_idx if j == 0 else int(out_idx[j - 1])
            row[f"target_h_{j}"] = float(peak_height[idx])
            row[f"target_dt_{j}"] = float(peak_time[idx] - peak_time[prev])
        row["input_start_peak"] = int(start)
        row["input_end_peak"] = int(in_idx[-1])
        row["target_start_peak"] = int(out_idx[0])
        row["target_end_peak"] = int(out_idx[-1])
        row["target_start_time"] = float(peak_time[out_idx[0]])
        row["target_end_time"] = float(peak_time[out_idx[-1]])
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, float]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_peaks(time: np.ndarray, raw: np.ndarray, smooth: np.ndarray, peak_idx: np.ndarray, path: Path, max_points: int) -> None:
    import matplotlib.pyplot as plt

    if max_points > 0 and time.size > max_points:
        step = max(1, time.size // max_points)
        sl = slice(None, None, step)
    else:
        sl = slice(None)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(time[sl], raw[sl], linewidth=0.5, alpha=0.4, label="raw")
    ax.plot(time[sl], smooth[sl], linewidth=1.0, label="smoothed")
    ax.scatter(time[peak_idx], smooth[peak_idx], s=18, color="#d62728", label="peaks", zorder=3)
    ax.set_xlabel("time / s")
    ax.set_ylabel("pressure")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare peak-sequence forecasting samples from a pressure CSV.")
    parser.add_argument("--csv", required=True, type=str)
    parser.add_argument("--pressure-column", default="p00", type=str)
    parser.add_argument("--time-start", default=0.16, type=float)
    parser.add_argument("--time-end", default=0.50, type=float)
    parser.add_argument("--smooth-window", default=512, type=int)
    parser.add_argument("--smooth-mode", choices=["centered", "causal"], default="centered")
    parser.add_argument("--peak-distance", default=3000, type=int, help="Minimum peak distance in samples.")
    parser.add_argument("--peak-prominence", default=40.0, type=float, help="Approximate local prominence threshold.")
    parser.add_argument("--input-peaks", default=8, type=int)
    parser.add_argument("--output-peaks", default=3, type=int)
    parser.add_argument("--train-ratio", default=0.7, type=float)
    parser.add_argument("--val-ratio", default=0.15, type=float)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--max-plot-points", default=50000, type=int)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header, columns = _load_columns(csv_path)
    if args.pressure_column not in columns:
        raise ValueError(f"Pressure column '{args.pressure_column}' not found. Available columns: {header}")
    cropped = _crop(columns, args.time_start, args.time_end)
    time = cropped["time"]
    raw = cropped[args.pressure_column]
    smooth = _moving_average(raw, args.smooth_window, args.smooth_mode)
    peaks = _find_peaks(smooth, args.peak_distance, args.peak_prominence)
    if peaks.size < args.input_peaks + args.output_peaks + 1:
        raise ValueError(
            f"Only found {peaks.size} peaks; need at least {args.input_peaks + args.output_peaks + 1}. "
            "Try lowering --peak-distance or --peak-prominence."
        )

    peak_time = time[peaks]
    peak_height = smooth[peaks]
    rows = _build_samples(peak_time, peak_height, args.input_peaks, args.output_peaks)
    split = _split_labels(len(rows), args.train_ratio, args.val_ratio)
    for row, label in zip(rows, split):
        row["split"] = str(label)

    input_cols = [f"in_h_{i}" for i in range(args.input_peaks)] + [f"in_dt_{i}" for i in range(args.input_peaks)]
    target_cols = [f"target_h_{i}" for i in range(args.output_peaks)] + [f"target_dt_{i}" for i in range(args.output_peaks)]
    meta_cols = ["split", "input_start_peak", "input_end_peak", "target_start_peak", "target_end_peak", "target_start_time", "target_end_time"]
    sample_csv = output_dir / f"{csv_path.stem}_{args.pressure_column}_peaks_ip{args.input_peaks}_op{args.output_peaks}.csv"
    _write_csv(sample_csv, rows, meta_cols + input_cols + target_cols)

    peak_rows = [
        {"peak_id": int(i), "source_idx": int(idx), "time": float(time[idx]), "height": float(smooth[idx]), "raw_height": float(raw[idx])}
        for i, idx in enumerate(peaks)
    ]
    peak_csv = output_dir / f"{csv_path.stem}_{args.pressure_column}_peaks.csv"
    _write_csv(peak_csv, peak_rows, ["peak_id", "source_idx", "time", "height", "raw_height"])

    config = {
        "source_csv": str(csv_path.resolve()),
        "pressure_column": args.pressure_column,
        "time_start": float(args.time_start),
        "time_end": float(args.time_end),
        "smooth_window": int(args.smooth_window),
        "smooth_mode": args.smooth_mode,
        "peak_distance": int(args.peak_distance),
        "peak_prominence": float(args.peak_prominence),
        "input_peaks": int(args.input_peaks),
        "output_peaks": int(args.output_peaks),
        "num_cropped_points": int(time.size),
        "num_peaks": int(peaks.size),
        "num_samples": int(len(rows)),
        "train_samples": int(np.sum(split == "train")),
        "val_samples": int(np.sum(split == "val")),
        "test_samples": int(np.sum(split == "test")),
        "sample_csv": str(sample_csv.resolve()),
        "peak_csv": str(peak_csv.resolve()),
        "input_columns": input_cols,
        "target_columns": target_cols,
    }
    config_path = output_dir / "peak_prepare_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = output_dir / "peak_prepare_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# pressure 峰值序列数据准备报告",
                "",
                f"- 输入文件: `{csv_path}`",
                f"- 压力列: `{args.pressure_column}`",
                f"- 时间段: `{args.time_start}` ~ `{args.time_end}` s",
                f"- 平滑窗口/模式: `{args.smooth_window}` / `{args.smooth_mode}`",
                f"- 峰最小间隔: `{args.peak_distance}` 点",
                f"- 峰 prominence 阈值: `{args.peak_prominence}`",
                f"- 峰数量: `{peaks.size}`",
                f"- 样本数量: `{len(rows)}`",
                f"- 训练/验证/测试: `{config['train_samples']}` / `{config['val_samples']}` / `{config['test_samples']}`",
                f"- 样本 CSV: `{sample_csv}`",
                f"- 峰列表 CSV: `{peak_csv}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    plot_error = None
    plot_path = output_dir / "pressure_peaks.png"
    try:
        _plot_peaks(time, raw, smooth, peaks, plot_path, args.max_plot_points)
    except ModuleNotFoundError as exc:
        plot_error = str(exc)

    print(f"Peak samples CSV: {sample_csv}")
    print(f"Peak list CSV: {peak_csv}")
    print(f"Config JSON: {config_path}")
    print(f"Report MD: {report_path}")
    if plot_error is None:
        print(f"Peak plot: {plot_path}")
    else:
        print(f"Peak plot skipped: {plot_error}")
    print(f"Peaks: {peaks.size}, samples: {len(rows)}")


if __name__ == "__main__":
    main()
