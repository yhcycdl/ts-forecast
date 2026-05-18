#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _causal_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1:
        return values.copy()
    n = values.size
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    ends = np.arange(1, n + 1, dtype=np.int64)
    starts = np.maximum(0, ends - int(window))
    counts = ends - starts
    return (cumsum[ends] - cumsum[starts]) / counts


def _centered_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if window <= 1:
        return values.copy()
    window = int(window)
    half_left = window // 2
    half_right = window - half_left - 1
    cumsum = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    centers = np.arange(values.size, dtype=np.int64)
    starts = np.maximum(0, centers - half_left)
    ends = np.minimum(values.size, centers + half_right + 1)
    counts = ends - starts
    return (cumsum[ends] - cumsum[starts]) / counts


def _moving_average(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    if mode == "causal":
        return _causal_moving_average(values, window)
    if mode == "centered":
        return _centered_moving_average(values, window)
    raise ValueError(f"Unsupported moving-average mode: {mode}")


def _split_bounds(length: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    return train_end, val_end


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a model-ready pressure trend target from a pressure/qdot CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--pressure-column", type=str, default="p00", help="Target pressure column.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save prepared CSV and report files.")
    parser.add_argument("--time-start", type=float, default=None, help="Optional crop start time in seconds.")
    parser.add_argument("--time-end", type=float, default=None, help="Optional crop end time in seconds.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--smooth-window", type=int, default=512, help="Moving-average window for the pressure target.")
    parser.add_argument(
        "--smooth-mode",
        choices=["causal", "centered"],
        default="causal",
        help="Target smoothing mode. centered is better for offline waveform-shape labels because it avoids phase lag.",
    )
    parser.add_argument(
        "--input-smooth-window",
        type=int,
        default=0,
        help="Optional moving-average window for all pressure input channels; 0 disables input smoothing columns.",
    )
    parser.add_argument(
        "--input-smooth-mode",
        choices=["causal", "centered"],
        default="causal",
        help="Smoothing mode for optional pressure input smoothing columns.",
    )
    parser.add_argument("--max-plot-points", type=int, default=20000, help="Maximum plotted points for long curves.")
    return parser


def _load_csv_columns(csv_path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        buffers: dict[str, list] = {name: [] for name in fieldnames}
        for row in reader:
            for name in fieldnames:
                buffers[name].append(float(row[name]))
    columns = {name: np.asarray(values, dtype=np.float64) for name, values in buffers.items()}
    return fieldnames, columns


def _crop_time_range(
    columns: dict[str, np.ndarray],
    time_start: float | None,
    time_end: float | None,
) -> tuple[dict[str, np.ndarray], int]:
    time = np.asarray(columns["time"], dtype=np.float64)
    mask = np.ones(time.shape[0], dtype=bool)
    if time_start is not None:
        mask &= time >= float(time_start)
    if time_end is not None:
        mask &= time <= float(time_end)
    kept = int(np.sum(mask))
    if kept <= 0:
        raise ValueError(
            f"No samples remain after time cropping: time_start={time_start}, time_end={time_end}."
        )
    cropped = {name: np.asarray(values)[mask] for name, values in columns.items()}
    return cropped, kept


def _write_prepared_csv(
    output_path: Path,
    original_header: list[str],
    original_columns: dict[str, np.ndarray],
    extra_columns: list[tuple[str, np.ndarray]],
    split: np.ndarray,
) -> None:
    header = list(original_header) + ["split"] + [name for name, _ in extra_columns]
    length = len(next(iter(original_columns.values())))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(length):
            row = [original_columns[name][i] for name in original_header]
            row.append(str(split[i]))
            row.extend(values[i] for _, values in extra_columns)
            writer.writerow(row)


def _make_summary_plot(
    columns: dict[str, np.ndarray],
    pressure_col: str,
    smooth_col: str,
    output_path: Path,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    time = np.asarray(columns["time"])
    raw = np.asarray(columns[pressure_col])
    smoothed = np.asarray(columns[smooth_col])

    if max_points > 0 and len(time) > max_points:
        step = max(1, len(time) // max_points)
        sl = slice(None, None, step)
        time = time[sl]
        raw = raw[sl]
        smoothed = smoothed[sl]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(time, raw, linewidth=0.8, label=f"{pressure_col} raw")
    axes[0].plot(time, smoothed, linewidth=1.2, alpha=0.9, label=smooth_col)
    axes[0].set_ylabel("pressure")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(time, smoothed, linewidth=1.2, label=smooth_col)
    axes[1].set_xlabel("time / s")
    axes[1].set_ylabel("smoothed")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _make_hist_plot(
    columns: dict[str, np.ndarray],
    pressure_col: str,
    smooth_col: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(columns[pressure_col], bins=200, color="#4c78a8")
    axes[0].set_title(f"{pressure_col} raw")
    axes[1].hist(columns[smooth_col], bins=200, color="#f58518")
    axes[1].set_title(smooth_col)
    for ax in axes:
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _build_parser().parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    original_header, original_columns = _load_csv_columns(csv_path)
    if "time" not in original_columns:
        raise ValueError("Input CSV must contain a time column.")
    if args.pressure_column not in original_columns:
        raise ValueError(f"Target pressure column '{args.pressure_column}' not found in {csv_path}.")

    original_total_len = len(next(iter(original_columns.values())))
    original_time_start = float(original_columns["time"][0])
    original_time_end = float(original_columns["time"][-1])

    original_columns, kept_after_crop = _crop_time_range(
        original_columns,
        time_start=args.time_start,
        time_end=args.time_end,
    )

    total_len = len(next(iter(original_columns.values())))
    train_end, val_end = _split_bounds(total_len, args.train_ratio, args.val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= total_len:
        raise ValueError("Invalid train/val ratios for the available series length.")

    split = np.full(total_len, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"

    raw = np.asarray(original_columns[args.pressure_column], dtype=np.float64)
    smooth_prefix = "cma" if args.smooth_mode == "centered" else "ma"
    smooth_col = f"{args.pressure_column}_{smooth_prefix}{int(args.smooth_window)}"
    smoothed = _moving_average(raw, int(args.smooth_window), args.smooth_mode)

    extra_columns = [(smooth_col, smoothed)]
    prepared_columns = dict(original_columns)
    prepared_columns[smooth_col] = smoothed

    input_smooth_cols: list[str] = []
    extra_column_names = {name for name, _ in extra_columns}
    if int(args.input_smooth_window) > 1:
        pressure_cols = sorted(name for name in original_header if name.startswith("p"))
        for col in pressure_cols:
            input_smooth_prefix = "cma" if args.input_smooth_mode == "centered" else "ma"
            ma_col = f"{col}_{input_smooth_prefix}{int(args.input_smooth_window)}"
            ma_values = _moving_average(np.asarray(original_columns[col], dtype=np.float64), int(args.input_smooth_window), args.input_smooth_mode)
            if ma_col not in extra_column_names:
                extra_columns.append((ma_col, ma_values))
                extra_column_names.add(ma_col)
            prepared_columns[ma_col] = ma_values
            input_smooth_cols.append(ma_col)

    output_csv = output_dir / f"{csv_path.stem}_{args.pressure_column}_prepared.csv"
    _write_prepared_csv(output_csv, original_header, original_columns, extra_columns, split)

    config = {
        "source_csv": str(csv_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "pressure_column": args.pressure_column,
        "original_total_rows": int(original_total_len),
        "total_rows": int(total_len),
        "original_time_start": original_time_start,
        "original_time_end": original_time_end,
        "crop_time_start": None if args.time_start is None else float(args.time_start),
        "crop_time_end": None if args.time_end is None else float(args.time_end),
        "cropped_time_start": float(original_columns["time"][0]),
        "cropped_time_end": float(original_columns["time"][-1]),
        "kept_after_crop": int(kept_after_crop),
        "train_rows": int(train_end),
        "val_rows": int(val_end - train_end),
        "test_rows": int(total_len - val_end),
        "smooth_window": int(args.smooth_window),
        "smooth_mode": args.smooth_mode,
        "smooth_column": smooth_col,
        "input_smooth_window": int(args.input_smooth_window),
        "input_smooth_mode": args.input_smooth_mode,
        "input_smooth_columns": input_smooth_cols,
        "raw_mean": float(np.mean(raw)),
        "raw_std": float(np.std(raw)),
        "smoothed_mean": float(np.mean(smoothed)),
        "smoothed_std": float(np.std(smoothed)),
    }
    config_path = output_dir / f"{csv_path.stem}_{args.pressure_column}_prepare_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# pressure 趋势目标预处理报告",
        "",
        f"- 输入文件: `{csv_path}`",
        f"- 目标列: `{args.pressure_column}`",
        f"- 输出文件: `{output_csv}`",
        "",
        "## 时间裁剪",
        f"- 原始时间范围: `{original_time_start:.9g}` ~ `{original_time_end:.9g}` s",
        f"- 裁剪时间范围: `{config['cropped_time_start']:.9g}` ~ `{config['cropped_time_end']:.9g}` s",
        f"- 裁剪后样本数: `{total_len}`",
        "",
        "## 数据划分",
        f"- 训练集: `{train_end}` 行",
        f"- 验证集: `{val_end - train_end}` 行",
        f"- 测试集: `{total_len - val_end}` 行",
        "",
        "## 目标定义",
        f"- 平滑窗口: `{int(args.smooth_window)}`",
        f"- 平滑模式: `{args.smooth_mode}`",
        f"- 平滑列名: `{smooth_col}`",
        f"- 输入平滑窗口: `{int(args.input_smooth_window)}`",
        f"- 输入平滑模式: `{args.input_smooth_mode}`",
        f"- 输入平滑列数: `{len(input_smooth_cols)}`",
        f"- 原始均值/标准差: `{config['raw_mean']:.6g}` / `{config['raw_std']:.6g}`",
        f"- 平滑均值/标准差: `{config['smoothed_mean']:.6g}` / `{config['smoothed_std']:.6g}`",
    ]
    report_path = output_dir / f"{csv_path.stem}_{args.pressure_column}_prepare_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_plot_path = output_dir / f"{csv_path.stem}_{args.pressure_column}_summary.png"
    hist_plot_path = output_dir / f"{csv_path.stem}_{args.pressure_column}_hist.png"
    plot_error = None
    try:
        _make_summary_plot(prepared_columns, args.pressure_column, smooth_col, summary_plot_path, max_points=args.max_plot_points)
        _make_hist_plot(prepared_columns, args.pressure_column, smooth_col, hist_plot_path)
    except ModuleNotFoundError as exc:
        plot_error = str(exc)

    print(f"Prepared CSV: {output_csv}")
    print(f"Config JSON: {config_path}")
    print(f"Report MD: {report_path}")
    if plot_error is None:
        print(f"Summary plot: {summary_plot_path}")
        print(f"Histogram plot: {hist_plot_path}")
    else:
        print(f"Plots skipped: {plot_error}")


if __name__ == "__main__":
    main()
