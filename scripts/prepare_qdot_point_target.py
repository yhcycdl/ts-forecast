#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute quantile on an empty array.")
    return float(np.quantile(values, q))


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sign(values) * np.log1p(np.abs(values))


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


def _split_bounds(length: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    train_end = int(length * train_ratio)
    val_end = int(length * (train_ratio + val_ratio))
    return train_end, val_end


def _repair_short_zero_gaps(
    values: np.ndarray,
    max_gap: int,
    neighbor_min_abs: float,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    repaired = np.asarray(values, dtype=np.float64).copy()
    segments: list[dict[str, float]] = []
    if max_gap <= 0:
        return repaired, segments

    n = repaired.size
    i = 0
    while i < n:
        if repaired[i] != 0.0:
            i += 1
            continue

        start = i
        while i < n and repaired[i] == 0.0:
            i += 1
        end = i
        gap_len = end - start

        if start == 0 or end >= n or gap_len > max_gap:
            continue

        left = repaired[start - 1]
        right = repaired[end]
        if abs(left) < neighbor_min_abs or abs(right) < neighbor_min_abs:
            continue

        fill = np.linspace(left, right, gap_len + 2)[1:-1]
        repaired[start:end] = fill
        segments.append(
            {
                "start_idx": int(start),
                "end_idx": int(end - 1),
                "gap_len": int(gap_len),
                "left_value": float(left),
                "right_value": float(right),
            }
        )

    return repaired, segments


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a model-ready qdot point target from a cleaned pressure/qdot CSV.")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV path.")
    parser.add_argument("--qdot-column", type=str, default="qdot00", help="Target qdot column.")
    parser.add_argument(
        "--qdot-aggregate",
        type=str,
        default="none",
        choices=["none", "mean16", "sum16", "active_mean16", "active_sum16"],
        help="Optional aggregated qdot target built from all qdot* columns.",
    )
    parser.add_argument(
        "--aggregate-active-quantile",
        type=float,
        default=0.2,
        help="For active_mean16/active_sum16, only channels above this train-split nonzero-abs quantile are aggregated.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save prepared CSV and report files.")
    parser.add_argument("--time-start", type=float, default=None, help="Optional crop start time in seconds.")
    parser.add_argument("--time-end", type=float, default=None, help="Optional crop end time in seconds.")
    parser.add_argument(
        "--drop-zero-rows",
        type=int,
        default=0,
        choices=[0, 1],
        help="1 drops rows whose resolved qdot target is treated as zero; this is an ablation and breaks strict time continuity.",
    )
    parser.add_argument(
        "--drop-zero-eps",
        type=float,
        default=0.0,
        help="Rows with |resolved qdot target| <= this threshold are dropped when --drop-zero-rows=1.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--clip-low-quantile", type=float, default=0.001, help="Lower clipping quantile fitted on train split.")
    parser.add_argument("--clip-high-quantile", type=float, default=0.999, help="Upper clipping quantile fitted on train split.")
    parser.add_argument("--repair-zero-gap-max-len", type=int, default=2, help="Repair zero gaps up to this length by linear interpolation.")
    parser.add_argument("--repair-threshold-quantile", type=float, default=0.2, help="Use train nonzero abs-value quantile as neighbor threshold for zero-gap repair.")
    parser.add_argument("--activation-quantile", type=float, default=0.1, help="Activation threshold quantile on repaired train nonzero abs values.")
    parser.add_argument(
        "--amp-weight-high-quantile",
        type=float,
        default=0.95,
        help="Train-split quantile used to normalize the qdot amplitude weight column.",
    )
    parser.add_argument(
        "--amp-weight-gamma",
        type=float,
        default=1.0,
        help="Exponent applied to the normalized qdot amplitude weight column; >1 focuses more on extreme peaks/valleys.",
    )
    parser.add_argument("--smooth-window", type=int, default=512, help="Causal moving-average window applied on qdot log1p target; 1 disables smoothing.")
    parser.add_argument(
        "--input-smooth-window",
        type=int,
        default=0,
        help="Optional causal moving-average window for pressure input columns; 0 disables pressure smoothing columns.",
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


def _filter_rows_by_mask(columns: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    return {name: np.asarray(values)[mask] for name, values in columns.items()}


def _resolve_qdot_series(
    columns: dict[str, np.ndarray],
    qdot_column: str,
    aggregate_mode: str,
    train_end: int,
    aggregate_active_quantile: float,
) -> tuple[str, np.ndarray]:
    aggregate_mode = str(aggregate_mode).lower()
    if aggregate_mode == "none":
        if qdot_column not in columns:
            raise ValueError(f"Target qdot column '{qdot_column}' not found in columns.")
        return qdot_column, np.asarray(columns[qdot_column], dtype=np.float64)

    qdot_cols = sorted(name for name in columns.keys() if name.startswith("qdot"))
    if not qdot_cols:
        raise ValueError("No qdot* columns found for aggregation.")
    stacked = np.stack([np.asarray(columns[name], dtype=np.float64) for name in qdot_cols], axis=1)
    if aggregate_mode == "mean16":
        return "qdot_mean16", np.mean(stacked, axis=1)
    if aggregate_mode == "sum16":
        return "qdot_sum16", np.sum(stacked, axis=1)
    if aggregate_mode in {"active_mean16", "active_sum16"}:
        train_stack = stacked[:train_end]
        train_nonzero_abs = np.abs(train_stack[np.abs(train_stack) > 0.0])
        if train_nonzero_abs.size == 0:
            raise ValueError("Train split contains no nonzero qdot values for active aggregation.")
        eps = _quantile(train_nonzero_abs, float(aggregate_active_quantile))
        active_mask = np.abs(stacked) >= eps
        active_count = np.sum(active_mask, axis=1)
        masked_sum = np.sum(np.where(active_mask, stacked, 0.0), axis=1)
        if aggregate_mode == "active_mean16":
            series = np.divide(
                masked_sum,
                np.maximum(active_count, 1),
                out=np.zeros_like(masked_sum, dtype=np.float64),
                where=active_count > 0,
            )
            return "qdot_mean_active16", series
        return "qdot_sum_active16", masked_sum
    raise ValueError(f"Unknown qdot aggregation mode: {aggregate_mode}")


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


def _write_segment_csv(output_path: Path, rows: list[dict[str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["start_idx", "end_idx", "gap_len", "left_value", "right_value"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_summary_plot(
    columns: dict[str, np.ndarray],
    qdot_col: str,
    smooth_col: str | None,
    output_path: Path,
    max_points: int,
) -> None:
    import matplotlib.pyplot as plt

    time = np.asarray(columns["time"])
    raw = np.asarray(columns[qdot_col])
    repaired = np.asarray(columns[f"{qdot_col}_repaired"])
    log1p_values = np.asarray(columns[f"{qdot_col}_log1p"])
    smoothed = np.asarray(columns[smooth_col]) if smooth_col is not None else None
    target = np.asarray(columns[f"{qdot_col}_target"])
    active = np.asarray(columns[f"{qdot_col}_active"], dtype=np.float64)

    if max_points > 0 and len(time) > max_points:
        step = max(1, len(time) // max_points)
        sl = slice(None, None, step)
        time = time[sl]
        raw = raw[sl]
        repaired = repaired[sl]
        log1p_values = log1p_values[sl]
        if smoothed is not None:
            smoothed = smoothed[sl]
        target = target[sl]
        active = active[sl]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    axes[0].plot(time, raw, linewidth=0.8, label=f"{qdot_col} raw")
    axes[0].plot(time, repaired, linewidth=0.8, alpha=0.8, label=f"{qdot_col} repaired")
    axes[0].set_ylabel("raw")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(time, log1p_values, linewidth=0.8, alpha=0.9, label=f"{qdot_col} log1p")
    if smoothed is not None:
        axes[1].plot(time, smoothed, linewidth=1.2, alpha=0.9, label=smooth_col)
    axes[1].set_ylabel("log/smoothed")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(time, target, linewidth=0.8, label=f"{qdot_col} target")
    axes[2].plot(time, active, linewidth=0.8, alpha=0.7, label=f"{qdot_col} active")
    axes[2].set_xlabel("time / s")
    axes[2].set_ylabel("target")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def _make_hist_plot(
    columns: dict[str, np.ndarray],
    qdot_col: str,
    smooth_col: str | None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    n_panels = 4 if smooth_col is not None else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    axes[0].hist(columns[f"{qdot_col}_repaired"], bins=200, color="#4c78a8")
    axes[0].set_title("repaired")
    axes[1].hist(columns[f"{qdot_col}_log1p"], bins=200, color="#f58518")
    axes[1].set_title("signed log1p")
    offset = 0
    if smooth_col is not None:
        axes[2].hist(columns[smooth_col], bins=200, color="#e45756")
        axes[2].set_title(smooth_col)
        offset = 1
    axes[2 + offset].hist(columns[f"{qdot_col}_target"], bins=200, color="#54a24b")
    axes[2 + offset].set_title("standardized target")
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

    original_total_len = len(next(iter(original_columns.values())))
    original_time_start = float(original_columns["time"][0])
    original_time_end = float(original_columns["time"][-1])

    original_columns, kept_after_crop = _crop_time_range(
        original_columns,
        time_start=args.time_start,
        time_end=args.time_end,
    )

    total_len = len(next(iter(original_columns.values())))
    provisional_train_end, _ = _split_bounds(total_len, args.train_ratio, args.val_ratio)
    qdot_cols = sorted(name for name in original_columns.keys() if name.startswith("qdot"))
    resolved_qdot_name, raw = _resolve_qdot_series(
        original_columns,
        qdot_column=args.qdot_column,
        aggregate_mode=args.qdot_aggregate,
        train_end=provisional_train_end,
        aggregate_active_quantile=args.aggregate_active_quantile,
    )
    removed_zero_rows = 0
    if int(args.drop_zero_rows) == 1:
        keep_mask = np.abs(raw) > float(args.drop_zero_eps)
        removed_zero_rows = int(np.sum(~keep_mask))
        if removed_zero_rows >= total_len:
            raise ValueError("Dropping zero rows would remove every sample. Lower --drop-zero-eps or disable --drop-zero-rows.")
        if removed_zero_rows > 0:
            original_columns = _filter_rows_by_mask(original_columns, keep_mask)

    total_len = len(next(iter(original_columns.values())))
    train_end, val_end = _split_bounds(total_len, args.train_ratio, args.val_ratio)
    if train_end <= 0 or val_end <= train_end or val_end >= total_len:
        raise ValueError("Invalid train/val ratios for the available series length.")

    split = np.full(total_len, "test", dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"

    qdot_cols = sorted(name for name in original_columns.keys() if name.startswith("qdot"))
    resolved_qdot_name, raw = _resolve_qdot_series(
        original_columns,
        qdot_column=args.qdot_column,
        aggregate_mode=args.qdot_aggregate,
        train_end=train_end,
        aggregate_active_quantile=args.aggregate_active_quantile,
    )
    if resolved_qdot_name not in original_columns:
        original_columns[resolved_qdot_name] = raw
        if resolved_qdot_name not in original_header:
            original_header.append(resolved_qdot_name)
    aggregate_eps = None
    if str(args.qdot_aggregate).lower() in {"active_mean16", "active_sum16"}:
        train_stack = np.stack([np.asarray(original_columns[name], dtype=np.float64) for name in qdot_cols], axis=1)[:train_end]
        train_nonzero_abs_all = np.abs(train_stack[np.abs(train_stack) > 0.0])
        aggregate_eps = _quantile(train_nonzero_abs_all, float(args.aggregate_active_quantile))
    train_raw = raw[:train_end]
    train_nonzero_abs = np.abs(train_raw[np.abs(train_raw) > 0.0])
    if train_nonzero_abs.size == 0:
        raise ValueError(f"Train split of {resolved_qdot_name} contains no nonzero samples.")

    neighbor_min_abs = _quantile(train_nonzero_abs, args.repair_threshold_quantile)
    repaired, repaired_segments = _repair_short_zero_gaps(
        raw,
        max_gap=int(args.repair_zero_gap_max_len),
        neighbor_min_abs=neighbor_min_abs,
    )

    repaired_train = repaired[:train_end]
    clip_low = _quantile(repaired_train, args.clip_low_quantile)
    clip_high = _quantile(repaired_train, args.clip_high_quantile)
    clipped = np.clip(repaired, clip_low, clip_high)
    log1p_values = _signed_log1p(clipped)
    smooth_col = None
    smoothed_values = None
    if int(args.smooth_window) > 1:
        smooth_col = f"{resolved_qdot_name}_log1p_ma{int(args.smooth_window)}"
        smoothed_values = _causal_moving_average(log1p_values, int(args.smooth_window))

    log_train = log1p_values[:train_end]
    target_mean = float(np.mean(log_train))
    target_std = float(np.std(log_train))
    if target_std <= 1e-12:
        target_std = 1.0
    target = (log1p_values - target_mean) / target_std

    repaired_train_nonzero_abs = np.abs(repaired_train[np.abs(repaired_train) > 0.0])
    activation_abs_threshold_raw = _quantile(repaired_train_nonzero_abs, args.activation_quantile)
    active = (np.abs(repaired) >= activation_abs_threshold_raw).astype(np.int64)

    weight_source = smoothed_values if smoothed_values is not None else log1p_values
    weight_train = np.asarray(weight_source[:train_end], dtype=np.float64)
    amp_center = float(np.median(weight_train))
    amp = np.abs(np.asarray(weight_source, dtype=np.float64) - amp_center)
    train_amp = np.abs(weight_train - amp_center)
    amp_scale = _quantile(train_amp, args.amp_weight_high_quantile)
    if amp_scale <= 1e-12:
        amp_scale = 1.0
    amp_weight = np.clip(amp / amp_scale, 0.0, 1.0)
    amp_gamma = max(float(args.amp_weight_gamma), 1e-12)
    if abs(amp_gamma - 1.0) > 1e-12:
        amp_weight = np.power(amp_weight, amp_gamma)

    extra_columns = [
        (f"{resolved_qdot_name}_repaired", repaired),
        (f"{resolved_qdot_name}_clipped", clipped),
        (f"{resolved_qdot_name}_log1p", log1p_values),
        (f"{resolved_qdot_name}_target", target),
        (f"{resolved_qdot_name}_active", active.astype(np.float64)),
        (f"{resolved_qdot_name}_amp_weight", amp_weight.astype(np.float64)),
    ]
    if smooth_col is not None and smoothed_values is not None:
        extra_columns.insert(3, (smooth_col, smoothed_values))

    input_smooth_cols: list[str] = []
    if int(args.input_smooth_window) > 1:
        pressure_cols = sorted(name for name in original_header if name.startswith("p"))
        for col in pressure_cols:
            ma_col = f"{col}_ma{int(args.input_smooth_window)}"
            ma_values = _causal_moving_average(np.asarray(original_columns[col], dtype=np.float64), int(args.input_smooth_window))
            extra_columns.append((ma_col, ma_values))
            input_smooth_cols.append(ma_col)

    prepared_columns = dict(original_columns)
    for name, values in extra_columns:
        prepared_columns[name] = values

    output_csv = output_dir / f"{csv_path.stem}_{args.qdot_column}_prepared.csv"
    _write_prepared_csv(output_csv, original_header, original_columns, extra_columns, split)

    repair_segments_path = output_dir / f"{csv_path.stem}_{args.qdot_column}_repaired_segments.csv"
    _write_segment_csv(repair_segments_path, repaired_segments)

    config = {
        "source_csv": str(csv_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        "qdot_column": args.qdot_column,
        "resolved_qdot_name": resolved_qdot_name,
        "qdot_aggregate": str(args.qdot_aggregate).lower(),
        "aggregate_active_quantile": float(args.aggregate_active_quantile),
        "aggregate_active_threshold": aggregate_eps,
        "original_total_rows": int(original_total_len),
        "total_rows": int(total_len),
        "original_time_start": original_time_start,
        "original_time_end": original_time_end,
        "crop_time_start": None if args.time_start is None else float(args.time_start),
        "crop_time_end": None if args.time_end is None else float(args.time_end),
        "cropped_time_start": float(original_columns["time"][0]),
        "cropped_time_end": float(original_columns["time"][-1]),
        "kept_after_crop": int(kept_after_crop),
        "drop_zero_rows": int(args.drop_zero_rows),
        "drop_zero_eps": float(args.drop_zero_eps),
        "removed_zero_rows": int(removed_zero_rows),
        "train_rows": int(train_end),
        "val_rows": int(val_end - train_end),
        "test_rows": int(total_len - val_end),
        "zero_ratio_raw": float(np.mean(raw == 0.0)),
        "repair_zero_gap_max_len": int(args.repair_zero_gap_max_len),
        "repair_threshold_quantile": float(args.repair_threshold_quantile),
        "repair_neighbor_min_abs": float(neighbor_min_abs),
        "repaired_segments": int(len(repaired_segments)),
        "clip_low_quantile": float(args.clip_low_quantile),
        "clip_high_quantile": float(args.clip_high_quantile),
        "clip_low_value": float(clip_low),
        "clip_high_value": float(clip_high),
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "activation_quantile": float(args.activation_quantile),
        "activation_abs_threshold_raw": float(activation_abs_threshold_raw),
        "active_ratio_total": float(np.mean(active)),
        "active_ratio_train": float(np.mean(active[:train_end])),
        "active_ratio_val": float(np.mean(active[train_end:val_end])),
        "active_ratio_test": float(np.mean(active[val_end:])),
        "amp_weight_column": f"{resolved_qdot_name}_amp_weight",
        "amp_weight_source_column": smooth_col if smooth_col is not None else f"{resolved_qdot_name}_log1p",
        "amp_weight_center": float(amp_center),
        "amp_weight_high_quantile": float(args.amp_weight_high_quantile),
        "amp_weight_scale": float(amp_scale),
        "amp_weight_gamma": float(amp_gamma),
        "smooth_window": int(args.smooth_window),
        "smooth_column": smooth_col,
        "input_smooth_window": int(args.input_smooth_window),
        "input_smooth_columns": input_smooth_cols,
    }
    config_path = output_dir / f"{csv_path.stem}_{args.qdot_column}_prepare_config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    report_lines = [
        "# qdot 单点标签预处理报告",
        "",
        f"- 输入文件: `{csv_path}`",
        f"- 目标列: `{resolved_qdot_name}`",
        f"- 聚合方式: `{str(args.qdot_aggregate).lower()}`",
        f"- 活跃聚合分位数: `{float(args.aggregate_active_quantile):.6g}`",
        f"- 活跃聚合阈值: `{aggregate_eps if aggregate_eps is not None else 'N/A'}`",
        f"- 输出文件: `{output_csv}`",
        "",
        "## 时间裁剪",
        f"- 原始时间范围: `{original_time_start:.9g}` ~ `{original_time_end:.9g}` s",
        f"- 裁剪时间范围: `{config['cropped_time_start']:.9g}` ~ `{config['cropped_time_end']:.9g}` s",
        f"- 裁剪后样本数: `{total_len}`",
        f"- 是否删除 0 行: `{int(args.drop_zero_rows)}`",
        f"- 删除阈值 eps: `{float(args.drop_zero_eps):.6g}`",
        f"- 删除的 0 行数: `{removed_zero_rows}`",
        "",
        "## 数据划分",
        f"- 训练集: `{train_end}` 行",
        f"- 验证集: `{val_end - train_end}` 行",
        f"- 测试集: `{total_len - val_end}` 行",
        "",
        "## 原始目标统计",
        f"- 原始零值比例: `{config['zero_ratio_raw']:.6f}`",
        f"- 零缺口修补阈值(训练集非零绝对值分位数): `{neighbor_min_abs:.6g}`",
        f"- 被修补的短零缺口数量: `{len(repaired_segments)}`",
        "",
        "## 变换参数",
        f"- 下截断分位数: `{args.clip_low_quantile}` -> `{clip_low:.6g}`",
        f"- 上截断分位数: `{args.clip_high_quantile}` -> `{clip_high:.6g}`",
        f"- 标准化均值: `{target_mean:.6g}`",
        f"- 标准化标准差: `{target_std:.6g}`",
        f"- 因果平滑窗口: `{int(args.smooth_window)}`",
        f"- 平滑列名: `{smooth_col}`",
        f"- 输入压力平滑窗口: `{int(args.input_smooth_window)}`",
        f"- 输入压力平滑列数: `{len(input_smooth_cols)}`",
        "",
        "## 激活标签",
        f"- 激活阈值分位数: `{args.activation_quantile}`",
        f"- 原始幅值阈值: `{activation_abs_threshold_raw:.6g}`",
        f"- 总激活比例: `{config['active_ratio_total']:.6f}`",
        f"- 训练激活比例: `{config['active_ratio_train']:.6f}`",
        f"- 验证激活比例: `{config['active_ratio_val']:.6f}`",
        f"- 测试激活比例: `{config['active_ratio_test']:.6f}`",
        "",
        "## 幅值权重",
        f"- 权重列名: `{config['amp_weight_column']}`",
        f"- 权重来源列: `{config['amp_weight_source_column']}`",
        f"- 中心值(训练集 median): `{config['amp_weight_center']:.6g}`",
        f"- 归一化分位数: `{config['amp_weight_high_quantile']}`",
        f"- 归一化尺度: `{config['amp_weight_scale']:.6g}`",
        f"- gamma: `{config['amp_weight_gamma']:.6g}`",
    ]
    report_path = output_dir / f"{csv_path.stem}_{args.qdot_column}_prepare_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    summary_plot_path = output_dir / f"{csv_path.stem}_{args.qdot_column}_summary.png"
    hist_plot_path = output_dir / f"{csv_path.stem}_{args.qdot_column}_hist.png"
    plot_error = None
    try:
        _make_summary_plot(prepared_columns, resolved_qdot_name, smooth_col, summary_plot_path, max_points=args.max_plot_points)
        _make_hist_plot(prepared_columns, resolved_qdot_name, smooth_col, hist_plot_path)
    except ModuleNotFoundError as exc:
        plot_error = str(exc)

    print(f"Prepared CSV: {output_csv}")
    print(f"Config JSON: {config_path}")
    print(f"Report MD: {report_path}")
    print(f"Repair segments CSV: {repair_segments_path}")
    if plot_error is None:
        print(f"Summary plot: {summary_plot_path}")
        print(f"Histogram plot: {hist_plot_path}")
    else:
        print(f"Plots skipped: {plot_error}")


if __name__ == "__main__":
    main()
