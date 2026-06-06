#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


SIGNAL_TYPE_MODULES = {
    "stable_single_freq": "cycle_adaptive_window",
    "noisy_single_freq": "main_residual_decomposition",
    "am_fm_modulated": "envelope_frequency_conditioning",
    "spike_event": "event_skeleton_constraint",
    "multi_freq": "frequency_band_decomposition",
    "weak_periodic": "predictability_rejection_or_target_switch",
}


TYPE_WINDOWS = {
    "stable_single_freq": (10, 4),
    "noisy_single_freq": (10, 3),
    "am_fm_modulated": (12, 3),
    "spike_event": (10, 3),
    "multi_freq": (12, 2),
    "weak_periodic": (6, 1),
}


SIGNAL_TYPE_TASKS = {
    "stable_single_freq": "main_waveform_forecast",
    "noisy_single_freq": "smooth_main_waveform_forecast",
    "am_fm_modulated": "conditioned_main_waveform_forecast",
    "spike_event": "event_timing_and_main_waveform_forecast",
    "multi_freq": "band_energy_or_multibranch_waveform_forecast",
    "weak_periodic": "short_horizon_or_rejection",
}


def _parse_cols(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _safe_name(value) -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value))
    text = text.strip("_")
    return text or "segment"


def _infer_sample_rate(df: pd.DataFrame, time_col: str | None, sample_rate: float | None) -> float:
    if sample_rate is not None and sample_rate > 0:
        return float(sample_rate)
    if time_col and time_col in df.columns:
        t = df[time_col].to_numpy(dtype=np.float64)
        diff = np.diff(t)
        diff = diff[np.isfinite(diff) & (diff > 0)]
        if diff.size:
            return float(1.0 / np.median(diff))
    raise ValueError("Cannot infer sample rate. Pass --sample-rate or provide a valid --time-col.")


def _infer_segment_sample_rate(
    df: pd.DataFrame,
    time_col: str | None,
    fs_col: str | None,
    sample_rate: float | None,
) -> float:
    if sample_rate is not None and sample_rate > 0:
        return float(sample_rate)
    if fs_col and fs_col in df.columns:
        fs_values = pd.to_numeric(df[fs_col], errors="coerce").to_numpy(dtype=np.float64)
        fs_values = fs_values[np.isfinite(fs_values) & (fs_values > 0)]
        if fs_values.size:
            return float(np.median(fs_values))
    return _infer_sample_rate(df, time_col, sample_rate)


def _clean_series(values: np.ndarray, max_samples: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError("Signal has no finite values.")
    fill = float(np.nanmedian(x[finite]))
    x = np.where(finite, x, fill)
    if max_samples > 0 and x.size > max_samples:
        step = int(math.ceil(x.size / max_samples))
        x = x[::step]
    return x


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1:
        return x.copy()
    cumsum = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(x.size)
    left_span = (window - 1) // 2
    right_span = window - left_span
    left = np.maximum(0, idx - left_span)
    right = np.minimum(x.size, idx + right_span)
    count = right - left
    return (cumsum[right] - cumsum[left]) / count


def _spectral_features(x: np.ndarray, fs: float) -> dict:
    y = x - np.mean(x)
    if y.size < 8 or np.std(y) <= 1e-12:
        return {
            "dominant_frequency_hz": 0.0,
            "dominant_period_samples": 0.0,
            "dominant_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "multi_peak_count": 0,
        }

    nperseg = min(y.size, 8192)
    freqs, power = signal.welch(y, fs=fs, nperseg=nperseg)
    if power.size <= 1:
        return {
            "dominant_frequency_hz": 0.0,
            "dominant_period_samples": 0.0,
            "dominant_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "multi_peak_count": 0,
        }

    freqs = freqs[1:]
    power = np.maximum(power[1:], 0.0)
    total = float(np.sum(power))
    if total <= 1e-20:
        return {
            "dominant_frequency_hz": 0.0,
            "dominant_period_samples": 0.0,
            "dominant_energy_ratio": 0.0,
            "spectral_entropy": 1.0,
            "multi_peak_count": 0,
        }

    idx = int(np.argmax(power))
    f_dom = float(freqs[idx])
    p = power / total
    entropy = float(-np.sum(p * np.log(p + 1e-12)) / np.log(max(2, p.size)))
    peak_idx, props = signal.find_peaks(power, height=float(np.max(power)) * 0.2)
    return {
        "dominant_frequency_hz": f_dom,
        "dominant_period_samples": float(fs / f_dom) if f_dom > 0 else 0.0,
        "dominant_energy_ratio": float(power[idx] / total),
        "spectral_entropy": entropy,
        "multi_peak_count": int(peak_idx.size),
    }


def _acf_features(x: np.ndarray, period_samples: float) -> dict:
    y = x - np.mean(x)
    std = float(np.std(y))
    if y.size < 8 or std <= 1e-12:
        return {"acf_peak": 0.0, "acf_peak_lag": 0}

    n = y.size
    fft_len = 1 << int(math.ceil(math.log2(max(2, n * 2 - 1))))
    acf = np.fft.irfft(np.fft.rfft(y, fft_len) * np.conj(np.fft.rfft(y, fft_len)))[:n]
    acf = acf / max(acf[0], 1e-20)

    max_lag = min(n - 1, int(max(period_samples * 3, 20))) if period_samples > 1 else min(n - 1, 2000)
    if max_lag <= 2:
        return {"acf_peak": 0.0, "acf_peak_lag": 0}
    search = acf[1:max_lag + 1]
    peaks, _ = signal.find_peaks(search)
    if peaks.size == 0:
        lag = int(np.argmax(search) + 1)
    else:
        lag = int(peaks[np.argmax(search[peaks])] + 1)
    return {"acf_peak": float(acf[lag]), "acf_peak_lag": lag}


def _peak_features(x: np.ndarray, period_samples: float) -> dict:
    y = x - np.median(x)
    scale = float(np.std(y))
    if y.size < 8 or scale <= 1e-12:
        return {
            "peak_count": 0,
            "peak_interval_cv": float("nan"),
            "peak_prominence_ratio": 0.0,
        }

    distance = max(1, int(period_samples * 0.35)) if period_samples > 1 else 1
    peaks, props = signal.find_peaks(y, distance=distance, prominence=scale * 0.5)
    if peaks.size < 2:
        return {
            "peak_count": int(peaks.size),
            "peak_interval_cv": float("nan"),
            "peak_prominence_ratio": 0.0,
        }
    intervals = np.diff(peaks).astype(np.float64)
    prominences = np.asarray(props.get("prominences", []), dtype=np.float64)
    prom_ratio = float(np.median(prominences) / scale) if prominences.size else 0.0
    return {
        "peak_count": int(peaks.size),
        "peak_interval_cv": float(np.std(intervals) / max(np.mean(intervals), 1e-12)),
        "peak_prominence_ratio": prom_ratio,
    }


def _envelope_cv(x: np.ndarray) -> float:
    y = x - np.mean(x)
    if y.size < 8 or np.std(y) <= 1e-12:
        return 0.0
    env = np.abs(signal.hilbert(y))
    return float(np.std(env) / max(np.mean(env), 1e-12))


def _residual_energy_ratio(x: np.ndarray, period_samples: float) -> float:
    if x.size < 8:
        return 0.0
    window = int(max(3, min(x.size // 2, round(period_samples / 6)))) if period_samples > 6 else 3
    smooth = _moving_average(x, window)
    var_total = float(np.var(x))
    if var_total <= 1e-20:
        return 0.0
    return float(np.var(x - smooth) / var_total)


def _classify(row: dict) -> str:
    energy = row["dominant_energy_ratio"]
    entropy = row["spectral_entropy"]
    acf = row["acf_peak"]
    interval_cv = row["peak_interval_cv"]
    envelope = row["envelope_cv"]
    residual = row["residual_energy_ratio"]
    prom = row["peak_prominence_ratio"]
    multi = row["multi_peak_count"]

    interval_cv_num = 999.0 if not np.isfinite(interval_cv) else float(interval_cv)

    if energy < 0.08 and acf < 0.20:
        return "weak_periodic"
    if prom >= 2.5 and interval_cv_num <= 0.50:
        return "spike_event"
    if multi >= 3 and entropy >= 0.45:
        return "multi_freq"
    if interval_cv_num >= 0.20 or envelope >= 0.45:
        return "am_fm_modulated"
    if residual >= 0.35 or entropy >= 0.55:
        return "noisy_single_freq"
    return "stable_single_freq"


def _predictability_score(row: dict) -> float:
    energy = float(row["dominant_energy_ratio"])
    acf = max(0.0, float(row["acf_peak"]))
    entropy_term = 1.0 - float(row["spectral_entropy"])
    residual_term = 1.0 - min(1.0, float(row["residual_energy_ratio"]))
    score = 0.35 * energy + 0.35 * acf + 0.15 * entropy_term + 0.15 * residual_term
    return float(np.clip(score, 0.0, 1.0))


def _recommend(signal_type: str, period_samples: float, fs: float) -> dict:
    in_cycles, out_cycles = TYPE_WINDOWS.get(signal_type, (10, 2))
    period = max(1, int(round(period_samples))) if period_samples > 0 else 1
    if signal_type == "spike_event":
        smooth = max(1, int(round(period / 20)))
    elif signal_type == "noisy_single_freq":
        smooth = max(3, int(round(period / 6)))
    elif signal_type == "am_fm_modulated":
        smooth = max(3, int(round(period / 8)))
    else:
        smooth = max(1, int(round(period / 10)))
    seq_len = int(max(16, period * in_cycles))
    pred_len = int(max(1, period * out_cycles))
    safe_fs = max(float(fs), 1e-12)
    return {
        "recommended_module": SIGNAL_TYPE_MODULES[signal_type],
        "recommended_task": SIGNAL_TYPE_TASKS[signal_type],
        "input_cycles": int(in_cycles),
        "output_cycles": int(out_cycles),
        "recommended_seq_len": seq_len,
        "recommended_pred_len": pred_len,
        "recommended_smooth_window": int(smooth),
        "recommended_seq_sec": float(seq_len / safe_fs),
        "recommended_pred_sec": float(pred_len / safe_fs),
        "recommended_smooth_sec": float(smooth / safe_fs),
    }


def analyze_segment(values: np.ndarray, fs: float, max_samples: int) -> dict:
    x = _clean_series(values, max_samples=max_samples)
    spec = _spectral_features(x, fs=fs)
    acf = _acf_features(x, spec["dominant_period_samples"])
    peaks = _peak_features(x, spec["dominant_period_samples"])
    row = {
        "n_samples_analyzed": int(x.size),
        "sample_rate_hz": float(fs),
        "duration_sec_analyzed": float(x.size / max(fs, 1e-12)),
        **spec,
        **acf,
        **peaks,
        "envelope_cv": _envelope_cv(x),
        "residual_energy_ratio": _residual_energy_ratio(x, spec["dominant_period_samples"]),
    }
    row["dominant_period_sec"] = float(row["dominant_period_samples"] / max(fs, 1e-12))
    row["cycles_analyzed"] = float(row["n_samples_analyzed"] / max(row["dominant_period_samples"], 1e-12))
    signal_type = _classify(row)
    row["signal_type"] = signal_type
    row["predictability_score"] = _predictability_score(row)
    row.update(_recommend(signal_type, row["dominant_period_samples"], fs=fs))
    return row


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Quasi-periodic Signal Profile",
        "",
        "## Summary by signal type",
        "",
    ]
    if not summary.empty:
        type_counts = summary["signal_type"].value_counts()
        for name, count in type_counts.items():
            module = SIGNAL_TYPE_MODULES.get(name, "")
            lines.append(f"- `{name}`: {int(count)} segment(s), module `{module}`")
    lines.extend(["", "## Recommended experiment defaults", ""])
    cols = [
        "segment_id",
        "signal_col",
        "signal_type",
        "dominant_frequency_hz",
        "dominant_period_samples",
        "dominant_period_sec",
        "predictability_score",
        "recommended_seq_len",
        "recommended_pred_len",
        "recommended_smooth_window",
        "recommended_pred_sec",
        "recommended_module",
    ]
    if not summary.empty:
        table = summary[cols].copy()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in table.iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.6g}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    (output_dir / "profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze quasi-periodic signal features and recommend forecasting settings.")
    parser.add_argument("--csv", required=True, help="Input prepared or raw CSV.")
    parser.add_argument("--signal-cols", required=True, help="Comma-separated signal columns to analyze.")
    parser.add_argument("--time-col", default="time", help="Time column used to infer sample rate when --sample-rate is omitted.")
    parser.add_argument("--fs-col", default="fs", help="Optional per-row sample-rate column in prepared CSVs.")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sample rate in Hz.")
    parser.add_argument("--segment-col", default=None, help="Optional segment id column.")
    parser.add_argument("--split-col", default=None, help="Optional split column.")
    parser.add_argument("--split-values", default=None, help="Optional comma-separated split labels to include, e.g. train,test.")
    parser.add_argument("--max-rows", type=int, default=10_000_000)
    parser.add_argument("--max-samples-per-segment", type=int, default=200_000)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, nrows=None if args.max_rows <= 0 else args.max_rows, low_memory=False)
    signal_cols = _parse_cols(args.signal_cols)
    missing = [col for col in signal_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing signal columns: {missing}. Available columns: {list(df.columns)}")

    if args.split_col and args.split_values:
        if args.split_col not in df.columns:
            raise ValueError(f"split_col '{args.split_col}' not found.")
        keep = {item.lower() for item in _parse_cols(args.split_values)}
        df = df[df[args.split_col].astype(str).str.lower().isin(keep)].copy()

    global_fs = None
    if args.sample_rate is not None:
        global_fs = float(args.sample_rate)
    elif args.fs_col and args.fs_col in df.columns:
        fs_values = pd.to_numeric(df[args.fs_col], errors="coerce").to_numpy(dtype=np.float64)
        fs_values = fs_values[np.isfinite(fs_values) & (fs_values > 0)]
        if fs_values.size:
            global_fs = float(np.median(fs_values))
    if global_fs is None:
        global_fs = _infer_sample_rate(df, args.time_col, args.sample_rate)

    if args.segment_col:
        if args.segment_col not in df.columns:
            raise ValueError(f"segment_col '{args.segment_col}' not found.")
        grouped = list(df.groupby(args.segment_col, sort=False))
    else:
        grouped = [("all", df)]

    rows = []
    for segment_id, seg_df in grouped:
        fs = _infer_segment_sample_rate(seg_df, args.time_col, args.fs_col, args.sample_rate)
        for col in signal_cols:
            if seg_df.empty:
                continue
            row = analyze_segment(
                seg_df[col].to_numpy(dtype=np.float64),
                fs=fs,
                max_samples=int(args.max_samples_per_segment),
            )
            row["segment_id"] = str(segment_id)
            row["signal_col"] = col
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No profile rows were produced.")

    by_segment_path = output_dir / "profile_by_segment.csv"
    summary.to_csv(by_segment_path, index=False)

    type_summary = (
        summary.groupby(["signal_type", "recommended_module"], dropna=False)
        .agg(
            segments=("segment_id", "count"),
            median_frequency_hz=("dominant_frequency_hz", "median"),
            median_period_samples=("dominant_period_samples", "median"),
            median_energy_ratio=("dominant_energy_ratio", "median"),
            median_spectral_entropy=("spectral_entropy", "median"),
            median_acf_peak=("acf_peak", "median"),
            median_residual_energy_ratio=("residual_energy_ratio", "median"),
            median_predictability_score=("predictability_score", "median"),
            median_recommended_seq_len=("recommended_seq_len", "median"),
            median_recommended_pred_len=("recommended_pred_len", "median"),
        )
        .reset_index()
    )
    type_summary.to_csv(output_dir / "profile_summary.csv", index=False)
    _write_report(summary, output_dir)

    metadata = {
        "csv": str(csv_path),
        "signal_cols": signal_cols,
        "sample_rate_hz": global_fs,
        "fs_col": args.fs_col,
        "segment_col": args.segment_col,
        "split_col": args.split_col,
        "split_values": _parse_cols(args.split_values),
        "rows_analyzed": int(len(df)),
        "outputs": {
            "profile_by_segment": str(by_segment_path),
            "profile_summary": str(output_dir / "profile_summary.csv"),
            "profile_report": str(output_dir / "profile_report.md"),
        },
    }
    (output_dir / "profile_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved profile rows: {by_segment_path}")
    print(f"Saved summary: {output_dir / 'profile_summary.csv'}")
    print(f"Saved report: {output_dir / 'profile_report.md'}")


if __name__ == "__main__":
    main()
