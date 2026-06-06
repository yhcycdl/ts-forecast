#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.qperiod_enhance import (
    band_features,
    clean_signal,
    envelope,
    estimate_dominant_period,
    event_skeleton,
    local_frequency,
    moving_average,
    phase_sin_cos,
)


MODULES = {
    "main_residual",
    "envelope_freq",
    "event_skeleton",
    "band_decomp",
    "predictability",
}

TYPE_IDS = {
    "stable_single_freq": 0,
    "noisy_single_freq": 1,
    "am_fm_modulated": 2,
    "spike_event": 3,
    "multi_freq": 4,
    "weak_periodic": 5,
}


def _parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _infer_fs(df: pd.DataFrame, time_col: str | None, fs_col: str | None, sample_rate: float | None) -> float:
    if sample_rate is not None and sample_rate > 0:
        return float(sample_rate)
    if fs_col and fs_col in df.columns:
        fs_values = pd.to_numeric(df[fs_col], errors="coerce").to_numpy(dtype=np.float64)
        fs_values = fs_values[np.isfinite(fs_values) & (fs_values > 0)]
        if fs_values.size:
            return float(np.median(fs_values))
    if time_col and time_col in df.columns:
        time = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=np.float64)
        diff = np.diff(time)
        diff = diff[np.isfinite(diff) & (diff > 0)]
        if diff.size:
            return float(1.0 / np.median(diff))
    raise ValueError("Cannot infer sample rate. Pass --sample-rate or provide --time-col/--fs-col.")


def _load_profile(profile_csv: str | None, signal_col: str) -> dict[str, dict]:
    if not profile_csv:
        return {}
    profile = pd.read_csv(profile_csv)
    if profile.empty or "segment_id" not in profile.columns:
        return {}
    if "signal_col" in profile.columns:
        matched = profile[profile["signal_col"].astype(str) == str(signal_col)]
        if not matched.empty:
            profile = matched
    mapping: dict[str, dict] = {}
    for _, row in profile.iterrows():
        mapping[str(row["segment_id"])] = row.to_dict()
    return mapping


def _profile_value(profile: dict | None, key: str, default):
    if not profile or key not in profile:
        return default
    value = profile[key]
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    return value


def _process_chunk(
    out: pd.DataFrame,
    idx,
    raw: np.ndarray,
    fs: float,
    profile: dict | None,
    args: argparse.Namespace,
    modules: set[str],
) -> dict:
    raw = clean_signal(raw)
    period = float(args.period_samples) if args.period_samples and args.period_samples > 0 else 0.0
    if period <= 0:
        period = float(_profile_value(profile, "dominant_period_samples", 0.0))
    if period <= 0:
        period = estimate_dominant_period(raw, fs)
    if period <= 0:
        period = max(8.0, fs)

    smooth_window = int(args.smooth_window) if args.smooth_window and args.smooth_window > 0 else 0
    if smooth_window <= 0:
        smooth_window = int(round(float(_profile_value(profile, "recommended_smooth_window", 0.0))))
    if smooth_window <= 0:
        smooth_window = max(3, int(round(period / 10.0)))

    main_input = None
    main_target = None
    if "main_residual" in modules:
        main_input = moving_average(raw, smooth_window, mode=args.input_smooth_mode)
        main_target = moving_average(raw, smooth_window, mode=args.target_smooth_mode)
        out.loc[idx, "qp_main_input"] = main_input
        out.loc[idx, "qp_main_target"] = main_target
        out.loc[idx, "qp_residual"] = raw - main_input
        out.loc[idx, "qp_abs_residual"] = np.abs(raw - main_input)

    base = main_input if main_input is not None else raw
    freq_window = max(3, int(round(period / 4.0)))
    if "envelope_freq" in modules:
        env = envelope(base, smooth_window=freq_window)
        local_freq = local_frequency(base, fs=fs, smooth_window=freq_window)
        phase_sin, phase_cos = phase_sin_cos(base)
        f_dom = fs / period if period > 0 else 0.0
        out.loc[idx, "qp_envelope"] = env
        out.loc[idx, "qp_local_freq"] = local_freq
        out.loc[idx, "qp_local_freq_ratio"] = local_freq / max(f_dom, 1e-12)
        out.loc[idx, "qp_phase_sin"] = phase_sin
        out.loc[idx, "qp_phase_cos"] = phase_cos

    if "event_skeleton" in modules:
        events = event_skeleton(
            raw,
            period_samples=period,
            prominence_z=float(args.event_prominence_z),
            distance_frac=float(args.event_distance_frac),
        )
        out.loc[idx, "qp_event_mask"] = events["mask"]
        out.loc[idx, "qp_event_prominence"] = events["prominence"]
        out.loc[idx, "qp_event_width"] = events["width"]
        out.loc[idx, "qp_event_proximity"] = events["proximity"]
        out.loc[idx, "qp_event_weight"] = 1.0 + float(args.event_weight_scale) * events["proximity"]

    if "band_decomp" in modules:
        bands = band_features(
            raw,
            fs=fs,
            period_samples=period,
            band_count=int(args.band_count),
            rms_window=max(3, int(round(period / 8.0))),
        )
        for name, values in bands.items():
            out.loc[idx, f"qp_{name}"] = values

    if "predictability" in modules:
        signal_type = str(_profile_value(profile, "signal_type", "unknown"))
        score = float(_profile_value(profile, "predictability_score", 0.0))
        out.loc[idx, "qp_predictability_score"] = score
        out.loc[idx, "qp_signal_type_id"] = TYPE_IDS.get(signal_type, -1)
        out.loc[idx, "qp_weak_periodic_flag"] = 1.0 if signal_type == "weak_periodic" else 0.0

    return {
        "fs": float(fs),
        "period_samples": float(period),
        "smooth_window": int(smooth_window),
        "rows": int(raw.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment prepared quasi-periodic CSVs with reusable feature-aware modules.")
    parser.add_argument("--csv", required=True, help="Input prepared CSV.")
    parser.add_argument("--output", required=True, help="Output augmented CSV.")
    parser.add_argument("--raw-col", default="raw")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--fs-col", default="fs")
    parser.add_argument("--sample-rate", type=float, default=None)
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--profile-csv", default=None, help="Optional profile_by_segment.csv.")
    parser.add_argument("--profile-signal-col", default="target_smooth")
    parser.add_argument("--modules", default="all", help="Comma-separated modules or 'all'.")
    parser.add_argument("--period-samples", type=float, default=None)
    parser.add_argument("--smooth-window", type=int, default=None)
    parser.add_argument("--input-smooth-mode", choices=["causal", "centered"], default="causal")
    parser.add_argument("--target-smooth-mode", choices=["causal", "centered"], default="centered")
    parser.add_argument("--event-prominence-z", type=float, default=0.75)
    parser.add_argument("--event-distance-frac", type=float, default=0.35)
    parser.add_argument("--event-weight-scale", type=float, default=2.0)
    parser.add_argument("--band-count", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    modules = MODULES if str(args.modules).lower() == "all" else set(_parse_list(args.modules))
    unknown = sorted(modules - MODULES)
    if unknown:
        raise ValueError(f"Unknown modules: {unknown}. Available: {sorted(MODULES)}")

    input_path = Path(args.csv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, nrows=None if args.max_rows <= 0 else args.max_rows, low_memory=False)
    if args.raw_col not in df.columns:
        raise ValueError(f"raw_col '{args.raw_col}' not found. Available columns: {list(df.columns)}")

    out = df.copy()
    profile_map = _load_profile(args.profile_csv, args.profile_signal_col)

    group_cols = []
    if args.segment_col and args.segment_col in df.columns:
        group_cols.append(args.segment_col)
    if args.split_col and args.split_col in df.columns:
        group_cols.append(args.split_col)

    if group_cols:
        groups = df.groupby(group_cols, sort=False)
    else:
        groups = [(("all",), df)]

    chunks = []
    for key, chunk in groups:
        idx = chunk.index
        segment_id = None
        if args.segment_col and args.segment_col in chunk.columns:
            segment_id = str(chunk[args.segment_col].iloc[0])
        profile = profile_map.get(segment_id, None)
        fs = _infer_fs(chunk, args.time_col, args.fs_col, args.sample_rate)
        info = _process_chunk(
            out,
            idx,
            raw=chunk[args.raw_col].to_numpy(dtype=np.float64),
            fs=fs,
            profile=profile,
            args=args,
            modules=modules,
        )
        info["group"] = str(key)
        info["segment_id"] = segment_id
        chunks.append(info)

    out.to_csv(output_path, index=False)
    metadata = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "modules": sorted(modules),
        "raw_col": args.raw_col,
        "profile_csv": args.profile_csv,
        "chunks": chunks,
        "recommended_input_sets": {
            "stable_single_freq": "qp_main_input",
            "noisy_single_freq": "qp_main_input,qp_residual,qp_abs_residual",
            "am_fm_modulated": "qp_main_input,qp_envelope,qp_local_freq_ratio,qp_phase_sin,qp_phase_cos",
            "spike_event": "qp_main_input,qp_event_proximity,qp_event_prominence,qp_event_weight",
            "multi_freq": "qp_main_input,qp_band0_rms,qp_band1_rms,qp_band2_rms",
            "weak_periodic": "qp_main_input,qp_predictability_score,qp_weak_periodic_flag",
        },
        "recommended_target": "qp_main_target",
    }
    metadata_path = output_path.with_suffix(".augment.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved augmented CSV: {output_path}")
    print(f"Saved metadata: {metadata_path}")
    print("Recommended target column: qp_main_target")


if __name__ == "__main__":
    main()
