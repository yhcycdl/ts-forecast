#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


SIGNAL_TYPES = [
    "stable_single_freq",
    "noisy_single_freq",
    "am_fm_modulated",
    "spike_event",
    "multi_freq",
    "weak_periodic",
]


def _parse_types(value: str) -> list[str]:
    if str(value).lower() == "all":
        return SIGNAL_TYPES.copy()
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(items) - set(SIGNAL_TYPES))
    if unknown:
        raise ValueError(f"Unknown synthetic types: {unknown}. Available: {SIGNAL_TYPES}")
    return items


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_") or "signal"


def _moving_average(values: Any, window: int, mode: str):
    import numpy as np

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    window = int(max(1, window))
    if window <= 1 or x.size <= 1:
        return x.copy()
    cumsum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(x.size, dtype=np.int64)
    if mode == "causal":
        ends = idx + 1
        starts = np.maximum(0, ends - window)
    elif mode == "centered":
        left = (window - 1) // 2
        right = window - left
        starts = np.maximum(0, idx - left)
        ends = np.minimum(x.size, idx + right)
    else:
        raise ValueError(f"Unsupported smoothing mode: {mode}")
    counts = np.maximum(1, ends - starts)
    return (cumsum[ends] - cumsum[starts]) / counts


def _moving_average_by_split(values: Any, window: int, mode: str, split: Any):
    import numpy as np

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    split = np.asarray(split, dtype=object).reshape(-1)
    if x.size != split.size:
        raise ValueError("values and split must have the same length.")
    out = np.empty_like(x, dtype=np.float64)
    start = 0
    while start < x.size:
        label = split[start]
        end = start + 1
        while end < x.size and split[end] == label:
            end += 1
        out[start:end] = _moving_average(x[start:end], window, mode)
        start = end
    return out


def _split_for_record(record_index: int, records_per_type: int, length: int, args: argparse.Namespace):
    import numpy as np

    if args.split_policy == "chronological":
        train_end = int(length * args.train_ratio)
        val_end = int(length * (args.train_ratio + args.val_ratio))
        if train_end <= 0 or val_end <= train_end or val_end >= length:
            raise ValueError("Invalid chronological split ratios.")
        split = np.full(length, "test", dtype=object)
        split[:train_end] = "train"
        split[train_end:val_end] = "val"
        return split

    train_end = int(records_per_type * args.train_ratio)
    val_end = int(records_per_type * (args.train_ratio + args.val_ratio))
    if record_index < train_end:
        label = "train"
    elif record_index < val_end:
        label = "val"
    else:
        label = "test"
    return np.full(length, label, dtype=object)


def _phase_from_frequency(freq_hz: Any, fs: float, phase0: float):
    import numpy as np

    return phase0 + 2.0 * math.pi * np.cumsum(freq_hz) / float(fs)


def _gaussian_pulses(t: Any, centers: Any, amps: Any, widths: Any):
    import numpy as np

    out = np.zeros_like(t, dtype=np.float64)
    for center, amp, width in zip(centers, amps, widths):
        out += float(amp) * np.exp(-0.5 * ((t - float(center)) / max(float(width), 1e-12)) ** 2)
    return out


def _generate_signal(signal_type: str, fs: float, period_sec: float, cycles: int, rng: Any) -> tuple[Any, dict]:
    import numpy as np

    period_sec = float(period_sec)
    duration = float(cycles) * period_sec
    n = max(16, int(round(duration * fs)))
    t = np.arange(n, dtype=np.float64) / float(fs)
    f0 = 1.0 / max(period_sec, 1e-12)
    phase0 = float(rng.uniform(0.0, 2.0 * math.pi))

    if signal_type == "stable_single_freq":
        freq = f0 * float(rng.uniform(0.96, 1.04))
        phase = 2.0 * math.pi * freq * t + phase0
        raw = np.sin(phase) + 0.18 * np.sin(2.0 * phase + 0.4) + 0.03 * rng.standard_normal(n)

    elif signal_type == "noisy_single_freq":
        freq = f0 * float(rng.uniform(0.94, 1.06))
        phase = 2.0 * math.pi * freq * t + phase0
        residual = 0.42 * rng.standard_normal(n) + 0.10 * np.sin(2.0 * math.pi * 8.0 * f0 * t)
        raw = np.sin(phase) + 0.12 * np.sin(2.0 * phase) + residual

    elif signal_type == "am_fm_modulated":
        mod1 = 2.0 * math.pi * t / max(duration * 0.75, period_sec)
        mod2 = 2.0 * math.pi * t / max(duration * 0.55, period_sec)
        amp = 1.0 + 0.42 * np.sin(mod1 + rng.uniform(0.0, 2.0 * math.pi))
        inst_freq = f0 * (1.0 + 0.28 * np.sin(mod2 + rng.uniform(0.0, 2.0 * math.pi)))
        inst_freq = np.maximum(inst_freq, f0 * 0.35)
        phase = _phase_from_frequency(inst_freq, fs, phase0)
        raw = amp * np.sin(phase) + 0.05 * rng.standard_normal(n)

    elif signal_type == "spike_event":
        freq = f0 * float(rng.uniform(0.96, 1.04))
        phase = 2.0 * math.pi * freq * t + phase0
        centers = []
        current = float(rng.uniform(0.2, 0.8) * period_sec)
        while current < duration:
            centers.append(current)
            current += period_sec * float(rng.normal(1.0, 0.06))
        centers_arr = np.asarray(centers, dtype=np.float64)
        amps = rng.normal(1.4, 0.18, size=centers_arr.size)
        widths = np.full(centers_arr.size, max(period_sec * 0.035, 1.0 / fs))
        raw = 0.22 * np.sin(phase) + _gaussian_pulses(t, centers_arr, amps, widths) + 0.04 * rng.standard_normal(n)

    elif signal_type == "multi_freq":
        switch = n // 2
        freq1 = f0 * float(rng.uniform(0.92, 1.08))
        freq2 = f0 * float(rng.uniform(1.65, 2.35))
        phase1 = 2.0 * math.pi * freq1 * t + phase0
        phase2 = 2.0 * math.pi * freq2 * t + rng.uniform(0.0, 2.0 * math.pi)
        gate = 1.0 / (1.0 + np.exp(-(np.arange(n) - switch) / max(8.0, 0.04 * n)))
        raw = (1.0 - gate) * np.sin(phase1) + gate * 0.95 * np.sin(phase2)
        raw += 0.35 * np.sin(phase1 + phase2 * 0.25) + 0.05 * rng.standard_normal(n)

    elif signal_type == "weak_periodic":
        components = np.zeros(n, dtype=np.float64)
        for _ in range(8):
            freq = f0 * float(rng.uniform(0.35, 5.5))
            components += rng.uniform(0.05, 0.18) * np.sin(2.0 * math.pi * freq * t + rng.uniform(0.0, 2.0 * math.pi))
        random_walk = np.cumsum(0.025 * rng.standard_normal(n))
        raw = 0.18 * np.sin(2.0 * math.pi * f0 * t + phase0) + components + random_walk + 0.55 * rng.standard_normal(n)

    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    raw = raw - float(np.mean(raw))
    std = float(np.std(raw))
    if std > 1e-12:
        raw = raw / std
    meta = {
        "expected_period_samples": float(fs / f0),
        "expected_frequency_hz": float(f0),
        "duration_sec": float(duration),
        "rows": int(n),
    }
    return raw.astype(np.float64), meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled synthetic quasi-periodic datasets.")
    parser.add_argument("--output", required=True, help="Output long-format CSV path.")
    parser.add_argument("--types", default="all", help="Comma-separated signal types or 'all'.")
    parser.add_argument("--records-per-type", type=int, default=8)
    parser.add_argument("--cycles-per-record", type=int, default=400)
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--period-sec", type=float, default=1.0)
    parser.add_argument("--input-smooth-sec", type=float, default=0.12)
    parser.add_argument("--input-smooth-mode", choices=["causal", "centered"], default="causal")
    parser.add_argument("--target-smooth-sec", type=float, default=0.12)
    parser.add_argument("--target-smooth-mode", choices=["causal", "centered"], default="centered")
    parser.add_argument("--split-policy", choices=["by_record", "chronological"], default="by_record")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    import numpy as np

    if args.records_per_type < 3:
        raise ValueError("--records-per-type should be at least 3 for train/val/test splits.")
    if args.sample_rate <= 0 or args.period_sec <= 0:
        raise ValueError("--sample-rate and --period-sec must be positive.")
    if not 0.0 < args.train_ratio < 1.0 or not 0.0 <= args.val_ratio < 1.0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("Require 0 < train_ratio, 0 <= val_ratio, and train_ratio + val_ratio < 1.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    signal_types = _parse_types(args.types)
    fs = float(args.sample_rate)
    input_window = max(1, int(round(float(args.input_smooth_sec) * fs)))
    target_window = max(1, int(round(float(args.target_smooth_sec) * fs)))

    fieldnames = [
        "time",
        "raw",
        "input_smooth",
        "target_smooth",
        "split",
        "segment_id",
        "record_id",
        "signal_name",
        "dataset",
        "fs",
        "synthetic_type",
        "expected_period_samples",
        "expected_frequency_hz",
    ]
    segments = []
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for signal_type in signal_types:
            for rec_idx in range(int(args.records_per_type)):
                raw, meta = _generate_signal(signal_type, fs, float(args.period_sec), int(args.cycles_per_record), rng)
                split = _split_for_record(rec_idx, int(args.records_per_type), raw.size, args)
                input_smooth = _moving_average_by_split(raw, input_window, args.input_smooth_mode, split)
                target_smooth = _moving_average_by_split(raw, target_window, args.target_smooth_mode, split)
                record_id = f"{_safe_name(signal_type)}_{rec_idx:03d}"
                segment_id = f"synthetic_{record_id}"
                time = np.arange(raw.size, dtype=np.float64) / fs
                for i in range(raw.size):
                    writer.writerow(
                        {
                            "time": float(time[i]),
                            "raw": float(raw[i]),
                            "input_smooth": float(input_smooth[i]),
                            "target_smooth": float(target_smooth[i]),
                            "split": str(split[i]),
                            "segment_id": segment_id,
                            "record_id": record_id,
                            "signal_name": signal_type,
                            "dataset": "synthetic_qp",
                            "fs": fs,
                            "synthetic_type": signal_type,
                            "expected_period_samples": float(meta["expected_period_samples"]),
                            "expected_frequency_hz": float(meta["expected_frequency_hz"]),
                        }
                    )
                rows_written += int(raw.size)
                segments.append(
                    {
                        "segment_id": segment_id,
                        "synthetic_type": signal_type,
                        "record_id": record_id,
                        "rows": int(raw.size),
                        "duration_sec": float(meta["duration_sec"]),
                        "fs": fs,
                        "expected_period_samples": float(meta["expected_period_samples"]),
                        "train_rows": int(np.sum(split == "train")),
                        "val_rows": int(np.sum(split == "val")),
                        "test_rows": int(np.sum(split == "test")),
                    }
                )

    config = {
        "dataset": "synthetic_qp",
        "output_csv": str(output_path.resolve()),
        "types": signal_types,
        "records_per_type": int(args.records_per_type),
        "cycles_per_record": int(args.cycles_per_record),
        "sample_rate": fs,
        "period_sec": float(args.period_sec),
        "input_smooth_sec": float(args.input_smooth_sec),
        "input_smooth_mode": args.input_smooth_mode,
        "target_smooth_sec": float(args.target_smooth_sec),
        "target_smooth_mode": args.target_smooth_mode,
        "smooth_isolated_by_split": True,
        "split_policy": args.split_policy,
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "rows_written": rows_written,
        "num_segments": len(segments),
        "segments": segments,
    }
    config_path = output_path.with_suffix(".config.json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated synthetic quasi-periodic CSV: {output_path}")
    print(f"Config JSON: {config_path}")
    print(f"Segments: {len(segments)}")
    print(f"Rows written: {rows_written}")
    print("Input column: input_smooth")
    print("Target column: target_smooth")
    print("Raw plot column: raw")


if __name__ == "__main__":
    main()
