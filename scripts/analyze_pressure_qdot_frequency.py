#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.data.pressure_qdot_loader import load_pressure_qdot_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze dominant frequencies and coupling for pressure/Qdot data.")
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="./data/pressure_qdot_csv_final_1us/manifest.json",
        help="Manifest JSON for final pressure/Qdot CSV files.",
    )
    parser.add_argument(
        "--csv-root",
        type=str,
        default=None,
        help="Optional local directory that contains the CSV files. Useful when manifest stores another machine's absolute paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/frequency_analysis_final_1us",
        help="Directory to save analysis artifacts.",
    )
    parser.add_argument(
        "--pressure-columns",
        type=str,
        default="p00,p05,p10,p15",
        help="Comma-separated pressure channels to analyze.",
    )
    parser.add_argument(
        "--qdot-columns",
        type=str,
        default="qdot00,qdot05,qdot10,qdot15",
        help="Comma-separated qdot channels to analyze.",
    )
    parser.add_argument(
        "--exclude-conditions",
        type=str,
        default="BR_0p945",
        help="Comma-separated condition ids to skip. Default skips the known duplicate BR_0p945.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=262144,
        help="Maximum number of points to load per condition/channel for FFT analysis.",
    )
    parser.add_argument(
        "--downsample-step",
        type=int,
        default=1,
        help="Subsample every N points before analysis.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of strongest spectral peaks to save per channel.",
    )
    parser.add_argument(
        "--detrend-window",
        type=int,
        default=2001,
        help="Moving-average window for detrending before FFT. Must be >= 3.",
    )
    parser.add_argument(
        "--min-frequency-hz",
        type=float,
        default=100.0,
        help="Ignore peaks below this frequency when selecting dominant oscillation frequency.",
    )
    return parser


def _parse_csv_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_selected_columns(csv_path: Path, columns: list[str], max_points: int, downsample_step: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx_map = {name: header.index(name) for name in columns}
        time_idx = header.index("time")
        times: list[float] = []
        values: dict[str, list[float]] = {name: [] for name in columns}
        for row_idx, row in enumerate(reader):
            if row_idx % downsample_step != 0:
                continue
            times.append(float(row[time_idx]))
            for name, idx in idx_map.items():
                values[name].append(float(row[idx]))
            if len(times) >= max_points:
                break
    return np.asarray(times, dtype=np.float64), {name: np.asarray(v, dtype=np.float64) for name, v in values.items()}


def _resolve_csv_path(condition: dict, manifest_path: Path, csv_root: str | None) -> Path:
    raw = Path(condition["csv_path"])
    if raw.exists():
        return raw

    if csv_root is not None:
        candidate = Path(csv_root) / Path(condition["csv_filename"]).name
        if candidate.exists():
            return candidate

    manifest_dir = manifest_path.resolve().parent
    candidate = manifest_dir / Path(condition["csv_filename"]).name
    if candidate.exists():
        return candidate

    output_dir = condition.get("output_dir") or None
    if output_dir:
        candidate = Path(output_dir) / Path(condition["csv_filename"]).name
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot resolve csv_path for condition {condition['condition_id']}. "
        f"manifest csv_path={condition['csv_path']!r}, csv_filename={condition['csv_filename']!r}"
    )


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    window = int(max(3, window))
    if window % 2 == 0:
        window += 1
    if len(signal) < window:
        return np.full_like(signal, np.mean(signal))
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(signal, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _dominant_frequency(
    signal: np.ndarray,
    dt: float,
    top_k: int,
    detrend_window: int,
    min_frequency_hz: float,
) -> tuple[float, float, list[tuple[float, float]], float]:
    signal = np.asarray(signal, dtype=np.float64)
    trend = _moving_average(signal, detrend_window)
    signal_hp = signal - trend
    signal_hp = signal_hp - np.mean(signal_hp)
    n = len(signal)
    if n < 8:
        return 0.0, 0.0, [], 0.0

    window = np.hanning(n)
    spec = np.fft.rfft(signal_hp * window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, d=dt)
    if len(freqs) <= 1:
        return 0.0, 0.0, [], float(np.sqrt(np.mean(signal_hp ** 2)))

    power[0] = 0.0
    valid = freqs >= float(min_frequency_hz)
    if not np.any(valid):
        return 0.0, 0.0, [], float(np.sqrt(np.mean(signal_hp ** 2)))
    masked_power = power.copy()
    masked_power[~valid] = 0.0
    peak_idx = int(np.argmax(masked_power))
    dom_freq = float(freqs[peak_idx])
    dom_power = float(masked_power[peak_idx])

    top_indices = np.argsort(masked_power)[::-1]
    peaks: list[tuple[float, float]] = []
    for idx in top_indices[:top_k]:
        if masked_power[idx] <= 0:
            continue
        peaks.append((float(freqs[idx]), float(masked_power[idx])))
    return dom_freq, dom_power, peaks, float(np.sqrt(np.mean(signal_hp ** 2)))


def _corr_with_best_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> tuple[int, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - np.mean(x)
    y = y - np.mean(y)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return 0, 0.0
    x = x / sx
    y = y / sy

    best_lag = 0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            xs = x[: len(x) - lag]
            ys = y[lag:]
        else:
            xs = x[-lag:]
            ys = y[: len(y) + lag]
        if len(xs) < 64:
            continue
        corr = float(np.mean(xs * ys))
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    return best_lag, best_corr


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_report(
    path: Path,
    rows: list[dict],
    coupling_rows: list[dict],
    pressure_cols: list[str],
    qdot_cols: list[str],
    excluded: list[str],
    downsample_step: int,
) -> None:
    lines: list[str] = []
    lines.append("# 压力-释热率频率分析报告")
    lines.append("")
    lines.append(f"- 分析压力通道: `{', '.join(pressure_cols)}`")
    lines.append(f"- 分析释热率通道: `{', '.join(qdot_cols)}`")
    lines.append(f"- 排除工况: `{', '.join(excluded) if excluded else '无'}`")
    lines.append(f"- 下采样步长: `{downsample_step}`")
    lines.append("")
    lines.append("## 主频汇总")
    lines.append("")
    lines.append("| 工况编号 | 通道 | 类型 | 样本数 | dt(s) | 主频(Hz) | 主周期(ms) | 均值 | 标准差 | RMS |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| `{row['condition_id']}` | `{row['channel']}` | `{row['signal_type']}` | `{row['n_samples']}` | "
            f"`{float(row['dt_seconds']):.9g}` | `{float(row['dominant_frequency_hz']):.6g}` | `{float(row['dominant_period_ms']):.6g}` | "
            f"`{float(row['mean']):.6g}` | `{float(row['std']):.6g}` | `{float(row['rms']):.6g}` |"
        )
    lines.append("")
    lines.append("## 压力-释热率耦合")
    lines.append("")
    lines.append("| 工况编号 | 压力通道 | 释热率通道 | 零时滞相关 | 最佳时滞样本 | 最佳时滞(ms) | 最佳相关 |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in coupling_rows:
        lines.append(
            f"| `{row['condition_id']}` | `{row['pressure_channel']}` | `{row['qdot_channel']}` | "
            f"`{float(row['corr_zero_lag']):.6g}` | `{row['best_lag_samples']}` | "
            f"`{float(row['best_lag_ms']):.6g}` | `{float(row['best_lag_corr']):.6g}` |"
        )
    lines.append("")
    lines.append("## 说明")
    lines.append("- 主频基于窗口化 FFT 的最大谱峰。")
    lines.append("- 这里先做工况级时间尺度摸底，用于反推窗口参数，不作为最终论文频谱图。")
    lines.append("- `BR_0p945` 已默认排除，因为已确认与 `BR_0p315` 完全重复。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_json)
    manifest = load_pressure_qdot_manifest(manifest_path)
    pressure_cols = _parse_csv_list(args.pressure_columns)
    qdot_cols = _parse_csv_list(args.qdot_columns)
    excluded = set(_parse_csv_list(args.exclude_conditions))

    rows: list[dict] = []
    peak_rows: list[dict] = []
    coupling_rows: list[dict] = []

    for condition in manifest["conditions"]:
        condition_id = condition["condition_id"]
        if condition_id in excluded:
            continue
        csv_path = _resolve_csv_path(condition, manifest_path, args.csv_root)
        all_cols = pressure_cols + qdot_cols
        times, values = _load_selected_columns(
            csv_path=csv_path,
            columns=all_cols,
            max_points=args.max_points,
            downsample_step=args.downsample_step,
        )
        if len(times) < 8:
            continue
        dt = float(np.median(np.diff(times)))

        for channel in pressure_cols + qdot_cols:
            signal = values[channel]
            dom_freq, dom_power, peaks, hp_rms = _dominant_frequency(
                signal,
                dt,
                top_k=args.top_k,
                detrend_window=args.detrend_window,
                min_frequency_hz=args.min_frequency_hz,
            )
            mean_val = float(np.mean(signal))
            std_val = float(np.std(signal))
            rms_val = float(np.sqrt(np.mean(signal ** 2)))
            rows.append(
                {
                    "condition_id": condition_id,
                    "channel": channel,
                    "signal_type": "pressure" if channel.startswith("p") else "qdot",
                    "n_samples": len(signal),
                    "dt_seconds": dt,
                    "dominant_frequency_hz": dom_freq,
                    "dominant_period_ms": (1000.0 / dom_freq) if dom_freq > 0 else 0.0,
                    "dominant_power": dom_power,
                    "mean": mean_val,
                    "std": std_val,
                    "rms": rms_val,
                    "highpass_rms": hp_rms,
                }
            )
            for rank, (freq, power) in enumerate(peaks, start=1):
                peak_rows.append(
                    {
                        "condition_id": condition_id,
                        "channel": channel,
                        "signal_type": "pressure" if channel.startswith("p") else "qdot",
                        "peak_rank": rank,
                        "frequency_hz": freq,
                        "power": power,
                    }
                )

        for p_ch, q_ch in zip(pressure_cols, qdot_cols, strict=True):
            p = values[p_ch]
            q = values[q_ch]
            p0 = p - np.mean(p)
            q0 = q - np.mean(q)
            corr_zero = float(np.corrcoef(p0, q0)[0, 1]) if np.std(p0) > 0 and np.std(q0) > 0 else 0.0
            best_lag, best_corr = _corr_with_best_lag(p, q, max_lag=min(2000, len(p) // 4))
            coupling_rows.append(
                {
                    "condition_id": condition_id,
                    "pressure_channel": p_ch,
                    "qdot_channel": q_ch,
                    "corr_zero_lag": corr_zero,
                    "best_lag_samples": best_lag,
                    "best_lag_ms": best_lag * dt * 1000.0,
                    "best_lag_corr": best_corr,
                }
            )

    _write_csv(
        output_dir / "dominant_frequency_summary.csv",
        rows,
        [
            "condition_id",
            "channel",
            "signal_type",
            "n_samples",
            "dt_seconds",
            "dominant_frequency_hz",
            "dominant_period_ms",
            "dominant_power",
            "mean",
            "std",
            "rms",
            "highpass_rms",
        ],
    )
    _write_csv(
        output_dir / "top_frequency_peaks.csv",
        peak_rows,
        ["condition_id", "channel", "signal_type", "peak_rank", "frequency_hz", "power"],
    )
    _write_csv(
        output_dir / "pressure_qdot_coupling.csv",
        coupling_rows,
        ["condition_id", "pressure_channel", "qdot_channel", "corr_zero_lag", "best_lag_samples", "best_lag_ms", "best_lag_corr"],
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "manifest_json": str(Path(args.manifest_json).resolve()),
                "pressure_columns": pressure_cols,
                "qdot_columns": qdot_cols,
                "exclude_conditions": sorted(excluded),
                "max_points": args.max_points,
                "downsample_step": args.downsample_step,
                "top_k": args.top_k,
                "detrend_window": args.detrend_window,
                "min_frequency_hz": args.min_frequency_hz,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        output_dir / "frequency_report.md",
        rows=rows,
        coupling_rows=coupling_rows,
        pressure_cols=pressure_cols,
        qdot_cols=qdot_cols,
        excluded=sorted(excluded),
        downsample_step=args.downsample_step,
    )

    print("Frequency analysis finished.")
    print(f"Output dir: {output_dir}")
    print(f"Summary: {output_dir / 'dominant_frequency_summary.csv'}")
    print(f"Peaks: {output_dir / 'top_frequency_peaks.csv'}")
    print(f"Coupling: {output_dir / 'pressure_qdot_coupling.csv'}")
    print(f"Report: {output_dir / 'frequency_report.md'}")


if __name__ == "__main__":
    main()
