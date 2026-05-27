#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # Keep --help usable on minimal local environments.
    pd = None


def _parse_cols(raw: str | None) -> list[str] | None:
    if raw is None or raw.strip().lower() in {"", "auto"}:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def _downsample(df, max_points: int):
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)


def _load_window_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
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
    keys = (str(csv_path), str(csv_path.resolve()), csv_path.name, csv_path.stem)
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


def _crop_by_time(df, time_start: float | None, time_end: float | None):
    if "time" not in df.columns:
        raise ValueError("Input CSV must contain a 'time' column.")
    if time_start is None and time_end is None:
        return df
    time = df["time"].to_numpy(dtype=np.float64)
    mask = np.ones(len(df), dtype=bool)
    if time_start is not None:
        mask &= time >= float(time_start)
    if time_end is not None:
        mask &= time <= float(time_end)
    out = df.loc[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows remain after crop: time_start={time_start}, time_end={time_end}")
    return out


def _read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def _load_condition(path: Path, time_start: float | None, time_end: float | None, window_config: dict):
    header = _read_header(path)
    p_cols = [col for col in header if col.startswith("p")]
    if not p_cols:
        raise ValueError(f"No pressure columns found in {path}")
    usecols = ["time"] + p_cols
    df = pd.read_csv(path, usecols=usecols)
    resolved_start, resolved_end = _resolve_time_window(path, time_start, time_end, window_config)
    df = _crop_by_time(df, resolved_start, resolved_end)
    return df, p_cols, resolved_start, resolved_end


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = int(max(1, window))
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot cross-condition combustion pressure overviews for state-clustering analysis."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        default=[
            "./data/cond00_BR_0p315.csv",
            "./data/cond01_BR_0p63.csv",
            "./data/cond03_BR_1p26.csv",
        ],
        help="Condition CSV files to compare.",
    )
    parser.add_argument(
        "--condition-names",
        default=None,
        help="Optional comma-separated display names. Defaults to CSV stems.",
    )
    parser.add_argument("--output-dir", default="./outputs/combustion_state_analysis/pressure_compare")
    parser.add_argument(
        "--overview-cols",
        default="p00,p05,p10,p15",
        help="Pressure channels shown in the multi-channel overview.",
    )
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--window-config", default=None, help="Optional per-file crop window JSON.")
    parser.add_argument("--max-points", type=int, default=50000)
    parser.add_argument("--rms-window", type=int, default=8192)
    parser.add_argument("--figsize", default="18,10")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if pd is None:
        raise RuntimeError("This script requires pandas. Install it in the training environment first.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csvs = [Path(path) for path in args.csvs]
    names = _parse_cols(args.condition_names)
    if names is None:
        names = [path.stem for path in csvs]
    if len(names) != len(csvs):
        raise ValueError("--condition-names must have the same length as --csvs.")

    overview_cols = _parse_cols(args.overview_cols) or []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_config = _load_window_config(args.window_config)
    width, height = [float(x.strip()) for x in args.figsize.split(",", 1)]

    loaded = []
    stats: list[dict] = []
    for name, path in zip(names, csvs, strict=True):
        df, p_cols, resolved_start, resolved_end = _load_condition(path, args.time_start, args.time_end, window_config)
        p = df[p_cols].to_numpy(dtype=np.float64)
        p_mean = p.mean(axis=1)
        p_fluct = p - p.mean(axis=0, keepdims=True)
        p_rms_each_time = np.sqrt(np.mean(p_fluct**2, axis=1))
        loaded.append(
            {
                "name": name,
                "path": path,
                "df": df,
                "p_cols": p_cols,
                "p_mean": p_mean,
                "p_mean_z": (p_mean - p_mean.mean()) / (p_mean.std() + 1e-12),
                "p_rms_envelope": _moving_average(p_rms_each_time, args.rms_window),
            }
        )
        stats.append(
            {
                "condition": name,
                "csv": str(path),
                "rows": int(len(df)),
                "time_start": float(df["time"].iloc[0]),
                "time_end": float(df["time"].iloc[-1]),
                "resolved_time_start": resolved_start,
                "resolved_time_end": resolved_end,
                "n_pressure_cols": int(len(p_cols)),
                "p_mean": float(p.mean()),
                "p_std_fluct": float(p_fluct.std()),
                "p_rms_fluct": float(np.sqrt(np.mean(p_fluct**2))),
                "p_peak_to_peak_mean": float(np.ptp(p, axis=0).mean()),
                "p_peak_to_peak_max": float(np.ptp(p, axis=0).max()),
            }
        )

    # 1. Selected pressure channels.
    fig, axes = plt.subplots(len(loaded), 1, figsize=(width, height), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, loaded, strict=True):
        df = item["df"]
        p_cols = item["p_cols"]
        cols = [col for col in overview_cols if col in p_cols] or p_cols[: min(4, len(p_cols))]
        d = _downsample(df[["time"] + cols], args.max_points)
        for col in cols:
            ax.plot(d["time"], d[col], lw=0.7, label=col)
        ax.set_title(str(item["name"]))
        ax.set_ylabel("pressure")
        ax.grid(alpha=0.25)
        ax.legend(ncol=min(4, len(cols)), fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "pressure_channels_overview.png", dpi=180)
    plt.close(fig)

    # 2. Mean pressure in physical units.
    fig, axes = plt.subplots(len(loaded), 1, figsize=(width, max(6.0, height * 0.9)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, loaded, strict=True):
        d = _downsample(pd.DataFrame({"time": item["df"]["time"], "p_mean": item["p_mean"]}), args.max_points)
        ax.plot(d["time"], d["p_mean"], lw=0.8)
        ax.set_title(f"{item['name']} | mean pressure")
        ax.set_ylabel("p_mean")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "pressure_mean_overview.png", dpi=180)
    plt.close(fig)

    # 3. Mean pressure shape after z-score.
    fig, axes = plt.subplots(len(loaded), 1, figsize=(width, max(6.0, height * 0.9)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, loaded, strict=True):
        d = _downsample(pd.DataFrame({"time": item["df"]["time"], "p_mean_z": item["p_mean_z"]}), args.max_points)
        ax.plot(d["time"], d["p_mean_z"], lw=0.8)
        ax.set_title(f"{item['name']} | mean pressure z-score")
        ax.set_ylabel("z")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "pressure_mean_z_compare.png", dpi=180)
    plt.close(fig)

    # 4. Pressure fluctuation RMS envelope.
    fig, axes = plt.subplots(len(loaded), 1, figsize=(width, max(6.0, height * 0.9)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, item in zip(axes, loaded, strict=True):
        d = _downsample(
            pd.DataFrame({"time": item["df"]["time"], "rms": item["p_rms_envelope"]}),
            args.max_points,
        )
        ax.plot(d["time"], d["rms"], lw=0.8)
        ax.set_title(f"{item['name']} | pressure fluctuation RMS envelope, window={args.rms_window}")
        ax.set_ylabel("RMS")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "pressure_rms_envelope_compare.png", dpi=180)
    plt.close(fig)

    stats_path = output_dir / "pressure_condition_stats.csv"
    pd.DataFrame(stats).to_csv(stats_path, index=False)

    print(f"Saved to: {output_dir}")
    print(f"  {output_dir / 'pressure_channels_overview.png'}")
    print(f"  {output_dir / 'pressure_mean_overview.png'}")
    print(f"  {output_dir / 'pressure_mean_z_compare.png'}")
    print(f"  {output_dir / 'pressure_rms_envelope_compare.png'}")
    print(f"  {stats_path}")


if __name__ == "__main__":
    main()
