#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from array import array
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pressure_qdot_loader import load_pressure_qdot_manifest


@dataclass(slots=True)
class ResampleStats:
    condition_id: str
    source_csv_path: str
    output_csv_path: str
    original_rows: int
    resampled_rows: int
    original_time_start: float
    original_time_end: float
    resampled_time_start: float
    resampled_time_end: float
    target_dt_seconds: float
    target_sample_rate_hz: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resample cleaned pressure/Qdot CSV files onto a regular time grid.")
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="./data/pressure_qdot_csv_clean/manifest.json",
        help="Manifest JSON for cleaned pressure/Qdot CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/pressure_qdot_csv_resampled_1us",
        help="Directory to save resampled CSV files.",
    )
    parser.add_argument(
        "--target-dt",
        type=float,
        default=1e-6,
        help="Target resampling interval in seconds.",
    )
    return parser


def _load_csv_columns(path: Path) -> tuple[list[str], array, list[array]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        cols = [array("d") for _ in header]
        for row in reader:
            for col, value in zip(cols, row, strict=True):
                col.append(float(value))
    return header, cols[0], cols[1:]


def _build_regular_grid(start: float, end: float, dt: float) -> array:
    n_steps = int(math.floor((end - start) / dt + 1e-12))
    grid = array("d", (start + i * dt for i in range(n_steps + 1)))
    if grid[-1] < end:
        grid.append(end)
    return grid


def _resample_linear(times: array, values: array, target_times: array) -> array:
    out = array("d")
    n = len(times)
    j = 0
    for t in target_times:
        while j + 1 < n and times[j + 1] < t:
            j += 1
        if j + 1 >= n:
            out.append(values[-1])
            continue
        t0 = times[j]
        t1 = times[j + 1]
        v0 = values[j]
        v1 = values[j + 1]
        if t1 == t0:
            out.append(v0)
            continue
        alpha = (t - t0) / (t1 - t0)
        out.append(v0 + alpha * (v1 - v0))
    return out


def _write_resampled_csv(path: Path, header: list[str], times: array, columns: list[array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, t in enumerate(times):
            writer.writerow([t, *[col[i] for col in columns]])


def _write_manifest(output_dir: Path, conditions: list[dict]) -> None:
    payload = {
        "source_type": "pressure_qdot_csv_resampled",
        "output_dir": str(output_dir.resolve()),
        "conditions": conditions,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(output_dir: Path, stats: list[ResampleStats]) -> None:
    fieldnames = [
        "工况编号",
        "原始CSV路径",
        "重采样后CSV路径",
        "原始行数",
        "重采样后行数",
        "原始起始时间_秒",
        "原始结束时间_秒",
        "重采样起始时间_秒",
        "重采样结束时间_秒",
        "目标时间步长_秒",
        "目标采样率_Hz",
    ]
    with (output_dir / "resample_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in stats:
            writer.writerow(
                {
                    "工况编号": s.condition_id,
                    "原始CSV路径": s.source_csv_path,
                    "重采样后CSV路径": s.output_csv_path,
                    "原始行数": s.original_rows,
                    "重采样后行数": s.resampled_rows,
                    "原始起始时间_秒": s.original_time_start,
                    "原始结束时间_秒": s.original_time_end,
                    "重采样起始时间_秒": s.resampled_time_start,
                    "重采样结束时间_秒": s.resampled_time_end,
                    "目标时间步长_秒": s.target_dt_seconds,
                    "目标采样率_Hz": s.target_sample_rate_hz,
                }
            )


def _write_report(output_dir: Path, stats: list[ResampleStats], target_dt: float) -> None:
    lines = []
    lines.append("# 重采样报告")
    lines.append("")
    lines.append(f"- 目标时间步长: `{target_dt}` 秒")
    lines.append(f"- 目标采样率: `{1.0 / target_dt}` Hz")
    lines.append("- 重采样方法: 分通道线性插值")
    lines.append("- 当前仅统一时间网格，不裁共同有效时间段。")
    lines.append("")
    lines.append("| 工况编号 | 原始行数 | 重采样后行数 | 原始起始时间(s) | 原始结束时间(s) | 重采样起始时间(s) | 重采样结束时间(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            f"| `{s.condition_id}` | `{s.original_rows}` | `{s.resampled_rows}` | `{s.original_time_start:.9g}` | "
            f"`{s.original_time_end:.9g}` | `{s.resampled_time_start:.9g}` | `{s.resampled_time_end:.9g}` |"
        )
    (output_dir / "resample_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_pressure_qdot_manifest(args.manifest_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[ResampleStats] = []
    resampled_conditions: list[dict] = []

    for condition in manifest["conditions"]:
        source_csv = Path(condition["csv_path"])
        header, times, columns = _load_csv_columns(source_csv)
        if len(times) < 2:
            raise ValueError(f"Not enough samples in {source_csv}")

        target_times = _build_regular_grid(times[0], times[-1], args.target_dt)
        resampled_columns = [_resample_linear(times, col, target_times) for col in columns]

        output_csv_name = Path(condition["csv_filename"]).name
        output_csv_path = output_dir / output_csv_name
        _write_resampled_csv(output_csv_path, header, target_times, resampled_columns)

        resampled_conditions.append(
            {
                "condition_id": condition["condition_id"],
                "bypass_ratio": condition["bypass_ratio"],
                "csv_filename": output_csv_name,
                "csv_path": str(output_csv_path.resolve()),
                "source_csv_path": str(source_csv.resolve()),
                "n_probes": condition["n_probes"],
                "n_samples": len(target_times),
                "dt_seconds": args.target_dt,
                "sample_rate_hz": 1.0 / args.target_dt,
                "time_start": target_times[0],
                "time_end": target_times[-1],
                "duration_seconds": target_times[-1] - target_times[0],
                "probe_coordinates": condition["probe_coordinates"],
                "resampling": {
                    "method": "linear_interpolation",
                    "target_dt_seconds": args.target_dt,
                    "target_sample_rate_hz": 1.0 / args.target_dt,
                },
            }
        )

        stats.append(
            ResampleStats(
                condition_id=condition["condition_id"],
                source_csv_path=str(source_csv.resolve()),
                output_csv_path=str(output_csv_path.resolve()),
                original_rows=len(times),
                resampled_rows=len(target_times),
                original_time_start=times[0],
                original_time_end=times[-1],
                resampled_time_start=target_times[0],
                resampled_time_end=target_times[-1],
                target_dt_seconds=args.target_dt,
                target_sample_rate_hz=1.0 / args.target_dt,
            )
        )

    _write_manifest(output_dir, resampled_conditions)
    _write_summary(output_dir, stats)
    _write_report(output_dir, stats, args.target_dt)

    print("Resampling finished.")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Summary: {output_dir / 'resample_summary.csv'}")
    print(f"Report: {output_dir / 'resample_report.md'}")


if __name__ == "__main__":
    main()
