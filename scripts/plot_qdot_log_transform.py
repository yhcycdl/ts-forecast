#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def _signed_log1p(value: float) -> float:
    if value > 0:
        return math.log1p(value)
    if value < 0:
        return -math.log1p(abs(value))
    return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot qdot before and after signed log1p as two separate SVG figures.")
    parser.add_argument("--csv", required=True, help="Input pressure/qdot CSV.")
    parser.add_argument("--qdot-column", default="qdot00", help="qdot column to plot.")
    parser.add_argument("--time-start", type=float, default=None, help="Optional start time.")
    parser.add_argument("--time-end", type=float, default=None, help="Optional end time.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--max-points", type=int, default=8000, help="Maximum points to draw in each SVG.")
    return parser.parse_args()


def _load_series(path: Path, qdot_col: str, time_start: float | None, time_end: float | None) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if "time" not in reader.fieldnames:
            raise ValueError("CSV must contain a time column.")
        if qdot_col not in reader.fieldnames:
            raise ValueError(f"Column not found: {qdot_col}")
        for row in reader:
            t = float(row["time"])
            if time_start is not None and t < time_start:
                continue
            if time_end is not None and t > time_end:
                continue
            times.append(t)
            values.append(float(row[qdot_col]))
    if not times:
        raise ValueError("No samples selected.")
    return times, values


def _downsample(times: list[float], values: list[float], max_points: int) -> tuple[list[float], list[float]]:
    if len(times) <= max_points:
        return times, values
    step = max(1, math.ceil(len(times) / max_points))
    return times[::step], values[::step]


def _nice_ticks(lo: float, hi: float, count: int = 6) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    return [lo + (hi - lo) * i / (count - 1) for i in range(count)]


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _write_svg(
    path: Path,
    times: list[float],
    values: list[float],
    title: str,
    y_label: str,
    line_color: str,
) -> None:
    width = 1600
    height = 700
    left = 95
    right = 35
    top = 60
    bottom = 75
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = min(times), max(times)
    y_min, y_max = min(values), max(values)
    y_pad = 0.03 * (y_max - y_min) if y_max > y_min else 1.0
    y_min -= y_pad
    y_max += y_pad

    def sx(x: float) -> float:
        return left + (x - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def sy(y: float) -> float:
        return top + (y_max - y) / max(y_max - y_min, 1e-12) * plot_h

    points = [(sx(t), sy(v)) for t, v in zip(times, values, strict=True)]
    x_ticks = _nice_ticks(x_min, x_max, 6)
    y_ticks = _nice_ticks(y_min, y_max, 7)

    lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect x="0" y="0" width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="32" text-anchor="middle" font-family="Arial" font-size="26">{title}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222" stroke-width="1.5"/>',
    ]

    for x in x_ticks:
        px = sx(x)
        lines.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_h}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{px:.2f}" y="{height - 32}" text-anchor="middle" font-family="Arial" font-size="16">{x:.3f}</text>')
    for y in y_ticks:
        py = sy(y)
        lines.append(f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_w}" y2="{py:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{py + 5:.2f}" text-anchor="end" font-family="Arial" font-size="16">{y:.4g}</text>')

    lines.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{line_color}" stroke-width="1.4"/>')
    lines.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" font-family="Arial" font-size="18">time / s</text>')
    lines.append(
        f'<text x="25" y="{top + plot_h / 2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="18" transform="rotate(-90 25 {top + plot_h / 2:.1f})">{y_label}</text>'
    )
    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    times, raw_values = _load_series(csv_path, args.qdot_column, args.time_start, args.time_end)
    log_values = [_signed_log1p(v) for v in raw_values]

    ds_times, ds_raw = _downsample(times, raw_values, args.max_points)
    _, ds_log = _downsample(times, log_values, args.max_points)

    stem = f"{csv_path.stem}_{args.qdot_column}"
    raw_path = output_dir / f"{stem}_before_log.svg"
    log_path = output_dir / f"{stem}_after_signed_log1p.svg"

    _write_svg(raw_path, ds_times, ds_raw, f"{args.qdot_column} before log", args.qdot_column, "#1f77b4")
    _write_svg(log_path, ds_times, ds_log, f"{args.qdot_column} after signed log1p", f"signed_log1p({args.qdot_column})", "#ff7f0e")

    print(f"Raw qdot plot: {raw_path}")
    print(f"Log qdot plot: {log_path}")
    print(f"Raw range: min={min(raw_values):.9g}, max={max(raw_values):.9g}")
    print(f"Log range: min={min(log_values):.9g}, max={max(log_values):.9g}")


if __name__ == "__main__":
    main()
