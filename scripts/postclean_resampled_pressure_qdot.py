#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pressure_qdot_loader import load_pressure_qdot_manifest


@dataclass(slots=True)
class PostCleanStats:
    condition_id: str
    source_csv_path: str
    output_csv_path: str
    original_rows: int
    qdot_all_zero_rows_removed: int
    cleaned_rows: int
    time_start: float
    time_end: float
    duration_seconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove residual all-zero Qdot rows from resampled pressure/Qdot CSV files.")
    parser.add_argument(
        "--manifest-json",
        type=str,
        default="./data/pressure_qdot_csv_resampled_1us/manifest.json",
        help="Manifest JSON for resampled pressure/Qdot CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/pressure_qdot_csv_final_1us",
        help="Directory to save final post-cleaned CSV files.",
    )
    return parser


def _write_csv(path: Path, header: list[str], rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def _write_manifest(output_dir: Path, conditions: list[dict]) -> None:
    payload = {
        "source_type": "pressure_qdot_csv_final",
        "output_dir": str(output_dir.resolve()),
        "conditions": conditions,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(output_dir: Path, stats: list[PostCleanStats]) -> None:
    with (output_dir / "postclean_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "工况编号",
                "重采样CSV路径",
                "最终CSV路径",
                "原始行数",
                "Qdot全零删除行数",
                "最终行数",
                "起始时间_秒",
                "结束时间_秒",
                "持续时长_秒",
            ],
        )
        writer.writeheader()
        for s in stats:
            writer.writerow(
                {
                    "工况编号": s.condition_id,
                    "重采样CSV路径": s.source_csv_path,
                    "最终CSV路径": s.output_csv_path,
                    "原始行数": s.original_rows,
                    "Qdot全零删除行数": s.qdot_all_zero_rows_removed,
                    "最终行数": s.cleaned_rows,
                    "起始时间_秒": s.time_start,
                    "结束时间_秒": s.time_end,
                    "持续时长_秒": s.duration_seconds,
                }
            )


def _write_report(output_dir: Path, stats: list[PostCleanStats]) -> None:
    lines = []
    lines.append("# 重采样后轻清洗报告")
    lines.append("")
    lines.append("## 处理规则")
    lines.append("1. 保留重采样后的规则时间网格。")
    lines.append("2. 删除重采样后残余的 `Qdot` 16 路全零行。")
    lines.append("")
    lines.append("| 工况编号 | 原始行数 | 删除Qdot全零 | 最终行数 | 起始时间(s) | 结束时间(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s in stats:
        lines.append(
            f"| `{s.condition_id}` | `{s.original_rows}` | `{s.qdot_all_zero_rows_removed}` | `{s.cleaned_rows}` | "
            f"`{s.time_start:.9g}` | `{s.time_end:.9g}` |"
        )
    (output_dir / "postclean_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    manifest = load_pressure_qdot_manifest(args.manifest_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats: list[PostCleanStats] = []
    final_conditions: list[dict] = []

    for condition in manifest["conditions"]:
        source_csv = Path(condition["csv_path"])
        with source_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            q_idx = [i for i, name in enumerate(header) if name.startswith("qdot")]
            rows = [[float(v) for v in row] for row in reader]

        kept: list[list[float]] = []
        removed = 0
        for row in rows:
            q_vals = [row[i] for i in q_idx]
            if all(v == 0.0 for v in q_vals):
                removed += 1
                continue
            kept.append(row)

        if not kept:
            raise ValueError(f"Post-cleaning removed all rows for {condition['condition_id']}")

        output_csv_name = Path(condition["csv_filename"]).name
        output_csv_path = output_dir / output_csv_name
        _write_csv(output_csv_path, header, kept)

        final_conditions.append(
            {
                "condition_id": condition["condition_id"],
                "bypass_ratio": condition["bypass_ratio"],
                "csv_filename": output_csv_name,
                "csv_path": str(output_csv_path.resolve()),
                "source_csv_path": str(source_csv.resolve()),
                "n_probes": condition["n_probes"],
                "n_samples": len(kept),
                "dt_seconds": condition["dt_seconds"],
                "sample_rate_hz": condition["sample_rate_hz"],
                "time_start": kept[0][0],
                "time_end": kept[-1][0],
                "duration_seconds": kept[-1][0] - kept[0][0],
                "probe_coordinates": condition["probe_coordinates"],
                "postclean": {
                    "residual_qdot_all_zero_rows_removed": removed,
                },
            }
        )
        stats.append(
            PostCleanStats(
                condition_id=condition["condition_id"],
                source_csv_path=str(source_csv.resolve()),
                output_csv_path=str(output_csv_path.resolve()),
                original_rows=len(rows),
                qdot_all_zero_rows_removed=removed,
                cleaned_rows=len(kept),
                time_start=kept[0][0],
                time_end=kept[-1][0],
                duration_seconds=kept[-1][0] - kept[0][0],
            )
        )

    _write_manifest(output_dir, final_conditions)
    _write_summary(output_dir, stats)
    _write_report(output_dir, stats)

    print("Post-cleaning finished.")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Summary: {output_dir / 'postclean_summary.csv'}")
    print(f"Report: {output_dir / 'postclean_report.md'}")


if __name__ == "__main__":
    main()
