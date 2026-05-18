#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pressure_qdot_loader import discover_condition_records, generate_data_dictionary_markdown


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the pressure/Qdot combustion dataset.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/压力释热变化数据",
        help="Root directory containing condition folders like 涵道比0.315.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/audit_pressure_qdot",
        help="Directory to save audit artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_condition_records(args.data_root)
    rows = [
        {
            "工况编号": r.condition_id,
            "涵道比": r.bypass_ratio,
            "源目录": r.source_dir,
            "压力文件路径": r.p_path,
            "释热率文件路径": r.qdot_path,
            "测点数": r.n_probes,
            "采样点数": r.n_samples,
            "采样步长_秒": r.dt_seconds,
            "采样率_Hz": r.sample_rate_hz,
            "起始时间_秒": r.time_start,
            "结束时间_秒": r.time_end,
            "持续时长_秒": r.duration_seconds,
        }
        for r in records
    ]

    _write_csv(
        output_dir / "condition_manifest.csv",
        rows,
        [
            "工况编号",
            "涵道比",
            "源目录",
            "压力文件路径",
            "释热率文件路径",
            "测点数",
            "采样点数",
            "采样步长_秒",
            "采样率_Hz",
            "起始时间_秒",
            "结束时间_秒",
            "持续时长_秒",
        ],
    )

    dictionary_md = generate_data_dictionary_markdown(records)
    (output_dir / "data_dictionary.md").write_text(dictionary_md, encoding="utf-8")

    print("Audit finished.")
    print(f"Output dir: {output_dir}")
    print(f"Condition manifest: {output_dir / 'condition_manifest.csv'}")
    print(f"Data dictionary: {output_dir / 'data_dictionary.md'}")


if __name__ == "__main__":
    main()
