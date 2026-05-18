#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.pressure_qdot_loader import (
    discover_condition_records,
    export_condition_csv,
    write_manifest_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export pressure/Qdot text files into per-condition CSV files.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/压力释热变化数据",
        help="Root directory containing condition folders like 涵道比0.315.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/pressure_qdot_csv",
        help="Directory to save merged CSV files.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = discover_condition_records(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_filenames: dict[str, str] = {}
    for idx, record in enumerate(records):
        filename = f"cond{idx:02d}_{record.condition_id}.csv"
        csv_filenames[record.condition_id] = filename
        export_condition_csv(record, output_dir / filename)

    write_manifest_json(records, output_dir, csv_filenames, output_dir / "manifest.json")

    print("Export finished.")
    print(f"Output dir: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    for condition_id, filename in csv_filenames.items():
        print(f"- {condition_id}: {output_dir / filename}")


if __name__ == "__main__":
    main()
