#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge prepared per-condition CSV files without losing segment identity.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Prepared CSV files to merge.")
    parser.add_argument("--output", required=True, help="Merged CSV path.")
    parser.add_argument("--segment-col", default="segment_id", help="Column used to mark the source condition/segment.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    input_paths = [Path(path) for path in args.inputs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_header: list[str] | None = None
    rows_written = 0
    with output_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = None
        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(path)
            with path.open("r", encoding="utf-8", newline="") as in_f:
                reader = csv.DictReader(in_f)
                if reader.fieldnames is None:
                    raise ValueError(f"CSV has no header: {path}")
                header = list(reader.fieldnames)
                if base_header is None:
                    base_header = header
                    output_header = list(base_header)
                    if args.segment_col not in output_header:
                        output_header.append(args.segment_col)
                    writer = csv.DictWriter(out_f, fieldnames=output_header)
                    writer.writeheader()
                elif header != base_header:
                    raise ValueError(f"Header mismatch in {path}. Expected {base_header}, got {header}")

                assert writer is not None
                segment_id = path.stem
                for row in reader:
                    row[args.segment_col] = segment_id
                    writer.writerow(row)
                    rows_written += 1

    print(f"Merged CSV: {output_path}")
    print(f"Input files: {len(input_paths)}")
    print(f"Rows written: {rows_written}")
    print(f"Segment column: {args.segment_col}")


if __name__ == "__main__":
    main()
