#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.future_indicator import SignalColumns, WindowParams, extract_indicator_rows, ms_to_samples


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_csv_root(manifest: dict[str, Any], manifest_path: Path, cli_csv_root: str | None) -> Path:
    if cli_csv_root:
        root = Path(cli_csv_root)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        return root

    candidates: list[Path] = []
    output_dir = manifest.get("output_dir")
    if output_dir:
        path = Path(str(output_dir))
        if not path.is_absolute():
            path = (manifest_path.parent / path).resolve()
        candidates.append(path)
    candidates.append(manifest_path.parent.resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return manifest_path.parent.resolve()


def _tag_piece(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace(".", "p")


def _load_csv_matrix(csv_path: Path) -> tuple[list[str], np.ndarray]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    matrix = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    return header, matrix


def _condition_id(op: dict[str, Any]) -> str:
    ph2 = str(op["hydrogen_power_fraction_PH2"]).replace(".", "p")
    lc = str(op["chamber_length_L_c"]).replace(".", "p")
    return f"PH2_{ph2}_Lc_{lc}"


def _run_id(op: dict[str, Any]) -> str:
    return f"run_{int(op['index']):02d}_{_condition_id(op)}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No indicator rows generated; refusing to write an empty table.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_run_indices(raw: str | None) -> set[int] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return {int(piece.strip()) for piece in str(raw).split(",") if piece.strip()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract history/future window indicator tables from Zenodo CSV runs.")
    parser.add_argument("--manifest-json", type=str, default="./data/zenodo_timeseries_csv/manifest.json")
    parser.add_argument(
        "--csv-root",
        type=str,
        default="",
        help="Optional override for the directory containing the exported Zenodo CSV files.",
    )
    parser.add_argument("--output-dir", type=str, default="./outputs/future_indicators")
    parser.add_argument("--history-ms", type=float, default=100.0)
    parser.add_argument("--delta-ms", type=float, default=10.0)
    parser.add_argument("--future-ms", type=float, default=30.0)
    parser.add_argument("--stride-ms", type=float, default=-1.0, help="If <= 0, use future-ms.")
    parser.add_argument("--pressure-column", type=str, default="P1")
    parser.add_argument("--q-column", type=str, default="Q")
    parser.add_argument("--band-low-hz", type=float, default=0.0)
    parser.add_argument("--band-high-hz", type=float, default=0.0)
    parser.add_argument("--band-half-bins", type=int, default=2)
    parser.add_argument("--permutation-order", type=int, default=5)
    parser.add_argument("--permutation-delay", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--split-mode", type=str, default="total")
    parser.add_argument("--run-indices", type=str, default="")
    parser.add_argument("--include-cross-boundary", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = _load_json(args.manifest_json)
    manifest_path = Path(args.manifest_json).resolve()
    csv_root = _resolve_csv_root(manifest, manifest_path, args.csv_root)
    selected_indices = _parse_run_indices(args.run_indices)
    signal_columns = SignalColumns(pressure_column=args.pressure_column, q_column=args.q_column)

    all_rows: list[dict[str, Any]] = []
    counts_by_run: dict[str, dict[str, int]] = {}
    params_by_rate: dict[float, WindowParams] = {}

    for op in manifest.get("operating_points", []):
        op_index = int(op["index"])
        if selected_indices is not None and op_index not in selected_indices:
            continue

        sample_rate = float(op["sampling_rate_hz"])
        if sample_rate not in params_by_rate:
            stride_ms = args.future_ms if float(args.stride_ms) <= 0 else float(args.stride_ms)
            params_by_rate[sample_rate] = WindowParams(
                history_length=ms_to_samples(args.history_ms, sample_rate),
                lead_gap=ms_to_samples(args.delta_ms, sample_rate),
                future_length=ms_to_samples(args.future_ms, sample_rate),
                stride=ms_to_samples(stride_ms, sample_rate),
            )
        params = params_by_rate[sample_rate]

        csv_path = csv_root / op["filename"]
        columns, matrix = _load_csv_matrix(csv_path)
        run_id = _run_id(op)
        condition_id = _condition_id(op)

        rows = extract_indicator_rows(
            values=matrix,
            columns=columns,
            sample_rate=sample_rate,
            params=params,
            run_id=run_id,
            condition_id=condition_id,
            signal_columns=signal_columns,
            condition_metadata={
                "hydrogen_power_fraction_ph2": float(op["hydrogen_power_fraction_PH2"]),
                "chamber_length_l_c": float(op["chamber_length_L_c"]),
                "bulk_velocity_um": float(op["bulk_velocity_Um"]),
                "equivalence_ratio": float(op["equivalence_ratio"]),
                "thermal_power": float(op["thermal_power"]),
                "source_csv": str(csv_path),
            },
            band_low_hz=float(args.band_low_hz) if float(args.band_high_hz) > float(args.band_low_hz) else None,
            band_high_hz=float(args.band_high_hz) if float(args.band_high_hz) > float(args.band_low_hz) else None,
            band_half_bins=int(args.band_half_bins),
            permutation_order=int(args.permutation_order),
            permutation_delay=int(args.permutation_delay),
            train_ratio=float(args.train_ratio),
            val_ratio=float(args.val_ratio),
            split_mode=args.split_mode,
        )

        if not bool(int(args.include_cross_boundary)):
            rows = [row for row in rows if row["split"] != "cross_boundary"]

        counts = {"train": 0, "val": 0, "test": 0, "cross_boundary": 0}
        for row in rows:
            counts[row["split"]] = counts.get(row["split"], 0) + 1
        counts_by_run[run_id] = counts
        all_rows.extend(rows)

    if not all_rows:
        raise ValueError("No indicator rows were generated. Check window parameters and run selection.")

    stride_ms = args.future_ms if float(args.stride_ms) <= 0 else float(args.stride_ms)
    setting_tag = (
        f"L{_tag_piece(args.history_ms)}ms_"
        f"D{_tag_piece(args.delta_ms)}ms_"
        f"W{_tag_piece(args.future_ms)}ms_"
        f"S{_tag_piece(stride_ms)}ms"
    )
    output_dir = Path(args.output_dir) / setting_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "indicator_table.csv"
    summary_path = output_dir / "indicator_summary.json"
    config_path = output_dir / "indicator_config.json"

    _write_csv(table_path, all_rows)
    summary = {
        "manifest_json": str(manifest_path),
        "num_rows": len(all_rows),
        "num_runs": len(counts_by_run),
        "counts_by_run": counts_by_run,
    }
    config = {
        "history_ms": float(args.history_ms),
        "delta_ms": float(args.delta_ms),
        "future_ms": float(args.future_ms),
        "stride_ms": float(stride_ms),
        "pressure_column": args.pressure_column,
        "q_column": args.q_column,
        "band_low_hz": float(args.band_low_hz),
        "band_high_hz": float(args.band_high_hz),
        "band_half_bins": int(args.band_half_bins),
        "permutation_order": int(args.permutation_order),
        "permutation_delay": int(args.permutation_delay),
        "train_ratio": float(args.train_ratio),
        "val_ratio": float(args.val_ratio),
        "split_mode": args.split_mode,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Indicator extraction finished.")
    print(f"Output dir: {output_dir}")
    print(f"Indicator table: {table_path}")
    print(f"Summary: {summary_path}")
    print(f"Config: {config_path}")


if __name__ == "__main__":
    main()
