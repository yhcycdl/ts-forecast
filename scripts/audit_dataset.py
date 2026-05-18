#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.zenodo_loader import (
    build_run_records_from_summary,
    build_transition_candidates,
    generate_data_dictionary_markdown,
    load_summary_json,
    summarize_conditions,
)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the Zenodo combustion dataset and export run-level manifests.")
    parser.add_argument(
        "--summary-json",
        type=str,
        default="./analysis/zenodo_timeseries_summary.json",
        help="Path to the precomputed dataset structure summary JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/audit",
        help="Directory to save audit artifacts.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary_json(args.summary_json)
    runs = build_run_records_from_summary(summary)
    condition_rows = summarize_conditions(runs)
    transition_rows = build_transition_candidates(runs)

    run_rows = [
        {
            "run_id": run.run_id,
            "condition_id": run.condition_id,
            "index": run.index,
            "hydrogen_power_fraction_ph2": run.hydrogen_power_fraction_ph2,
            "chamber_length_l_c": run.chamber_length_l_c,
            "bulk_velocity_um": run.bulk_velocity_um,
            "equivalence_ratio": run.equivalence_ratio,
            "thermal_power": run.thermal_power,
            "sampling_rate_hz": run.sampling_rate,
            "signal_length_samples": run.signal_length_samples,
            "duration_seconds": run.duration_seconds,
            "signal_fields": ",".join(run.signal_fields),
            "labels_present": int(run.labels_present),
            "transition_labels_present": int(run.transition_labels_present),
            "source_path": run.source_path,
        }
        for run in runs
    ]

    _write_csv(
        output_dir / "run_manifest.csv",
        run_rows,
        [
            "run_id",
            "condition_id",
            "index",
            "hydrogen_power_fraction_ph2",
            "chamber_length_l_c",
            "bulk_velocity_um",
            "equivalence_ratio",
            "thermal_power",
            "sampling_rate_hz",
            "signal_length_samples",
            "duration_seconds",
            "signal_fields",
            "labels_present",
            "transition_labels_present",
            "source_path",
        ],
    )

    _write_csv(
        output_dir / "condition_summary.csv",
        condition_rows,
        [
            "condition_id",
            "hydrogen_power_fraction_ph2",
            "chamber_length_l_c",
            "num_runs",
            "total_duration_seconds",
            "sampling_rate_hz",
            "signal_fields",
        ],
    )

    _write_csv(
        output_dir / "transition_candidates.csv",
        transition_rows,
        ["run_id", "condition_id", "candidate_type", "confidence", "start_time_sec", "end_time_sec", "reason"],
    )

    dictionary_md = generate_data_dictionary_markdown(summary, runs)
    (output_dir / "data_dictionary.md").write_text(dictionary_md, encoding="utf-8")

    print("Audit finished.")
    print(f"Output dir: {output_dir}")
    print(f"Run manifest: {output_dir / 'run_manifest.csv'}")
    print(f"Condition summary: {output_dir / 'condition_summary.csv'}")
    print(f"Transition candidates: {output_dir / 'transition_candidates.csv'}")
    print(f"Data dictionary: {output_dir / 'data_dictionary.md'}")


if __name__ == "__main__":
    main()
