#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _parse_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _safe_name(value) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return text or "unknown"


def _load_type_map(args: argparse.Namespace, df: Any) -> tuple[Any, dict]:
    import pandas as pd

    if args.type_col and args.type_col in df.columns:
        labels = df[args.type_col].astype(str)
        return labels, {"source": "csv_type_col", "type_col": args.type_col}

    if not args.profile_csv:
        raise ValueError("Pass --type-col from the CSV or --profile-csv with segment-level signal_type.")
    if args.segment_col not in df.columns:
        raise ValueError(f"segment_col '{args.segment_col}' not found in CSV.")

    profile = pd.read_csv(args.profile_csv)
    if args.profile_segment_col not in profile.columns:
        raise ValueError(f"profile_segment_col '{args.profile_segment_col}' not found in profile CSV.")
    if args.profile_type_col not in profile.columns:
        raise ValueError(f"profile_type_col '{args.profile_type_col}' not found in profile CSV.")
    mapping = (
        profile[[args.profile_segment_col, args.profile_type_col]]
        .dropna()
        .drop_duplicates(subset=[args.profile_segment_col])
        .set_index(args.profile_segment_col)[args.profile_type_col]
        .astype(str)
        .to_dict()
    )
    labels = df[args.segment_col].map(mapping).fillna(args.unknown_label).astype(str)
    return labels, {
        "source": "profile_csv",
        "profile_csv": args.profile_csv,
        "profile_segment_col": args.profile_segment_col,
        "profile_type_col": args.profile_type_col,
        "mapped_segments": len(mapping),
    }


def _write_profile_subset(args: argparse.Namespace, output_dir: Path, signal_type: str, segment_ids: set[str]) -> str | None:
    if not args.profile_csv or not segment_ids:
        return None
    import pandas as pd

    profile = pd.read_csv(args.profile_csv)
    if args.profile_segment_col not in profile.columns:
        return None
    subset = profile[profile[args.profile_segment_col].astype(str).isin(segment_ids)].copy()
    if subset.empty:
        return None
    path = output_dir / f"profile_{_safe_name(signal_type)}.csv"
    subset.to_csv(path, index=False)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a prepared/augmented quasi-periodic CSV into one file per signal type.")
    parser.add_argument("--csv", required=True, help="Prepared or augmented long-format CSV.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--type-col", default=None, help="Type label already present in the CSV, e.g. synthetic_type.")
    parser.add_argument("--profile-csv", default=None, help="profile_by_segment.csv from analyze_quasiperiodic_profile.py.")
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--profile-segment-col", default="segment_id")
    parser.add_argument("--profile-type-col", default="signal_type")
    parser.add_argument("--include-types", default=None, help="Optional comma-separated type labels to keep.")
    parser.add_argument("--unknown-label", default="unknown")
    parser.add_argument("--drop-unknown", action="store_true")
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    import pandas as pd

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, nrows=None if args.max_rows <= 0 else args.max_rows, low_memory=False)
    if df.empty:
        raise ValueError("Input CSV is empty.")
    labels, label_meta = _load_type_map(args, df)
    include = set(_parse_list(args.include_types))

    work = df.copy()
    work["_qp_split_type"] = labels.to_numpy()
    if include:
        work = work[work["_qp_split_type"].isin(include)].copy()
    if args.drop_unknown:
        work = work[work["_qp_split_type"] != args.unknown_label].copy()
    if work.empty:
        raise ValueError("No rows left after type filtering.")

    outputs = []
    for signal_type, subset in work.groupby("_qp_split_type", sort=True):
        signal_type = str(signal_type)
        clean = subset.drop(columns=["_qp_split_type"])
        out_path = output_dir / f"{csv_path.stem}_{_safe_name(signal_type)}.csv"
        clean.to_csv(out_path, index=False)
        segment_ids: set[str] = set()
        if args.segment_col in clean.columns:
            segment_ids = set(clean[args.segment_col].astype(str).unique().tolist())
        profile_subset = _write_profile_subset(args, output_dir, signal_type, segment_ids)
        outputs.append(
            {
                "signal_type": signal_type,
                "csv": str(out_path),
                "profile_csv": profile_subset,
                "rows": int(len(clean)),
                "segments": int(len(segment_ids)) if segment_ids else None,
            }
        )

    metadata = {
        "input_csv": str(csv_path),
        "output_dir": str(output_dir),
        "label_meta": label_meta,
        "include_types": sorted(include),
        "drop_unknown": bool(args.drop_unknown),
        "outputs": outputs,
    }
    metadata_path = output_dir / "split_by_type_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved split metadata: {metadata_path}")
    for item in outputs:
        print(
            f"{item['signal_type']}: rows={item['rows']} "
            f"segments={item['segments']} csv={item['csv']}"
        )


if __name__ == "__main__":
    main()
