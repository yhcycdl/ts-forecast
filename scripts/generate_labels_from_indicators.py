#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.labels.label_generator import IndicatorLabelGenerator, export_label_artifacts, load_indicator_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate margin scores and labels from future-window indicator tables.")
    parser.add_argument("--indicator-table", type=str, required=True, help="Path to indicator_table.csv")
    parser.add_argument("--output-dir", type=str, default="./outputs/labels")
    parser.add_argument("--margin-mode", type=str, default="train-fitted", choices=["train-fitted", "fixed-rule"])
    parser.add_argument(
        "--weight-mode",
        type=str,
        default="physics_rule",
        choices=["physics_rule", "equal", "train_fitted_unsupervised"],
    )
    parser.add_argument("--threshold-mode", type=str, default="train-fitted", choices=["train-fitted", "fixed-rule"])
    parser.add_argument("--binary-quantile", type=float, default=0.80)
    parser.add_argument("--low-quantile", type=float, default=0.50)
    parser.add_argument("--high-quantile", type=float, default=0.85)
    parser.add_argument("--fixed-binary-threshold", type=float, default=0.65)
    parser.add_argument("--fixed-low-threshold", type=float, default=0.45)
    parser.add_argument("--fixed-high-threshold", type=float, default=0.70)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    indicator_table = Path(args.indicator_table).resolve()
    setting_tag = indicator_table.parent.name
    run_tag = f"{setting_tag}_{args.margin_mode}_{args.weight_mode}_{args.threshold_mode}"
    output_dir = Path(args.output_dir).resolve() / run_tag

    rows = load_indicator_rows(indicator_table)
    generator = IndicatorLabelGenerator(
        margin_mode=args.margin_mode,
        weight_mode=args.weight_mode,
        threshold_mode=args.threshold_mode,
        binary_quantile=float(args.binary_quantile),
        low_quantile=float(args.low_quantile),
        high_quantile=float(args.high_quantile),
        fixed_binary_threshold=float(args.fixed_binary_threshold),
        fixed_low_threshold=float(args.fixed_low_threshold),
        fixed_high_threshold=float(args.fixed_high_threshold),
    )
    paths = export_label_artifacts(
        rows=rows,
        generator=generator,
        output_dir=output_dir,
        source_indicator_csv=indicator_table,
    )

    print("Label generation finished.")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
