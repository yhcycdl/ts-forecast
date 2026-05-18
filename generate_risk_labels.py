import argparse
import json
import os

import numpy as np
import pandas as pd

from data_provider.processing import load_dataframe
from data_provider.risk_labels import compute_risk_window_stats, fit_risk_label_config, assign_risk_labels


def main():
    parser = argparse.ArgumentParser(description="Generate explicit risk labels from a raw time-series CSV.")
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True, help="Output .csv or .npz file path")
    parser.add_argument("--feature_cols", type=str, default=None, help="Comma-separated feature columns; default uses all numeric columns")
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--risk_label_channel", type=int, default=0)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--risk_band_low", type=float, default=0.0)
    parser.add_argument("--risk_band_high", type=float, default=0.0)
    parser.add_argument("--risk_use_ber", type=int, default=1)
    parser.add_argument("--risk_rms_low_quantile", type=float, default=0.5)
    parser.add_argument("--risk_rms_high_quantile", type=float, default=0.85)
    parser.add_argument("--risk_ber_low_quantile", type=float, default=0.5)
    parser.add_argument("--risk_ber_high_quantile", type=float, default=0.85)
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    args = parser.parse_args()

    df = load_dataframe(args.input_csv)
    if args.feature_cols:
        feature_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
        values = df[feature_cols].to_numpy(dtype=np.float32)
    else:
        values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)

    label_config, stats = fit_risk_label_config(values, args)
    labels = assign_risk_labels(stats, label_config)
    start_indices = np.asarray(stats["start_indices"], dtype=np.int64)

    ext = os.path.splitext(args.output_path)[1].lower()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    if ext == ".csv":
        out_df = pd.DataFrame({
            "start_idx": start_indices,
            "label": labels,
            "rms": stats["rms"],
            "ber": stats["ber"],
        })
        out_df.to_csv(args.output_path, index=False)
    elif ext == ".npz":
        np.savez(
            args.output_path,
            start_indices=start_indices,
            labels=labels,
            rms=stats["rms"],
            ber=stats["ber"],
        )
    else:
        raise ValueError("output_path must end with .csv or .npz")

    meta_path = os.path.splitext(args.output_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(label_config, f, ensure_ascii=False, indent=2)

    print(f"Saved labels to: {args.output_path}")
    print(f"Saved label config to: {meta_path}")


if __name__ == "__main__":
    main()
