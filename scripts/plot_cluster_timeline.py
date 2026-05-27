#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:  # Keep --help usable on minimal local environments.
    np = None
    pd = None


def _parse_k_values(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            out.extend(range(int(left), int(right) + 1))
        else:
            out.append(int(part))
    out = sorted(set(k for k in out if k >= 2))
    if not out:
        raise ValueError("--k-values must contain at least one K >= 2.")
    return out


def _downsample(df, max_points: int):
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot KMeans cluster labels back on the original combustion pressure timeline."
    )
    parser.add_argument(
        "--cluster-dir",
        required=True,
        help="Output directory from cluster_combustion_states.py.",
    )
    parser.add_argument("--k-values", default="2-4", help="K values to plot, e.g. 2-4 or 2,3,4.")
    parser.add_argument("--raw-col", default="p00", help="Raw pressure column shown as background.")
    parser.add_argument("--max-points", type=int, default=60000)
    parser.add_argument("--output-dir", default=None, help="Defaults to --cluster-dir.")
    parser.add_argument("--fig-width", type=float, default=18.0)
    parser.add_argument("--row-height", type=float, default=4.0)
    parser.add_argument("--point-size", type=float, default=12.0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if pd is None or np is None:
        raise RuntimeError("This script requires numpy and pandas. Install them in the training environment first.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cluster_dir = Path(args.cluster_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else cluster_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    k_values = _parse_k_values(args.k_values)

    for k in k_values:
        label_path = cluster_dir / f"window_labels_k{k}.csv"
        if not label_path.exists():
            print(f"[WARN] missing: {label_path}")
            continue

        labels = pd.read_csv(label_path)
        required = {"record_id", "source_csv", "time_start", "time_end", "cluster"}
        missing = required - set(labels.columns)
        if missing:
            raise ValueError(f"{label_path} is missing required columns: {sorted(missing)}")

        records = labels["record_id"].drop_duplicates().tolist()
        if not records:
            print(f"[WARN] no records in {label_path}")
            continue

        fig, axes = plt.subplots(
            len(records),
            1,
            figsize=(args.fig_width, max(args.row_height, args.row_height * len(records))),
            sharex=False,
        )
        if len(records) == 1:
            axes = [axes]

        cmap = ListedColormap(plt.get_cmap("tab10").colors[:k])
        norm = BoundaryNorm(np.arange(-0.5, k + 0.5, 1), cmap.N)
        scatter = None
        for ax, record in zip(axes, records, strict=True):
            sub = labels[labels["record_id"] == record].copy()
            csv_path = sub["source_csv"].iloc[0]

            raw = pd.read_csv(csv_path, usecols=["time", args.raw_col])
            raw = _downsample(raw, args.max_points)
            ax.plot(
                raw["time"],
                raw[args.raw_col],
                lw=0.6,
                color="#1f77b4",
                alpha=0.75,
                label=args.raw_col,
            )

            ymin, ymax = ax.get_ylim()
            y_band = ymin + 0.06 * (ymax - ymin)
            centers = (sub["time_start"].to_numpy() + sub["time_end"].to_numpy()) / 2.0
            clusters = sub["cluster"].to_numpy()
            scatter = ax.scatter(
                centers,
                [y_band] * len(centers),
                c=clusters,
                cmap=cmap,
                norm=norm,
                s=args.point_size,
                marker="s",
                alpha=0.9,
                label="cluster label",
            )

            ax.set_title(f"{record} | K={k}")
            ax.set_ylabel("pressure")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("time (s)")
        if scatter is not None:
            fig.subplots_adjust(right=0.90, hspace=0.35)
            cax = fig.add_axes([0.92, 0.18, 0.018, 0.64])
            cbar = fig.colorbar(scatter, cax=cax, ticks=list(range(k)))
            cbar.set_label("cluster")
        else:
            fig.tight_layout()
        out_path = output_dir / f"cluster_timeline_k{k}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
