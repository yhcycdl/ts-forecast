#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import pandas as pd
except ModuleNotFoundError:  # Keep --help usable on minimal local environments.
    pd = None


def _parse_cols(value: str | None, header: list[str], prefix: str) -> list[str]:
    if value is None or value.strip().lower() == "auto":
        return [name for name in header if name.startswith(prefix)]
    if value.strip().lower() in {"", "none", "null"}:
        return []
    cols = [part.strip() for part in value.split(",") if part.strip()]
    missing = [name for name in cols if name not in header]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return cols


def _parse_k_values(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
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


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def _load_window_config(config_path: str | None) -> dict:
    if config_path is None:
        return {}
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("--window-config must be a JSON object.")
    windows = raw.get("windows", raw)
    if not isinstance(windows, dict):
        raise ValueError("--window-config JSON must be a mapping or contain a 'windows' mapping.")
    return windows


def _resolve_time_window(
    csv_path: Path,
    default_start: float | None,
    default_end: float | None,
    window_config: dict,
) -> tuple[float | None, float | None]:
    keys = (str(csv_path), str(csv_path.resolve()), csv_path.name, csv_path.stem)
    spec = None
    for key in keys:
        if key in window_config:
            spec = window_config[key]
            break
    if spec is None:
        return default_start, default_end
    if isinstance(spec, (list, tuple)):
        if len(spec) != 2:
            raise ValueError(f"Window list for {csv_path} must contain [time_start, time_end].")
        return spec[0], spec[1]
    if not isinstance(spec, dict):
        raise ValueError(f"Window spec for {csv_path} must be a dict or [time_start, time_end].")
    return spec.get("time_start", default_start), spec.get("time_end", default_end)


def _crop_df(df: pd.DataFrame, time_start: float | None, time_end: float | None) -> pd.DataFrame:
    if "time" not in df.columns:
        return df
    mask = np.ones(len(df), dtype=bool)
    time = df["time"].to_numpy(dtype=np.float64)
    if time_start is not None:
        mask &= time >= float(time_start)
    if time_end is not None:
        mask &= time <= float(time_end)
    cropped = df.loc[mask].reset_index(drop=True)
    if cropped.empty:
        raise ValueError(f"No rows remain after crop: time_start={time_start}, time_end={time_end}")
    return cropped


def _estimate_sample_rate(time: np.ndarray, fallback: float | None) -> float:
    if fallback is not None and fallback > 0:
        return float(fallback)
    if time.size < 2:
        return 1.0
    dt = float(np.median(np.diff(time)))
    if dt <= 0:
        return 1.0
    return 1.0 / dt


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _moments(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    std = float(np.std(x))
    if std <= 1e-12:
        return 0.0, 0.0
    z = (x - float(np.mean(x))) / std
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    return skew, kurt


def _spectral_features(signal: np.ndarray, fs: float, min_freq: float, max_freq: float | None) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - float(np.mean(signal))
    if signal.size < 8 or float(np.std(signal)) <= 1e-12:
        return {
            "dom_freq_hz": 0.0,
            "dom_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_entropy": 0.0,
        }
    window = np.hanning(signal.size)
    spec = np.fft.rfft(signal * window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    power[0] = 0.0
    valid = freqs >= float(min_freq)
    if max_freq is not None and max_freq > 0:
        valid &= freqs <= float(max_freq)
    if not np.any(valid):
        valid = freqs > 0
    p = power.copy()
    p[~valid] = 0.0
    total = float(np.sum(p))
    if total <= 1e-20:
        return {
            "dom_freq_hz": 0.0,
            "dom_power_ratio": 0.0,
            "spectral_centroid_hz": 0.0,
            "spectral_entropy": 0.0,
        }
    idx = int(np.argmax(p))
    prob = p[valid] / total
    entropy = -float(np.sum(prob * np.log(prob + 1e-20))) / float(np.log(max(2, prob.size)))
    centroid = float(np.sum(freqs * p) / total)
    return {
        "dom_freq_hz": float(freqs[idx]),
        "dom_power_ratio": float(p[idx] / total),
        "spectral_centroid_hz": centroid,
        "spectral_entropy": entropy,
    }


def _band_energy_ratio(signal: np.ndarray, fs: float, low: float, high: float) -> float:
    signal = np.asarray(signal, dtype=np.float64)
    signal = signal - float(np.mean(signal))
    if signal.size < 8 or float(np.std(signal)) <= 1e-12:
        return 0.0
    spec = np.fft.rfft(signal * np.hanning(signal.size))
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    power[0] = 0.0
    total = float(np.sum(power))
    if total <= 1e-20:
        return 0.0
    band = (freqs >= low) & (freqs <= high)
    return float(np.sum(power[band]) / total)


def _extract_window_features(
    p_win: np.ndarray,
    q_win: np.ndarray | None,
    fs: float,
    min_freq: float,
    max_freq: float | None,
    band_ranges: list[tuple[float, float]],
) -> dict[str, float]:
    p_win = np.asarray(p_win, dtype=np.float64)
    p_mean = np.mean(p_win, axis=1)
    p_centered = p_win - np.mean(p_win, axis=0, keepdims=True)
    p_flat = p_centered.reshape(-1)
    p_rms = float(np.sqrt(np.mean(p_flat**2)))
    p_max_abs = float(np.max(np.abs(p_flat))) if p_flat.size else 0.0
    p_skew, p_kurt = _moments(p_flat)

    feats: dict[str, float] = {
        "p_mean": float(np.mean(p_win)),
        "p_std": float(np.std(p_flat)),
        "p_rms": p_rms,
        "p_peak_to_peak_mean": float(np.mean(np.ptp(p_win, axis=0))),
        "p_peak_to_peak_max": float(np.max(np.ptp(p_win, axis=0))),
        "p_max_abs_fluct": p_max_abs,
        "p_crest_factor": p_max_abs / (p_rms + 1e-12),
        "p_skew": p_skew,
        "p_kurtosis": p_kurt,
        "p_channel_mean_std": float(np.std(np.mean(p_win, axis=0))),
    }
    feats.update({f"p_{k}": v for k, v in _spectral_features(p_mean, fs, min_freq, max_freq).items()})
    for low, high in band_ranges:
        key = f"p_band_{int(low)}_{int(high)}_ratio"
        feats[key] = _band_energy_ratio(p_mean, fs, low, high)

    if q_win is not None and q_win.size:
        q_mean = np.mean(q_win, axis=1)
        q_centered = q_win - np.mean(q_win, axis=0, keepdims=True)
        q_flat = q_centered.reshape(-1)
        q_rms = float(np.sqrt(np.mean(q_flat**2)))
        q_max_abs = float(np.max(np.abs(q_flat))) if q_flat.size else 0.0
        q_skew, q_kurt = _moments(q_flat)
        feats.update(
            {
                "q_mean": float(np.mean(q_win)),
                "q_std": float(np.std(q_flat)),
                "q_rms": q_rms,
                "q_peak_to_peak_mean": float(np.mean(np.ptp(q_win, axis=0))),
                "q_peak_to_peak_max": float(np.max(np.ptp(q_win, axis=0))),
                "q_max_abs_fluct": q_max_abs,
                "q_crest_factor": q_max_abs / (q_rms + 1e-12),
                "q_skew": q_skew,
                "q_kurtosis": q_kurt,
                "pq_corr": _safe_corr(p_mean - np.mean(p_mean), q_mean - np.mean(q_mean)),
                "rayleigh_proxy": float(np.mean((p_mean - np.mean(p_mean)) * (q_mean - np.mean(q_mean)))),
            }
        )
        feats.update({f"q_{k}": v for k, v in _spectral_features(q_mean, fs, min_freq, max_freq).items()})
    return feats


def _iter_starts(length: int, window_size: int, stride: int, max_windows: int | None) -> Iterable[int]:
    starts = list(range(0, max(0, length - window_size + 1), stride))
    if max_windows is not None and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]
    return starts


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_plots(
    output_dir: Path,
    x_scaled: np.ndarray,
    labels: np.ndarray,
    scores: list[dict],
    feature_names: list[str],
    best_k: int,
    random_state: int,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[WARN] Plotting skipped: {exc}")
        return

    if scores:
        ks = [int(row["k"]) for row in scores]
        sil = [float(row["silhouette"]) for row in scores]
        db = [float(row["davies_bouldin"]) for row in scores]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(ks, sil, marker="o", label="Silhouette")
        ax1.set_xlabel("K")
        ax1.set_ylabel("Silhouette (higher is better)")
        ax2 = ax1.twinx()
        ax2.plot(ks, db, marker="s", color="#d62728", label="Davies-Bouldin")
        ax2.set_ylabel("Davies-Bouldin (lower is better)")
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "cluster_scores.png", dpi=160)
        plt.close(fig)

    if x_scaled.shape[0] >= 2:
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(x_scaled)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=labels, s=8, cmap="tab10", alpha=0.75)
        ax.set_title(f"PCA view of KMeans clusters (K={best_k})")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.25)
        fig.colorbar(sc, ax=ax, label="cluster")
        fig.tight_layout()
        fig.savefig(output_dir / f"pca_clusters_k{best_k}.png", dpi=180)
        plt.close(fig)

    # Feature importance proxy: absolute distance between cluster means.
    cluster_ids = sorted(set(int(x) for x in labels))
    if len(cluster_ids) >= 2:
        means = []
        for cid in cluster_ids:
            means.append(np.mean(x_scaled[labels == cid], axis=0))
        spread = np.std(np.stack(means, axis=0), axis=0)
        top_idx = np.argsort(spread)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh([feature_names[i] for i in top_idx[::-1]], spread[top_idx][::-1])
        ax.set_title("Top cluster-separating features")
        ax.set_xlabel("Std. of cluster means in standardized feature space")
        fig.tight_layout()
        fig.savefig(output_dir / f"top_features_k{best_k}.png", dpi=180)
        plt.close(fig)


def _save_example_plot(
    output_dir: Path,
    label_rows: list[dict],
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    centers: np.ndarray,
    pressure_cols: dict[str, list[str]],
    qdot_cols: dict[str, list[str]],
    best_k: int,
    examples_per_cluster: int,
) -> None:
    if examples_per_cluster <= 0:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Example plot skipped: {exc}")
        return

    selected: list[int] = []
    for cid in range(best_k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        dist = np.linalg.norm(feature_matrix[idx] - centers[cid], axis=1)
        order = idx[np.argsort(dist)[:examples_per_cluster]]
        selected.extend(int(x) for x in order)

    if not selected:
        return

    n = len(selected)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(2.2, 1.8 * n)), squeeze=False)
    cache: dict[str, pd.DataFrame] = {}
    for ax, row_idx in zip(axes[:, 0], selected, strict=False):
        row = label_rows[row_idx]
        source = str(row["source_csv"])
        t0 = float(row["time_start"])
        t1 = float(row["time_end"])
        cols = ["time"] + pressure_cols[source] + qdot_cols.get(source, [])
        if source not in cache:
            cache[source] = pd.read_csv(source, usecols=cols)
        df = cache[source]
        seg = df[(df["time"] >= t0) & (df["time"] <= t1)]
        if seg.empty:
            continue
        x = (seg["time"].to_numpy(dtype=np.float64) - t0) * 1000.0
        p = seg[pressure_cols[source]].to_numpy(dtype=np.float64)
        p_mean = np.mean(p, axis=1)
        p_plot = (p_mean - np.mean(p_mean)) / (np.std(p_mean) + 1e-12)
        ax.plot(x, p_plot, lw=0.9, label="pressure mean (z)")
        q_cols = qdot_cols.get(source, [])
        if q_cols:
            q = seg[q_cols].to_numpy(dtype=np.float64)
            q_mean = np.mean(q, axis=1)
            q_plot = (q_mean - np.mean(q_mean)) / (np.std(q_mean) + 1e-12)
            ax.plot(x, q_plot, lw=0.8, alpha=0.75, label="qdot mean (z)")
        ax.set_title(f"cluster={row['cluster']} | {Path(source).name} | t={t0:.5f}-{t1:.5f}s")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1, 0].set_xlabel("time in window (ms)")
    fig.tight_layout()
    fig.savefig(output_dir / f"cluster_examples_k{best_k}.png", dpi=180)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Discover combustion states with sliding-window features and KMeans clustering."
    )
    parser.add_argument("--csvs", nargs="+", required=True, help="Input combustion pressure/qdot CSV files.")
    parser.add_argument("--output-dir", default="./outputs/combustion_state_clustering")
    parser.add_argument("--pressure-cols", default="auto", help="Comma-separated pressure columns, auto, or none.")
    parser.add_argument("--qdot-cols", default="auto", help="Comma-separated qdot columns, auto, or none.")
    parser.add_argument("--window-config", default=None, help="Optional per-file time window JSON.")
    parser.add_argument("--time-start", type=float, default=None, help="Global crop start in seconds.")
    parser.add_argument("--time-end", type=float, default=None, help="Global crop end in seconds.")
    parser.add_argument("--window-size", type=int, default=8192, help="Sliding window length in samples.")
    parser.add_argument("--stride", type=int, default=1024, help="Sliding window stride in samples.")
    parser.add_argument("--max-windows-per-file", type=int, default=2500)
    parser.add_argument("--sample-rate", type=float, default=None, help="Override sample rate in Hz.")
    parser.add_argument("--min-frequency-hz", type=float, default=20.0)
    parser.add_argument("--max-frequency-hz", type=float, default=None)
    parser.add_argument(
        "--band-ranges",
        default="50-500,500-2000,2000-8000",
        help="Comma-separated spectral bands in Hz, e.g. 50-500,500-2000.",
    )
    parser.add_argument("--k-values", default="2-5", help="K values, e.g. 2-5 or 2,3,4.")
    parser.add_argument("--random-state", type=int, default=2026)
    parser.add_argument("--metric-sample-size", type=int, default=5000)
    parser.add_argument("--examples-per-cluster", type=int, default=2)
    return parser


def _parse_band_ranges(raw: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        left, right = part.split("-", 1)
        low = float(left)
        high = float(right)
        if high > low >= 0:
            bands.append((low, high))
    return bands


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if pd is None:
        raise RuntimeError("This script requires pandas. Install it in the training environment first.")

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        raise RuntimeError(
            "This script requires scikit-learn. Install it in the training environment first."
        ) from exc

    window_config = _load_window_config(args.window_config)
    k_values = _parse_k_values(args.k_values)
    band_ranges = _parse_band_ranges(args.band_ranges)

    label_rows: list[dict] = []
    feature_rows: list[dict[str, float]] = []
    pressure_cols_by_source: dict[str, list[str]] = {}
    qdot_cols_by_source: dict[str, list[str]] = {}
    resolved_windows: dict[str, dict[str, float | None]] = {}

    for csv_name in args.csvs:
        csv_path = Path(csv_name)
        header = _read_header(csv_path)
        p_cols = _parse_cols(args.pressure_cols, header, "p")
        q_cols = _parse_cols(args.qdot_cols, header, "qdot")
        if not p_cols:
            raise ValueError(f"No pressure columns selected for {csv_path}")
        usecols = ["time"] + p_cols + q_cols if "time" in header else p_cols + q_cols
        df = pd.read_csv(csv_path, usecols=usecols)
        t_start, t_end = _resolve_time_window(csv_path, args.time_start, args.time_end, window_config)
        df = _crop_df(df, t_start, t_end)
        time = df["time"].to_numpy(dtype=np.float64) if "time" in df.columns else np.arange(len(df), dtype=np.float64)
        fs = _estimate_sample_rate(time, args.sample_rate)
        p_values = df[p_cols].to_numpy(dtype=np.float64)
        q_values = df[q_cols].to_numpy(dtype=np.float64) if q_cols else None

        source = str(csv_path)
        pressure_cols_by_source[source] = p_cols
        qdot_cols_by_source[source] = q_cols
        resolved_windows[source] = {"time_start": t_start, "time_end": t_end, "sample_rate_hz": fs}

        for start in _iter_starts(len(df), args.window_size, args.stride, args.max_windows_per_file):
            end = start + args.window_size
            feats = _extract_window_features(
                p_values[start:end],
                None if q_values is None else q_values[start:end],
                fs=fs,
                min_freq=args.min_frequency_hz,
                max_freq=args.max_frequency_hz,
                band_ranges=band_ranges,
            )
            label_rows.append(
                {
                    "source_csv": source,
                    "record_id": csv_path.stem,
                    "start_index": start,
                    "end_index": end,
                    "time_start": float(time[start]),
                    "time_end": float(time[end - 1]),
                }
            )
            feature_rows.append(feats)

    if len(feature_rows) < max(3, max(k_values)):
        raise ValueError(f"Not enough windows for clustering: {len(feature_rows)}")

    feature_names = sorted(feature_rows[0].keys())
    x = np.asarray([[row[name] for name in feature_names] for row in feature_rows], dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    scores: list[dict] = []
    labels_by_k: dict[int, np.ndarray] = {}
    models: dict[int, KMeans] = {}
    rng = np.random.default_rng(args.random_state)
    metric_idx = np.arange(x_scaled.shape[0])
    if args.metric_sample_size and x_scaled.shape[0] > args.metric_sample_size:
        metric_idx = rng.choice(x_scaled.shape[0], size=args.metric_sample_size, replace=False)

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=args.random_state, n_init=20)
        labels = model.fit_predict(x_scaled)
        labels_by_k[k] = labels
        models[k] = model
        if len(set(labels[metric_idx])) > 1:
            sil = float(silhouette_score(x_scaled[metric_idx], labels[metric_idx]))
        else:
            sil = 0.0
        scores.append(
            {
                "k": k,
                "n_windows": int(x_scaled.shape[0]),
                "silhouette": sil,
                "calinski_harabasz": float(calinski_harabasz_score(x_scaled, labels)),
                "davies_bouldin": float(davies_bouldin_score(x_scaled, labels)),
            }
        )

    best_k = max(scores, key=lambda row: (float(row["silhouette"]), -float(row["davies_bouldin"])))["k"]
    best_k = int(best_k)
    best_labels = labels_by_k[best_k]
    best_model = models[best_k]

    _write_csv(output_dir / "cluster_scores.csv", scores)

    feature_table_rows: list[dict] = []
    for base, feats, label in zip(label_rows, feature_rows, best_labels, strict=True):
        row = dict(base)
        row["cluster"] = int(label)
        row.update(feats)
        feature_table_rows.append(row)
    _write_csv(output_dir / f"window_clusters_k{best_k}.csv", feature_table_rows)

    for k, labels in labels_by_k.items():
        rows = []
        for base, label in zip(label_rows, labels, strict=True):
            row = dict(base)
            row["cluster"] = int(label)
            rows.append(row)
        _write_csv(output_dir / f"window_labels_k{k}.csv", rows)

    summary_rows: list[dict] = []
    for cid in sorted(set(int(x) for x in best_labels)):
        idx = np.where(best_labels == cid)[0]
        row: dict[str, float | int] = {"cluster": cid, "n_windows": int(idx.size)}
        for name_idx, name in enumerate(feature_names):
            vals = x[idx, name_idx]
            row[f"{name}_mean"] = float(np.mean(vals))
            row[f"{name}_std"] = float(np.std(vals))
        summary_rows.append(row)
    _write_csv(output_dir / f"cluster_summary_k{best_k}.csv", summary_rows)

    config = {
        "csvs": [str(Path(p).resolve()) for p in args.csvs],
        "output_dir": str(output_dir.resolve()),
        "pressure_cols": args.pressure_cols,
        "qdot_cols": args.qdot_cols,
        "resolved_windows": resolved_windows,
        "window_size": args.window_size,
        "stride": args.stride,
        "max_windows_per_file": args.max_windows_per_file,
        "k_values": k_values,
        "best_k": best_k,
        "feature_names": feature_names,
        "scores": scores,
    }
    (output_dir / "cluster_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_plots(output_dir, x_scaled, best_labels, scores, feature_names, best_k, args.random_state)
    _save_example_plot(
        output_dir=output_dir,
        label_rows=label_rows,
        feature_matrix=x_scaled,
        labels=best_labels,
        centers=best_model.cluster_centers_,
        pressure_cols=pressure_cols_by_source,
        qdot_cols=qdot_cols_by_source,
        best_k=best_k,
        examples_per_cluster=args.examples_per_cluster,
    )

    print(f"Windows: {len(label_rows)}")
    print(f"Best K: {best_k}")
    print(f"Scores: {output_dir / 'cluster_scores.csv'}")
    print(f"Labels: {output_dir / f'window_clusters_k{best_k}.csv'}")
    print(f"Summary: {output_dir / f'cluster_summary_k{best_k}.csv'}")
    print(f"Config: {output_dir / 'cluster_config.json'}")


if __name__ == "__main__":
    main()
