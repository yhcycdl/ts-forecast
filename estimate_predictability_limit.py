#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _optional_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_builtin(obj):
    if isinstance(obj, dict):
        return {k: _as_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _positive_int(value: int, fallback: int) -> int:
    value = int(value)
    return value if value > 0 else int(fallback)


def _prefix_horizon(mask: Iterable[bool]) -> int:
    horizon = 0
    for i, ok in enumerate(mask, start=1):
        if not ok:
            break
        horizon = i
    return horizon


def _next_pow_two(n: int) -> int:
    n = int(n)
    if n <= 1:
        return 1
    return 1 << int(math.ceil(math.log2(n)))


def _load_series(args) -> Tuple[np.ndarray, Dict[str, object]]:
    if bool(int(args.demo_logistic)):
        n_total = int(args.demo_points)
        x = np.empty(n_total, dtype=np.float64)
        x[0] = float(args.demo_x0)
        r = float(args.demo_r)
        for i in range(1, n_total):
            x[i] = r * x[i - 1] * (1.0 - x[i - 1])
        return x, {
            "source": "demo_logistic",
            "column": "demo_logistic",
            "original_length": int(n_total),
        }

    csv_path = args.data_path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.root_path, csv_path)

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"No header found in CSV: {csv_path}")

        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"CSV contains no data rows: {csv_path}")

        if args.column is not None:
            if args.column not in fieldnames:
                raise ValueError(f"column '{args.column}' not found in CSV.")
            column = args.column
        elif args.target in fieldnames:
            column = args.target
        else:
            column = None
            for name in fieldnames:
                try:
                    float(first_row[name])
                    column = name
                    break
                except Exception:
                    continue
            if column is None:
                raise ValueError("No numeric column found in input CSV.")

        values: List[float] = []

        def _push(row_value):
            try:
                v = float(row_value)
            except Exception:
                return
            if math.isfinite(v):
                values.append(v)

        _push(first_row[column])
        row_count = 1
        for row in reader:
            if args.max_rows is not None and row_count >= int(args.max_rows):
                break
            _push(row[column])
            row_count += 1

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Selected series is empty after removing NaN/Inf.")
    loaded_length = int(values.size)

    split = str(args.split).lower()
    if split != "all":
        values_2d = values[:, None]
        train_raw, val_raw, test_raw = _time_split(
            values_2d, args.train_ratio, args.val_ratio, args.split_mode
        )
        split_map = {
            "train": train_raw,
            "val": val_raw,
            "test": test_raw,
        }
        picked = split_map.get(split)
        if picked is None or len(picked) == 0:
            raise ValueError(f"Split '{split}' is empty.")
        values = picked.reshape(-1)

    if int(args.start_index) > 0:
        values = values[int(args.start_index) :]
    if int(args.end_index) > 0:
        values = values[: int(args.end_index)]
    if int(args.downsample) > 1:
        values = values[:: int(args.downsample)]
    if int(args.max_points) > 0 and len(values) > int(args.max_points):
        values = values[: int(args.max_points)]

    if len(values) < 100:
        raise ValueError(f"Series is too short for predictability analysis. Need >= 100 points, got {len(values)}.")

    return values, {
        "source": csv_path,
        "column": column,
        "loaded_length_before_split": loaded_length,
    }


def _time_split(values: np.ndarray, train_ratio: float, val_ratio: float, split_mode: str):
    values = np.asarray(values)
    total = len(values)
    if total <= 0:
        raise ValueError("values must contain at least one row.")
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")

    split_mode = str(split_mode).lower()
    if split_mode == "legacy_rest":
        tr_end = int(total * train_ratio)
        train = values[:tr_end]
        rest = values[tr_end:]
        if val_ratio and val_ratio > 0:
            val_end = int(len(rest) * val_ratio)
            return train, rest[:val_end], rest[val_end:]
        return train, None, rest

    if float(train_ratio) + float(val_ratio) >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1 when split_mode='total'.")

    tr_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    train = values[:tr_end]
    val = values[tr_end:val_end] if val_ratio and val_ratio > 0 else None
    test = values[val_end:]
    return train, val, test


def _normalize_series(values: np.ndarray, mode: str) -> Tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(np.mean(values))
    std = float(np.std(values))
    std = std if std > 1e-12 else 1.0
    mode = str(mode).lower()

    if mode == "none":
        return values.copy(), mean, std
    if mode == "center":
        return values - mean, mean, std
    if mode == "zscore":
        return (values - mean) / std, mean, std
    raise ValueError(f"Unknown normalize mode: {mode}")


def _acf_fft(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    if len(x) < 2:
        raise ValueError("Need at least 2 points for ACF.")
    max_lag = min(int(max_lag), len(x) - 1)
    nfft = _next_pow_two(2 * len(x) - 1)
    spec = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(spec * np.conjugate(spec), n=nfft)[: max_lag + 1]
    if acf[0] == 0:
        return np.ones(max_lag + 1, dtype=np.float64)
    acf = acf / acf[0]
    return acf


def _dominant_period_fft(x: np.ndarray, effective_sample_rate: float) -> Dict[str, Optional[float]]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    if len(x) < 8:
        return {
            "dominant_frequency_hz": None,
            "dominant_period_steps": None,
            "dominant_period_seconds": None,
        }

    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / effective_sample_rate)
    if len(spec) <= 1:
        return {
            "dominant_frequency_hz": None,
            "dominant_period_steps": None,
            "dominant_period_seconds": None,
        }

    spec[0] = 0.0
    peak = int(np.argmax(spec))
    if peak <= 0 or freqs[peak] <= 0:
        return {
            "dominant_frequency_hz": None,
            "dominant_period_steps": None,
            "dominant_period_seconds": None,
        }

    dominant_frequency_hz = float(freqs[peak])
    dt_sec = 1.0 / effective_sample_rate
    dominant_period_seconds = 1.0 / dominant_frequency_hz
    dominant_period_steps = dominant_period_seconds / dt_sec
    return {
        "dominant_frequency_hz": dominant_frequency_hz,
        "dominant_period_steps": float(dominant_period_steps),
        "dominant_period_seconds": float(dominant_period_seconds),
    }


def _choose_delay(acf: np.ndarray, dominant_period_steps: Optional[float]) -> Dict[str, Optional[int]]:
    tau_1e = None
    tau_zero = None

    for lag in range(1, len(acf)):
        if tau_1e is None and acf[lag] <= math.exp(-1.0):
            tau_1e = lag
        if tau_zero is None and acf[lag] <= 0.0:
            tau_zero = lag
        if tau_1e is not None and tau_zero is not None:
            break

    quarter_period = None
    if dominant_period_steps is not None and dominant_period_steps > 0:
        quarter_period = max(1, int(round(float(dominant_period_steps) / 4.0)))

    if tau_1e is not None:
        tau = tau_1e
    elif tau_zero is not None:
        tau = tau_zero
    elif quarter_period is not None:
        tau = quarter_period
    else:
        tau = 1

    if quarter_period is not None:
        tau = min(tau, quarter_period)
    tau = max(1, int(tau))

    return {
        "tau_selected": int(tau),
        "tau_acf_1_over_e": None if tau_1e is None else int(tau_1e),
        "tau_first_zero_crossing": None if tau_zero is None else int(tau_zero),
        "tau_quarter_period_cap": None if quarter_period is None else int(quarter_period),
    }


def _delay_embed(
    x: np.ndarray,
    emb_dim: int,
    tau: int,
    max_future: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    emb_dim = int(emb_dim)
    tau = int(tau)
    max_future = int(max_future)
    n = len(x) - (emb_dim - 1) * tau - max_future
    if n <= 1:
        raise ValueError(
            f"Not enough samples for embedding: len={len(x)}, emb_dim={emb_dim}, tau={tau}, max_future={max_future}."
        )
    starts = np.arange(n, dtype=np.int64)
    offsets = np.arange(emb_dim, dtype=np.int64) * tau
    emb = x[starts[:, None] + offsets[None, :]]
    current_idx = starts + (emb_dim - 1) * tau
    return emb, starts, current_idx


def _subsample_indices(length: int, max_points: int) -> np.ndarray:
    length = int(length)
    if max_points <= 0 or length <= max_points:
        return np.arange(length, dtype=np.int64)
    return np.unique(np.linspace(0, length - 1, max_points, dtype=np.int64))


def _nearest_neighbor_indices(
    ref_emb: np.ndarray,
    cand_emb: np.ndarray,
    ref_starts: np.ndarray,
    cand_starts: np.ndarray,
    theiler_window: int,
    chunk_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ref = ref_emb.shape[0]
    best_idx = np.full(n_ref, -1, dtype=np.int64)
    best_d2 = np.full(n_ref, np.inf, dtype=np.float64)
    cand_norm = np.sum(cand_emb * cand_emb, axis=1)
    theiler_window = int(theiler_window)
    chunk_size = max(1, int(chunk_size))

    for s in range(0, n_ref, chunk_size):
        e = min(n_ref, s + chunk_size)
        block = ref_emb[s:e]
        block_norm = np.sum(block * block, axis=1, keepdims=True)
        d2 = block_norm + cand_norm[None, :] - 2.0 * (block @ cand_emb.T)
        np.maximum(d2, 0.0, out=d2)
        invalid = np.abs(ref_starts[s:e, None] - cand_starts[None, :]) <= theiler_window
        d2[invalid] = np.inf
        idx = np.argmin(d2, axis=1)
        best_idx[s:e] = idx
        best_d2[s:e] = d2[np.arange(e - s), idx]

    valid = np.isfinite(best_d2) & (best_idx >= 0)
    return best_idx, best_d2, valid


def _estimate_fnn(
    x: np.ndarray,
    tau: int,
    max_dim: int,
    fnn_threshold: float,
    theiler_window: int,
    max_refs: int,
    max_candidates: int,
    chunk_size: int,
    rtol: float,
    atol: float,
) -> Dict[str, object]:
    sigma = float(np.std(x))
    sigma = sigma if sigma > 1e-12 else 1.0
    ratios: List[Dict[str, float]] = []
    selected_dim = int(max_dim)
    eps = 1e-12

    for dim in range(1, int(max_dim)):
        n = len(x) - dim * int(tau)
        if n <= 8:
            break

        starts = np.arange(n, dtype=np.int64)
        offsets = np.arange(dim, dtype=np.int64) * int(tau)
        emb = x[starts[:, None] + offsets[None, :]]

        ref_sel = _subsample_indices(n, max_refs)
        cand_sel = _subsample_indices(n, max_candidates)

        ref_emb = emb[ref_sel]
        cand_emb = emb[cand_sel]
        ref_starts = starts[ref_sel]
        cand_starts = starts[cand_sel]

        nn_pos, d2, valid = _nearest_neighbor_indices(
            ref_emb, cand_emb, ref_starts, cand_starts, theiler_window, chunk_size
        )
        if not np.any(valid):
            continue

        ref_valid_starts = ref_starts[valid]
        nn_valid_starts = cand_starts[nn_pos[valid]]
        base_dist = np.sqrt(np.maximum(d2[valid], eps))
        extra_ref = x[ref_valid_starts + dim * int(tau)]
        extra_nn = x[nn_valid_starts + dim * int(tau)]
        extra_diff = np.abs(extra_ref - extra_nn)
        dist_next = np.sqrt(base_dist * base_dist + extra_diff * extra_diff)

        false_neighbor = (extra_diff / base_dist > rtol) | (dist_next / sigma > atol)
        ratio = float(np.mean(false_neighbor))
        ratios.append({"embedding_dim": int(dim), "fnn_ratio": ratio})

        if ratio <= float(fnn_threshold):
            selected_dim = int(dim)
            break

    return {
        "selected_embedding_dim": int(selected_dim),
        "fnn_curve": ratios,
    }


def _estimate_lyapunov(
    x: np.ndarray,
    emb_dim: int,
    tau: int,
    max_horizon: int,
    theiler_window: int,
    max_refs: int,
    max_candidates: int,
    chunk_size: int,
    fit_start: int,
    fit_end: int,
    dt_seconds: float,
) -> Dict[str, object]:
    emb_full, starts_full, _ = _delay_embed(x, emb_dim, tau, max_future=0)
    usable = len(starts_full) - int(max_horizon)
    if usable <= 8:
        raise ValueError("Series is too short for Lyapunov estimation at the requested horizon.")

    emb = emb_full[:usable]
    starts = starts_full[:usable]
    ref_sel = _subsample_indices(usable, max_refs)
    cand_sel = _subsample_indices(usable, max_candidates)

    ref_emb = emb[ref_sel]
    cand_emb = emb[cand_sel]
    ref_starts = starts[ref_sel]
    cand_starts = starts[cand_sel]

    nn_pos, d2, valid = _nearest_neighbor_indices(
        ref_emb, cand_emb, ref_starts, cand_starts, theiler_window, chunk_size
    )
    if not np.any(valid):
        raise ValueError("No valid neighbors found for Lyapunov estimation. Increase data length or relax Theiler window.")

    ref_valid = ref_starts[valid]
    nn_valid = cand_starts[nn_pos[valid]]
    eps = 1e-12
    divergence = np.empty(int(max_horizon) + 1, dtype=np.float64)

    for k in range(int(max_horizon) + 1):
        diff = emb_full[ref_valid + k] - emb_full[nn_valid + k]
        dist = np.linalg.norm(diff, axis=1)
        divergence[k] = float(np.mean(np.log(np.maximum(dist, eps))))

    fit_start = _positive_int(fit_start, 1)
    if fit_end <= 0:
        fit_end = min(int(max_horizon), max(fit_start + 2, min(int(max_horizon) // 4, max(8, 2 * int(tau)))))
    fit_end = max(fit_start + 1, min(int(max_horizon), int(fit_end)))

    fit_x = np.arange(fit_start, fit_end + 1, dtype=np.float64)
    fit_y = divergence[fit_start : fit_end + 1]
    slope, intercept = np.polyfit(fit_x, fit_y, deg=1)
    lyap_per_step = float(slope)
    lyap_per_second = lyap_per_step / dt_seconds
    lyap_time_steps = float("inf") if lyap_per_step <= 0 else 1.0 / lyap_per_step
    lyap_time_seconds = float("inf") if lyap_per_second <= 0 else 1.0 / lyap_per_second

    fitted = intercept + slope * np.arange(len(divergence), dtype=np.float64)
    return {
        "divergence_curve": divergence,
        "fit_curve": fitted,
        "fit_start": int(fit_start),
        "fit_end": int(fit_end),
        "lambda_max_per_step": lyap_per_step,
        "lambda_max_per_second": lyap_per_second,
        "lyapunov_time_steps": lyap_time_steps,
        "lyapunov_time_seconds": lyap_time_seconds,
        "num_neighbor_pairs": int(len(ref_valid)),
    }


def _analog_forecast_curve(
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
    emb_dim: int,
    tau: int,
    max_horizon: int,
    theiler_window: int,
    max_refs: int,
    max_candidates: int,
    chunk_size: int,
) -> Dict[str, object]:
    emb, starts, current_idx = _delay_embed(x_scaled, emb_dim, tau, max_future=max_horizon)
    ref_sel = _subsample_indices(len(starts), max_refs)
    cand_sel = _subsample_indices(len(starts), max_candidates)

    ref_emb = emb[ref_sel]
    cand_emb = emb[cand_sel]
    ref_starts = starts[ref_sel]
    cand_starts = starts[cand_sel]

    nn_pos, _, valid = _nearest_neighbor_indices(
        ref_emb, cand_emb, ref_starts, cand_starts, theiler_window, chunk_size
    )
    if not np.any(valid):
        raise ValueError("No valid neighbors found for analog forecast analysis.")

    ref_current = current_idx[ref_sel][valid]
    nn_current = current_idx[cand_sel][nn_pos[valid]]

    scaled_std = float(np.std(x_scaled))
    scaled_std = scaled_std if scaled_std > 1e-12 else 1.0
    raw_std = float(np.std(x_raw))
    raw_std = raw_std if raw_std > 1e-12 else 1.0

    horizons = np.arange(1, int(max_horizon) + 1, dtype=np.int64)
    rmse_scaled = np.empty_like(horizons, dtype=np.float64)
    persistence_rmse_scaled = np.empty_like(horizons, dtype=np.float64)
    skill = np.empty_like(horizons, dtype=np.float64)

    for i, h in enumerate(horizons):
        actual = x_scaled[ref_current + h]
        pred = x_scaled[nn_current + h]
        persistence = x_scaled[ref_current]

        rmse_h = float(np.sqrt(np.mean((pred - actual) ** 2)))
        persistence_h = float(np.sqrt(np.mean((persistence - actual) ** 2)))
        rmse_scaled[i] = rmse_h
        persistence_rmse_scaled[i] = persistence_h
        skill[i] = 0.0 if persistence_h <= 1e-12 else 1.0 - rmse_h / persistence_h

    rmse_raw = rmse_scaled * raw_std / scaled_std
    persistence_rmse_raw = persistence_rmse_scaled * raw_std / scaled_std
    nrmse = rmse_raw / raw_std

    return {
        "horizons": horizons,
        "rmse_scaled": rmse_scaled,
        "rmse_raw": rmse_raw,
        "nrmse": nrmse,
        "persistence_rmse_scaled": persistence_rmse_scaled,
        "persistence_rmse_raw": persistence_rmse_raw,
        "skill_vs_persistence": skill,
        "num_neighbor_pairs": int(np.sum(valid)),
    }


def _summarize_horizons(curve: Dict[str, object]) -> Dict[str, int]:
    nrmse = np.asarray(curve["nrmse"], dtype=np.float64)
    skill = np.asarray(curve["skill_vs_persistence"], dtype=np.float64)
    return {
        "horizon_nrmse_le_0_10": _prefix_horizon(nrmse <= 0.10),
        "horizon_nrmse_le_0_20": _prefix_horizon(nrmse <= 0.20),
        "horizon_nrmse_le_0_50": _prefix_horizon(nrmse <= 0.50),
        "horizon_nrmse_le_1_00": _prefix_horizon(nrmse <= 1.00),
        "horizon_skill_ge_0_20": _prefix_horizon(skill >= 0.20),
        "horizon_skill_ge_0_00": _prefix_horizon(skill >= 0.00),
    }


def _write_curve_csv(path: str, curve: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "horizon",
                "rmse_scaled",
                "rmse_raw",
                "nrmse",
                "persistence_rmse_scaled",
                "persistence_rmse_raw",
                "skill_vs_persistence",
            ]
        )
        rows = zip(
            curve["horizons"],
            curve["rmse_scaled"],
            curve["rmse_raw"],
            curve["nrmse"],
            curve["persistence_rmse_scaled"],
            curve["persistence_rmse_raw"],
            curve["skill_vs_persistence"],
        )
        for row in rows:
            writer.writerow([float(v) if i > 0 else int(v) for i, v in enumerate(row)])


def _plot_outputs(
    output_dir: str,
    acf: np.ndarray,
    delay_info: Dict[str, Optional[int]],
    fnn_info: Dict[str, object],
    lyap_info: Dict[str, object],
    analog_info: Dict[str, object],
    dt_seconds: float,
) -> List[str]:
    plt = _optional_matplotlib()
    if plt is None:
        return []

    saved = []

    fig, ax = plt.subplots(figsize=(8, 4.5))
    lags = np.arange(len(acf))
    ax.plot(lags, acf, lw=1.8)
    tau = int(delay_info["tau_selected"])
    ax.axvline(tau, color="tab:red", ls="--", lw=1.2, label=f"selected tau={tau}")
    ax.axhline(math.exp(-1.0), color="tab:green", ls=":", lw=1.0, label="1/e")
    ax.set_title("Autocorrelation and Suggested Delay")
    ax.set_xlabel("Lag (steps)")
    ax.set_ylabel("ACF")
    ax.grid(alpha=0.25)
    ax.legend()
    path = os.path.join(output_dir, "acf_delay.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fnn_curve = fnn_info.get("fnn_curve", [])
    if fnn_curve:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        dims = [item["embedding_dim"] for item in fnn_curve]
        ratios = [item["fnn_ratio"] for item in fnn_curve]
        ax.plot(dims, ratios, marker="o", lw=1.6)
        ax.axvline(int(fnn_info["selected_embedding_dim"]), color="tab:red", ls="--", lw=1.2)
        ax.set_title("False Nearest Neighbors")
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("FNN Ratio")
        ax.grid(alpha=0.25)
        path = os.path.join(output_dir, "fnn_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        saved.append(path)

    divergence = np.asarray(lyap_info["divergence_curve"], dtype=np.float64)
    fit_curve = np.asarray(lyap_info["fit_curve"], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    steps = np.arange(len(divergence))
    time_axis = steps * dt_seconds
    ax.plot(time_axis, divergence, lw=1.8, label="mean log divergence")
    ax.plot(time_axis, fit_curve, ls="--", lw=1.2, label="linear fit")
    ax.axvspan(
        lyap_info["fit_start"] * dt_seconds,
        lyap_info["fit_end"] * dt_seconds,
        color="tab:orange",
        alpha=0.18,
        label="fit range",
    )
    ax.set_title("Lyapunov Divergence (Rosenstein)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean log distance")
    ax.grid(alpha=0.25)
    ax.legend()
    path = os.path.join(output_dir, "lyapunov_divergence.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    horizons = np.asarray(analog_info["horizons"], dtype=np.float64)
    axes[0].plot(horizons, analog_info["nrmse"], lw=1.8, label="analog NRMSE")
    axes[0].axhline(0.2, color="tab:green", ls=":", lw=1.0, label="NRMSE=0.2")
    axes[0].axhline(0.5, color="tab:orange", ls=":", lw=1.0, label="NRMSE=0.5")
    axes[0].axhline(1.0, color="tab:red", ls=":", lw=1.0, label="NRMSE=1.0")
    axes[0].set_ylabel("NRMSE")
    axes[0].set_title("State-Space Analog Forecast Curve")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(horizons, analog_info["skill_vs_persistence"], lw=1.8, label="skill vs persistence")
    axes[1].axhline(0.0, color="tab:red", ls=":", lw=1.0, label="skill=0")
    axes[1].axhline(0.2, color="tab:green", ls=":", lw=1.0, label="skill=0.2")
    axes[1].set_xlabel("Forecast Horizon (steps)")
    axes[1].set_ylabel("Skill")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    path = os.path.join(output_dir, "analog_forecast_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    saved.append(path)

    return saved


def build_parser():
    parser = argparse.ArgumentParser(description="Estimate predictability upper bound from a scalar time series.")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--column", type=str, default=None)
    parser.add_argument("--target", type=str, default="value")
    parser.add_argument("--max_rows", type=int, default=10_000_000)
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_mode", type=str, default="total", choices=["total", "legacy_rest"])
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=0)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=50000)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "center", "none"])
    parser.add_argument("--max_acf_lag", type=int, default=2048)
    parser.add_argument("--tau", type=int, default=0, help="Delay embedding lag. <=0 means auto.")
    parser.add_argument("--max_dim", type=int, default=10)
    parser.add_argument("--fnn_threshold", type=float, default=0.05)
    parser.add_argument("--fnn_rtol", type=float, default=15.0)
    parser.add_argument("--fnn_atol", type=float, default=2.0)
    parser.add_argument("--theiler_window", type=int, default=0, help="Temporal exclusion window. <=0 means auto.")
    parser.add_argument("--max_horizon", type=int, default=256)
    parser.add_argument("--fit_start", type=int, default=1)
    parser.add_argument("--fit_end", type=int, default=0, help="<=0 means auto.")
    parser.add_argument("--max_refs", type=int, default=1500)
    parser.add_argument("--max_candidates", type=int, default=4000)
    parser.add_argument("--distance_chunk", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="./analysis/predictability")
    parser.add_argument("--demo_logistic", type=int, default=0)
    parser.add_argument("--demo_points", type=int, default=8000)
    parser.add_argument("--demo_r", type=float, default=4.0)
    parser.add_argument("--demo_x0", type=float, default=0.123456)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    raw_values, source_info = _load_series(args)
    x_scaled, raw_mean, raw_std = _normalize_series(raw_values, args.normalize)

    effective_sample_rate = float(args.sample_rate) / max(1, int(args.downsample))
    dt_seconds = 1.0 / effective_sample_rate if effective_sample_rate > 0 else 1.0

    period_info = _dominant_period_fft(x_scaled, effective_sample_rate)
    acf = _acf_fft(x_scaled, min(int(args.max_acf_lag), len(x_scaled) - 1))

    delay_info = _choose_delay(acf, period_info["dominant_period_steps"])
    tau = int(args.tau) if int(args.tau) > 0 else int(delay_info["tau_selected"])
    delay_info["tau_selected"] = int(tau)

    theiler_window = int(args.theiler_window)
    if theiler_window <= 0:
        period_steps = period_info["dominant_period_steps"]
        if period_steps is not None and period_steps > 0:
            theiler_window = max(int(round(period_steps)), int(tau) * 2)
        else:
            theiler_window = max(10, int(tau) * 2)

    fnn_info = _estimate_fnn(
        x_scaled,
        tau=tau,
        max_dim=int(args.max_dim),
        fnn_threshold=float(args.fnn_threshold),
        theiler_window=theiler_window,
        max_refs=int(args.max_refs),
        max_candidates=int(args.max_candidates),
        chunk_size=int(args.distance_chunk),
        rtol=float(args.fnn_rtol),
        atol=float(args.fnn_atol),
    )
    emb_dim = int(fnn_info["selected_embedding_dim"])

    lyap_info = _estimate_lyapunov(
        x_scaled,
        emb_dim=emb_dim,
        tau=tau,
        max_horizon=int(args.max_horizon),
        theiler_window=theiler_window,
        max_refs=int(args.max_refs),
        max_candidates=int(args.max_candidates),
        chunk_size=int(args.distance_chunk),
        fit_start=int(args.fit_start),
        fit_end=int(args.fit_end),
        dt_seconds=dt_seconds,
    )

    analog_info = _analog_forecast_curve(
        x_scaled=x_scaled,
        x_raw=raw_values,
        emb_dim=emb_dim,
        tau=tau,
        max_horizon=int(args.max_horizon),
        theiler_window=theiler_window,
        max_refs=int(args.max_refs),
        max_candidates=int(args.max_candidates),
        chunk_size=int(args.distance_chunk),
    )

    horizon_summary = _summarize_horizons(analog_info)
    if period_info["dominant_period_steps"] is not None and lyap_info["lyapunov_time_steps"] != float("inf"):
        lyap_cycles = lyap_info["lyapunov_time_steps"] / float(period_info["dominant_period_steps"])
    else:
        lyap_cycles = None

    summary = {
        "source_info": source_info,
        "analysis_config": {
            "split": args.split,
            "downsample": int(args.downsample),
            "effective_sample_rate": effective_sample_rate,
            "dt_seconds": dt_seconds,
            "normalize": args.normalize,
            "series_length_after_preprocess": int(len(raw_values)),
            "raw_mean": raw_mean,
            "raw_std": raw_std,
            "tau": tau,
            "embedding_dim": emb_dim,
            "theiler_window": int(theiler_window),
            "max_horizon": int(args.max_horizon),
        },
        "period_info": period_info,
        "delay_info": delay_info,
        "fnn_info": fnn_info,
        "lyapunov_info": {
            "lambda_max_per_step": lyap_info["lambda_max_per_step"],
            "lambda_max_per_second": lyap_info["lambda_max_per_second"],
            "lyapunov_time_steps": lyap_info["lyapunov_time_steps"],
            "lyapunov_time_seconds": lyap_info["lyapunov_time_seconds"],
            "lyapunov_time_cycles": lyap_cycles,
            "fit_start": lyap_info["fit_start"],
            "fit_end": lyap_info["fit_end"],
            "num_neighbor_pairs": lyap_info["num_neighbor_pairs"],
        },
        "predictability_horizons": horizon_summary,
        "interpretation": {
            "notes": [
                "Lyapunov time is a local exponential divergence estimate, not an exact forecast limit.",
                "State-space analog forecast is a best-case proxy based on nearest historical analogs, useful as a practical upper-bound surrogate.",
                "If best classification or regression models are already close to the analog curve, the dataset may be near its intrinsic predictability limit.",
            ]
        },
    }

    if args.demo_logistic:
        stem = "demo_logistic"
    else:
        src = os.path.splitext(os.path.basename(str(source_info["source"])))[0]
        stem = f"{src}_{source_info['column']}_{args.split}"
    output_dir = os.path.join(args.output_dir, stem)
    _ensure_dir(output_dir)

    curve_csv_path = os.path.join(output_dir, "analog_forecast_curve.csv")
    _write_curve_csv(curve_csv_path, analog_info)

    summary_path = os.path.join(output_dir, "predictability_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_as_builtin(summary), f, ensure_ascii=False, indent=2)

    plots = _plot_outputs(
        output_dir=output_dir,
        acf=acf,
        delay_info=delay_info,
        fnn_info=fnn_info,
        lyap_info=lyap_info,
        analog_info=analog_info,
        dt_seconds=dt_seconds,
    )

    print("Predictability analysis finished.")
    print(f"Output dir: {output_dir}")
    print(f"Summary JSON: {summary_path}")
    print(f"Analog curve CSV: {curve_csv_path}")
    if plots:
        for path in plots:
            print(f"Plot: {path}")

    print(
        "Key results | "
        f"tau={tau} | emb_dim={emb_dim} | "
        f"lambda_max/step={lyap_info['lambda_max_per_step']:.6f} | "
        f"lyapunov_time_steps={lyap_info['lyapunov_time_steps']:.3f} | "
        f"horizon_skill_ge_0={horizon_summary['horizon_skill_ge_0_00']} | "
        f"horizon_nrmse_le_0.5={horizon_summary['horizon_nrmse_le_0_50']}"
    )


if __name__ == "__main__":
    main()
