from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.features.complexity import cecp_complexity, permutation_entropy, temporal_kurtosis
from src.features.coupling import band_limited_coherence, phase_difference
from src.features.spectral import (
    FrequencyBand,
    band_energy_ratio,
    dominant_frequency,
    envelope_slope,
    peak_to_peak,
    rms,
)


@dataclass(slots=True)
class WindowParams:
    history_length: int
    lead_gap: int
    future_length: int
    stride: int

    def __post_init__(self) -> None:
        self.history_length = int(self.history_length)
        self.lead_gap = int(self.lead_gap)
        self.future_length = int(self.future_length)
        self.stride = int(self.stride)
        if self.history_length <= 0:
            raise ValueError("history_length must be positive.")
        if self.lead_gap < 0:
            raise ValueError("lead_gap must be >= 0.")
        if self.future_length <= 0:
            raise ValueError("future_length must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")


@dataclass(slots=True)
class SignalColumns:
    pressure_column: str = "P1"
    q_column: str = "Q"


def ms_to_samples(milliseconds: float, sample_rate: float) -> int:
    milliseconds = float(milliseconds)
    sample_rate = float(sample_rate)
    if milliseconds < 0:
        raise ValueError(f"milliseconds must be >= 0, got {milliseconds}.")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if milliseconds == 0:
        return 0
    return max(int(round(milliseconds * sample_rate / 1000.0)), 1)


def _seconds(index: int, sample_rate: float) -> float:
    return float(index) / float(sample_rate)


def iter_window_indices(total_length: int, params: WindowParams) -> Iterable[dict[str, int]]:
    total_length = int(total_length)
    window_index = 0
    start = 0
    while True:
        history_start = start
        history_end = history_start + params.history_length
        future_start = history_end + params.lead_gap
        future_end = future_start + params.future_length
        if future_end > total_length:
            break
        yield {
            "window_index": window_index,
            "history_start": history_start,
            "history_end": history_end,
            "future_start": future_start,
            "future_end": future_end,
        }
        window_index += 1
        start += params.stride


def assign_window_split(
    history_start: int,
    future_end: int,
    total_length: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_mode: str = "total",
) -> str:
    split_mode = str(split_mode).lower()
    total_length = int(total_length)
    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)

    if split_mode == "legacy_rest":
        tr_end = int(total_length * train_ratio)
        rest = total_length - tr_end
        val_len = int(rest * val_ratio) if val_ratio > 0 else 0
        val_end = tr_end + val_len
    else:
        if train_ratio + val_ratio >= 1.0:
            raise ValueError("train_ratio + val_ratio must be < 1.0.")
        tr_end = int(total_length * train_ratio)
        val_end = int(total_length * (train_ratio + val_ratio))

    sample_start = int(history_start)
    sample_end = int(future_end)
    if sample_end <= tr_end:
        return "train"
    if sample_start >= tr_end and sample_end <= val_end:
        return "val"
    if sample_start >= val_end and sample_end <= total_length:
        return "test"
    return "cross_boundary"


def _column_index(columns: list[str], column_name: str) -> int:
    try:
        return columns.index(column_name)
    except ValueError as exc:
        raise ValueError(f"column '{column_name}' not found in columns {columns}.") from exc


def _prefixed_indicator_block(
    pressure_signal: np.ndarray,
    q_signal: np.ndarray,
    sample_rate: float,
    prefix: str,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    band_half_bins: int = 2,
    permutation_order: int = 5,
    permutation_delay: int = 1,
) -> dict[str, float]:
    p_band_ratio, band = band_energy_ratio(
        pressure_signal,
        sample_rate=sample_rate,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        half_band_bins=band_half_bins,
    )
    q_band_ratio, _ = band_energy_ratio(
        q_signal,
        sample_rate=sample_rate,
        band_low_hz=band.low_hz,
        band_high_hz=band.high_hz,
        half_band_bins=band_half_bins,
    )
    pq_coh, _ = band_limited_coherence(
        pressure_signal,
        q_signal,
        sample_rate=sample_rate,
        band_low_hz=band.low_hz,
        band_high_hz=band.high_hz,
        half_band_bins=band_half_bins,
    )
    pq_phase, _ = phase_difference(
        pressure_signal,
        q_signal,
        sample_rate=sample_rate,
        band_low_hz=band.low_hz,
        band_high_hz=band.high_hz,
        half_band_bins=band_half_bins,
    )

    return {
        f"{prefix}p_rms": rms(pressure_signal),
        f"{prefix}p_band_energy_ratio": p_band_ratio,
        f"{prefix}p_dom_freq_hz": dominant_frequency(pressure_signal, sample_rate),
        f"{prefix}q_rms": rms(q_signal),
        f"{prefix}q_band_energy_ratio": q_band_ratio,
        f"{prefix}p_env_slope": envelope_slope(pressure_signal, sample_rate),
        f"{prefix}pq_coherence": pq_coh,
        f"{prefix}pq_phase_diff_rad": pq_phase,
        f"{prefix}p_peak_to_peak": peak_to_peak(pressure_signal),
        f"{prefix}temporal_kurtosis": temporal_kurtosis(pressure_signal),
        f"{prefix}permutation_entropy": permutation_entropy(
            pressure_signal,
            order=permutation_order,
            delay=permutation_delay,
            normalize=True,
        ),
        f"{prefix}cecp_complexity": cecp_complexity(
            pressure_signal,
            order=permutation_order,
            delay=permutation_delay,
        ),
        f"{prefix}band_low_hz": band.low_hz,
        f"{prefix}band_high_hz": band.high_hz,
        f"{prefix}band_center_hz": band.center_hz,
        f"{prefix}band_source": band.source,
    }


def extract_indicator_rows(
    values: np.ndarray,
    columns: list[str],
    sample_rate: float,
    params: WindowParams,
    run_id: str,
    condition_id: str,
    signal_columns: SignalColumns | None = None,
    condition_metadata: dict[str, Any] | None = None,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    band_half_bins: int = 2,
    permutation_order: int = 5,
    permutation_delay: int = 1,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    split_mode: str = "total",
) -> list[dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"values must be a 2D array, got shape {matrix.shape}.")

    signal_columns = signal_columns or SignalColumns()
    pressure_idx = _column_index(columns, signal_columns.pressure_column)
    q_idx = _column_index(columns, signal_columns.q_column)
    total_length = matrix.shape[0]
    rows: list[dict[str, Any]] = []
    condition_metadata = dict(condition_metadata or {})

    for window in iter_window_indices(total_length=total_length, params=params):
        history_slice = slice(window["history_start"], window["history_end"])
        future_slice = slice(window["future_start"], window["future_end"])

        history_p = matrix[history_slice, pressure_idx]
        history_q = matrix[history_slice, q_idx]
        future_p = matrix[future_slice, pressure_idx]
        future_q = matrix[future_slice, q_idx]

        split = assign_window_split(
            history_start=window["history_start"],
            future_end=window["future_end"],
            total_length=total_length,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            split_mode=split_mode,
        )

        row: dict[str, Any] = {
            "run_id": run_id,
            "condition_id": condition_id,
            "window_index": int(window["window_index"]),
            "history_start": int(window["history_start"]),
            "history_end": int(window["history_end"]),
            "future_start": int(window["future_start"]),
            "future_end": int(window["future_end"]),
            "history_start_sec": _seconds(window["history_start"], sample_rate),
            "history_end_sec": _seconds(window["history_end"], sample_rate),
            "future_start_sec": _seconds(window["future_start"], sample_rate),
            "future_end_sec": _seconds(window["future_end"], sample_rate),
            "history_length": params.history_length,
            "lead_gap": params.lead_gap,
            "future_length": params.future_length,
            "stride": params.stride,
            "sample_rate_hz": float(sample_rate),
            "pressure_column": signal_columns.pressure_column,
            "q_column": signal_columns.q_column,
            "split": split,
        }
        row.update(condition_metadata)
        row.update(
            _prefixed_indicator_block(
                history_p,
                history_q,
                sample_rate=sample_rate,
                prefix="hist_",
                band_low_hz=band_low_hz,
                band_high_hz=band_high_hz,
                band_half_bins=band_half_bins,
                permutation_order=permutation_order,
                permutation_delay=permutation_delay,
            )
        )
        row.update(
            _prefixed_indicator_block(
                future_p,
                future_q,
                sample_rate=sample_rate,
                prefix="future_",
                band_low_hz=band_low_hz,
                band_high_hz=band_high_hz,
                band_half_bins=band_half_bins,
                permutation_order=permutation_order,
                permutation_delay=permutation_delay,
            )
        )
        rows.append(row)
    return rows
