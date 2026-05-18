from __future__ import annotations

import numpy as np

from src.features.spectral import EPS, FrequencyBand, as_1d, infer_frequency_band


def _validate_pair(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    x_arr = as_1d(x)
    y_arr = as_1d(y)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"signals must have the same shape, got {x_arr.shape} and {y_arr.shape}.")
    return x_arr, y_arr


def band_limited_coherence(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    sample_rate: float,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    half_band_bins: int = 2,
    min_hz: float = 1e-6,
) -> tuple[float, FrequencyBand]:
    x_arr, y_arr = _validate_pair(x, y)
    if x_arr.size < 4 or sample_rate <= 0:
        return 0.0, FrequencyBand(0.0, 0.0, 0.0, "degenerate")

    band = infer_frequency_band(
        x_arr,
        sample_rate=sample_rate,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        half_band_bins=half_band_bins,
        min_hz=min_hz,
    )
    x_spec = np.fft.rfft(x_arr - float(np.mean(x_arr)))
    y_spec = np.fft.rfft(y_arr - float(np.mean(y_arr)))
    freqs = np.fft.rfftfreq(x_arr.size, d=1.0 / float(sample_rate))
    mask = (freqs >= band.low_hz) & (freqs <= band.high_hz)
    if mask.size:
        mask[0] = False
    if not bool(np.any(mask)):
        return 0.0, band

    cross = x_spec[mask] * np.conjugate(y_spec[mask])
    num = float(np.abs(np.sum(cross)) ** 2)
    den = float(np.sum(np.abs(x_spec[mask]) ** 2) * np.sum(np.abs(y_spec[mask]) ** 2))
    if den <= EPS:
        return 0.0, band
    coherence = min(max(num / den, 0.0), 1.0)
    return float(coherence), band


def phase_difference(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    sample_rate: float,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    half_band_bins: int = 2,
    min_hz: float = 1e-6,
) -> tuple[float, FrequencyBand]:
    x_arr, y_arr = _validate_pair(x, y)
    if x_arr.size < 4 or sample_rate <= 0:
        return 0.0, FrequencyBand(0.0, 0.0, 0.0, "degenerate")

    band = infer_frequency_band(
        x_arr,
        sample_rate=sample_rate,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        half_band_bins=half_band_bins,
        min_hz=min_hz,
    )
    x_spec = np.fft.rfft(x_arr - float(np.mean(x_arr)))
    y_spec = np.fft.rfft(y_arr - float(np.mean(y_arr)))
    freqs = np.fft.rfftfreq(x_arr.size, d=1.0 / float(sample_rate))
    mask = (freqs >= band.low_hz) & (freqs <= band.high_hz)
    if mask.size:
        mask[0] = False
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return 0.0, band
    cross = x_spec[valid] * np.conjugate(y_spec[valid])
    idx = valid[int(np.argmax(np.abs(cross)))]
    return float(np.angle(x_spec[idx] * np.conjugate(y_spec[idx]))), band
