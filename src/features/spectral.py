from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(slots=True)
class FrequencyBand:
    low_hz: float
    high_hz: float
    center_hz: float
    source: str


def as_1d(signal: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(signal, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("signal must contain at least one value.")
    return arr


def rms(signal: np.ndarray | list[float]) -> float:
    arr = as_1d(signal)
    return float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))


def peak_to_peak(signal: np.ndarray | list[float]) -> float:
    arr = as_1d(signal)
    return float(np.max(arr) - np.min(arr))


def fft_power_spectrum(signal: np.ndarray | list[float], sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    arr = as_1d(signal)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    centered = arr - float(np.mean(arr))
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(arr.size, d=1.0 / float(sample_rate))
    return freqs, power


def dominant_frequency(
    signal: np.ndarray | list[float],
    sample_rate: float,
    min_hz: float = 1e-6,
) -> float:
    freqs, power = fft_power_spectrum(signal, sample_rate)
    if power.size <= 1:
        return 0.0
    mask = freqs >= float(min_hz)
    if mask.size:
        mask[0] = False
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return 0.0
    idx = valid[int(np.argmax(power[valid]))]
    return float(freqs[idx])


def infer_frequency_band(
    signal: np.ndarray | list[float],
    sample_rate: float,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    half_band_bins: int = 2,
    min_hz: float = 1e-6,
) -> FrequencyBand:
    if band_low_hz is not None and band_high_hz is not None and band_high_hz > band_low_hz:
        center = 0.5 * (float(band_low_hz) + float(band_high_hz))
        return FrequencyBand(
            low_hz=float(band_low_hz),
            high_hz=float(band_high_hz),
            center_hz=center,
            source="explicit",
        )

    freqs, power = fft_power_spectrum(signal, sample_rate)
    if power.size <= 1:
        return FrequencyBand(low_hz=0.0, high_hz=0.0, center_hz=0.0, source="degenerate")

    mask = freqs >= float(min_hz)
    if mask.size:
        mask[0] = False
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        return FrequencyBand(low_hz=0.0, high_hz=0.0, center_hz=0.0, source="degenerate")

    dom_idx = valid[int(np.argmax(power[valid]))]
    half_band_bins = max(int(half_band_bins), 1)
    low_idx = max(1, dom_idx - half_band_bins)
    high_idx = min(len(freqs) - 1, dom_idx + half_band_bins)
    return FrequencyBand(
        low_hz=float(freqs[low_idx]),
        high_hz=float(freqs[high_idx]),
        center_hz=float(freqs[dom_idx]),
        source="dominant_bin_neighborhood",
    )


def band_energy_ratio(
    signal: np.ndarray | list[float],
    sample_rate: float,
    band_low_hz: float | None = None,
    band_high_hz: float | None = None,
    half_band_bins: int = 2,
    min_hz: float = 1e-6,
) -> tuple[float, FrequencyBand]:
    arr = as_1d(signal)
    if arr.size < 4 or sample_rate <= 0:
        return 0.0, FrequencyBand(0.0, 0.0, 0.0, "degenerate")

    band = infer_frequency_band(
        arr,
        sample_rate=sample_rate,
        band_low_hz=band_low_hz,
        band_high_hz=band_high_hz,
        half_band_bins=half_band_bins,
        min_hz=min_hz,
    )
    freqs, power = fft_power_spectrum(arr, sample_rate)
    if power.size <= 1:
        return 0.0, band
    total_power = float(np.sum(power[1:]))
    if total_power <= EPS:
        return 0.0, band
    mask = (freqs >= band.low_hz) & (freqs <= band.high_hz)
    if mask.size:
        mask[0] = False
    band_power = float(np.sum(power[mask]))
    return band_power / total_power, band


def analytic_signal(signal: np.ndarray | list[float]) -> np.ndarray:
    arr = as_1d(signal)
    n = arr.size
    spec = np.fft.fft(arr)
    h = np.zeros(n, dtype=np.float64)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spec * h)


def envelope_slope(signal: np.ndarray | list[float], sample_rate: float) -> float:
    arr = as_1d(signal)
    if arr.size < 4 or sample_rate <= 0:
        return 0.0
    envelope = np.abs(analytic_signal(arr))
    time_axis = np.arange(arr.size, dtype=np.float64) / float(sample_rate)
    slope = np.polyfit(time_axis, envelope, deg=1)[0]
    return float(slope)
