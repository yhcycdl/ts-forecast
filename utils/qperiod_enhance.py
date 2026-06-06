from __future__ import annotations

import math

import numpy as np
from scipy import signal


def clean_signal(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if not np.any(finite):
        raise ValueError("Signal has no finite values.")
    fill = float(np.nanmedian(x[finite]))
    return np.where(finite, x, fill)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    x = clean_signal(values)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = 1.4826 * mad if mad > 1e-12 else float(np.std(x))
    if scale <= 1e-12:
        return np.zeros_like(x)
    return (x - med) / scale


def moving_average(values: np.ndarray, window: int, mode: str = "centered") -> np.ndarray:
    x = clean_signal(values)
    window = int(max(1, window))
    if window <= 1 or x.size <= 1:
        return x.copy()

    cumsum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    idx = np.arange(x.size, dtype=np.int64)
    mode = str(mode).lower()
    if mode == "causal":
        ends = idx + 1
        starts = np.maximum(0, ends - window)
    elif mode == "centered":
        left_span = (window - 1) // 2
        right_span = window - left_span
        starts = np.maximum(0, idx - left_span)
        ends = np.minimum(x.size, idx + right_span)
    else:
        raise ValueError(f"Unsupported moving average mode: {mode}")
    counts = np.maximum(1, ends - starts)
    return (cumsum[ends] - cumsum[starts]) / counts


def rolling_rms(values: np.ndarray, window: int, mode: str = "centered") -> np.ndarray:
    return np.sqrt(np.maximum(moving_average(np.square(clean_signal(values)), window, mode), 0.0))


def estimate_dominant_period(values: np.ndarray, fs: float) -> float:
    x = clean_signal(values)
    y = x - np.mean(x)
    if y.size < 8 or np.std(y) <= 1e-12 or fs <= 0:
        return 0.0
    nperseg = min(y.size, 8192)
    freqs, power = signal.welch(y, fs=float(fs), nperseg=nperseg)
    if power.size <= 1:
        return 0.0
    freqs = freqs[1:]
    power = np.maximum(power[1:], 0.0)
    if power.size == 0 or float(np.sum(power)) <= 1e-20:
        return 0.0
    f_dom = float(freqs[int(np.argmax(power))])
    if f_dom <= 0:
        return 0.0
    return float(fs / f_dom)


def analytic_phase(values: np.ndarray) -> np.ndarray:
    x = clean_signal(values)
    if x.size < 4 or float(np.std(x)) <= 1e-12:
        return np.zeros_like(x)
    return np.unwrap(np.angle(signal.hilbert(x - np.mean(x))))


def envelope(values: np.ndarray, smooth_window: int = 1) -> np.ndarray:
    x = clean_signal(values)
    if x.size < 4 or float(np.std(x)) <= 1e-12:
        env = np.zeros_like(x)
    else:
        env = np.abs(signal.hilbert(x - np.mean(x)))
    if smooth_window > 1:
        env = moving_average(env, smooth_window, mode="centered")
    return env


def local_frequency(values: np.ndarray, fs: float, smooth_window: int = 1) -> np.ndarray:
    phase = analytic_phase(values)
    if phase.size <= 1 or fs <= 0:
        return np.zeros_like(phase)
    freq = np.diff(phase, prepend=phase[0]) * float(fs) / (2.0 * math.pi)
    freq = np.maximum(freq, 0.0)
    if smooth_window > 1:
        freq = moving_average(freq, smooth_window, mode="centered")
    return freq


def causal_zero_cross_frequency(values: np.ndarray, fs: float, period_samples: float, smooth_window: int = 1) -> np.ndarray:
    x = clean_signal(values)
    if x.size <= 1 or fs <= 0:
        return np.zeros_like(x)
    trend_window = max(3, int(round(max(period_samples, 1.0) * 2.0)))
    centered = x - moving_average(x, trend_window, mode="causal")
    signs = centered >= 0
    default_freq = float(fs / period_samples) if period_samples > 0 else 0.0
    freq = np.full(x.size, default_freq, dtype=np.float64)
    last_cross = None
    current_freq = default_freq
    for i in range(1, x.size):
        if not signs[i - 1] and signs[i]:
            if last_cross is not None and i > last_cross:
                current_freq = float(fs / max(i - last_cross, 1))
            last_cross = i
        freq[i] = current_freq
    if smooth_window > 1:
        freq = moving_average(freq, smooth_window, mode="causal")
    return freq


def cycle_phase_sin_cos(length: int, period_samples: float) -> tuple[np.ndarray, np.ndarray]:
    length = int(length)
    if length <= 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty
    period = max(float(period_samples), 1.0)
    phase = 2.0 * math.pi * (np.arange(length, dtype=np.float64) / period)
    return np.sin(phase), np.cos(phase)


def phase_sin_cos(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phase = analytic_phase(values)
    return np.sin(phase), np.cos(phase)


def event_skeleton(
    values: np.ndarray,
    period_samples: float,
    prominence_z: float = 0.75,
    distance_frac: float = 0.35,
) -> dict[str, np.ndarray]:
    z = robust_zscore(values)
    n = z.size
    if n == 0:
        empty = np.asarray([], dtype=np.float64)
        return {"mask": empty, "prominence": empty, "width": empty, "proximity": empty}

    distance = max(1, int(round(max(period_samples, 1.0) * float(distance_frac))))
    peaks, props = signal.find_peaks(z, distance=distance, prominence=float(prominence_z))
    mask = np.zeros(n, dtype=np.float64)
    prominence = np.zeros(n, dtype=np.float64)
    width_arr = np.zeros(n, dtype=np.float64)
    proximity = np.zeros(n, dtype=np.float64)

    if peaks.size == 0:
        return {"mask": mask, "prominence": prominence, "width": width_arr, "proximity": proximity}

    widths = signal.peak_widths(z, peaks, rel_height=0.5)[0]
    prom = np.asarray(props.get("prominences", np.ones_like(peaks, dtype=np.float64)), dtype=np.float64)
    for peak, peak_prom, width in zip(peaks, prom, widths):
        peak = int(peak)
        radius = max(1, int(round(width)))
        left = max(0, peak - radius)
        right = min(n, peak + radius + 1)
        mask[peak] = 1.0
        prominence[peak] = float(peak_prom)
        width_arr[peak] = float(width)
        distances = np.abs(np.arange(left, right) - peak) / max(radius, 1)
        proximity[left:right] = np.maximum(proximity[left:right], 1.0 - distances)
    return {"mask": mask, "prominence": prominence, "width": width_arr, "proximity": proximity}


def relative_bands(f_dom: float, fs: float, band_count: int = 3) -> list[tuple[float, float]]:
    nyq = max(float(fs) * 0.5, 1e-12)
    if f_dom <= 0:
        edges = np.linspace(0.02 * nyq, 0.9 * nyq, int(band_count) + 1)
        return [(float(edges[i]), float(edges[i + 1])) for i in range(int(band_count))]

    multipliers = [(0.5, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 10.0), (10.0, 16.0)]
    bands: list[tuple[float, float]] = []
    for lo_mul, hi_mul in multipliers[: int(band_count)]:
        lo = max(1e-6, lo_mul * float(f_dom))
        hi = min(0.98 * nyq, hi_mul * float(f_dom))
        if hi <= lo:
            hi = min(0.98 * nyq, lo * 1.5)
        if hi > lo and lo < nyq:
            bands.append((float(lo), float(hi)))
    while len(bands) < int(band_count):
        bands.append((0.0, 0.0))
    return bands


def bandpass_signal(
    values: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
    zero_phase: bool = False,
) -> np.ndarray:
    x = clean_signal(values)
    nyq = float(fs) * 0.5
    if x.size < 16 or fs <= 0 or low_hz <= 0 or high_hz <= low_hz or low_hz >= nyq:
        return np.zeros_like(x)
    high_hz = min(float(high_hz), 0.98 * nyq)
    if high_hz <= low_hz:
        return np.zeros_like(x)
    sos = signal.butter(int(order), [float(low_hz) / nyq, high_hz / nyq], btype="bandpass", output="sos")
    padlen = min(x.size - 1, max(0, 3 * (2 * sos.shape[0] + 1)))
    if not zero_phase or padlen <= 0:
        return signal.sosfilt(sos, x)
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def band_features(
    values: np.ndarray,
    fs: float,
    period_samples: float,
    band_count: int = 3,
    rms_window: int = 1,
    zero_phase: bool = False,
    rms_mode: str = "causal",
) -> dict[str, np.ndarray]:
    x = clean_signal(values)
    f_dom = float(fs / period_samples) if period_samples > 0 and fs > 0 else 0.0
    out: dict[str, np.ndarray] = {}
    for i, (low, high) in enumerate(relative_bands(f_dom, fs, band_count=band_count)):
        band = bandpass_signal(x, fs, low, high, zero_phase=zero_phase)
        out[f"band{i}"] = band
        out[f"band{i}_rms"] = rolling_rms(band, max(1, int(rms_window)), mode=rms_mode)
    return out
