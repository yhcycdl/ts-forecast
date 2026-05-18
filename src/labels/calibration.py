from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

EPS = 1e-6


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(slots=True)
class QuantileCalibrator:
    center: float
    scale: float
    clip_value: float = 6.0

    def transform(self, value: float) -> float:
        z = (float(value) - self.center) / max(float(self.scale), EPS)
        z = float(np.clip(z, -float(self.clip_value), float(self.clip_value)))
        return float(_sigmoid(z))

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def fit_quantile_calibrator(
    values: np.ndarray,
    center_quantile: float = 0.5,
    low_quantile: float = 0.1,
    high_quantile: float = 0.9,
    clip_value: float = 6.0,
) -> QuantileCalibrator:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return QuantileCalibrator(center=0.0, scale=1.0, clip_value=float(clip_value))

    center = float(np.quantile(values, float(center_quantile)))
    low = float(np.quantile(values, float(low_quantile)))
    high = float(np.quantile(values, float(high_quantile)))
    scale = max(high - low, EPS)
    return QuantileCalibrator(center=center, scale=scale, clip_value=float(clip_value))
