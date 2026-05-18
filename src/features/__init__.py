"""Feature extraction utilities for the Zenodo combustion rebuild."""

from src.features.complexity import cecp_complexity, permutation_entropy, temporal_kurtosis
from src.features.coupling import band_limited_coherence, phase_difference
from src.features.future_indicator import (
    SignalColumns,
    WindowParams,
    assign_window_split,
    extract_indicator_rows,
    iter_window_indices,
    ms_to_samples,
)
from src.features.spectral import (
    FrequencyBand,
    band_energy_ratio,
    dominant_frequency,
    envelope_slope,
    peak_to_peak,
    rms,
)

__all__ = [
    "FrequencyBand",
    "SignalColumns",
    "WindowParams",
    "assign_window_split",
    "band_energy_ratio",
    "band_limited_coherence",
    "cecp_complexity",
    "dominant_frequency",
    "envelope_slope",
    "extract_indicator_rows",
    "iter_window_indices",
    "ms_to_samples",
    "peak_to_peak",
    "permutation_entropy",
    "phase_difference",
    "rms",
    "temporal_kurtosis",
]
