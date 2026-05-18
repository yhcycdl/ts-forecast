from __future__ import annotations

import math

import numpy as np

from src.features.spectral import as_1d

EPS = 1e-12


def temporal_kurtosis(signal: np.ndarray | list[float]) -> float:
    arr = as_1d(signal)
    std = float(np.std(arr))
    if std <= EPS:
        return 0.0
    centered = (arr - float(np.mean(arr))) / std
    return float(np.mean(centered ** 4) - 3.0)


def permutation_entropy(
    signal: np.ndarray | list[float],
    order: int = 5,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    arr = as_1d(signal)
    order = int(order)
    delay = int(delay)
    if order < 3:
        raise ValueError(f"order must be >= 3, got {order}.")
    if delay < 1:
        raise ValueError(f"delay must be >= 1, got {delay}.")

    n_vectors = arr.size - delay * (order - 1)
    if n_vectors <= 0:
        return 0.0

    tie_breaker = np.linspace(0.0, 1e-12, num=order, endpoint=False)
    counts: dict[tuple[int, ...], int] = {}
    for start in range(n_vectors):
        pattern = arr[start : start + order * delay : delay]
        ordinal = tuple(np.argsort(pattern + tie_breaker, kind="mergesort"))
        counts[ordinal] = counts.get(ordinal, 0) + 1

    probs = np.asarray(list(counts.values()), dtype=np.float64)
    probs /= float(n_vectors)
    entropy = -float(np.sum(probs * np.log(probs + EPS)))
    if not normalize:
        return entropy
    return float(entropy / max(math.log(math.factorial(order)), EPS))


def cecp_complexity(
    signal: np.ndarray | list[float],
    order: int = 5,
    delay: int = 1,
) -> float:
    # TODO: implement full CECP statistical complexity once the label pipeline is stable.
    _ = (signal, order, delay)
    return float("nan")
