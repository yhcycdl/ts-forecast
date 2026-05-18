from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from src.labels.calibration import QuantileCalibrator, fit_quantile_calibrator

EPS = 1e-6
DEFAULT_WEIGHT_MODE = "physics_rule"


@dataclass(slots=True)
class FeatureSpec:
    name: str
    direction: int
    weight: float
    transform: str = "identity"
    fixed_center: float = 0.0
    fixed_scale: float = 1.0

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def default_feature_specs() -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name="future_p_rms",
            direction=1,
            weight=0.30,
            transform="log1p",
            fixed_center=np.log1p(100.0),
            fixed_scale=max(np.log1p(400.0) - np.log1p(50.0), EPS),
        ),
        FeatureSpec(
            name="future_p_band_energy_ratio",
            direction=1,
            weight=0.20,
            transform="identity",
            fixed_center=0.35,
            fixed_scale=0.25,
        ),
        FeatureSpec(
            name="future_q_rms",
            direction=1,
            weight=0.10,
            transform="identity",
            fixed_center=1.0,
            fixed_scale=0.03,
        ),
        FeatureSpec(
            name="future_pq_coherence",
            direction=1,
            weight=0.20,
            transform="identity",
            fixed_center=0.50,
            fixed_scale=0.20,
        ),
        FeatureSpec(
            name="future_p_env_slope",
            direction=1,
            weight=0.10,
            transform="signed_log1p",
            fixed_center=0.0,
            fixed_scale=np.log1p(50.0),
        ),
        FeatureSpec(
            name="future_permutation_entropy",
            direction=-1,
            weight=0.10,
            transform="identity",
            fixed_center=0.60,
            fixed_scale=0.10,
        ),
    ]


def transform_feature_value(value: float, transform: str) -> float:
    value = float(value)
    transform = str(transform).lower()
    if transform == "identity":
        return value
    if transform == "log1p":
        return float(np.log1p(max(value, 0.0)))
    if transform == "signed_log1p":
        return float(np.sign(value) * np.log1p(abs(value)))
    raise ValueError(f"Unsupported feature transform: {transform}")


def _safe_float(row: dict[str, object], key: str) -> float:
    raw = row.get(key, float("nan"))
    try:
        value = float(raw)
    except Exception:
        return float("nan")
    return value


def _calibrator_probability(value: float, calibrator: QuantileCalibrator, direction: int) -> float:
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, got {direction}.")
    return calibrator.transform(direction * float(value))


def _normalize_weights(raw_weights: list[float]) -> list[float]:
    weights = np.asarray(raw_weights, dtype=np.float64)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights))
    if total <= EPS:
        weights = np.ones_like(weights, dtype=np.float64)
        total = float(np.sum(weights))
    return [float(value / total) for value in weights]


def _train_probability_matrix(
    train_rows: list[dict[str, object]],
    feature_specs: list[FeatureSpec],
    calibrators: dict[str, QuantileCalibrator],
) -> tuple[np.ndarray, list[int]]:
    columns: list[list[float]] = [[] for _ in feature_specs]
    row_keep_mask: list[bool] = []
    for row in train_rows:
        row_values: list[float] = []
        valid = True
        for spec in feature_specs:
            raw = _safe_float(row, spec.name)
            if not np.isfinite(raw):
                valid = False
                break
            transformed = transform_feature_value(raw, spec.transform)
            probability = _calibrator_probability(transformed, calibrators[spec.name], spec.direction)
            row_values.append(float(probability))
        if valid:
            row_keep_mask.append(True)
            for idx, value in enumerate(row_values):
                columns[idx].append(value)
        else:
            row_keep_mask.append(False)
    if not any(row_keep_mask):
        return np.zeros((0, len(feature_specs)), dtype=np.float64), []
    matrix = np.asarray(columns, dtype=np.float64).T
    keep_indices = [idx for idx, keep in enumerate(row_keep_mask) if keep]
    return matrix, keep_indices


def _fit_unsupervised_weights(
    train_rows: list[dict[str, object]],
    feature_specs: list[FeatureSpec],
    calibrators: dict[str, QuantileCalibrator],
) -> list[float]:
    matrix, _ = _train_probability_matrix(train_rows, feature_specs, calibrators)
    if matrix.shape[0] < 2 or matrix.shape[1] == 0:
        return _normalize_weights([spec.weight for spec in feature_specs])

    dispersion = np.std(matrix, axis=0, ddof=0)
    if matrix.shape[1] == 1:
        return _normalize_weights(dispersion.tolist())

    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    mean_abs_corr = np.mean(np.abs(corr - np.eye(corr.shape[0], dtype=np.float64)), axis=1)
    raw_weights = dispersion / (1.0 + mean_abs_corr)
    return _normalize_weights(raw_weights.tolist())


def _resolved_feature_specs(
    train_rows: list[dict[str, object]],
    feature_specs: list[FeatureSpec],
    calibrators: dict[str, QuantileCalibrator],
    weight_mode: str,
) -> list[FeatureSpec]:
    weight_mode = str(weight_mode).lower()
    specs = [FeatureSpec(**spec.to_dict()) for spec in feature_specs]
    if weight_mode == "physics_rule":
        weights = _normalize_weights([spec.weight for spec in specs])
    elif weight_mode == "equal":
        weights = _normalize_weights([1.0 for _ in specs])
    elif weight_mode == "train_fitted_unsupervised":
        weights = _fit_unsupervised_weights(train_rows, specs, calibrators)
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    for spec, weight in zip(specs, weights, strict=True):
        spec.weight = float(weight)
    return specs


def fit_margin_config(
    rows: Iterable[dict[str, object]],
    feature_specs: list[FeatureSpec] | None = None,
    clip_value: float = 6.0,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
) -> dict[str, object]:
    feature_specs = feature_specs or default_feature_specs()
    train_rows = [row for row in rows if str(row.get("split", "")).lower() == "train"]
    if not train_rows:
        raise ValueError("fit_margin_config requires at least one row from the train split.")

    calibrators: dict[str, QuantileCalibrator] = {}
    for spec in feature_specs:
        values = [
            transform_feature_value(_safe_float(row, spec.name), spec.transform)
            for row in train_rows
            if np.isfinite(_safe_float(row, spec.name))
        ]
        calibrator = fit_quantile_calibrator(np.asarray(values, dtype=np.float64), clip_value=clip_value)
        calibrators[spec.name] = calibrator

    resolved_specs = _resolved_feature_specs(
        train_rows=train_rows,
        feature_specs=feature_specs,
        calibrators=calibrators,
        weight_mode=weight_mode,
    )

    return {
        "mode": "train-fitted",
        "weight_mode": str(weight_mode).lower(),
        "clip_value": float(clip_value),
        "feature_specs": [spec.to_dict() for spec in resolved_specs],
        "calibrators": {key: value.to_dict() for key, value in calibrators.items()},
    }


def build_fixed_rule_margin_config(
    feature_specs: list[FeatureSpec] | None = None,
    clip_value: float = 6.0,
    weight_mode: str = DEFAULT_WEIGHT_MODE,
) -> dict[str, object]:
    feature_specs = feature_specs or default_feature_specs()
    calibrators = {
        spec.name: {
            "center": float(spec.fixed_center),
            "scale": max(float(spec.fixed_scale), EPS),
            "clip_value": float(clip_value),
        }
        for spec in feature_specs
    }
    calibrator_objs = {key: QuantileCalibrator(**value) for key, value in calibrators.items()}
    resolved_specs = _resolved_feature_specs(
        train_rows=[],
        feature_specs=feature_specs,
        calibrators=calibrator_objs,
        weight_mode=weight_mode if weight_mode != "train_fitted_unsupervised" else "physics_rule",
    )
    return {
        "mode": "fixed-rule",
        "weight_mode": str(weight_mode).lower(),
        "clip_value": float(clip_value),
        "feature_specs": [spec.to_dict() for spec in resolved_specs],
        "calibrators": calibrators,
    }


def score_row(row: dict[str, object], margin_config: dict[str, object]) -> tuple[float, dict[str, float]]:
    feature_specs = [FeatureSpec(**spec) for spec in margin_config["feature_specs"]]
    calibrators = {
        key: QuantileCalibrator(**value_dict)
        for key, value_dict in dict(margin_config["calibrators"]).items()
    }
    components: dict[str, float] = {}
    weighted_sum = 0.0
    total_weight = 0.0

    for spec in feature_specs:
        raw = _safe_float(row, spec.name)
        if not np.isfinite(raw):
            probability = float("nan")
        else:
            transformed = transform_feature_value(raw, spec.transform)
            probability = _calibrator_probability(transformed, calibrators[spec.name], spec.direction)
            weighted_sum += float(spec.weight) * probability
            total_weight += float(spec.weight)
        components[f"margin_component_{spec.name}"] = probability

    score = weighted_sum / max(total_weight, EPS)
    return float(score), components


def score_rows(rows: Iterable[dict[str, object]], margin_config: dict[str, object]) -> list[dict[str, object]]:
    scored_rows: list[dict[str, object]] = []
    for row in rows:
        enriched = dict(row)
        score, components = score_row(enriched, margin_config)
        enriched["margin_score"] = float(score)
        enriched.update(components)
        scored_rows.append(enriched)
    return scored_rows
