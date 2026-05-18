"""Label generation utilities for future-window instability margin tasks."""

from src.labels.label_generator import IndicatorLabelGenerator
from src.labels.margin_score import default_feature_specs, fit_margin_config, score_rows

__all__ = [
    "IndicatorLabelGenerator",
    "default_feature_specs",
    "fit_margin_config",
    "score_rows",
]
