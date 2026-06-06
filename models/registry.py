"""Model registry with lazy imports.

The runner only needs model names during argument parsing. Keeping the actual
imports lazy prevents optional dependencies or heavyweight modules from
breaking unrelated experiments.
"""

from __future__ import annotations

from importlib import import_module


MODEL_MODULES = {
    "tcn_claude": "models.tcn_claude",
    "smooth_pecnet": "models.smooth_pecnet",
    "qpenhanced_tcn": "models.qpenhanced_tcn",
    "cycle_residual_tcn": "models.cycle_residual_tcn",
    "DLinear": "models.Dlinear",
    "PatchTST": "models.PatchTST",
}

MODEL_NAMES = tuple(MODEL_MODULES.keys())


def get_model_module(name: str):
    if name not in MODEL_MODULES:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_NAMES)}")
    return import_module(MODEL_MODULES[name])
