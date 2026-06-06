# utils/metrics.py
import numpy as np
import torch

from utils.losses import _align_pred_target


def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    MSE with the same shape-alignment rule used by training losses.
    """
    pred, true = _align_pred_target(pred, true)
    return float(torch.mean((pred - true) ** 2).item())
