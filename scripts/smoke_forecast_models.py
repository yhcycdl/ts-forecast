#!/usr/bin/env python3
"""Smoke-test forecasting model shape compatibility.

This is intentionally small and CPU-friendly. It checks that registered models
can consume the repository's current dataloader layout (B, C, L), also accept
(B, L, C), and return (B, pred_len, c_out).
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.registry import MODEL_NAMES, get_model_module


def _parse_models(value: str | None) -> list[str]:
    if not value:
        return list(MODEL_NAMES)
    models = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in models if name not in MODEL_NAMES]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {list(MODEL_NAMES)}")
    return models


def _make_args(model: str, seq_len: int, pred_len: int, enc_in: int, c_out: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        model=model,
        seq_len=int(seq_len),
        pred_len=int(pred_len),
        enc_in=int(enc_in),
        c_out=int(c_out),
        out_in=int(c_out),
        # Shared neural dimensions.
        d_model=32,
        d_ff=64,
        e_layers=1,
        dropout=0.05,
        # TCN-family knobs.
        kernel_size=3,
        num_layers=4,
        base_ch=16,
        max_ch=64,
        top_k_freq=16,
        freq_dim=16,
        use_revin=1,
        residual_output=1,
        # SmoothPEC / enhanced wrappers.
        smoothpec_window=5,
        smoothpec_mode="smooth_raw",
        qpenhance_gate=1,
        qpenhance_gate_hidden=16,
        qpenhance_input_dropout=0.0,
        # Cycle residual.
        period_len=max(2, int(seq_len) // 8),
        cycle_base_cycles=2,
        cycle_base_mode="mean",
        cycle_backbone_revin=0,
        # DLinear / TimeMixer decomposition.
        moving_avg=5,
        individual=0,
        decomp_method="moving_avg",
        top_k=3,
        down_sampling_window=2,
        down_sampling_layers=2,
        down_sampling_method="avg",
        channel_independence=0,
        use_norm=1,
        # PatchTST.
        patch_len=8,
        patch_stride=4,
        n_heads=4,
        factor=1,
        activation="gelu",
        # Classification-only fields some restored modules still define.
        num_classes=2,
        cls_hidden_dim=32,
        cls_pool_bins=4,
    )


def _check_model(model_name: str, args: argparse.Namespace) -> None:
    import torch

    cfg = _make_args(model_name, args.seq_len, args.pred_len, args.enc_in, args.c_out)
    module = get_model_module(model_name)
    model = module.Model(cfg).float().eval()

    expected = (args.batch_size, args.pred_len, args.c_out)
    x_bcl = torch.randn(args.batch_size, args.enc_in, args.seq_len)
    x_blc = x_bcl.permute(0, 2, 1).contiguous()

    with torch.no_grad():
        for label, x in (("BCL", x_bcl), ("BLC", x_blc)):
            y = model(x)
            if tuple(y.shape) != expected:
                raise AssertionError(
                    f"{model_name} {label} output shape {tuple(y.shape)} != expected {expected}"
                )
            if not torch.isfinite(y).all():
                raise AssertionError(f"{model_name} {label} output contains NaN/Inf")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CPU shape smoke tests for registered forecasting models.")
    parser.add_argument("--models", default=None, help="Comma-separated subset; default tests all registered models.")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--pred-len", type=int, default=16)
    parser.add_argument("--enc-in", type=int, default=3)
    parser.add_argument("--c-out", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    failures: list[str] = []
    for model_name in _parse_models(args.models):
        try:
            _check_model(model_name, args)
            print(f"[OK] {model_name}")
        except Exception as exc:
            failures.append(model_name)
            print(f"[FAIL] {model_name}: {exc}", file=sys.stderr)
            traceback.print_exc()

    if failures:
        print(f"Failed models: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("All requested models passed shape smoke tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
