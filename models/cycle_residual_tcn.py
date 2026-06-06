from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from models import tcn_claude


class Model(nn.Module):
    """Cycle-template residual wrapper for quasi-periodic forecasting.

    The model builds a non-parametric future prior by repeating the latest
    dominant cycle, or the average of several latest cycles. A TCN then predicts
    only the correction around that prior:

        forecast = repeated_cycle_template + TCN(history)

    This is meant for same-quantity main-waveform forecasting. It is deliberately
    simple so it can be ablated against plain QPWave-TCN.
    """

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.pred_len = int(args.pred_len)
        self.in_channels = int(getattr(args, "enc_in", 1))
        self.out_channels = int(getattr(args, "c_out", 1))
        self.period_len = int(getattr(args, "period_len", 0))
        self.base_cycles = int(getattr(args, "cycle_base_cycles", 1))
        self.base_mode = str(getattr(args, "cycle_base_mode", "last")).lower()

        if self.period_len <= 0:
            raise ValueError("--period_len must be positive for cycle_residual_tcn.")
        if self.period_len > self.seq_len:
            raise ValueError(
                f"--period_len={self.period_len} cannot exceed seq_len={self.seq_len}."
            )
        if self.base_cycles <= 0:
            raise ValueError("--cycle_base_cycles must be positive.")
        if self.base_mode not in {"last", "mean"}:
            raise ValueError("--cycle_base_mode must be 'last' or 'mean'.")
        if self.out_channels > self.in_channels:
            raise ValueError(
                "cycle_residual_tcn uses the first c_out input channels as the "
                f"cycle template, but enc_in={self.in_channels}, c_out={self.out_channels}."
            )

        backbone_args = copy.copy(args)
        backbone_args.c_out = self.out_channels
        backbone_args.residual_output = 0
        backbone_args.use_revin = int(getattr(args, "cycle_backbone_revin", 0))
        self.backbone = tcn_claude.Model(backbone_args)

        print(
            "[CycleResidualTCN] "
            f"period_len={self.period_len} base_mode={self.base_mode} "
            f"base_cycles={self.base_cycles} backbone_revin={backbone_args.use_revin}"
        )

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if self.in_channels != 1 or x.shape[1] != self.seq_len:
                raise ValueError(
                    "cycle_residual_tcn 2D input is only valid for single-channel "
                    f"(B,{self.seq_len}) tensors; got {tuple(x.shape)} with "
                    f"enc_in={self.in_channels}."
                )
            return x.unsqueeze(1)
        if x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                return x.permute(0, 2, 1).contiguous()
            if x.shape[1] == self.in_channels and x.shape[2] == self.seq_len:
                return x
            raise ValueError(
                "cycle_residual_tcn input shape not recognized. Expected "
                f"(B,{self.in_channels},{self.seq_len}) or "
                f"(B,{self.seq_len},{self.in_channels}), got {tuple(x.shape)}."
            )
        raise ValueError(f"Expected x.dim() in [2, 3], got {tuple(x.shape)}")

    def _cycle_template(self, x_bcl: torch.Tensor) -> torch.Tensor:
        base_channels = x_bcl[:, : self.out_channels, :]
        if self.base_mode == "mean" and self.base_cycles > 1:
            usable_cycles = min(self.base_cycles, self.seq_len // self.period_len)
            usable = usable_cycles * self.period_len
            cycles = base_channels[:, :, -usable:].reshape(
                x_bcl.shape[0],
                self.out_channels,
                usable_cycles,
                self.period_len,
            )
            template = cycles.mean(dim=2)
        else:
            template = base_channels[:, :, -self.period_len :]

        repeats = int(math.ceil(self.pred_len / self.period_len))
        base = template.repeat(1, 1, repeats)[..., : self.pred_len]
        return base.permute(0, 2, 1).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bcl = self._to_bcl(x)
        base = self._cycle_template(x_bcl)
        correction = self.backbone(x_bcl)
        return base + correction
