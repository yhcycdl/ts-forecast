"""
SmoothPECNet: Smooth Peak-Envelope-Cycle forecasting wrapper.

The model keeps the existing tcn_claude backbone, but moves the causal
main-wave extraction into the network:

    raw input -> causal moving average -> [smooth, raw/residual] -> backbone

For smooth-wave forecasting, putting the smooth branch first is important:
tcn_claude uses the first output channel for residual output and RevIN
denormalization when c_out=1.
"""

import copy

import torch
import torch.nn as nn

from models import tcn_claude


class CausalMovingAverage(nn.Module):
    def __init__(self, window: int):
        super().__init__()
        self.window = max(1, int(window))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L). Average over the available causal history.
        if self.window <= 1:
            return x

        length = x.shape[-1]
        if length <= 1:
            return x

        k = min(self.window, length)
        cumsum = torch.cumsum(x, dim=-1)
        window_sum = cumsum.clone()
        window_sum[..., k:] = cumsum[..., k:] - cumsum[..., :-k]

        counts = torch.arange(1, length + 1, device=x.device, dtype=x.dtype)
        counts = torch.clamp(counts, max=float(k)).view(1, 1, length)
        return window_sum / counts


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.raw_in_channels = int(getattr(args, "enc_in", 1))
        self.mode = str(getattr(args, "smoothpec_mode", "smooth_raw")).lower()
        self.smoother = CausalMovingAverage(int(getattr(args, "smoothpec_window", 1)))

        if self.mode not in {"smooth_raw", "raw_smooth", "smooth_residual", "smooth_only"}:
            raise ValueError(
                "--smoothpec_mode must be one of: smooth_raw, raw_smooth, "
                "smooth_residual, smooth_only"
            )

        if self.mode == "smooth_only":
            backbone_in = self.raw_in_channels
        else:
            backbone_in = self.raw_in_channels * 2

        backbone_args = copy.copy(args)
        backbone_args.enc_in = backbone_in
        self.backbone = tcn_claude.Model(backbone_args)

        print(
            "[SmoothPECNet] "
            f"window={self.smoother.window} mode={self.mode} "
            f"raw_in={self.raw_in_channels} backbone_in={backbone_in}"
        )

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.raw_in_channels:
                return x.permute(0, 2, 1).contiguous()
            return x
        raise ValueError(f"Expected x.dim() in [2, 3], got {tuple(x.shape)}")

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        raw = self._to_bcl(x)
        smooth = self.smoother(raw)

        if self.mode == "smooth_only":
            return smooth
        if self.mode == "raw_smooth":
            return torch.cat([raw, smooth], dim=1)
        if self.mode == "smooth_residual":
            return torch.cat([smooth, raw - smooth], dim=1)
        return torch.cat([smooth, raw], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self._augment(x))
