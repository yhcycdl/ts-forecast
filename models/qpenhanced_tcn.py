from __future__ import annotations

import copy

import torch
import torch.nn as nn

from models import tcn_claude


class ChannelGate(nn.Module):
    def __init__(self, channels: int, hidden: int = 32, preserve_first: bool = True):
        super().__init__()
        channels = int(channels)
        hidden = max(4, int(hidden))
        self.preserve_first = bool(preserve_first)
        self.net = nn.Sequential(
            nn.Linear(channels * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        last = x[..., -1]
        gate = self.net(torch.cat([mean, std, last], dim=-1)).unsqueeze(-1)
        if self.preserve_first and gate.shape[1] > 0:
            gate = gate.clone()
            gate[:, :1, :] = 1.0
        return x * gate


class Model(nn.Module):
    """Feature-aware wrapper around QPWave-TCN.

    Intended input columns are produced by scripts/augment_quasiperiodic_dataset.py,
    for example:

      qp_main_input, qp_envelope, qp_local_freq_ratio, qp_phase_sin, qp_phase_cos

    The wrapper keeps the proven TCN backbone and only adds a small channel gate,
    so the module is easy to ablate against plain tcn_claude with the same
    enhanced input columns.
    """

    def __init__(self, args):
        super().__init__()
        self.seq_len = int(args.seq_len)
        self.in_channels = int(getattr(args, "enc_in", 1))
        self.use_gate = bool(int(getattr(args, "qpenhance_gate", 1)))
        self.input_dropout_p = float(getattr(args, "qpenhance_input_dropout", 0.0))
        self.input_dropout = nn.Dropout(self.input_dropout_p)
        hidden = int(getattr(args, "qpenhance_gate_hidden", 32))
        self.gate = ChannelGate(self.in_channels, hidden=hidden, preserve_first=True) if self.use_gate else nn.Identity()

        backbone_args = copy.copy(args)
        backbone_args.enc_in = self.in_channels
        self.backbone = tcn_claude.Model(backbone_args)
        print(
            "[QPEnhancedTCN] "
            f"enc_in={self.in_channels} gate={int(self.use_gate)} "
            f"input_dropout={self.input_dropout_p}"
        )

    def _to_bcl(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if self.in_channels != 1 or x.shape[1] != self.seq_len:
                raise ValueError(
                    "qpenhanced_tcn 2D input is only valid for single-channel "
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
                "qpenhanced_tcn input shape not recognized. Expected "
                f"(B,{self.in_channels},{self.seq_len}) or "
                f"(B,{self.seq_len},{self.in_channels}), got {tuple(x.shape)}."
            )
        raise ValueError(f"Expected x.dim() in [2, 3], got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bcl(x)
        if self.training and self.input_dropout_p > 0 and x.shape[1] > 1:
            x = torch.cat([x[:, :1, :], self.input_dropout(x[:, 1:, :])], dim=1)
        x = self.gate(x)
        return self.backbone(x)
