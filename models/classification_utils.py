import torch
import torch.nn as nn
import torch.nn.functional as F


def to_bcl(x: torch.Tensor, seq_len: int, in_channels: int) -> torch.Tensor:
    """
    Normalize common time-series layouts to (B, C, L).
    Supports:
      - (B, L)
      - (B, C, L)
      - (B, L, C)
    """
    if x.dim() == 2:
        if int(in_channels) != 1 or x.shape[1] != int(seq_len):
            raise ValueError(
                "2D input is only valid for single-channel "
                f"(B,{int(seq_len)}) tensors; got {tuple(x.shape)} with "
                f"enc_in={int(in_channels)}."
            )
        return x.unsqueeze(1)
    if x.dim() != 3:
        raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")

    if x.shape[1] == int(in_channels) and x.shape[2] == int(seq_len):
        return x.contiguous()
    if x.shape[1] == int(seq_len) and x.shape[2] == int(in_channels):
        return x.permute(0, 2, 1).contiguous()

    raise ValueError(
        f"Cannot infer channel/time dims from {tuple(x.shape)} with seq_len={seq_len}, in_channels={in_channels}"
    )


class SequenceClassifierHead(nn.Module):
    """
    Classifier for sequence features in (B, C, T).
    Uses pooled bins + simple global statistics.
    """
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1,
                 pool_bins: int = 16):
        super().__init__()
        self.pool_bins = int(pool_bins)
        in_dim = int(feature_dim) * (self.pool_bins + 3)
        self.net = nn.Sequential(
            nn.Linear(in_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 3:
            raise ValueError(f"Expected feature shape (B,C,T), got {tuple(feat.shape)}")
        pooled = F.adaptive_avg_pool1d(feat, self.pool_bins).flatten(1)
        mean = feat.mean(dim=-1)
        maxv = feat.amax(dim=-1)
        last = feat[..., -1]
        summary = torch.cat([pooled, mean, maxv, last], dim=1)
        return self.net(summary)


class VectorClassifierHead(nn.Module):
    """
    Classifier for pooled vector features in (B, D).
    """
    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(feature_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.dim() != 2:
            raise ValueError(f"Expected feature shape (B,D), got {tuple(feat.shape)}")
        return self.net(feat)
