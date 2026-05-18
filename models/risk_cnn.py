import torch
import torch.nn as nn

from models.classification_utils import VectorClassifierHead, to_bcl


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: int = 2, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.MaxPool1d(pool) if pool and pool > 1 else nn.Identity(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Model(nn.Module):
    """
    轻量 1D CNN 风险分类器。

    输入:
      x: (B, L) / (B, L, C) / (B, C, L)
    输出:
      logits: (B, num_classes)
    """
    def __init__(self, configs):
        super().__init__()

        self.seq_len = int(getattr(configs, "seq_len", 4096))
        self.pred_len = int(getattr(configs, "pred_len", 128))
        self.in_channels = int(getattr(configs, "enc_in", getattr(configs, "in_channels", 1)))
        self.out_channels = int(getattr(configs, "c_out", getattr(configs, "out_channels", 1)))
        self.task_name = str(getattr(configs, "task_name", "risk_classification"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        channels = (32, 64, 128)
        self.backbone = nn.Sequential(
            ConvBlock(self.in_channels, channels[0], kernel_size=7, pool=2, dropout=0.1),
            ConvBlock(channels[0], channels[1], kernel_size=5, pool=2, dropout=0.1),
            ConvBlock(channels[1], channels[2], kernel_size=3, pool=2, dropout=0.1),
        )

        self.attn_score = nn.Conv1d(channels[-1], 1, kernel_size=1)
        rep_dim = channels[-1] * 2
        self.forecast_head = nn.Sequential(
            nn.Linear(rep_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.pred_len * self.out_channels),
        )
        self.cls_head = VectorClassifierHead(rep_dim, self.num_classes, hidden_dim=128, dropout=0.2)

    def _encode(self, x):
        x = to_bcl(x, self.seq_len, self.in_channels)

        feat = self.backbone(x)
        avg_pool = torch.mean(feat, dim=-1)

        attn = torch.softmax(self.attn_score(feat), dim=-1)
        attn_pool = torch.sum(feat * attn, dim=-1)

        rep = torch.cat([avg_pool, attn_pool], dim=1)
        last_point = x[:, 0, -1].view(-1, 1, 1)
        return rep, last_point

    def forecast(self, x):
        rep, last_point = self._encode(x)
        pred = self.forecast_head(rep).view(-1, self.pred_len, self.out_channels)
        return pred + last_point

    def classification(self, x):
        rep, _ = self._encode(x)
        return self.cls_head(rep)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
