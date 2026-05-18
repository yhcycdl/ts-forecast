import torch
import torch.nn as nn

from models.classification_utils import SequenceClassifierHead


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        channel = int(channel)
        reduction = int(reduction)
        hidden = max(1, channel // reduction)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B,C,L)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)        # (B,C)
        y = self.fc(y).view(b, c, 1)           # (B,C,1)
        return x * y.expand_as(x)


class SpectralBlockPro(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, se_reduction: int = 8):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            SELayer(out_channels, reduction=se_reduction),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        # (B,C,L)
        return self.conv(x) + self.downsample(x)


class Model(nn.Module):
    """
    SpectralCNN_Plus (encoder + residual forecasting)

    输入支持：
      - (B, L)
      - (B, C, L)
      - (B, L, C)（当 L==seq_len 且 C==in_channels 时自动转）

    输出统一：
      - (B, pred_len, out_channels)

    设计：
      - encoder 做 4 次 MaxPool1d(2)，时间长度缩小 16 倍：T' = seq_len / 16
      - 取最后时间步的特征 (B,256)，经 fc -> delta (B, pred_len*out_channels)
      - residual：输出 = last_point + delta（广播）
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))

        # 通道命名兼容：in_channels/out_channels 或 enc_in/c_out
        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", 1)))

        # 固定结构超参（你说不用传很多，就固定）
        se_reduction = 8

        self.encoder = nn.Sequential(
            SpectralBlockPro(self.in_channels, 32, se_reduction=se_reduction),
            nn.MaxPool1d(2),
            SpectralBlockPro(32, 64, se_reduction=se_reduction),
            nn.MaxPool1d(2),
            SpectralBlockPro(64, 128, se_reduction=se_reduction),
            nn.MaxPool1d(2),
            SpectralBlockPro(128, 256, se_reduction=se_reduction),
            nn.MaxPool1d(2),
        )

        # 输出头：用最后时刻的特征向量做预测
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self.pred_len * self.out_channels),
        )
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_dropout = float(getattr(configs, "dropout", 0.1))
        cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        self.cls_head = SequenceClassifierHead(
            256,
            self.num_classes,
            hidden_dim=cls_hidden,
            dropout=cls_dropout,
            pool_bins=cls_pool_bins,
        )

    def _encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,L)
        elif x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")
        feat = self.encoder(x)
        return x, feat

    def forecast(self, x):
        x, feat = self._encode(x)
        last_point = x[:, 0:1, -1:]  # (B,1,1)
        last_feat = feat[:, :, -1]    # (B,256)
        delta = self.fc(last_feat)    # (B, pred_len*out_channels)
        delta = delta.view(-1, self.pred_len, self.out_channels)  # (B,P,Cout)
        base = last_point.transpose(1, 2).contiguous()            # (B,1,1) -> (B,1,1)
        base = base.expand(-1, self.pred_len, self.out_channels)  # (B,P,Cout)
        return base + delta

    def classification(self, x):
        _, feat = self._encode(x)
        return self.cls_head(feat)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
