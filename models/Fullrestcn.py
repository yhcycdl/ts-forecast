import torch
import torch.nn as nn

from models.classification_utils import SequenceClassifierHead


class Chomp1d(nn.Module):
    """剪裁层：把 padding 产生的多余时间步裁掉，保证因果/长度对齐"""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        # x: (B,C,L+pad) -> (B,C,L)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        dropout = float(dropout)

        padding = (kernel_size - 1) * dilation  # 关键：保证输出长度等于输入长度
        self.padding = padding

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class Model(nn.Module):
    """
    Full-Res TCN + Inertia Trend（你原版逻辑保留）

    输入支持：
      - (B, L)
      - (B, C, L)
      - (B, L, C) （当 L==seq_len 且 C==in_channels 时自动转）
    输出统一：
      - (B, pred_len, c_out)
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        # 必要参数（由 args/configs 给）
        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))

        # 通道命名兼容：in_channels/out_channels 或 enc_in/c_out
        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", 1)))

        # 固定结构超参（你说不想传太多，就写死默认；想改再加 args）
        num_channels = [32, 64, 128, 128, 128, 128]
        kernel_size = 3
        dropout = 0.1

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = self.in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(DilatedBlock(in_ch, out_ch, kernel_size, dilation=dilation, dropout=dropout))

        self.network = nn.Sequential(*layers)
        self.out_dim = num_channels[-1]

        # 输出头：预测 delta，维度为 pred_len * out_channels
        self.fc = nn.Linear(self.out_dim, self.pred_len * self.out_channels)
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_dropout = float(getattr(configs, "dropout", dropout))
        cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        self.cls_head = SequenceClassifierHead(
            self.out_dim,
            self.num_classes,
            hidden_dim=cls_hidden,
            dropout=cls_dropout,
            pool_bins=cls_pool_bins,
        )

    def _encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")
        feat = self.network(x)
        return x, feat

    def forecast(self, x):
        x, feat = self._encode(x)
        last_feat = feat[:, :, -1]       # (B, H)
        delta = self.fc(last_feat)       # (B, pred_len*out_channels)
        delta = delta.view(-1, self.pred_len, self.out_channels)  # (B, P, C_out)

        last_10 = x[:, 0:1, -10:]  # (B,1,10)
        slope = (last_10[:, :, -1] - last_10[:, :, 0]) / 10.0    # (B,1)
        slope = slope.unsqueeze(-1)                                 # (B,1,1)

        steps = torch.arange(1, self.pred_len + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)  # (1,1,P)
        last_val = x[:, 0:1, -1:].to(x.dtype)  # (B,1,1)
        trend_1ch = last_val + slope * steps   # (B,1,P)
        trend_1ch = trend_1ch.transpose(1, 2).contiguous()  # (B,P,1)

        # 把 trend 扩展到 out_channels（如果 out_channels>1，同一趋势广播到各输出通道）
        trend = trend_1ch.expand(-1, -1, self.out_channels)  # (B,P,C_out)

        return trend + delta

    def classification(self, x):
        _, feat = self._encode(x)
        return self.cls_head(feat)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
