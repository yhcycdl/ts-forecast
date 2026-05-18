import torch
import torch.nn as nn

from models.classification_utils import SequenceClassifierHead


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        # x: (B,C,L+pad) -> (B,C,L)
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        kernel_size = int(kernel_size)
        dilation = int(dilation)

        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.act2 = nn.GELU()

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.act2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res


class Model(nn.Module):
    """
    Fast TCN (causal dilated conv) for forecasting.

    输入支持：
      - (B, L)
      - (B, C, L)
      - (B, L, C)  （如果你想支持，也可打开自动判断）

    输出统一：
      - (B, pred_len, c_out)

    说明：
      - 你的原版是取最后一个时间点特征 last_feat -> fc -> pred
      - 这里保持一致
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

        # ===== 固定的结构超参（你说不用传）=====
        num_channels = [32, 64, 128, 128, 128, 128]
        kernel_size = 3

        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_ch = self.in_channels if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(DilatedBlock(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation))

        self.network = nn.Sequential(*layers)
        self.out_dim = num_channels[-1]

        # 输出头：预测 pred_len * out_channels
        self.fc = nn.Linear(self.out_dim, self.pred_len * self.out_channels)
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_dropout = float(getattr(configs, "dropout", 0.1))
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
        return feat

    def forecast(self, x):
        feat = self._encode(x)
        last_feat = feat[:, :, -1]         # (B,H)
        y = self.fc(last_feat)             # (B, pred_len*out_channels)
        y = y.view(-1, self.pred_len, self.out_channels)  # (B,P,Cout)
        return y

    def classification(self, x):
        feat = self._encode(x)
        return self.cls_head(feat)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
