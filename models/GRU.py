import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPool1D(nn.Module):
    """对 (B, T, H) 做注意力池化 -> (B, H)"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, H)
        w = self.score(x)                 # (B, T, 1)
        w = torch.softmax(w, dim=1)       # (B, T, 1)
        pooled = (x * w).sum(dim=1)       # (B, H)
        return pooled


class Model(nn.Module):
    """
    CNN downsample + GRU/LSTM + AttnPool + Residual Head

    你的原始版本假设输入是 (B, L, 1)，输出是 (B, pred_len)  
    这里为了适配你项目：
      - 支持输入 (B,L) / (B,L,C) / (B,C,L)
      - 输出统一 (B, pred_len, c_out)
    """
    def __init__(self, configs):
        super().__init__()

        # ====== 必要参数（从 args/configs 里拿）======
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        # 输入/输出通道：你项目里可能叫 in_channels/out_channels 或 enc_in/c_out
        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", 1)))

        # ====== 下面这些模型内部超参：你说“固定就行”，我就写死默认值 ======
        conv_channels = (64, 128, 256)
        rnn_hidden = 512
        rnn_layers = 2
        rnn_dropout = 0.2
        use_gru = True       # 默认 GRU
        mlp_ratio = 1.0      # 预测头宽度倍率

        c1, c2, c3 = conv_channels

        # CNN 下采样：stride=2 逐层缩短时间长度
        self.conv1 = nn.Conv1d(self.in_channels, c1, kernel_size=5, padding=2, stride=2)
        self.bn1   = nn.BatchNorm1d(c1)

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, padding=2, stride=2)
        self.bn2   = nn.BatchNorm1d(c2)

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2, stride=2)
        self.bn3   = nn.BatchNorm1d(c3)

        self.act = nn.ReLU()

        rnn_cls = nn.GRU if use_gru else nn.LSTM
        self.rnn = rnn_cls(
            input_size=c3,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )

        self.pool = AttnPool1D(rnn_hidden)

        mid = int(rnn_hidden * mlp_ratio)
        # 输出维度：pred_len * out_channels
        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, mid),
            nn.ReLU(),
            nn.Linear(mid, self.pred_len * self.out_channels),
        )

        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_dropout = float(getattr(configs, "dropout", 0.1))
        self.cls_head = nn.Sequential(
            nn.Linear(rnn_hidden, cls_hidden),
            nn.ReLU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

    def _to_bcl(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")
        return x

    def _encode(self, x):
        x = self._to_bcl(x)
        last_value = x[:, 0, -1]

        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)

        out = self.rnn(x)
        if isinstance(out, tuple):
            out = out[0]

        pooled = self.pool(out)
        return pooled, last_value

    def forecast(self, x):
        pooled, last_value = self._encode(x)
        delta = self.head(pooled)
        delta = delta.view(-1, self.pred_len, self.out_channels)
        return delta + last_value.view(-1, 1, 1)

    def classification(self, x):
        pooled, _ = self._encode(x)
        return self.cls_head(pooled)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
