
import torch
import torch.nn as nn

from models.classification_utils import VectorClassifierHead, to_bcl


class SELayer(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)      # (B, C)
        y = self.fc(y).view(b, c, 1)         # (B, C, 1)
        return x * y                         # broadcast


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        mid = max(1, hidden_size // 2)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, mid),
            nn.Tanh(),
            nn.Linear(mid, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor):
        # x: (B, T, H)
        weights = self.attention(x)              # (B, T, 1)
        context = torch.sum(x * weights, dim=1)  # (B, H)
        return context, weights


class ConvBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        drop_out: float = 0.3,
        pool_size: int = 2,
        se_reduction: int = 16,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=1, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.se = SELayer(out_channels, reduction=se_reduction)
        self.pool = nn.MaxPool1d(pool_size) if pool_size and pool_size > 1 else nn.Identity()
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        x = self.drop(x)
        return x


def _get(args, key, default):
    return getattr(args, key, default)


class Model(nn.Module):
    """
    TimeMixer 风格入口：Model(args)

    args 里只需要提供：
      - seq_len: 输入长度
      - pred_len: 预测长度
      - enc_in: 输入通道数（特征数）
      - c_out: 输出通道数（目标维度）

    其它结构超参全部固定在代码里（顶会 repo 常用做法）。
    """
    def __init__(self, args):
        super().__init__()
        self.task_name = str(_get(args, "task_name", "long_term_forecast"))
        self.num_classes = int(_get(args, "num_classes", 2))

        # ===== 从 args 读取“必须参数” =====
        self.seq_len = int(_get(args, "seq_len", 4096))
        self.pred_len = int(_get(args, "pred_len", 196))

        # 输入/输出通道：优先 enc_in/c_out（TimeMixer 通用命名）
        self.in_channels = int(_get(args, "enc_in", 1))
        self.out_channels = int(_get(args, "c_out", 1))

        # ===== 固定的模型结构超参（不从 args 读取）=====
        cnn_channels = (64, 128, 256)
        kernel_size = (5, 3, 3)
        pool_size = (2, 2, 2)

        lstm_hidden = 512
        lstm_layer = 2

        dropout_cnn = 0.3
        dropout_lstm = 0.3

        se_reduction = 16

        fc_hidden = 256
        dropout_fc = 0.2

        assert len(cnn_channels) == len(kernel_size) == len(pool_size), \
            "cnn_channels/kernel_size/pool_size length must match"

        # ===== CNN backbone =====
        blocks = []
        prev = self.in_channels
        for out_ch, k, p in zip(cnn_channels, kernel_size, pool_size):
            blocks.append(
                ConvBlock1D(
                    in_channels=prev,
                    out_channels=out_ch,
                    kernel_size=int(k),
                    drop_out=dropout_cnn,
                    pool_size=int(p),
                    se_reduction=se_reduction
                )
            )
            prev = out_ch
        self.cnn = nn.Sequential(*blocks)

        # ===== LSTM =====
        lstm_in = cnn_channels[-1]
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layer,
            batch_first=True,
            dropout=dropout_lstm if lstm_layer > 1 else 0.0,
            bidirectional=False
        )

        # ===== Attention =====
        self.attn = AttentionBlock(lstm_hidden)

        # ===== Output head =====
        # 输出维度 = pred_len * c_out
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.GELU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc_hidden, self.pred_len * self.out_channels)
        )
        cls_hidden = int(_get(args, "cls_hidden_dim", 256))
        self.cls_head = VectorClassifierHead(
            lstm_hidden,
            self.num_classes,
            hidden_dim=cls_hidden,
            dropout=dropout_fc,
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = to_bcl(x, self.seq_len, self.in_channels)

        # CNN: (B,C,L)->(B,C',L')
        x = self.cnn(x)

        # LSTM: (B,L',C')
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)  # (B,L',H)

        context, _ = self.attn(x)  # (B,H)
        return context

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        context = self._encode(x)
        y = self.fc(context)       # (B, pred_len*c_out)
        return y.view(-1, self.pred_len, self.out_channels)

    def classification(self, x: torch.Tensor) -> torch.Tensor:
        context = self._encode(x)
        return self.cls_head(context)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
