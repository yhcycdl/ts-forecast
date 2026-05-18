import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classification_utils import SequenceClassifierHead


class RevIN(nn.Module):
    """
    x: (B, C, L)
    norm/denorm 都基于 dim=-1 (L) 计算统计量
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

        self.mean = None
        self.stdev = None

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise NotImplementedError(f"Unknown mode: {mode}")

    def _get_statistics(self, x):
        # x: (B,C,L)
        self.mean = torch.mean(x, dim=-1, keepdim=True).detach()
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            w = self.affine_weight.view(1, -1, 1)
            b = self.affine_bias.view(1, -1, 1)
            x = x * w + b
        return x

    def _denormalize(self, x):
        if self.affine:
            w = self.affine_weight.view(1, -1, 1)
            b = self.affine_bias.view(1, -1, 1)
            x = (x - b) / (w + 1e-10)
        x = x * self.stdev + self.mean
        return x


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 5, 7, 11), bottleneck_channels=32):
        super().__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        kernel_sizes = list(kernel_sizes)

        # 保护：保证 branch_channels 是整数
        if out_channels % len(kernel_sizes) != 0:
            raise ValueError(
                f"out_channels({out_channels}) must be divisible by num_kernels({len(kernel_sizes)})"
            )

        self.use_bottleneck = in_channels > 1
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1) if self.use_bottleneck else nn.Identity()

        input_channels = bottleneck_channels if self.use_bottleneck else in_channels
        branch_channels = out_channels // len(kernel_sizes)

        self.convs = nn.ModuleList([
            nn.Conv1d(input_channels, branch_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])

        # MaxPool 分支输出也用 branch_channels
        self.maxpool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        )

        concat_channels = branch_channels * (len(kernel_sizes) + 1)

        self.bn = nn.BatchNorm1d(concat_channels)
        self.act = nn.GELU()
        self.project = nn.Conv1d(concat_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B,C,L)
        x_bottle = self.bottleneck(x)
        outputs = [conv(x_bottle) for conv in self.convs]
        outputs.append(self.maxpool(x))
        out = torch.cat(outputs, dim=1)   # (B, concat, L)
        out = self.act(self.bn(out))
        out = self.project(out)           # (B, out_channels, L)
        return out + x if x.shape[1] == out.shape[1] else out


class Model(nn.Module):
    """
    InceptionTime for forecasting + RevIN + linear trend baseline

    输入支持：
      - (B, L)
      - (B, C, L)
      - (B, L, C)（当 L==seq_len 且 C==in_channels 时自动转）

    输出统一：
      - (B, pred_len, out_channels)
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))

        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", 1)))

        # 固定超参（你不想传太多就固定）
        hidden_dim = int(getattr(configs, "d_model", 64))   # 兼容你问的 d_model
        layers = int(getattr(configs, "e_layers", 6))       # 兼容 e_layers
        self.cls_use_input_norm = bool(int(getattr(configs, "cls_use_input_norm", 0)))

        self.revin = RevIN(self.in_channels, eps=1e-5, affine=True)

        self.blocks = nn.Sequential(
            InceptionBlock(self.in_channels, hidden_dim),
            *[InceptionBlock(hidden_dim, hidden_dim) for _ in range(max(0, layers - 1))]
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        # 预测头：输出 pred_len*out_channels
        self.fc = nn.Linear(hidden_dim, self.pred_len * self.out_channels)

        # 线性趋势：用输入第0通道做 baseline（单变量等价），再广播到 out_channels
        self.linear_res = nn.Linear(self.seq_len, self.pred_len)
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_dropout = float(getattr(configs, "dropout", 0.1))
        cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        self.cls_pool_bins = cls_pool_bins
        cls_in_dim = hidden_dim * (cls_pool_bins + 1)
        self.cls_attn_score = nn.Conv1d(hidden_dim, 1, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_in_dim),
            nn.Linear(cls_in_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

    def _encode(self, x, apply_input_norm: bool):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B,1,L)
        elif x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                x = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")

        x_raw = x
        x_norm = self.revin(x, "norm") if apply_input_norm else x
        feat = self.blocks(x_norm)          # (B, hidden, L)
        pooled = self.gap(feat).squeeze(-1)   # (B, hidden)
        return x_raw, feat, pooled

    def forecast(self, x):
        x_raw, _, pooled = self._encode(x, apply_input_norm=True)
        delta = self.fc(pooled).view(-1, self.pred_len, self.out_channels)  # (B,P,Cout)

        if self.out_channels != self.in_channels:
            delta_denorm = delta
        else:
            delta_bc = delta.permute(0, 2, 1).contiguous()          # (B,C,P)
            delta_bc = self.revin(delta_bc, "denorm")                # (B,C,P) 回到原尺度
            delta_denorm = delta_bc.permute(0, 2, 1).contiguous()    # (B,P,C)

        trend = self.linear_res(x_raw[:, 0, :])    # (B,P)
        trend = trend.unsqueeze(-1).expand(-1, -1, self.out_channels)  # (B,P,Cout)
        return trend + delta_denorm

    def classification(self, x):
        _, feat, _ = self._encode(x, apply_input_norm=self.cls_use_input_norm)
        pooled = F.adaptive_avg_pool1d(feat, self.cls_pool_bins).flatten(1)
        attn_w = torch.softmax(self.cls_attn_score(feat), dim=-1)
        attn_pool = (feat * attn_w).sum(dim=-1)
        summary = torch.cat([pooled, attn_pool], dim=1)
        return self.cls_head(summary)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
