# models/mamba_forecast.py
import torch
import torch.nn as nn

from models.classification_utils import SequenceClassifierHead, to_bcl

try:
    from mamba_ssm import Mamba
except Exception as e:
    raise ImportError(
        "mamba_ssm 没装或导入失败。请先安装：pip install mamba-ssm"
    ) from e


class _SeqNorm(nn.Module):
    """
    对输入 (B,C,L) 做逐样本、逐通道的标准化（沿时间维 L）
    """
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x: (B,C,L)
        mean = x.mean(dim=-1, keepdim=True).detach()
        var = x.var(dim=-1, keepdim=True, unbiased=False).detach()
        std = torch.sqrt(var + self.eps)
        x_norm = (x - mean) / std
        return x_norm, mean, std

    def denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class Model(nn.Module):
    """
    适配你项目的 Mamba 预测模型：
    输入 : (B, C_in, L)
    输出 : (B, pred_len, C_out)

    预测方式：
      - 编码整个历史序列 -> 得到 (B, L, d_model)
      - 取最后一个时间步表示 -> (B, d_model)
      - head -> (B, pred_len*C_out) -> reshape 成 (B, pred_len, C_out)

    可选：use_residual=True 时，输出为 last_value + delta（对波形更稳）
    当 use_norm=True 时，delta 和 residual 都在归一化空间中计算，最后再还原回输入尺度。
    """
    def __init__(
        self,
        configs,
        d_model: int = 128,
        d_state: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        norm: bool = True,
    ):
        super().__init__()
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        # ---- 从 configs 读取你项目常用参数 ----
        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))

        # 你的项目里一般叫 in_channels / out_channels
        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", 1)))

        self.use_residual = bool(use_residual)
        self.use_norm = bool(norm)
        self.cls_use_input_norm = bool(int(getattr(configs, "cls_use_input_norm", 0)))

        # ---- 归一化（逐样本/逐通道）----
        self.norm = _SeqNorm() if self.use_norm else None

        # ---- 输入投影：C_in -> d_model ----
        self.in_proj = nn.Linear(self.in_channels, d_model)

        # ---- Mamba backbone ----
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        # ---- 预测 head：d_model -> pred_len * C_out ----
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pred_len * self.out_channels),
        )
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        self.cls_head = SequenceClassifierHead(
            d_model,
            self.num_classes,
            hidden_dim=cls_hidden,
            dropout=dropout,
            pool_bins=cls_pool_bins,
        )

    def _encode(self, x: torch.Tensor, apply_input_norm: bool):
        x = to_bcl(x, self.seq_len, self.in_channels)
        B, C, L = x.shape
        if C != self.in_channels:
            raise ValueError(f"in_channels mismatch: expect {self.in_channels}, got {C}")

        if self.use_norm and apply_input_norm:
            x_norm, mean, std = self.norm(x)  # (B,C,L), (B,C,1), (B,C,1)
        else:
            x_norm = x
            mean = std = None

        x_tok = x_norm.transpose(1, 2).contiguous()  # (B,L,C)
        h = self.in_proj(x_tok)  # (B,L,d_model)
        h = self.mamba(h)        # (B,L,d_model)
        return x, x_norm, h, mean, std

    def _forecast_stats(self, mean: torch.Tensor, std: torch.Tensor):
        if mean is None or std is None:
            return None, None

        mean = mean.squeeze(-1)  # (B,C)
        std = std.squeeze(-1)    # (B,C)
        if self.out_channels == self.in_channels:
            return mean, std

        # 与 residual 的默认行为保持一致：通道数不匹配时，回退到第一个输入通道。
        mean = mean[:, :1].expand(-1, self.out_channels)
        std = std[:, :1].expand(-1, self.out_channels)
        return mean, std

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        x_raw, x_model, h, mean, std = self._encode(x, apply_input_norm=True)
        B = x_raw.size(0)
        h_last = h[:, -1, :]     # (B,d_model)
        y = self.head(h_last)    # (B, pred_len*C_out)
        y = y.view(B, self.pred_len, self.out_channels)  # (B,P,Cout)

        if self.use_residual:
            last = x_model[:, :, -1]  # (B,C_in), 与 head 输出处于同一空间
            if self.out_channels == self.in_channels:
                base = last.unsqueeze(1)  # (B,1,C) -> broadcast 到 (B,P,C)
            else:
                base = last[:, :1].unsqueeze(1).expand(B, 1, self.out_channels)  # (B,1,Cout)
            y = y + base  # broadcast 到 (B,P,Cout)

        if self.use_norm and mean is not None and std is not None:
            mean_out, std_out = self._forecast_stats(mean, std)
            y = self.norm.denorm(y, mean_out.unsqueeze(1), std_out.unsqueeze(1))
        return y

    def classification(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, _, _ = self._encode(x, apply_input_norm=self.cls_use_input_norm)
        feat = h.permute(0, 2, 1).contiguous()
        return self.cls_head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
