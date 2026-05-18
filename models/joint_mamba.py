import torch
import torch.nn as nn
import torch.nn.functional as F

from models.classification_utils import to_bcl

try:
    from mamba_ssm import Mamba
except Exception as e:
    Mamba = None
    _MAMBA_IMPORT_ERROR = e
else:
    _MAMBA_IMPORT_ERROR = None


class _SeqNorm(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True).detach()
        var = x.var(dim=-1, keepdim=True, unbiased=False).detach()
        std = torch.sqrt(var + self.eps)
        return (x - mean) / std, mean, std

    def denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class Model(nn.Module):
    """
    True M4 for Mamba:
      - 单次编码历史窗口
      - forecast head 预测未来波形
      - classification head 融合历史 latent + 未来预测摘要
    """

    def __init__(self, configs):
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm 没装或导入失败。只有使用 --model joint_mamba 时才需要安装："
                "pip install mamba-ssm"
            ) from _MAMBA_IMPORT_ERROR
        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", 1)))

        d_model = int(getattr(configs, "d_model", 128))
        d_state = int(getattr(configs, "d_state", 256))
        d_conv = int(getattr(configs, "d_conv", 4))
        expand = int(getattr(configs, "expand", 2))
        dropout = float(getattr(configs, "dropout", 0.1))

        self.use_residual = bool(int(getattr(configs, "use_residual", 1)))
        self.use_norm = bool(int(getattr(configs, "use_norm", 1)))
        self.norm = _SeqNorm() if self.use_norm else None

        self.in_proj = nn.Linear(self.in_channels, d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.forecast_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pred_len * self.out_channels),
        )

        self.cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        self.future_pool_bins = int(getattr(configs, "joint_future_pool_bins", self.cls_pool_bins))
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))

        self.sample_rate = float(getattr(configs, "sample_rate", 1.0))
        self.band_low = float(getattr(configs, "risk_band_low", 0.0))
        self.band_high = float(getattr(configs, "risk_band_high", 0.0))
        self.use_future_band_feature = bool(int(getattr(configs, "joint_use_future_band_feature", 1)))
        self.use_future_band_feature = (
            self.use_future_band_feature
            and self.pred_len >= 4
            and self.sample_rate > 0
            and self.band_high > self.band_low
        )
        if self.use_future_band_feature:
            freqs = torch.fft.rfftfreq(self.pred_len, d=1.0 / self.sample_rate)
            band_mask = (freqs >= self.band_low) & (freqs <= self.band_high)
            self.register_buffer("future_band_mask", band_mask, persistent=False)
            if bool(torch.any(band_mask).item()) is False:
                self.use_future_band_feature = False
        else:
            self.register_buffer("future_band_mask", torch.zeros(0, dtype=torch.bool), persistent=False)

        history_dim = d_model * (self.cls_pool_bins + 3) + 5 * self.in_channels
        future_dim = self.out_channels * self.future_pool_bins + 5 * self.out_channels
        if self.use_future_band_feature:
            future_dim += self.out_channels
        self.cls_head = nn.Sequential(
            nn.LayerNorm(history_dim + future_dim),
            nn.Linear(history_dim + future_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

    def _encode_once(self, x: torch.Tensor):
        x_raw = to_bcl(x, self.seq_len, self.in_channels)
        if self.use_norm:
            x_model, mean, std = self.norm(x_raw)
        else:
            x_model = x_raw
            mean = std = None

        x_tok = x_model.transpose(1, 2).contiguous()
        h = self.mamba(self.in_proj(x_tok))
        return x_raw, x_model, h, mean, std

    def _forecast_stats(self, mean: torch.Tensor, std: torch.Tensor):
        if mean is None or std is None:
            return None, None
        mean = mean.squeeze(-1)
        std = std.squeeze(-1)
        if self.out_channels == self.in_channels:
            return mean, std
        return mean[:, :1].expand(-1, self.out_channels), std[:, :1].expand(-1, self.out_channels)

    def _decode_forecast(self, x_model, h, mean, std):
        b = x_model.size(0)
        y = self.forecast_head(h[:, -1, :]).view(b, self.pred_len, self.out_channels)

        if self.use_residual:
            last = x_model[:, :, -1]
            if self.out_channels == self.in_channels:
                y = y + last.unsqueeze(1)
            else:
                y = y + last[:, :1].unsqueeze(1).expand(b, 1, self.out_channels)

        if self.use_norm and mean is not None and std is not None:
            mean_out, std_out = self._forecast_stats(mean, std)
            y = self.norm.denorm(y, mean_out.unsqueeze(1), std_out.unsqueeze(1))
        return y

    def _summarize_history(self, x_raw, h):
        feat = h.permute(0, 2, 1).contiguous()  # (B, D, L)
        pooled = F.adaptive_avg_pool1d(feat, self.cls_pool_bins).flatten(1)
        mean = feat.mean(dim=-1)
        maxv = feat.amax(dim=-1)
        last = feat[..., -1]

        hist = x_raw  # (B, C, L)
        raw_mean = hist.mean(dim=-1)
        raw_std = hist.std(dim=-1, unbiased=False)
        raw_max = hist.amax(dim=-1)
        raw_last = hist[..., -1]
        raw_rms = torch.sqrt(torch.mean(hist ** 2, dim=-1) + 1e-6)
        return torch.cat([pooled, mean, maxv, last, raw_mean, raw_std, raw_max, raw_last, raw_rms], dim=1)

    def _future_band_ratio(self, future_feat):
        if not self.use_future_band_feature:
            return None
        centered = future_feat - future_feat.mean(dim=-1, keepdim=True)
        power = torch.abs(torch.fft.rfft(centered, dim=-1)) ** 2
        total_power = power[..., 1:].sum(dim=-1).clamp_min(1e-6)
        band_power = power[..., self.future_band_mask].sum(dim=-1)
        return band_power / total_power

    def _summarize_future(self, forecast):
        future_feat = forecast.permute(0, 2, 1).contiguous()  # (B, C_out, P)
        pooled = F.adaptive_avg_pool1d(future_feat, self.future_pool_bins).flatten(1)
        mean = future_feat.mean(dim=-1)
        std = future_feat.std(dim=-1, unbiased=False)
        maxv = future_feat.amax(dim=-1)
        last = future_feat[..., -1]
        rms = torch.sqrt(torch.mean(future_feat ** 2, dim=-1) + 1e-6)
        parts = [pooled, mean, std, maxv, last, rms]
        ber = self._future_band_ratio(future_feat)
        if ber is not None:
            parts.append(ber)
        return torch.cat(parts, dim=1)

    def forward_joint(self, x):
        x_raw, x_model, h, mean, std = self._encode_once(x)
        forecast = self._decode_forecast(x_model, h, mean, std)
        hist_summary = self._summarize_history(x_raw, h)
        future_summary = self._summarize_future(forecast)
        logits = self.cls_head(torch.cat([hist_summary, future_summary], dim=1))
        return {
            "forecast": forecast,
            "classification": logits,
        }

    def forecast(self, x):
        return self.forward_joint(x)["forecast"]

    def classification(self, x):
        return self.forward_joint(x)["classification"]

    def forward(self, x):
        return self.forward_joint(x)
