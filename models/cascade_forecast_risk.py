import torch
import torch.nn as nn
import torch.nn.functional as F

from models.task_wrapper import _forecast_to_bcl


class _ConvBlock(nn.Module):
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


class FutureWaveformClassifier(nn.Module):
    """
    第二阶段分类器：
      输入是 forecast model 预测得到的未来波形 (B, C_out, pred_len)
      输出是风险分类 logits

    分类器既看原始预测波形的卷积特征，也看其统计/频带特征。
    """

    def __init__(self, configs):
        super().__init__()
        self.pred_len = int(getattr(configs, "pred_len", 128))
        self.in_channels = int(getattr(configs, "c_out", getattr(configs, "out_channels", 1)))
        self.num_classes = int(getattr(configs, "num_classes", 2))
        dropout = float(getattr(configs, "dropout", 0.1))
        hidden_dim = int(getattr(configs, "cls_hidden_dim", 256))

        ch1 = int(getattr(configs, "cascade_cls_ch1", 32))
        ch2 = int(getattr(configs, "cascade_cls_ch2", 64))
        ch3 = int(getattr(configs, "cascade_cls_ch3", 128))
        self.pool_bins = int(getattr(configs, "cls_pool_bins", 16))

        self.backbone = nn.Sequential(
            _ConvBlock(self.in_channels, ch1, kernel_size=7, pool=2, dropout=dropout),
            _ConvBlock(ch1, ch2, kernel_size=5, pool=2, dropout=dropout),
            _ConvBlock(ch2, ch3, kernel_size=3, pool=2, dropout=dropout),
        )
        self.attn_score = nn.Conv1d(ch3, 1, kernel_size=1)

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

        conv_summary_dim = ch3 * 2
        pooled_raw_dim = self.in_channels * self.pool_bins
        raw_stats_dim = 6 * self.in_channels
        if self.use_future_band_feature:
            raw_stats_dim += self.in_channels
        total_dim = conv_summary_dim + pooled_raw_dim + raw_stats_dim

        self.cls_head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def _future_band_ratio(self, future_feat):
        if not self.use_future_band_feature:
            return None
        centered = future_feat - future_feat.mean(dim=-1, keepdim=True)
        power = torch.abs(torch.fft.rfft(centered, dim=-1)) ** 2
        total_power = power[..., 1:].sum(dim=-1).clamp_min(1e-6)
        band_power = power[..., self.future_band_mask].sum(dim=-1)
        return band_power / total_power

    def forward(self, future_feat):
        if future_feat.dim() != 3:
            raise ValueError(f"Expected future waveform shape (B,C,P), got {tuple(future_feat.shape)}")

        conv_feat = self.backbone(future_feat)
        avg_pool = conv_feat.mean(dim=-1)
        attn = torch.softmax(self.attn_score(conv_feat), dim=-1)
        attn_pool = torch.sum(conv_feat * attn, dim=-1)

        pooled_raw = F.adaptive_avg_pool1d(future_feat, self.pool_bins).flatten(1)
        mean = future_feat.mean(dim=-1)
        std = future_feat.std(dim=-1, unbiased=False)
        maxv = future_feat.amax(dim=-1)
        minv = future_feat.amin(dim=-1)
        last = future_feat[..., -1]
        rms = torch.sqrt(torch.mean(future_feat ** 2, dim=-1) + 1e-6)

        parts = [avg_pool, attn_pool, pooled_raw, mean, std, maxv, minv, last, rms]
        ber = self._future_band_ratio(future_feat)
        if ber is not None:
            parts.append(ber)

        summary = torch.cat(parts, dim=1)
        return self.cls_head(summary)


class Model(nn.Module):
    """
    级联串联模型：
      x(history) -> forecast_model -> predicted future waveform -> classifier -> risk logits

    分类模型的输入不是历史窗口，而是第一阶段模型预测出的未来窗口。
    """

    def __init__(self, forecast_model: nn.Module, configs):
        super().__init__()
        self.forecast_model = forecast_model
        self.pred_len = int(getattr(configs, "pred_len", 128))
        self.detach_forecast_for_cls = bool(int(getattr(configs, "cascade_detach_forecast", 1)))
        self.classifier = FutureWaveformClassifier(configs)

    def forecast(self, x):
        return self.forecast_model(x)

    def classify_from_forecast(self, forecast):
        future_feat = _forecast_to_bcl(forecast, self.pred_len)
        if self.detach_forecast_for_cls:
            future_feat = future_feat.detach()
        return self.classifier(future_feat)

    def classification(self, x):
        forecast = self.forecast(x)
        return self.classify_from_forecast(forecast)

    def forward(self, x):
        forecast = self.forecast(x)
        logits = self.classify_from_forecast(forecast)
        return {
            "forecast": forecast,
            "classification": logits,
        }
