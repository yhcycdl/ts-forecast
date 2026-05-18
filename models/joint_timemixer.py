import torch
import torch.nn as nn
import torch.nn.functional as F

from models import timemixer


class Model(timemixer.Model):
    """
    True M4 for TimeMixer:
      - 单次多尺度编码
      - 共享 encoder 产出 forecast
      - risk head 融合多尺度历史 latent + 未来预测摘要
    """

    def __init__(self, configs):
        super().__init__(configs)

        self.future_pool_bins = int(getattr(configs, "joint_future_pool_bins", self.cls_pool_bins))
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

        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_feat_dim = self.d_model if self.channel_independence == 0 else self.in_channels * self.d_model
        hist_latent_dim = cls_feat_dim * (self.cls_pool_bins + 3) * self.num_scales
        hist_raw_dim = 5 * self.in_channels
        future_dim = self.out_channels * self.future_pool_bins + 5 * self.out_channels
        if self.use_future_band_feature:
            future_dim += self.out_channels
        total_dim = hist_latent_dim + hist_raw_dim + future_dim

        self.cls_head = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(cls_hidden, cls_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

    def _decode_forecast(self, B, enc_out_list, x_list):
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        return self.normalize_layers[0](dec_out, "denorm")

    def _summarize_history(self, B, x_raw, enc_out_list):
        summaries = []
        for enc_out in enc_out_list:
            if self.channel_independence == 1:
                t = enc_out.size(1)
                enc_out = enc_out.view(B, self.in_channels, t, self.d_model)
                feat = enc_out.permute(0, 1, 3, 2).contiguous().view(B, self.in_channels * self.d_model, t)
            else:
                feat = enc_out.permute(0, 2, 1).contiguous()

            pooled = F.adaptive_avg_pool1d(feat, self.cls_pool_bins).flatten(1)
            mean = feat.mean(dim=-1)
            maxv = feat.amax(dim=-1)
            last = feat[..., -1]
            summaries.append(torch.cat([pooled, mean, maxv, last], dim=1))

        x_hist = x_raw.permute(0, 2, 1).contiguous()  # (B, C, L)
        raw_mean = x_hist.mean(dim=-1)
        raw_std = x_hist.std(dim=-1, unbiased=False)
        raw_max = x_hist.amax(dim=-1)
        raw_last = x_hist[..., -1]
        raw_rms = torch.sqrt(torch.mean(x_hist ** 2, dim=-1) + 1e-6)
        summaries.append(torch.cat([raw_mean, raw_std, raw_max, raw_last, raw_rms], dim=1))
        return torch.cat(summaries, dim=1)

    def _future_band_ratio(self, future_feat):
        if not self.use_future_band_feature:
            return None
        centered = future_feat - future_feat.mean(dim=-1, keepdim=True)
        power = torch.abs(torch.fft.rfft(centered, dim=-1)) ** 2
        total_power = power[..., 1:].sum(dim=-1).clamp_min(1e-6)
        band_power = power[..., self.future_band_mask].sum(dim=-1)
        return band_power / total_power

    def _summarize_future(self, forecast):
        future_feat = forecast.permute(0, 2, 1).contiguous()
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
        x_raw = timemixer._to_BLC(x, self.in_channels)
        B, enc_out_list, x_list = self._encode_inputs(x)
        forecast = self._decode_forecast(B, enc_out_list, x_list)
        hist_summary = self._summarize_history(B, x_raw, enc_out_list)
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
