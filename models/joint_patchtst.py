import torch
from torch import nn
import torch.nn.functional as F

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Encoder, EncoderLayer
from models.PatchTST import FlattenHead, Transpose, _to_BLC


class Model(nn.Module):
    """
    True M4 for PatchTST:
      - 单次编码历史窗口
      - forecast head 预测未来波形
      - risk head 同时读取历史 latent 与预测未来摘要
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", self.in_channels)))

        patch_len = int(getattr(configs, "patch_len", 16))
        stride = int(getattr(configs, "patch_stride", 8))
        padding = stride

        d_model = int(getattr(configs, "d_model", 128))
        n_heads = int(getattr(configs, "n_heads", 8))
        e_layers = int(getattr(configs, "e_layers", 3))
        d_ff = int(getattr(configs, "d_ff", 512))
        factor = int(getattr(configs, "factor", 1))
        dropout = float(getattr(configs, "dropout", 0.2))
        activation = str(getattr(configs, "activation", "gelu"))

        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2)),
        )

        patch_num = int((self.seq_len - patch_len) / stride + 2)
        head_nf = d_model * patch_num
        self.forecast_head = FlattenHead(self.in_channels, head_nf, self.pred_len, head_dropout=dropout)
        self.forecast_channel_proj = (
            nn.Linear(self.in_channels, self.out_channels) if self.out_channels != self.in_channels else nn.Identity()
        )

        self.cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        self.future_pool_bins = int(getattr(configs, "joint_future_pool_bins", self.cls_pool_bins))
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        self.hist_attn_score = nn.Conv1d(self.in_channels * d_model, 1, kernel_size=1)

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

        hist_summary_dim = self.in_channels * d_model * self.cls_pool_bins + self.in_channels * d_model + 5 * self.in_channels
        future_summary_dim = self.out_channels * self.future_pool_bins + 5 * self.out_channels
        if self.use_future_band_feature:
            future_summary_dim += self.out_channels
        cls_in_dim = hist_summary_dim + future_summary_dim

        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_in_dim),
            nn.Linear(cls_in_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, cls_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

    def _encode_once(self, x):
        x = _to_BLC(x, self.in_channels)  # (B, L, C)
        x_raw = x

        means = x.mean(dim=1, keepdim=True).detach()
        x_centered = x - means
        stdev = torch.sqrt(torch.var(x_centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_centered / stdev

        x_patch = x_norm.permute(0, 2, 1).contiguous()
        enc_in, n_vars = self.patch_embedding(x_patch)
        enc_out, _ = self.encoder(enc_in)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2).contiguous()  # (B, C, D, P)
        return x_raw, enc_out, means, stdev

    def _decode_forecast(self, enc_out, means, stdev):
        out = self.forecast_head(enc_out).permute(0, 2, 1).contiguous()  # (B, P, C_in)
        out = out * stdev[:, 0, :].unsqueeze(1)
        out = out + means[:, 0, :].unsqueeze(1)
        return self.forecast_channel_proj(out)

    def _summarize_history(self, x_raw, enc_out):
        b, c, d, p = enc_out.shape
        feat = enc_out.reshape(b, c * d, p)
        pooled = F.adaptive_avg_pool1d(feat, self.cls_pool_bins).flatten(1)
        attn_w = torch.softmax(self.hist_attn_score(feat), dim=-1)
        attn_pool = (feat * attn_w).sum(dim=-1)

        x_hist = x_raw.permute(0, 2, 1).contiguous()
        mean = x_hist.mean(dim=-1)
        std = x_hist.std(dim=-1, unbiased=False)
        maxv = x_hist.amax(dim=-1)
        last = x_hist[..., -1]
        rms = torch.sqrt(torch.mean(x_hist ** 2, dim=-1) + 1e-6)
        return torch.cat([pooled, attn_pool, mean, std, maxv, last, rms], dim=1)

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
        x_raw, enc_out, means, stdev = self._encode_once(x)
        forecast = self._decode_forecast(enc_out, means, stdev)
        hist_summary = self._summarize_history(x_raw, enc_out)
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
