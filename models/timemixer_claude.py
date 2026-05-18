
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────

def _to_BLC(x, in_channels: int):
    if x.dim() == 2:
        return x.unsqueeze(-1)
    if x.dim() != 3:
        raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")
    if x.shape[1] == in_channels:
        return x.permute(0, 2, 1).contiguous()
    if x.shape[2] == in_channels:
        return x.contiguous()
    raise ValueError(f"Cannot infer channel dim. got {tuple(x.shape)} with in_channels={in_channels}")


# ─────────────────────────────────────────────────────────
# 改进1：RevIN（状态不存实例变量，随forward传递）
# ─────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """
    可逆实例归一化。
    mean/std 作为返回值传递，不存实例变量，线程安全。
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, non_norm=False):
        super().__init__()
        self.eps      = eps
        self.affine   = affine
        self.non_norm = non_norm
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias   = nn.Parameter(torch.zeros(1, 1, num_features))

    def norm(self, x):
        # x: (B, L, C)
        if self.non_norm:
            return x, None, None
        mean = x.mean(dim=1, keepdim=True).detach()
        std  = (x.var(dim=1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        x    = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x, mean, std

    def denorm(self, x, mean, std):
        if self.non_norm or mean is None:
            return x
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps)
        return x * std + mean


# ─────────────────────────────────────────────────────────
# 改进2：FreqInjection（燃烧谐波特征注入）
# ─────────────────────────────────────────────────────────

class FreqInjection(nn.Module):
    """
    将 FFT 幅值谱特征注入到时域嵌入中。
    燃烧信号的谐波结构在频域比时域稳定，
    这个模块让 TimeMixer 的每个时间步都"知道"当前窗口的频率分布。
    """
    def __init__(self, seq_len: int, d_model: int, top_k: int = 64, dropout: float = 0.1):
        super().__init__()
        self.top_k = min(top_k, seq_len // 2 + 1)
        self.proj  = nn.Sequential(
            nn.Linear(self.top_k, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_raw: torch.Tensor, x_emb: torch.Tensor) -> torch.Tensor:
        """
        x_raw : (B, L, C)     归一化后的原始输入
        x_emb : (B, L, d_model)  时域嵌入
        return: (B, L, d_model)  注入频域信息后的嵌入
        """
        sig  = x_raw[..., 0]                                   # (B, L)
        amp  = torch.fft.rfft(sig, dim=-1).abs()               # (B, L//2+1)
        amp  = torch.log1p(amp)[:, :self.top_k]                # (B, top_k)
        freq = self.proj(amp).unsqueeze(1)                     # (B, 1, d_model)
        return x_emb + freq                                    # 广播到全部时间步


# ─────────────────────────────────────────────────────────
# 原版模块（保留）
# ─────────────────────────────────────────────────────────

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end   = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x):
        trend    = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DFTSeriesDecomp(nn.Module):
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf   = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0
        top_k_freq, _ = torch.topk(freq, self.top_k, dim=1)
        threshold = top_k_freq.min(dim=1, keepdim=True).values
        # 修复：< 而非 <=，避免误删等于阈值的分量
        xf        = xf.masked_fill(freq < threshold, 0)
        x_season  = torch.fft.irfft(xf, n=x.size(1), dim=1)
        x_trend   = x - x_season
        return x_season, x_trend


class ValueEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x))


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** i),
                          seq_len // (down_sampling_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                          seq_len // (down_sampling_window ** (i + 1))),
            )
            for i in range(down_sampling_layers)
        ])

    def forward(self, season_list):
        if len(season_list) <= 1:
            return [season_list[0].permute(0, 2, 1)]
        out_high = season_list[0]
        out_low  = season_list[1]
        out_list = [out_high.permute(0, 2, 1)]
        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low  = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_list.append(out_high.permute(0, 2, 1))
        return out_list


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_sampling_window ** (i + 1)),
                          seq_len // (down_sampling_window ** i)),
                nn.GELU(),
                nn.Linear(seq_len // (down_sampling_window ** i),
                          seq_len // (down_sampling_window ** i)),
            )
            for i in reversed(range(down_sampling_layers))
        ])

    def forward(self, trend_list):
        if len(trend_list) <= 1:
            return [trend_list[0].permute(0, 2, 1)]
        rev  = trend_list.copy()
        rev.reverse()
        out_low  = rev[0]
        out_high = rev[1]
        out_list = [out_low.permute(0, 2, 1)]
        for i in range(len(rev) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low  = out_high
            if i + 2 <= len(rev) - 1:
                out_high = rev[i + 2]
            out_list.append(out_low.permute(0, 2, 1))
        out_list.reverse()
        return out_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len              = int(configs.seq_len)
        self.down_sampling_window = int(getattr(configs, "down_sampling_window", 2))
        self.channel_independence = int(getattr(configs, "channel_independence", 0))
        d_model      = int(getattr(configs, "d_model",   128))
        d_ff         = int(getattr(configs, "d_ff",      256))
        dropout      = float(getattr(configs, "dropout", 0.1))
        decomp_method = str(getattr(configs, "decomp_method", "moving_avg"))
        moving_avg   = int(getattr(configs, "moving_avg", 25))
        top_k        = int(getattr(configs, "top_k",     5))
        n_layers     = int(getattr(configs, "down_sampling_layers", 2))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

        if decomp_method == "moving_avg":
            self.decomposition = SeriesDecomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomposition = DFTSeriesDecomp(top_k)
        else:
            raise ValueError(f"Unknown decomp_method: {decomp_method}")

        if self.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
            )
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            self.seq_len, self.down_sampling_window, n_layers)
        self.mixing_multi_scale_trend  = MultiScaleTrendMixing(
            self.seq_len, self.down_sampling_window, n_layers)
        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x_list):
        length_list  = [x.size(1) for x in x_list]
        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decomposition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend  = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list  = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, s, t, L in zip(x_list, out_season_list, out_trend_list, length_list):
            out = s + t
            if self.channel_independence == 1:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :L, :])
        return out_list


# ─────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.seq_len   = int(configs.seq_len)
        self.pred_len  = int(configs.pred_len)
        self.in_channels  = int(getattr(configs, "enc_in",  1))
        self.out_channels = int(getattr(configs, "c_out",   self.in_channels))

        self.down_sampling_window  = int(getattr(configs, "down_sampling_window",  2))
        self.down_sampling_layers  = int(getattr(configs, "down_sampling_layers",  2))
        self.down_sampling_method  = str(getattr(configs, "down_sampling_method",  "avg"))
        self.channel_independence  = int(getattr(configs, "channel_independence",  0))
        self.e_layers  = int(getattr(configs, "e_layers",   2))
        self.d_model   = int(getattr(configs, "d_model",    128))
        self.d_ff      = int(getattr(configs, "d_ff",       256))
        self.dropout   = float(getattr(configs, "dropout",  0.1))
        self.use_norm  = int(getattr(configs, "use_norm",   1))
        self.cls_use_input_norm = bool(int(getattr(configs, "cls_use_input_norm", 0)))
        self.cls_use_pre_enc = bool(int(getattr(configs, "cls_use_pre_enc", 0)))
        top_k_freq     = int(getattr(configs, "top_k_freq", 64))

        # ── 改进1：RevIN 替换 SimpleNormalize ────────────────
        non_norm = (self.use_norm == 0)
        self.normalize_layers = nn.ModuleList([
            RevIN(self.in_channels, non_norm=non_norm)
            for _ in range(self.down_sampling_layers + 1)
        ])
        # 存储每层的 mean/std（随 forward 更新，非实例状态）
        self._norm_stats = [None] * (self.down_sampling_layers + 1)

        # ── PDM 块 ────────────────────────────────────────────
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(self.e_layers)
        ])

        moving_avg = int(getattr(configs, "moving_avg", 25))
        self.preprocess = SeriesDecomp(moving_avg)

        # ── Embedding ─────────────────────────────────────────
        enc_in = 1 if self.channel_independence == 1 else self.in_channels
        self.enc_embedding = ValueEmbedding(enc_in, self.d_model, self.dropout)

        # ── 改进2：FreqInjection（针对燃烧谐波）──────────────
        self.freq_injection = FreqInjection(
            seq_len=self.seq_len,
            d_model=self.d_model,
            top_k=top_k_freq,
            dropout=self.dropout,
        )

        # ── 预测层 ────────────────────────────────────────────
        self.predict_layers = nn.ModuleList([
            nn.Linear(self.seq_len // (self.down_sampling_window ** i), self.pred_len)
            for i in range(self.down_sampling_layers + 1)
        ])

        # ── 改进3：可学习多尺度加权融合（替换等权 sum）─────────
        n_scales = self.down_sampling_layers + 1
        self.scale_weights = nn.Parameter(torch.ones(n_scales))
        cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_feat_dim = self.d_model if self.channel_independence == 0 else self.in_channels * self.d_model
        cls_in_dim = cls_feat_dim * (cls_pool_bins + 1) * n_scales
        self.cls_pool_bins = cls_pool_bins
        self.cls_attn_score = nn.Conv1d(cls_feat_dim, 1, kernel_size=1)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(cls_in_dim),
            nn.Linear(cls_in_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

        # ── 输出投影 ──────────────────────────────────────────
        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(self.d_model, 1)
            self.out_channel_proj = (nn.Linear(self.in_channels, self.out_channels)
                                     if self.in_channels != self.out_channels
                                     else nn.Identity())
        else:
            self.projection_layer = nn.Linear(self.d_model, self.out_channels)
            self.out_res_layers = nn.ModuleList([
                nn.Linear(self.seq_len // (self.down_sampling_window ** i),
                          self.seq_len // (self.down_sampling_window ** i))
                for i in range(self.down_sampling_layers + 1)
            ])
            self.regression_layers = nn.ModuleList([
                nn.Linear(self.seq_len // (self.down_sampling_window ** i), self.pred_len)
                for i in range(self.down_sampling_layers + 1)
            ])
            self.res_channel_proj = (nn.Linear(self.in_channels, self.out_channels)
                                     if self.in_channels != self.out_channels
                                     else nn.Identity())

        # ── 修复：conv 下采样层移到 __init__ 注册 ────────────
        if self.down_sampling_method == "conv":
            self.down_conv = nn.Conv1d(
                self.in_channels, self.in_channels,
                kernel_size=3, padding=1,
                stride=self.down_sampling_window,
                padding_mode="circular", bias=False,
                groups=self.in_channels,
            )

    # ── 下采样 ────────────────────────────────────────────────
    def _multi_scale_process_inputs(self, x):
        # x: (B, L, C)
        if self.down_sampling_method == "max":
            down_pool = nn.MaxPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "avg":
            down_pool = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            down_pool = self.down_conv   # 修复：使用注册的层
        else:
            raise ValueError(f"Unknown: {self.down_sampling_method}")

        x_BCL  = x.permute(0, 2, 1).contiguous()   # (B, C, L)
        x_list = [x]                                 # 原始分辨率
        x_cur  = x_BCL
        for _ in range(self.down_sampling_layers):
            x_cur  = down_pool(x_cur)               # (B, C, L//w)
            x_list.append(x_cur.permute(0, 2, 1))   # (B, L//w, C)
        return x_list

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return x_list, None
        out1, out2 = [], []
        for x in x_list:
            s, t = self.preprocess(x)
            out1.append(s)
            out2.append(t)
        return out1, out2

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = self.out_res_layers[i](out_res.permute(0, 2, 1))
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        out_res = self.res_channel_proj(out_res)
        return dec_out + out_res

    def future_multi_mixing(self, B, enc_out_list, x_list_tuple):
        seasonal_list, trend_list = x_list_tuple
        dec_out_list = []

        if self.channel_independence == 1:
            for i, enc_out in enumerate(enc_out_list):
                dec = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec = self.projection_layer(dec)                          # (B*N, P, 1)
                dec = dec.reshape(B, self.in_channels, self.pred_len).permute(0, 2, 1)
                dec = self.out_channel_proj(dec)
                dec_out_list.append(dec)
        else:
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, trend_list)):
                dec = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
                dec = self.out_projection(dec, i, out_res)
                dec_out_list.append(dec)

        return dec_out_list

    def _encode_inputs(self, x):
        x = _to_BLC(x, self.in_channels)   # (B, L, C)
        B = x.size(0)

        x_scales = self._multi_scale_process_inputs(x)

        x_list, norm_stats = [], []
        for i, x_i in enumerate(x_scales):
            x_i, mean_i, std_i = self.normalize_layers[i].norm(x_i)
            norm_stats.append((mean_i, std_i))
            if self.channel_independence == 1:
                B0, T0, N0 = x_i.shape
                x_i = x_i.permute(0, 2, 1).reshape(B0 * N0, T0, 1)
            x_list.append(x_i)

        x_list_tuple = self.pre_enc(x_list)

        enc_out_list = []
        src_list = x_list_tuple[0]
        for i, x_i in enumerate(src_list):
            enc = self.enc_embedding(x_i)
            if i == 0:
                x_raw = x_scales[0]
                if self.channel_independence == 1:
                    enc_reshape = enc.reshape(B, self.in_channels, x_i.shape[1], self.d_model)
                    enc_mean = enc_reshape.mean(dim=1)
                    enc_mean = self.freq_injection(x_raw, enc_mean)
                    enc_reshape = enc_reshape + enc_mean.unsqueeze(1)
                    enc = enc_reshape.reshape(B * self.in_channels, x_i.shape[1], self.d_model)
                else:
                    enc = self.freq_injection(x_raw, enc)
            enc_out_list.append(enc)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list)

        return B, enc_out_list, x_list_tuple, norm_stats

    def _encode_for_classification(self, x):
        x = _to_BLC(x, self.in_channels)
        B = x.size(0)

        x_scales = self._multi_scale_process_inputs(x)

        x_list = []
        for i, x_i in enumerate(x_scales):
            if self.cls_use_input_norm:
                x_i, _, _ = self.normalize_layers[i].norm(x_i)
            if self.channel_independence == 1:
                B0, T0, N0 = x_i.shape
                x_i = x_i.permute(0, 2, 1).reshape(B0 * N0, T0, 1)
            x_list.append(x_i)

        if self.cls_use_pre_enc:
            x_list = self.pre_enc(x_list)[0]

        enc_out_list = []
        for i, x_i in enumerate(x_list):
            enc = self.enc_embedding(x_i)
            if i == 0:
                x_raw = x_scales[0]
                if self.channel_independence == 1:
                    enc_reshape = enc.reshape(B, self.in_channels, x_i.shape[1], self.d_model)
                    enc_mean = enc_reshape.mean(dim=1)
                    enc_mean = self.freq_injection(x_raw, enc_mean)
                    enc_reshape = enc_reshape + enc_mean.unsqueeze(1)
                    enc = enc_reshape.reshape(B * self.in_channels, x_i.shape[1], self.d_model)
                else:
                    enc = self.freq_injection(x_raw, enc)
            enc_out_list.append(enc)

        for pdm in self.pdm_blocks:
            enc_out_list = pdm(enc_out_list)

        return B, enc_out_list

    def forecast(self, x):
        B, enc_out_list, x_list_tuple, norm_stats = self._encode_inputs(x)
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list_tuple)

        w = torch.softmax(self.scale_weights, dim=0)           # (n_scales,)
        stacked  = torch.stack(dec_out_list, dim=-1)           # (B, P, C, n_scales)
        dec_out  = (stacked * w).sum(dim=-1)                   # (B, P, C)

        mean0, std0 = norm_stats[0]
        if mean0 is not None:
            dec_out = self.normalize_layers[0].denorm(dec_out, mean0, std0)

        return dec_out   # (B, pred_len, C_out)

    def classification(self, x):
        B, enc_out_list = self._encode_for_classification(x)
        summaries = []
        for enc_out in enc_out_list:
            if self.channel_independence == 1:
                t = enc_out.size(1)
                enc_out = enc_out.view(B, self.in_channels, t, self.d_model)
                feat = enc_out.permute(0, 1, 3, 2).contiguous().view(B, self.in_channels * self.d_model, t)
            else:
                feat = enc_out.permute(0, 2, 1).contiguous()

            pooled = F.adaptive_avg_pool1d(feat, self.cls_pool_bins).flatten(1)
            attn_w = torch.softmax(self.cls_attn_score(feat), dim=-1)
            attn_pool = (feat * attn_w).sum(dim=-1)
            summaries.append(torch.cat([pooled, attn_pool], dim=1))

        summary = torch.cat(summaries, dim=1)
        return self.cls_head(summary)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
