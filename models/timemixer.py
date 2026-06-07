import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_BLC(x, in_channels: int, seq_len: int):
    """
    支持:
      - (B, L)
      - (B, C, L)
      - (B, L, C)
    统一成 (B, L, C)
    """
    if x.dim() == 2:
        if int(in_channels) != 1 or x.shape[1] != int(seq_len):
            raise ValueError(
                "TimeMixer 2D input is only valid for single-channel "
                f"(B,{int(seq_len)}) tensors; got {tuple(x.shape)} with "
                f"enc_in={int(in_channels)}."
            )
        return x.unsqueeze(-1)  # (B,L,1)

    if x.dim() != 3:
        raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")

    # (B,C,L) -> (B,L,C)
    if x.shape[1] == int(in_channels) and x.shape[2] == int(seq_len):
        return x.permute(0, 2, 1).contiguous()

    # (B,L,C)
    if x.shape[1] == int(seq_len) and x.shape[2] == int(in_channels):
        return x.contiguous()

    raise ValueError(
        "TimeMixer input shape not recognized. Expected "
        f"(B,{int(in_channels)},{int(seq_len)}) or "
        f"(B,{int(seq_len)},{int(in_channels)}), got {tuple(x.shape)}."
    )


class SimpleNormalize(nn.Module):
    """
    兼容 TimeMixer 里 norm/denorm 的简单版。
    对每个 batch 的每个通道按时间维做归一化。
    """
    def __init__(self, num_features, affine=True, eps=1e-5, non_norm=False):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.non_norm = non_norm

        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self._last_mean = None
        self._last_std = None

    def forward(self, x, mode="norm"):
        if self.non_norm:
            return x

        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            self._last_mean = mean
            self._last_std = std

            x = (x - mean) / std
            if self.affine:
                x = x * self.weight + self.bias
            return x

        elif mode == "denorm":
            if self._last_mean is None or self._last_std is None:
                return x
            if self.affine:
                x = (x - self.bias) / (self.weight + self.eps)
            x = x * self._last_std + self._last_mean
            return x

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def denorm_subset(self, x, out_channels: int):
        if self.non_norm:
            return x
        if self._last_mean is None or self._last_std is None:
            return x

        out_channels = int(out_channels)
        in_channels = int(self._last_mean.shape[-1])
        if out_channels > in_channels:
            mean = self._last_mean[..., :1].expand(*self._last_mean.shape[:-1], out_channels)
            std = self._last_std[..., :1].expand(*self._last_std.shape[:-1], out_channels)
            if self.affine:
                weight = self.weight[..., :1].expand(*self.weight.shape[:-1], out_channels)
                bias = self.bias[..., :1].expand(*self.bias.shape[:-1], out_channels)
                x = (x - bias) / (weight + self.eps)
            return x * std + mean
        if self.affine:
            weight = self.weight[..., :out_channels]
            bias = self.bias[..., :out_channels]
            x = (x - bias) / (weight + self.eps)
        mean = self._last_mean[..., :out_channels]
        std = self._last_std[..., :out_channels]
        return x * std + mean


class MovingAvg(nn.Module):
    """
    简单移动平均，输入输出: [B, L, C]
    """
    def __init__(self, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B,L,C]
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        front = x[:, 0:1, :].repeat(1, left, 1)
        end = x[:, -1:, :].repeat(1, right, 1)
        x = torch.cat([front, x, end], dim=1)           # [B, L+pad, C]
        x = x.permute(0, 2, 1)                          # [B, C, L]
        x = self.avg(x)                                 # [B, C, L]
        x = x.permute(0, 2, 1)                          # [B, L, C]
        return x


class SeriesDecomp(nn.Module):
    """
    moving average decomposition
    x = seasonal + trend
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DFTSeriesDecomp(nn.Module):
    """
    DFT decomposition
    """
    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        # x: [B, L, C]
        xf = torch.fft.rfft(x, dim=1)
        freq = torch.abs(xf)
        freq[:, 0, :] = 0
        k = min(int(self.top_k), int(freq.shape[1]))
        if k <= 0:
            return torch.zeros_like(x), x
        top_k_freq, _ = torch.topk(freq, k, dim=1)
        threshold = top_k_freq.min(dim=1, keepdim=True).values
        xf = xf.masked_fill(freq <= threshold, 0)
        x_season = torch.fft.irfft(xf, n=x.size(1), dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class ValueEmbedding(nn.Module):
    """
    替代原始 DataEmbedding_wo_pos
    只做 value embedding，不依赖 time feature
    """
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.value_embedding(x))


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    输入列表元素: [B, D, L_i]
    输出列表元素: [B, L_i, D]
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        if len(season_list) <= 1:
            return [season_list[0].permute(0, 2, 1)]

        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    输入列表元素: [B, D, L_i]
    输出列表元素: [B, L_i, D]
    """
    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        if len(trend_list) <= 1:
            return [trend_list[0].permute(0, 2, 1)]

        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()

        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.down_sampling_window = int(getattr(configs, "down_sampling_window", 2))
        self.channel_independence = int(getattr(configs, "channel_independence", 0))

        d_model = int(getattr(configs, "d_model", 128))
        d_ff = int(getattr(configs, "d_ff", 256))
        dropout = float(getattr(configs, "dropout", 0.1))

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        decomp_method = str(getattr(configs, "decomp_method", "moving_avg"))
        moving_avg = int(getattr(configs, "moving_avg", 25))
        top_k = int(getattr(configs, "top_k", 5))

        if decomp_method == "moving_avg":
            self.decomposition = SeriesDecomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomposition = DFTSeriesDecomp(top_k)
        else:
            raise ValueError(f"Unknown decomp_method: {decomp_method}")

        if self.channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            self.seq_len, self.down_sampling_window, int(getattr(configs, "down_sampling_layers", 2))
        )
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            self.seq_len, self.down_sampling_window, int(getattr(configs, "down_sampling_layers", 2))
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        length_list = [x.size(1) for x in x_list]

        season_list = []
        trend_list = []

        for x in x_list:
            season, trend = self.decomposition(x)  # [B, L, D]
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)

            season_list.append(season.permute(0, 2, 1))  # [B, D, L]
            trend_list.append(trend.permute(0, 2, 1))    # [B, D, L]

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list, length_list):
            out = out_season + out_trend
            if self.channel_independence == 1:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])

        return out_list


class Model(nn.Module):
    """
    适配你项目的 TimeMixer:
    - 输入: (B,L) / (B,C,L) / (B,L,C)
    - 输出预测: (B,pred_len,C_out)
    - 输出分类: (B,num_classes)

    推荐:
    - M / MS 任务: channel_independence=0
    - S 任务: channel_independence=1 或 0 都可以
    """
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.task_name = str(getattr(configs, "task_name", "long_term_forecast"))

        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.num_classes = int(getattr(configs, "num_classes", 2))

        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(
            getattr(configs, "out_channels",
                    getattr(configs, "c_out",
                            getattr(configs, "out_in", self.in_channels)))
        )

        self.down_sampling_window = int(getattr(configs, "down_sampling_window", 2))
        self.down_sampling_layers = int(getattr(configs, "down_sampling_layers", 2))
        self.down_sampling_method = str(getattr(configs, "down_sampling_method", "avg"))
        min_scale_len = self.seq_len // (self.down_sampling_window ** self.down_sampling_layers)
        if min_scale_len < 1:
            raise ValueError(
                "TimeMixer down-sampling is too deep for seq_len: "
                f"seq_len={self.seq_len}, down_sampling_window={self.down_sampling_window}, "
                f"down_sampling_layers={self.down_sampling_layers}."
            )

        self.channel_independence = int(getattr(configs, "channel_independence", 0))
        self.e_layers = int(getattr(configs, "e_layers", 2))
        self.d_model = int(getattr(configs, "d_model", 128))
        self.d_ff = int(getattr(configs, "d_ff", 256))
        self.dropout = float(getattr(configs, "dropout", 0.1))
        self.use_norm = int(getattr(configs, "use_norm", 1))
        self.cls_use_input_norm = bool(int(getattr(configs, "cls_use_input_norm", 0)))
        self.cls_use_pre_enc = bool(int(getattr(configs, "cls_use_pre_enc", 0)))

        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs) for _ in range(self.e_layers)])

        moving_avg = int(getattr(configs, "moving_avg", 25))
        self.preprocess = SeriesDecomp(moving_avg)

        if self.channel_independence == 1:
            self.enc_embedding = ValueEmbedding(1, self.d_model, self.dropout)
        else:
            self.enc_embedding = ValueEmbedding(self.in_channels, self.d_model, self.dropout)

        self.normalize_layers = nn.ModuleList(
            [
                SimpleNormalize(
                    self.in_channels,
                    affine=True,
                    non_norm=True if self.use_norm == 0 else False
                )
                for _ in range(self.down_sampling_layers + 1)
            ]
        )

        if self.down_sampling_method == "max":
            self.down_sampler = nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == "avg":
            self.down_sampler = nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == "conv":
            self.down_sampler = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                padding=1,
                stride=self.down_sampling_window,
                padding_mode="circular",
                bias=False,
                groups=self.in_channels,
            )
        else:
            raise ValueError(f"Unknown down_sampling_method: {self.down_sampling_method}")

        self.predict_layers = nn.ModuleList(
            [
                nn.Linear(
                    self.seq_len // (self.down_sampling_window ** i),
                    self.pred_len,
                )
                for i in range(self.down_sampling_layers + 1)
            ]
        )

        if self.channel_independence == 1:
            self.projection_layer = nn.Linear(self.d_model, 1, bias=True)
        else:
            self.projection_layer = nn.Linear(self.d_model, self.out_channels, bias=True)

            self.out_res_layers = nn.ModuleList(
                [
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.seq_len // (self.down_sampling_window ** i),
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

            self.regression_layers = nn.ModuleList(
                [
                    nn.Linear(
                        self.seq_len // (self.down_sampling_window ** i),
                        self.pred_len,
                    )
                    for i in range(self.down_sampling_layers + 1)
                ]
            )

            # 支持 enc_in != c_out
            if self.in_channels != self.out_channels:
                self.res_channel_proj = nn.Linear(self.in_channels, self.out_channels)
            else:
                self.res_channel_proj = nn.Identity()

        # channel_independence=1 时如果 in_channels != out_channels，也做一层输出通道映射
        if self.channel_independence == 1 and self.in_channels != self.out_channels:
            self.out_channel_proj = nn.Linear(self.in_channels, self.out_channels)
        else:
            self.out_channel_proj = nn.Identity()

        self.cls_pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        cls_hidden = int(getattr(configs, "cls_hidden_dim", 256))
        cls_feat_dim = self.d_model if self.channel_independence == 0 else self.in_channels * self.d_model
        self.num_scales = self.down_sampling_layers + 1
        cls_in_dim = cls_feat_dim * (self.cls_pool_bins + 3) * self.num_scales
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, cls_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(cls_hidden, self.num_classes),
        )

    def _multi_scale_process_inputs(self, x):
        # x: [B,L,C] -> [B,C,L]
        x = x.permute(0, 2, 1).contiguous()

        x_ori = x
        x_list = [x.permute(0, 2, 1).contiguous()]

        for _ in range(self.down_sampling_layers):
            x_down = self.down_sampler(x_ori)
            x_list.append(x_down.permute(0, 2, 1).contiguous())
            x_ori = x_down

        return x_list

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x1, x2 = self.preprocess(x)  # seasonal, trend
                out1_list.append(x1)
                out2_list.append(x2)
            return (out1_list, out2_list)

    def out_projection(self, dec_out, i, out_res):
        """
        dec_out: [B, pred_len, d_model]
        out_res: [B, T_i, C_in]
        """
        dec_out = self.projection_layer(dec_out)  # [B, pred_len, C_out]

        out_res = out_res.permute(0, 2, 1).contiguous()  # [B, C_in, T_i]
        out_res = self.out_res_layers[i](out_res)        # [B, C_in, T_i]
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1).contiguous()  # [B, pred_len, C_in]
        out_res = self.res_channel_proj(out_res)         # [B, pred_len, C_out]

        dec_out = dec_out + out_res
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []

        if self.channel_independence == 1:
            x_list = x_list[0]  # 没实际用到，只保持和原结构一致
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                # enc_out: [B*N, T_i, d_model]
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
                dec_out = self.projection_layer(dec_out)  # [B*N, pred_len, 1]

                # -> [B, pred_len, in_channels]
                dec_out = dec_out.reshape(B, self.in_channels, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out = self.out_channel_proj(dec_out)  # 允许输出通道数 != 输入通道数

                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                # enc_out: [B, T_i, d_model]
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def _encode_inputs(self, x):
        x = _to_BLC(x, self.in_channels, self.seq_len)  # [B, L, C]
        B = x.size(0)

        # 1) 多尺度
        x_scales = self._multi_scale_process_inputs(x)

        # 2) 归一化 + （可选）通道独立 reshape
        x_list = []
        for i, x_i in enumerate(x_scales):
            # x_i: [B, T_i, C]
            x_i = self.normalize_layers[i](x_i, "norm")
            if self.channel_independence == 1:
                B0, T0, N0 = x_i.size()
                x_i = x_i.permute(0, 2, 1).contiguous().reshape(B0 * N0, T0, 1)
            x_list.append(x_i)

        # 3) preprocess
        x_list = self.pre_enc(x_list)

        # 4) embedding
        enc_out_list = []
        for x_i in x_list[0]:
            enc_out = self.enc_embedding(x_i)  # [B, T, d_model] or [B*N, T, d_model]
            enc_out_list.append(enc_out)

        # 5) Past Decomposable Mixing
        for i in range(self.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        return B, enc_out_list, x_list

    def _encode_for_classification(self, x):
        x = _to_BLC(x, self.in_channels, self.seq_len)  # [B, L, C]
        B = x.size(0)

        x_scales = self._multi_scale_process_inputs(x)

        x_list = []
        for i, x_i in enumerate(x_scales):
            if self.cls_use_input_norm:
                x_i = self.normalize_layers[i](x_i, "norm")
            if self.channel_independence == 1:
                B0, T0, N0 = x_i.size()
                x_i = x_i.permute(0, 2, 1).contiguous().reshape(B0 * N0, T0, 1)
            x_list.append(x_i)

        if self.cls_use_pre_enc:
            x_list = self.pre_enc(x_list)[0]

        enc_out_list = []
        for x_i in x_list:
            enc_out = self.enc_embedding(x_i)  # [B, T, d_model] or [B*N, T, d_model]
            enc_out_list.append(enc_out)

        for i in range(self.e_layers):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        return B, enc_out_list

    def forecast(self, x):
        B, enc_out_list, x_list = self._encode_inputs(x)
        # 6) Future Multipredictor Mixing
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        # 7) 多尺度融合
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)  # [B, pred_len, C_out]

        # 8) 反归一化（用最高分辨率那一层）
        # 如果是 channel_independence=1，layer0 存的是原始 [B,L,C] 的均值方差，仍可直接 denorm
        if dec_out.size(-1) == self.in_channels:
            dec_out = self.normalize_layers[0](dec_out, "denorm")
        else:
            dec_out = self.normalize_layers[0].denorm_subset(dec_out, dec_out.size(-1))

        return dec_out

    def classification(self, x):
        B, enc_out_list = self._encode_for_classification(x)
        summaries = []

        for enc_out in enc_out_list:
            if self.channel_independence == 1:
                t = enc_out.size(1)
                enc_out = enc_out.view(B, self.in_channels, t, self.d_model)
                feat = enc_out.permute(0, 1, 3, 2).contiguous().view(B, self.in_channels * self.d_model, t)
            else:
                feat = enc_out.permute(0, 2, 1).contiguous()  # (B, d_model, T)

            pooled = F.adaptive_avg_pool1d(feat, self.cls_pool_bins).flatten(1)
            mean = feat.mean(dim=-1)
            maxv = feat.amax(dim=-1)
            last = feat[..., -1]
            summaries.append(torch.cat([pooled, mean, maxv, last], dim=1))

        summary = torch.cat(summaries, dim=1)
        return self.cls_head(summary)

    def forward(self, x):
        if self.task_name == "risk_classification":
            return self.classification(x)
        return self.forecast(x)
