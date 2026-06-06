import torch
import torch.nn as nn


class SeriesDecomp(nn.Module):
    """
    moving average decomposition:
      x = trend + seasonal
      trend = moving_avg(x)
      seasonal = x - trend

    输入:  (B, L, C)
    输出: seasonal, trend  (B, L, C)
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        k = int(kernel_size)
        if k <= 1:
            raise ValueError("moving_avg kernel_size must be > 1")
        self.kernel_size = k
        self.avg = nn.AvgPool1d(kernel_size=k, stride=1, padding=0)

    def forward(self, x):
        # x: (B,L,C)
        x_bcL = x.permute(0, 2, 1)            # (B,C,L)
        # Keep output length equal to input length for both odd and even kernels.
        # AvgPool1d's symmetric padding shortens even-kernel outputs by one.
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        front = x_bcL[:, :, :1].repeat(1, 1, left)
        end = x_bcL[:, :, -1:].repeat(1, 1, right)
        padded = torch.cat([front, x_bcL, end], dim=-1)
        trend = self.avg(padded)              # (B,C,L)
        trend = trend.permute(0, 2, 1)        # (B,L,C)
        seasonal = x - trend
        return seasonal, trend


class Model(nn.Module):
    """
    DLinear (Decomposition-Linear) for forecasting
    - 输入：x (B,L) or (B,C,L) or (B,L,C)
    - 输出：y (B,pred_len,C_out)

    configs 需要字段：
      - seq_len
      - pred_len
      - in_channels (或 enc_in)
      - out_channels (或 c_out)  [可选，默认=in_channels]
      - moving_avg  [必须>1]
      - out_indices [可选，list[int]，想预测哪些通道；默认全通道]
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))

        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", self.in_channels)))

        self.moving_avg = int(getattr(configs, "moving_avg", 25))
        self.decomp = SeriesDecomp(self.moving_avg)

        self.individual = bool(getattr(configs, "individual", False))

        # 预测哪些通道：默认全部
        out_indices = getattr(configs, "out_indices", None)
        if out_indices is None:
            self.out_indices = list(range(self.in_channels))
        else:
            self.out_indices = list(out_indices)

        # DLinear最自然的是输出通道数=输入通道数
        # 若你设置了 out_channels，但和 out_indices 不一致，按 out_indices 的数量为准更合理
        self.c_out = len(self.out_indices)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.c_out):
                ls = nn.Linear(self.seq_len, self.pred_len)
                lt = nn.Linear(self.seq_len, self.pred_len)
                # 初始化为均值投影（论文常用初始化）
                ls.weight = nn.Parameter((1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
                lt.weight = nn.Parameter((1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
                self.Linear_Seasonal.append(ls)
                self.Linear_Trend.append(lt)
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Seasonal.weight = nn.Parameter((1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))
            self.Linear_Trend.weight = nn.Parameter((1 / self.seq_len) * torch.ones(self.pred_len, self.seq_len))

    def _prepare(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B,L,1)
        elif x.dim() == 3:
            if x.shape[1] == self.in_channels and x.shape[2] == self.seq_len:
                x = x.permute(0, 2, 1).contiguous()
            elif x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                pass
            else:
                raise ValueError(
                    f"Input shape not recognized. Expect (B,L), (B,C,L) or (B,L,C). got {tuple(x.shape)}"
                )
        else:
            raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")
        x = x[:, :, self.out_indices]
        seasonal, trend = self.decomp(x)  # (B,L,C_out), (B,L,C_out)
        return seasonal, trend

    def forecast(self, x):
        seasonal, trend = self._prepare(x)

        seasonal = seasonal.permute(0, 2, 1).contiguous()  # (B,C,L)
        trend = trend.permute(0, 2, 1).contiguous()        # (B,C,L)

        if self.individual:
            seasonal_out = torch.zeros(seasonal.size(0), seasonal.size(1), self.pred_len, device=seasonal.device, dtype=seasonal.dtype)
            trend_out = torch.zeros_like(seasonal_out)
            for i in range(self.c_out):
                seasonal_out[:, i, :] = self.Linear_Seasonal[i](seasonal[:, i, :])
                trend_out[:, i, :] = self.Linear_Trend[i](trend[:, i, :])
        else:
            seasonal_out = self.Linear_Seasonal(seasonal)  # (B,C,P)
            trend_out = self.Linear_Trend(trend)           # (B,C,P)

        out = seasonal_out + trend_out  # (B,C,P)

        out = out.permute(0, 2, 1).contiguous()  # (B,P,C_out)
        return out

    def forward(self, x):
        return self.forecast(x)
