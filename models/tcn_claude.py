"""
QPWave-TCN: long-receptive-field TCN for smooth quasi-periodic waveform forecasting.

架构：
  ┌─────────────────────────────────────────┐
  │  输入 (B, 1, 4096)                       │
  │      ↓ 实例归一化（消除非平稳均值漂移）    │
  │  ┌───┴────────────────────────────┐      │
  │  │ 时域：多尺度 TCN               │      │
  │  │   3路并行卷积核(3/7/15)        │      │
  │  │   + 膨胀残差块 × N 层          │      │
  │  │   → 最后一步特征 (B, H)        │      │
  │  └───────────────────────────────┘      │
  │  ┌───────────────────────────────┐      │
  │  │ 频域：FFT幅值谱               │      │
  │  │   log压缩 → top-k → MLP       │      │
  │  │   → 频域特征 (B, freq_dim)     │      │
  │  └───────────────────────────────┘      │
  │      ↓ concat                           │
  │  融合 MLP → delta                        │
  │      ↓ + last_val（残差输出）            │
  │  预测 (B, pred_len, 1)                   │
  └─────────────────────────────────────────┘

关键设计：
  1. 实例归一化 (RevIN)：消除每个样本的均值/方差漂移
  2. 多尺度并行卷积核：同时捕获细节(k=3)、中程(k=7)、长程(k=15)模式
  3. 频域分支：直接提取准周期谐波结构，与时域互补
  4. 残差输出：只学delta，降低学习难度
  5. 感受野自动裁剪：不超过 seq_len

args 必须：seq_len, pred_len, enc_in, c_out
args 可选：
  num_layers   (int,   default=11)   TCN层数，自动裁剪
  base_ch      (int,   default=32)   起始通道
  max_ch       (int,   default=256)  最大通道
  dropout      (float, default=0.1)
  top_k_freq   (int,   default=128)  取前k个频率分量
  freq_dim     (int,   default=128)  频域特征维度
  use_revin    (int,   default=1)    是否使用实例归一化
  residual_output (int, default=1)    是否输出 last_value + delta
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 模块1：RevIN（可逆实例归一化）
# 解决燃烧信号的非平稳问题：每个样本窗口独立归一化
# 预测后再还原到原始尺度
# ─────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """
    Reversible Instance Normalization
    - 输入归一化：每个样本独立去均值、除方差
    - 输出反归一化：还原到原始尺度
    这样模型看到的永远是"零均值单位方差"的信号，
    大幅降低不同燃烧工况之间的分布差异
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        # x: (B, C, L)
        if mode == 'norm':
            self._mean = x.mean(dim=-1, keepdim=True)           # (B, C, 1)
            self._std  = x.std(dim=-1, keepdim=True) + self.eps # (B, C, 1)
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.weight[None, :, None] + self.bias[None, :, None]
            return x
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.bias[None, :, None]) / (self.weight[None, :, None] + self.eps)
            return x * self._std + self._mean
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")

    def denorm_subset(self, x, out_channels: int):
        out_channels = int(out_channels)
        if self.affine:
            bias = self.bias[:out_channels][None, :, None]
            weight = self.weight[:out_channels][None, :, None]
            x = (x - bias) / (weight + self.eps)
        mean = self._mean[:, :out_channels, :]
        std = self._std[:, :out_channels, :]
        return x * std + mean


# ─────────────────────────────────────────────────────────
# 模块2：因果膨胀卷积块
# ─────────────────────────────────────────────────────────

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation

        self.conv1  = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp1 = Chomp1d(pad)
        self.act1   = nn.GELU()
        self.drop1  = nn.Dropout(dropout)

        self.conv2  = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.chomp2 = Chomp1d(pad)
        self.act2   = nn.GELU()
        self.drop2  = nn.Dropout(dropout)

        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.drop1(self.act1(self.chomp1(self.conv1(x))))
        out = self.drop2(self.act2(self.chomp2(self.conv2(out))))
        res = x if self.proj is None else self.proj(x)
        return F.gelu(out + res)


# ─────────────────────────────────────────────────────────
# 模块3：多尺度输入嵌入
# 3路并行卷积核（3/7/15），同时捕获不同尺度的局部模式
# 燃烧信号既有高频细节（爆震）又有中低频趋势（燃烧包络）
# ─────────────────────────────────────────────────────────

class MultiScaleEmbedding(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        assert out_ch % 3 == 0, "out_ch 需要能被3整除"
        branch_ch = out_ch // 3

        # 3路并行因果卷积，不同核大小
        self.branches = nn.ModuleList([
            self._causal_conv(in_ch, branch_ch, k)
            for k in [3, 7, 15]
        ])
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _causal_conv(in_ch, out_ch, kernel_size):
        pad = kernel_size - 1
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad),
            Chomp1d(pad),
        )

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]  # 3×(B, branch_ch, L)
        out  = torch.cat(outs, dim=1)                    # (B, out_ch, L)
        return self.drop(self.act(out))


# ─────────────────────────────────────────────────────────
# 模块4：频域分支
# ─────────────────────────────────────────────────────────

class FreqBranch(nn.Module):
    def __init__(self, seq_len, top_k=128, freq_dim=128, dropout=0.1):
        super().__init__()
        self.top_k = min(top_k, seq_len // 2 + 1)

        self.mlp = nn.Sequential(
            nn.Linear(self.top_k, freq_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(freq_dim * 2, freq_dim),
            nn.GELU(),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, C, L)
        sig = x[:, 0, :]                           # (B, L)
        amp = torch.fft.rfft(sig, dim=-1).abs()    # (B, L//2+1)
        amp = torch.log1p(amp)                     # log 压缩
        amp = amp[:, :self.top_k]                  # (B, top_k)
        return self.mlp(amp)                       # (B, freq_dim)


# ─────────────────────────────────────────────────────────
# 主模型
# ─────────────────────────────────────────────────────────

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.seq_len      = int(args.seq_len)
        self.pred_len     = int(args.pred_len)
        self.in_channels  = int(getattr(args, "enc_in",     1))
        self.out_channels = int(getattr(args, "c_out",      1))

        kernel_size  = int(getattr(args,   "kernel_size",  3))
        dropout      = float(getattr(args, "dropout",      0.1))
        base_ch      = int(getattr(args,   "base_ch",      32))
        max_ch       = int(getattr(args,   "max_ch",       256))
        top_k_freq   = int(getattr(args,   "top_k_freq",   128))
        freq_dim     = int(getattr(args,   "freq_dim",     128))
        self.use_revin = bool(getattr(args, "use_revin",   1))
        self.residual_output = bool(getattr(args, "residual_output", 1))

        # ── RevIN ───────────────────────────────────────────
        if self.use_revin:
            self.revin = RevIN(self.in_channels)

        # ── 多尺度嵌入层 ─────────────────────────────────────
        # out_ch 必须能被3整除
        embed_ch = 96   # 32×3
        self.embed = MultiScaleEmbedding(self.in_channels, embed_ch, dropout=dropout)

        # ── TCN 主干（感受野自动裁剪） ───────────────────────
        max_layers = int(math.floor(math.log2(max((self.seq_len - 1) / max(kernel_size - 1, 1), 1))))
        max_layers = max(1, max_layers)
        num_layers = min(int(getattr(args, "num_layers", 11)), max_layers)
        rf = (kernel_size - 1) * (2 ** num_layers - 1) + 1
        print(f"[QPWaveTCN] layers={num_layers}  RF={rf} samples  seq_len={self.seq_len}")

        ch = [min(base_ch * (2 ** i), max_ch) for i in range(num_layers)]
        tcn_layers = []
        for i in range(num_layers):
            in_ch = embed_ch if i == 0 else ch[i - 1]
            tcn_layers.append(DilatedBlock(in_ch, ch[i], kernel_size,
                                           dilation=2 ** i, dropout=dropout))
        self.tcn    = nn.Sequential(*tcn_layers)
        tcn_hidden  = ch[-1]

        # ── 频域分支 ──────────────────────────────────────────
        self.freq_branch = FreqBranch(self.seq_len,
                                      top_k=top_k_freq,
                                      freq_dim=freq_dim,
                                      dropout=dropout)

        # ── 融合解码头 ────────────────────────────────────────
        fused_dim = tcn_hidden + freq_dim
        mid_dim   = min(fused_dim * 2, 1024)
        self.decoder = nn.Sequential(
            nn.Linear(fused_dim, mid_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim // 2, self.pred_len * self.out_channels),
        )
        for m in self.decoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _to_BCL(self, x):
        if x.dim() == 2:
            if self.in_channels != 1 or x.shape[1] != self.seq_len:
                raise ValueError(
                    "QPWaveTCN 2D input is only valid for single-channel "
                    f"(B,{self.seq_len}) tensors; got {tuple(x.shape)} with "
                    f"enc_in={self.in_channels}."
                )
            return x.unsqueeze(1)
        if x.dim() == 3:
            if x.shape[1] == self.seq_len and x.shape[2] == self.in_channels:
                return x.permute(0, 2, 1).contiguous()
            if x.shape[1] == self.in_channels and x.shape[2] == self.seq_len:
                return x
            raise ValueError(
                "QPWaveTCN input shape not recognized. Expected "
                f"(B,{self.in_channels},{self.seq_len}) or "
                f"(B,{self.seq_len},{self.in_channels}), got {tuple(x.shape)}."
            )
        raise ValueError(f"Expected x.dim() in [2,3], got {tuple(x.shape)}")

    def _extract_features(self, x, apply_input_norm: bool):
        x = self._to_BCL(x)                              # (B, C, L)

        if self.use_revin and apply_input_norm:
            x = self.revin(x, 'norm')

        # 2. 记录最后已知值（用于残差输出）
        last_val = x[:, :self.out_channels, -1].unsqueeze(1)  # (B, 1, C_out)

        # 3. 时域：多尺度嵌入 → TCN → 最后一步
        feat      = self.embed(x)                         # (B, embed_ch, L)
        tcn_out   = self.tcn(feat)                        # (B, H, L)
        time_feat = tcn_out[:, :, -1]                     # (B, H)

        # 4. 频域分支
        freq_feat = self.freq_branch(x)                   # (B, freq_dim)

        # 5. 融合解码
        fused = torch.cat([time_feat, freq_feat], dim=-1) # (B, H+freq_dim)
        return fused, last_val, tcn_out, freq_feat

    def forecast(self, x):
        fused, last_val, _, _ = self._extract_features(x, apply_input_norm=True)
        delta = self.decoder(fused)                       # (B, P*C_out)
        delta = delta.view(-1, self.pred_len, self.out_channels)  # (B, P, C_out)

        # 6. 残差输出：预测 = 最后已知值 + delta。
        # raw->smooth 或不同物理量预测时可用 --residual_output 0 关闭该先验。
        out = last_val + delta if self.residual_output else delta  # (B, P, C_out)

        # 7. RevIN 反归一化（还原到原始尺度）
        if self.use_revin:
            # out: (B, P, C_out) → (B, C_out, P) → denorm → (B, P, C_out)
            out = out.permute(0, 2, 1)                    # (B, C_out, P)
            if self.out_channels == self.in_channels:
                out = self.revin(out, 'denorm')
            else:
                out = self.revin.denorm_subset(out, self.out_channels)
            out = out.permute(0, 2, 1)                    # (B, P, C_out)

        return out                                        # (B, pred_len, C_out)

    def forward(self, x):
        return self.forecast(x)
