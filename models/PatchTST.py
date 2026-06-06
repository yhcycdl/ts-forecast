import torch
from torch import nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims = dims
        self.contiguous = contiguous

    def forward(self, x):
        x = x.transpose(*self.dims)
        return x.contiguous() if self.contiguous else x


class FlattenHead(nn.Module):
    """
    x: [B, C, D, P]  (vars, d_model, patch_num)
    out: [B, pred_len, C]
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0.0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)  # (D,P)->(D*P)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)      # [B,C,nf]
        x = self.linear(x)       # [B,C,pred_len]
        x = self.dropout(x)
        return x


def _to_BLC(x, in_channels: int):
    """
    支持:
      - (B,L)
      - (B,C,L)
      - (B,L,C)
    统一成 (B,L,C)
    """
    if x.dim() == 2:
        return x.unsqueeze(-1)  # (B,L,1)

    if x.dim() != 3:
        raise ValueError(f"Expected x dim 2 or 3, got {tuple(x.shape)}")

    # (B,C,L) -> (B,L,C)
    if x.shape[1] == in_channels:
        return x.permute(0, 2, 1).contiguous()

    # (B,L,C)
    if x.shape[2] == in_channels:
        return x

    raise ValueError(f"Cannot infer channel dim. got {tuple(x.shape)} with in_channels={in_channels}")


class Model(nn.Module):
    """
    PatchTST for forecasting (简化成你项目可用结构)
    - 输入: x (B,L) / (B,C,L) / (B,L,C)
    - 输出: y (B,pred_len,C_out)  (默认 C_out = in_channels)
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)

        # 兼容你项目常用命名
        self.in_channels = int(getattr(configs, "in_channels", getattr(configs, "enc_in", 1)))
        self.out_channels = int(getattr(configs, "out_channels", getattr(configs, "c_out", self.in_channels)))

        # PatchTST 默认是多变量同维预测，最自然 out_channels=in_channels
        # 如果你确实只预测部分通道，建议用 out_indices 在 data_loader 里控制
        if self.out_channels != self.in_channels:
            # 不直接禁止，但提醒：PatchTST 标准形式是 C_out=C_in
            pass

        # patch 参数（给默认值，避免你每个模型都要传一堆）
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

        # patch embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )

        # encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        )

        # head_nf = d_model * patch_num
        # patch_num = int((seq_len - patch_len)/stride + 2)  (原论文/实现)
        patch_num = int((self.seq_len - patch_len) / stride + 2)
        head_nf = d_model * patch_num

        self.head = FlattenHead(self.in_channels, head_nf, self.pred_len, head_dropout=dropout)
        self.patch_num = patch_num

    def _encode_backbone(self, x):
        x = _to_BLC(x, self.in_channels)

        means = x.mean(dim=1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x = x / stdev

        x = x.permute(0, 2, 1).contiguous()
        enc_in, n_vars = self.patch_embedding(x)
        enc_out, _ = self.encoder(enc_in)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2).contiguous()  # (B,C,D,P)
        return enc_out, means, stdev

    def forecast(self, x):
        enc_out, means, stdev = self._encode_backbone(x)
        if self.out_channels != self.in_channels:
            enc_out = enc_out[:, :self.out_channels, ...]
            means = means[..., :self.out_channels]
            stdev = stdev[..., :self.out_channels]
        out = self.head(enc_out)
        out = out.permute(0, 2, 1).contiguous()
        out = out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        out = out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return out

    def forward(self, x):
        return self.forecast(x)
