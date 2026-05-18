import torch
import torch.nn as nn
import torch.nn.functional as F


def _forecast_to_bcl(pred: torch.Tensor, pred_len: int) -> torch.Tensor:
    """
    把不同模型的预测输出统一转成 (B, C, L) 供分类头使用。

    支持:
      - (B, P)
      - (B, 1, P)
      - (B, P, C)
      - (B, C, P)
    """
    if pred.dim() == 2:
        return pred.unsqueeze(1)  # (B,1,P)

    if pred.dim() != 3:
        raise ValueError(f"Expected forecast output dim 2 or 3, got shape {tuple(pred.shape)}")

    if pred.shape[1] == 1:
        return pred  # (B,1,P)

    if pred.shape[1] == int(pred_len):
        return pred.permute(0, 2, 1).contiguous()  # (B,P,C) -> (B,C,P)

    return pred.contiguous()


class ForecastClassifierWrapper(nn.Module):
    """
    给任意“预测模型”补一个分类头。

    逻辑:
      - 先跑原模型得到 forecast 输出
      - 再把 forecast 输出做统计池化，映射成风险分类 logits

    这样不用把每个模型单独改成两套 forward。
    """
    def __init__(self, base_model: nn.Module, configs):
        super().__init__()
        self.base_model = base_model
        self.pred_len = int(getattr(configs, "pred_len", 128))
        self.num_classes = int(getattr(configs, "num_classes", 2))
        self.pool_bins = int(getattr(configs, "cls_pool_bins", 16))
        hidden_dim = int(getattr(configs, "cls_hidden_dim", 128))
        dropout = float(getattr(configs, "dropout", 0.1))

        self.cls_head = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )

    def forward(self, x):
        pred = self.base_model(x)
        feat = _forecast_to_bcl(pred, self.pred_len)  # (B,C,P)

        pooled = F.adaptive_avg_pool1d(feat, self.pool_bins).flatten(1)
        mean = feat.mean(dim=-1)
        std = feat.std(dim=-1, unbiased=False)
        maxv = feat.amax(dim=-1)
        minv = feat.amin(dim=-1)
        last = feat[..., -1]

        summary = torch.cat([pooled, mean, std, maxv, minv, last], dim=1)
        return self.cls_head(summary)
