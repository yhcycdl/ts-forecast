import numpy as np
import torch
from torch.utils.data import Dataset

from data_provider.data_factory import _build_risk_loader, _risk_data_provider


class JointWindowDataset(Dataset):
    """
    多任务数据集：
      - x: 历史窗口 (C, seq_len)
      - y_forecast: 未来波形 (pred_len, C)
      - y_cls: 未来风险标签 int64
    """

    def __init__(
        self,
        series,
        labels,
        seq_len,
        pred_len,
        stride=1,
        start_indices=None,
        label_config=None,
        window_stats=None,
    ):
        series = torch.as_tensor(series, dtype=torch.float32)
        if series.dim() == 1:
            series = series.unsqueeze(-1)
        if series.dim() != 2:
            raise ValueError(f"series must be 1D or 2D, got shape {tuple(series.shape)}")

        self.series = series.contiguous()  # (T, C)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)

        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if start_indices is None:
            expected = max(0, (self.series.shape[0] - self.seq_len - self.pred_len) // self.stride + 1)
            if labels.shape[0] != expected:
                raise ValueError(f"labels length {labels.shape[0]} does not match expected window count {expected}.")
            start_indices = np.arange(expected, dtype=np.int64) * self.stride
        else:
            start_indices = np.asarray(start_indices, dtype=np.int64).reshape(-1)
            if labels.shape[0] != start_indices.shape[0]:
                raise ValueError("labels and start_indices must have the same length.")
            if start_indices.size == 0:
                raise ValueError("No joint windows found for this split.")
            max_end = start_indices + self.seq_len + self.pred_len
            if np.any(start_indices < 0) or np.any(max_end > self.series.shape[0]):
                raise ValueError("Some start_indices fall outside the available series range.")

        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.start_indices = start_indices
        self.n_samples = int(start_indices.shape[0])
        self.label_config = dict(label_config or {})
        self.window_stats = dict(window_stats or {})

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = int(self.start_indices[idx])
        m = s + self.seq_len
        e = m + self.pred_len

        x = self.series[s:m, :].transpose(0, 1).contiguous()  # (C, seq_len)
        y_forecast = self.series[m:e, :]                       # (pred_len, C)
        y_cls = self.labels[idx]
        return x, y_forecast, y_cls


def joint_data_provider(args, flag):
    """
    复用风险分类的数据划分/打标逻辑，只把样本扩展成：
      x + future waveform + risk label
    """
    risk_ds, _, scaler = _risk_data_provider(args, flag)
    joint_ds = JointWindowDataset(
        series=risk_ds.series,
        labels=risk_ds.labels.detach().cpu().numpy(),
        seq_len=risk_ds.seq_len,
        pred_len=risk_ds.pred_len,
        stride=risk_ds.stride,
        start_indices=getattr(risk_ds, "start_indices", None),
        label_config=getattr(risk_ds, "label_config", None),
        window_stats=getattr(risk_ds, "window_stats", None),
    )

    for attr in ("class_weights", "class_counts", "label_config", "window_stats"):
        if hasattr(risk_ds, attr):
            setattr(joint_ds, attr, getattr(risk_ds, attr))

    return joint_ds, _build_risk_loader(args, joint_ds, flag), scaler
