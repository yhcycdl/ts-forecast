# data_provider/dataset_custom.py
import numpy as np
import torch
from torch.utils.data import Dataset

class WindowDataset(Dataset):
    """
    input series: (T,C) or (T,)
    returns:
      x: (C_in, seq_len)
      y: (pred_len, C_out)  # 注意：你的 FullResTCN 输出是 (B,1,pred_len)，要对齐你 loss/metric
    """
    def __init__(
        self,
        series,
        seq_len,
        pred_len,
        stride=1,
        in_indices=None,
        out_indices=None,
        target_shift=0,
        sample_weights=None,
        window_mode="past",
        center_left=None,
        start_indices=None,
    ):
        if series.ndim == 1:
            series = series[:, None]
        self.series = torch.tensor(series, dtype=torch.float32)  # (T,C)
        self.sample_weights = None if sample_weights is None else torch.tensor(sample_weights, dtype=torch.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.target_shift = int(target_shift)
        self.window_mode = str(window_mode).lower()
        self.center_left = self.seq_len // 2 if center_left is None else int(center_left)
        self.start_indices = None if start_indices is None else np.asarray(start_indices, dtype=np.int64).reshape(-1)
        if self.window_mode not in {"past", "center"}:
            raise ValueError(f"Unknown window_mode: {window_mode}")
        if self.window_mode == "past" and self.target_shift < 0:
            raise ValueError("target_shift must be non-negative when window_mode='past'.")
        if self.window_mode == "center":
            if self.center_left < 0 or self.center_left + self.pred_len > self.seq_len:
                raise ValueError(
                    "For window_mode='center', center_left must satisfy "
                    "0 <= center_left and center_left + pred_len <= seq_len."
                )
        n_channels = int(self.series.shape[1])
        if in_indices is None:
            in_indices = np.arange(n_channels, dtype=np.int64)
        if out_indices is None:
            out_indices = np.arange(n_channels, dtype=np.int64)
        self.in_indices = np.asarray(in_indices, dtype=np.int64).reshape(-1)
        self.out_indices = np.asarray(out_indices, dtype=np.int64).reshape(-1)

        T = self.series.shape[0]
        if self.start_indices is not None:
            self.n_samples = int(self.start_indices.size)
        else:
            if self.window_mode == "center":
                n = (T - self.seq_len) // self.stride + 1
            else:
                n = (T - self.seq_len - self.target_shift - self.pred_len) // self.stride + 1
            self.n_samples = max(0, int(n))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        s = int(self.start_indices[idx]) if self.start_indices is not None else idx * self.stride
        m = s + self.seq_len
        if self.window_mode == "center":
            y_start = s + self.center_left
        else:
            y_start = m + self.target_shift
        e = y_start + self.pred_len

        x = self.series[s:m, self.in_indices]      # (seq_len, C_in)
        y = self.series[y_start:e, self.out_indices]     # (pred_len, C_out)
        w = None if self.sample_weights is None else self.sample_weights[y_start:e]

        x = x.transpose(0, 1).contiguous()  # (C, seq_len)
        # y 先保持 (pred_len,C)，HybridLoss 里你可以决定怎么用
        if w is None:
            return x, y
        if w.dim() == 1:
            w = w.unsqueeze(-1)
        return x, y, w
