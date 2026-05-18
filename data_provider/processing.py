# data_provider/processing.py
import numpy as np
import pandas as pd

class DataScaler:
    """按列归一化（多变量推荐）"""
    def __init__(self, eps=1e-6):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, data: np.ndarray):
        if data.ndim == 1:
            data = data[:, None]
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std = np.maximum(self.std, self.eps)

    def transform(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            data = data[:, None]
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std, eps=np.array([self.eps], dtype=np.float32))

    def load(self, path: str):
        d = np.load(path)
        self.mean = d["mean"]
        self.std = d["std"]
        self.eps = float(d["eps"][0])


class GlobalScaler:
    """全局归一化（单变量推荐）"""
    def __init__(self, eps=1e-6):
        self.mean = 0.0
        self.std = 1.0
        self.eps = eps

    def fit(self, data: np.ndarray):
        self.mean = float(np.mean(data))
        self.std = float(np.std(data))

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / (self.std + self.eps)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * (self.std + self.eps) + self.mean

    def save(self, path: str):
        np.savez(path, mean=np.array([self.mean], dtype=np.float32),
                 std=np.array([self.std], dtype=np.float32),
                 eps=np.array([self.eps], dtype=np.float32))

    def load(self, path: str):
        d = np.load(path)
        self.mean = float(d["mean"][0])
        self.std = float(d["std"][0])
        self.eps = float(d["eps"][0])


def load_dataframe(csv_path: str, max_rows=None):
    df = pd.read_csv(csv_path)
    if max_rows is not None:
        df = df.iloc[:max_rows]
    return df


def select_feature_values(df: pd.DataFrame, col_names=None, dtype=np.float32) -> np.ndarray:
    if col_names is not None:
        df = df[col_names]
    else:
        df = df.select_dtypes(include=[np.number])
    return df.to_numpy(dtype=dtype)


def load_csv(csv_path: str, col_names=None, max_rows=None, dtype=np.float32) -> np.ndarray:
    df = load_dataframe(csv_path, max_rows=max_rows)
    return select_feature_values(df, col_names=col_names, dtype=dtype)  # (T,C)

def time_split(values: np.ndarray, train_ratio=0.8, val_ratio=0.0, split_mode="total"):
    T = len(values)
    if T <= 0:
        raise ValueError("values must contain at least one row.")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}.")

    split_mode = str(split_mode).lower()

    if split_mode == "legacy_rest":
        tr_end = int(T * train_ratio)
        train = values[:tr_end]
        rest = values[tr_end:]
        if val_ratio and val_ratio > 0:
            val_end = int(len(rest) * val_ratio)
            val = rest[:val_end]
            test = rest[val_end:]
            return train, val, test
        return train, None, rest

    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f"train_ratio + val_ratio must be < 1 when split_mode='total', got {train_ratio + val_ratio:.4f}."
        )

    tr_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    train = values[:tr_end]
    val = values[tr_end:val_end] if val_ratio and val_ratio > 0 else None
    test = values[val_end:]
    return train, val, test


def time_split_bounds(length: int, train_ratio=0.8, val_ratio=0.0, split_mode="total"):
    length = int(length)
    dummy = np.zeros((length, 1), dtype=np.float32)
    train, val, test = time_split(dummy, train_ratio=train_ratio, val_ratio=val_ratio, split_mode=split_mode)
    train_end = len(train)
    val_end = train_end + (0 if val is None else len(val))
    return (0, train_end), (train_end, val_end), (val_end, val_end + len(test))


def count_windows(length: int, seq_len: int, pred_len: int, stride: int) -> int:
    length = int(length)
    seq_len = int(seq_len)
    pred_len = int(pred_len)
    stride = int(stride)
    n = (length - seq_len - pred_len) // stride + 1
    return max(0, int(n))

