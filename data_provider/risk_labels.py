import os

import numpy as np
import pandas as pd

from data_provider.processing import count_windows


def _safe_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    q = float(np.clip(q, 0.0, 1.0))
    return float(np.quantile(values, q))


def _band_energy_ratio(window: np.ndarray, sample_rate: float, band_low: float, band_high: float) -> float:
    window = np.asarray(window, dtype=np.float32).reshape(-1)
    if window.size < 4 or sample_rate <= 0 or band_high <= band_low:
        return 0.0

    centered = window - float(window.mean())
    spec = np.fft.rfft(centered)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(window.size, d=1.0 / float(sample_rate))

    if power.size <= 1:
        return 0.0

    total_power = float(np.sum(power[1:]))
    if total_power <= 1e-12:
        return 0.0

    mask = (freqs >= float(band_low)) & (freqs <= float(band_high))
    band_power = float(np.sum(power[mask]))
    return band_power / total_power


def compute_risk_window_stats(
    values: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
    target_channel: int = 0,
    sample_rate: float = 1.0,
    band_low: float = 0.0,
    band_high: float = 0.0,
):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        signal = values
    elif values.ndim == 2:
        if not 0 <= int(target_channel) < values.shape[1]:
            raise ValueError(
                f"target_channel={target_channel} is out of range for values with {values.shape[1]} channels."
            )
        signal = values[:, int(target_channel)]
    else:
        raise ValueError(f"values must be 1D or 2D, got shape {values.shape}.")

    n_windows = count_windows(len(signal), seq_len, pred_len, stride)
    start_indices = np.arange(n_windows, dtype=np.int64) * int(stride)
    rms = np.zeros(n_windows, dtype=np.float32)
    ber = np.zeros(n_windows, dtype=np.float32)

    for idx, start in enumerate(start_indices):
        middle = int(start) + int(seq_len)
        end = middle + int(pred_len)
        future = signal[middle:end]
        rms[idx] = float(np.sqrt(np.mean(np.square(future))))
        ber[idx] = float(_band_energy_ratio(future, sample_rate, band_low, band_high))

    return {"start_indices": start_indices, "rms": rms, "ber": ber}


def compute_risk_stats_from_future_windows(
    future_windows: np.ndarray,
    target_channel: int = 0,
    sample_rate: float = 1.0,
    band_low: float = 0.0,
    band_high: float = 0.0,
):
    """
    直接对未来窗口集合计算风险统计量。

    输入支持:
      - (N, P)
      - (P,)
      - (N, P, C)
    返回:
      {"rms": (N,), "ber": (N,)}
    """
    windows = np.asarray(future_windows, dtype=np.float32)

    if windows.ndim == 1:
        signal = windows[None, :]
    elif windows.ndim == 2:
        signal = windows
    elif windows.ndim == 3:
        if not 0 <= int(target_channel) < windows.shape[2]:
            raise ValueError(
                f"target_channel={target_channel} is out of range for future_windows with {windows.shape[2]} channels."
            )
        signal = windows[:, :, int(target_channel)]
    else:
        raise ValueError(f"future_windows must be 1D/2D/3D, got shape {windows.shape}.")

    rms = np.sqrt(np.mean(np.square(signal), axis=1, dtype=np.float32)).astype(np.float32)
    ber = np.zeros(signal.shape[0], dtype=np.float32)
    for idx, window in enumerate(signal):
        ber[idx] = float(_band_energy_ratio(window, sample_rate, band_low, band_high))

    return {"rms": rms, "ber": ber}


def fit_risk_label_config(train_values: np.ndarray, args):
    train_stats = compute_risk_window_stats(
        train_values,
        seq_len=getattr(args, "seq_len"),
        pred_len=getattr(args, "pred_len"),
        stride=getattr(args, "stride"),
        target_channel=getattr(args, "risk_label_channel", 0),
        sample_rate=float(getattr(args, "sample_rate", 1.0)),
        band_low=float(getattr(args, "risk_band_low", 0.0)),
        band_high=float(getattr(args, "risk_band_high", 0.0)),
    )

    num_classes = int(getattr(args, "num_classes", 2))
    if num_classes not in (2, 3):
        raise ValueError(f"num_classes must be 2 or 3, got {num_classes}.")

    use_ber = bool(int(getattr(args, "risk_use_ber", 1)))
    use_ber = use_ber and float(getattr(args, "risk_band_high", 0.0)) > float(getattr(args, "risk_band_low", 0.0))
    use_ber = use_ber and bool(np.any(train_stats["ber"] > 1e-8))

    thresholds = {
        "rms_low": _safe_quantile(train_stats["rms"], getattr(args, "risk_rms_low_quantile", 0.5)),
        "rms_high": _safe_quantile(train_stats["rms"], getattr(args, "risk_rms_high_quantile", 0.85)),
        "ber_low": _safe_quantile(train_stats["ber"], getattr(args, "risk_ber_low_quantile", 0.5)) if use_ber else 0.0,
        "ber_high": _safe_quantile(train_stats["ber"], getattr(args, "risk_ber_high_quantile", 0.85)) if use_ber else 0.0,
    }

    class_names = ["stable", "unstable"] if num_classes == 2 else ["stable", "pre_instability", "unstable"]
    label_config = {
        "source": "generated",
        "num_classes": num_classes,
        "class_names": class_names,
        "use_ber": use_ber,
        "target_channel": int(getattr(args, "risk_label_channel", 0)),
        "sample_rate": float(getattr(args, "sample_rate", 1.0)),
        "band_low": float(getattr(args, "risk_band_low", 0.0)),
        "band_high": float(getattr(args, "risk_band_high", 0.0)),
        "thresholds": thresholds,
    }
    return label_config, train_stats


def assign_risk_labels(stats: dict, label_config: dict) -> np.ndarray:
    rms = np.asarray(stats["rms"], dtype=np.float32).reshape(-1)
    ber = np.asarray(stats.get("ber", np.zeros_like(rms)), dtype=np.float32).reshape(-1)
    thresholds = label_config["thresholds"]
    use_ber = bool(label_config.get("use_ber", False))
    num_classes = int(label_config["num_classes"])

    if num_classes == 2:
        positive = rms >= float(thresholds["rms_high"])
        if use_ber:
            positive = positive | (ber >= float(thresholds["ber_high"]))
        return positive.astype(np.int64)

    stable = rms < float(thresholds["rms_low"])
    unstable = rms >= float(thresholds["rms_high"])
    if use_ber:
        stable = stable & (ber < float(thresholds["ber_low"]))
        unstable = unstable & (ber >= float(thresholds["ber_high"]))

    labels = np.ones(rms.shape[0], dtype=np.int64)
    labels[stable] = 0
    labels[unstable] = 2
    return labels


def risk_probabilities_from_stats(stats: dict, label_config: dict) -> np.ndarray:
    """
    根据后处理统计量构造 soft probability，方便 M3 计算 AUROC/AUPRC。
    注意：hard label 仍然应以 assign_risk_labels(...) 为准。
    """
    rms = np.asarray(stats["rms"], dtype=np.float32).reshape(-1)
    ber = np.asarray(stats.get("ber", np.zeros_like(rms)), dtype=np.float32).reshape(-1)

    thresholds = dict(label_config.get("thresholds", {}))
    rms_low = float(thresholds.get("rms_low", thresholds.get("rms_high", 0.0)))
    rms_high = float(thresholds.get("rms_high", rms_low))
    ber_low = float(thresholds.get("ber_low", thresholds.get("ber_high", 0.0)))
    ber_high = float(thresholds.get("ber_high", ber_low))

    use_ber = bool(label_config.get("use_ber", False))
    num_classes = int(label_config.get("num_classes", 2))

    def _sigmoid(x):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax(logits):
        logits = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(np.clip(logits, -60.0, 60.0))
        probs_sum = np.maximum(np.sum(probs, axis=1, keepdims=True), 1e-6)
        return probs / probs_sum

    rms_scale = max(abs(rms_high - rms_low), 1e-6)
    ber_scale = max(abs(ber_high - ber_low), 1e-6)

    if num_classes == 2:
        score = (rms - rms_high) / rms_scale
        if use_ber:
            score = np.maximum(score, (ber - ber_high) / ber_scale)
        positive = _sigmoid(score)
        return np.stack([1.0 - positive, positive], axis=1).astype(np.float32)

    rms_mid = 0.5 * (rms_low + rms_high)
    stable_score = (rms_low - rms) / rms_scale
    unstable_score = (rms - rms_high) / rms_scale
    pre_score = 1.0 - np.abs(rms - rms_mid) / max(0.5 * abs(rms_high - rms_low), 1e-6)

    if use_ber:
        ber_mid = 0.5 * (ber_low + ber_high)
        stable_score = np.minimum(stable_score, (ber_low - ber) / ber_scale)
        unstable_score = np.minimum(unstable_score, (ber - ber_high) / ber_scale)
        pre_score = 0.5 * (
            pre_score
            + (1.0 - np.abs(ber - ber_mid) / max(0.5 * abs(ber_high - ber_low), 1e-6))
        )

    logits = np.stack([stable_score, pre_score, unstable_score], axis=1) * 4.0
    return _softmax(logits).astype(np.float32)


def compute_class_weights(labels: np.ndarray, num_classes: int, power: float = 1.0):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=int(num_classes)).astype(np.float32)
    safe_counts = np.maximum(counts, 1.0)
    weights = counts.sum() / safe_counts
    weights = np.power(weights, float(power))
    weights = weights / np.maximum(weights.mean(), 1e-6)
    return weights.astype(np.float32), counts.astype(np.int64)


def derive_window_labels_from_point_labels(
    label_sequence: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
    strategy: str = "last",
    num_classes: int = None,
):
    label_sequence = np.asarray(label_sequence, dtype=np.int64).reshape(-1)
    n_windows = count_windows(len(label_sequence), seq_len, pred_len, stride)
    start_indices = np.arange(n_windows, dtype=np.int64) * int(stride)
    labels = np.zeros(n_windows, dtype=np.int64)
    strategy = str(strategy).lower()

    for idx, start in enumerate(start_indices):
        future = label_sequence[int(start) + int(seq_len): int(start) + int(seq_len) + int(pred_len)]
        if future.size == 0:
            raise ValueError("Encountered empty future label window while deriving labels.")

        if strategy == "last":
            labels[idx] = int(future[-1])
        elif strategy == "max":
            labels[idx] = int(np.max(future))
        elif strategy == "majority":
            n_cls = int(num_classes) if num_classes is not None else int(np.max(label_sequence) + 1)
            counts = np.bincount(future, minlength=max(1, n_cls))
            labels[idx] = int(np.argmax(counts))
        else:
            raise ValueError(f"Unknown window_label_strategy: {strategy}")

    return start_indices, labels


def load_label_file(path: str, value_col: str = None, start_col: str = "start_idx"):
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npy":
        arr = np.load(path, allow_pickle=False)
        return {"labels": np.asarray(arr).reshape(-1), "granularity": "point"}

    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        labels = np.asarray(data["labels"]).reshape(-1)
        if "start_indices" in data:
            start_indices = np.asarray(data["start_indices"]).reshape(-1)
            return {"labels": labels, "start_indices": start_indices, "granularity": "window"}
        return {"labels": labels, "granularity": "point"}

    if ext in {".csv", ".txt", ".tsv"}:
        sep = "\t" if ext == ".tsv" else None
        df = pd.read_csv(path, sep=sep)
        if value_col is None:
            if "label" in df.columns:
                value_col = "label"
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError(f"No numeric label column found in {path}.")
                value_col = numeric_cols[0]
        labels = df[value_col].to_numpy()
        if start_col in df.columns:
            return {
                "labels": labels.reshape(-1),
                "start_indices": df[start_col].to_numpy(dtype=np.int64).reshape(-1),
                "granularity": "window",
            }
        return {"labels": labels.reshape(-1), "granularity": "point"}

    raise ValueError(f"Unsupported label file extension: {ext}")


def split_window_records_by_segment(
    start_indices: np.ndarray,
    labels: np.ndarray,
    segment_start: int,
    segment_length: int,
    seq_len: int,
    pred_len: int,
):
    start_indices = np.asarray(start_indices, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if start_indices.shape[0] != labels.shape[0]:
        raise ValueError("start_indices and labels must have the same length.")

    segment_end = int(segment_start) + int(segment_length)
    valid = (
        (start_indices >= int(segment_start)) &
        (start_indices + int(seq_len) + int(pred_len) <= segment_end)
    )

    local_starts = start_indices[valid] - int(segment_start)
    local_labels = labels[valid]
    return local_starts.astype(np.int64), local_labels.astype(np.int64)
