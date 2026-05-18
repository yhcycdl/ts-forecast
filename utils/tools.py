# utils/tools.py
import json
import torch
import matplotlib.pyplot as plt
import os 
import numpy as np

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model, path):
        # score 越小越好（loss/mse）
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.counter = 0
            save_model(model, path)
            if self.verbose:
                print(f"EarlyStopping: best score -> {score:.6f}, saved {path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
# utils/tools.py（新增：rolling 推理 + 画图）




def plot_rolling_forecast(
    save_path: str,
    true_seq,
    pred_seq,
    raw_seq=None,
    pred_len: int = None,
    title: str = None,
    max_points: int = 30000,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    true_seq = np.asarray(true_seq).reshape(-1)
    pred_seq = np.asarray(pred_seq).reshape(-1)

    L = min(len(true_seq), len(pred_seq))
    true_seq = true_seq[:L]
    pred_seq = pred_seq[:L]
    if raw_seq is not None:
        raw_seq = np.asarray(raw_seq).reshape(-1)[:L]

    x = np.arange(L)
    if max_points > 0 and L > max_points:
        step = max(1, int(np.ceil(L / max_points)))
        x_plot = x[::step]
        true_plot = true_seq[::step]
        pred_plot = pred_seq[::step]
        raw_plot = raw_seq[::step] if raw_seq is not None else None
    else:
        step = 1
        x_plot = x
        true_plot = true_seq
        pred_plot = pred_seq
        raw_plot = raw_seq

    plt.figure(figsize=(15, 6))
    if raw_plot is not None:
        plt.plot(x_plot, raw_plot, linewidth=0.7, alpha=0.25, label="Raw")
    plt.plot(x_plot, true_plot, linewidth=1.8, alpha=0.65, label="Ground Truth")
    plt.plot(x_plot, pred_plot, linestyle="--", linewidth=1.2, label="Prediction")

    if pred_len is not None and pred_len > 1 and (L // int(pred_len) <= 200):
        for i in range(0, L, int(pred_len)):
            plt.axvline(x=i, linestyle=":", alpha=0.15)

    if title:
        if step > 1:
            title = f"{title}\nOverview plot downsampled for readability: every {step} points"
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_pred_vs_true_scatter(
    save_path: str,
    true_seq,
    pred_seq,
    title: str | None = None,
    max_points: int = 10000,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    true_seq = np.asarray(true_seq).reshape(-1)
    pred_seq = np.asarray(pred_seq).reshape(-1)
    mask = np.isfinite(true_seq) & np.isfinite(pred_seq)
    true_seq = true_seq[mask]
    pred_seq = pred_seq[mask]
    if true_seq.size == 0:
        raise ValueError("No finite samples for scatter plot.")

    if max_points > 0 and true_seq.size > max_points:
        step = max(1, true_seq.size // max_points)
        true_seq = true_seq[::step]
        pred_seq = pred_seq[::step]

    lo = float(min(np.min(true_seq), np.min(pred_seq)))
    hi = float(max(np.max(true_seq), np.max(pred_seq)))

    plt.figure(figsize=(6, 6))
    plt.scatter(true_seq, pred_seq, s=8, alpha=0.18, edgecolors="none")
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, color="black", alpha=0.7)
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_prediction_zoom_panels(
    save_path: str,
    true_seq,
    pred_seq,
    raw_seq=None,
    title: str | None = None,
    window_size: int = 400,
    num_panels: int = 6,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    true_seq = np.asarray(true_seq).reshape(-1)
    pred_seq = np.asarray(pred_seq).reshape(-1)
    L = min(true_seq.size, pred_seq.size)
    if raw_seq is not None:
        raw_seq = np.asarray(raw_seq).reshape(-1)
        L = min(L, raw_seq.size)
    if L == 0:
        raise ValueError("Empty prediction sequence for zoom plot.")
    true_seq = true_seq[:L]
    pred_seq = pred_seq[:L]
    if raw_seq is not None:
        raw_seq = raw_seq[:L]

    window = max(1, min(int(window_size), L))
    if L <= window:
        starts = [0]
    else:
        panel_count = max(3, min(int(num_panels), int(np.ceil(L / window))))
        starts = np.linspace(0, L - window, num=panel_count, dtype=np.int64).tolist()
        starts = sorted(set(int(s) for s in starts))

    fig, axes = plt.subplots(len(starts), 1, figsize=(15, 3.2 * len(starts)), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])

    for ax, start in zip(axes, starts):
        end = min(L, start + window)
        idx = np.arange(start, end)
        if raw_seq is not None:
            ax.plot(idx, raw_seq[start:end], linewidth=0.7, alpha=0.32, label="Raw")
        ax.plot(idx, true_seq[start:end], linewidth=2, alpha=0.65, label="Ground Truth")
        ax.plot(idx, pred_seq[start:end], linestyle="--", linewidth=1.5, label="Prediction")
        ax.set_title(f"Zoom [{start}:{end}]")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(save_path: str, confusion_matrix, class_names=None, title: str = "Confusion Matrix"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = np.asarray(confusion_matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"confusion_matrix must be square, got shape {cm.shape}")

    n = cm.shape[0]
    if class_names is None or len(class_names) != n:
        class_names = [str(i) for i in range(n)]

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(n)
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    threshold = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color)

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def _as_numpy_series(series):
    """
    series: torch.Tensor or np.ndarray
    -> np.ndarray (T,C)
    """
    if isinstance(series, torch.Tensor):
        series = series.detach().cpu().numpy()
    series = np.asarray(series)
    if series.ndim == 1:
        series = series[:, None]
    if series.ndim != 2:
        raise ValueError(f"series must be (T,C) or (T,), got {series.shape}")
    return series


def _pred_to_1d(pred):
    """
    pred: torch.Tensor
    -> 1D np.ndarray (P,)
    支持 pred 形状：
      (B,P) / (B,1,P) / (B,P,C)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()

    if pred.dim() == 2:
        return pred[0].numpy()              # (P,)
    if pred.dim() == 3:
        if pred.shape[1] == 1:              # (B,1,P)
            return pred[0, 0, :].numpy()
        return pred[0, :, 0].numpy()        # (B,P,C) -> 取 C=0

    raise ValueError(f"Unexpected pred shape: {tuple(pred.shape)}")


def _inverse_target_1d(scaler, x_1d, channel_idx: int | None = None):
    """
    对单个目标通道做反归一化。

    之前直接把 1D 序列喂给 DataScaler.inverse_transform，在多通道场景下
    会因为 shape 不匹配而静默失败，导致 Rolling MSE 有时在原始尺度、
    有时在标准化尺度，口径不一致。
    """
    x_1d = np.asarray(x_1d).reshape(-1)
    if scaler is None:
        return x_1d

    mean = getattr(scaler, "mean", None)
    std = getattr(scaler, "std", None)
    if mean is not None and std is not None:
        mean = np.asarray(mean).reshape(-1)
        std = np.asarray(std).reshape(-1)
        if mean.size == 1 and std.size == 1:
            return x_1d * std[0] + mean[0]
        if channel_idx is not None and 0 <= int(channel_idx) < mean.size and int(channel_idx) < std.size:
            ch = int(channel_idx)
            return x_1d * std[ch] + mean[ch]

    try:
        return np.asarray(scaler.inverse_transform(x_1d)).reshape(-1)
    except Exception:
        return x_1d


def _pearson_corr(a, b) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if a.size < 2:
        return float("nan")
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 1e-12 or b_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def run_rolling_inference_and_plot(
    model,
    device,
    series,
    scaler,
    seq_len: int,
    pred_len: int,
    save_dir: str,
    setting: str,
    horizon: int = 2000,
    start_idx: int = 0,
    in_indices=None,
    out_indices=None,
    target_shift: int = 0,
    target_name: str | None = None,
    plot_filename: str = "rolling_forecast.png",
    raw_overlay=None,
    start_indices=None,
):
    """
    一站式：滚动推理 + 反归一化 + 指标 + 画图

    参数：
      - model/device: 你的模型和 device
      - series: (T,C) 或 (T,) 的原始序列（通常来自 test_set.series）
      - scaler: 用于 inverse_transform（可以 None）
      - in_indices/out_indices: 输入/输出通道选择（不传默认全通道；画图默认 out 的第0个）
      - horizon: 预测长度总点数（最终拼接出来的长度）
      - start_idx: 起始位置
      - target_shift: 输出目标相对历史窗口末端的偏移量（单位：采样点）
      - save_dir/setting: 保存图片和标题

    返回：
      mse_raw, mse_norm, fig_path, pred_seq_raw, true_seq_raw, diagnostics
    """
    os.makedirs(save_dir, exist_ok=True)

    series = _as_numpy_series(series)
    T, C_all = series.shape

    if in_indices is None:
        in_indices = list(range(C_all))
    if out_indices is None:
        out_indices = in_indices

    # 只评估/画第一个目标通道
    out0 = out_indices[0]

    raw_overlay = None if raw_overlay is None else np.asarray(raw_overlay).reshape(-1)
    preds_raw, trues_raw = [], []
    raw_overlay_seq = []
    preds_norm, trues_norm = [], []
    curr = int(start_idx)
    if start_indices is not None:
        start_indices = np.asarray(start_indices, dtype=np.int64).reshape(-1)
        start_indices = start_indices[start_indices >= int(start_idx)]
    produced = 0
    model.eval()

    target_shift = int(target_shift)
    max_steps = int(horizon)
    if max_steps <= 0:
        max_steps = max(0, T - seq_len - target_shift)

    start_iter = iter(start_indices.tolist()) if start_indices is not None else None
    while produced < max_steps:
        if start_iter is not None:
            try:
                curr = int(next(start_iter))
            except StopIteration:
                break
        if curr + seq_len + target_shift + pred_len > T:
            if start_iter is not None:
                continue
            break
        x_np = series[curr: curr + seq_len, in_indices]                     # (L,C_in)
        y_start = curr + seq_len + target_shift
        y_np = series[y_start: y_start + pred_len, out_indices]  # (P,C_out)
        y_1d = y_np[:, 0]                                                   # (P,)

        # (1,C_in,L)
        x_t = torch.tensor(x_np, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)

        pred = model(x_t)
        pred_1d = np.asarray(_pred_to_1d(pred)).reshape(-1)                 # (P,)
        true_1d = np.asarray(y_1d).reshape(-1)

        pred_raw = _inverse_target_1d(scaler, pred_1d, channel_idx=out0)
        true_raw = _inverse_target_1d(scaler, true_1d, channel_idx=out0)

        need = min(pred_len, max_steps - produced)
        preds_raw.append(pred_raw[:need])
        trues_raw.append(true_raw[:need])
        if raw_overlay is not None:
            raw_overlay_seq.append(raw_overlay[y_start: y_start + pred_len][:need])
        preds_norm.append(pred_1d[:need])
        trues_norm.append(true_1d[:need])

        produced += need
        if start_iter is None:
            curr += pred_len

    if len(preds_raw) == 0:
        raise RuntimeError("No rolling samples produced. Check start_idx/seq_len/pred_len and series length.")

    pred_seq_raw = np.concatenate(preds_raw)
    true_seq_raw = np.concatenate(trues_raw)
    raw_seq = np.concatenate(raw_overlay_seq) if raw_overlay_seq else None
    pred_seq_norm = np.concatenate(preds_norm)
    true_seq_norm = np.concatenate(trues_norm)

    mse_raw = float(np.mean((pred_seq_raw - true_seq_raw) ** 2))
    mse_norm = float(np.mean((pred_seq_norm - true_seq_norm) ** 2))
    mae_raw = float(np.mean(np.abs(pred_seq_raw - true_seq_raw)))
    mae_norm = float(np.mean(np.abs(pred_seq_norm - true_seq_norm)))
    pearson_raw = _pearson_corr(true_seq_raw, pred_seq_raw)
    pearson_norm = _pearson_corr(true_seq_norm, pred_seq_norm)

    fig_path = os.path.join(save_dir, plot_filename)
    target_desc = f"target={target_name}" if target_name else f"out_ch={out0}"
    title = (
        f"{setting}\n"
        f"Rolling MSE(raw): {mse_raw:.9f} | MSE(norm): {mse_norm:.9f} | "
        f"horizon={len(pred_seq_raw)} | shift={target_shift} | {target_desc}"
    )
    plot_rolling_forecast(
        save_path=fig_path,
        true_seq=true_seq_raw,
        pred_seq=pred_seq_raw,
        raw_seq=raw_seq,
        pred_len=pred_len,
        title=title
    )

    scatter_path = os.path.join(save_dir, "prediction_scatter.png")
    scatter_title = (
        f"{setting}\n"
        f"Pearson(raw): {pearson_raw:.6f} | MAE(raw): {mae_raw:.9f}"
    )
    plot_pred_vs_true_scatter(
        save_path=scatter_path,
        true_seq=true_seq_raw,
        pred_seq=pred_seq_raw,
        title=scatter_title,
    )

    zoom_path = os.path.join(save_dir, "prediction_zoom.png")
    zoom_window = min(len(pred_seq_raw), max(400, int(pred_len) * 2))
    plot_prediction_zoom_panels(
        save_path=zoom_path,
        true_seq=true_seq_raw,
        pred_seq=pred_seq_raw,
        raw_seq=raw_seq,
        title=f"{setting}\nLocal Zoom Views",
        window_size=zoom_window,
        num_panels=6,
    )

    metrics_path = os.path.join(save_dir, "point_metrics.json")
    diagnostics = {
        "mse_raw": mse_raw,
        "mse_norm": mse_norm,
        "mae_raw": mae_raw,
        "mae_norm": mae_norm,
        "pearson_raw": pearson_raw,
        "pearson_norm": pearson_norm,
        "rolling_plot": fig_path,
        "scatter_plot": scatter_path,
        "zoom_plot": zoom_path,
        "metrics_json": metrics_path,
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, ensure_ascii=False, indent=2)

    return mse_raw, mse_norm, fig_path, pred_seq_raw, true_seq_raw, diagnostics
