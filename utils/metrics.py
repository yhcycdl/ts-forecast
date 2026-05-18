# utils/metrics.py
import numpy as np
import torch

def mse(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    兼容你的 FullResTCN pred=(B,1,P) 和 dataset y=(B,P,C)
    这里先做一个最简单的对齐：
      - 若 true 是 (B,P,1)，转成 (B,1,P)
    """
    if pred.dim() == 3 and pred.shape[1] == 1:
        # pred: (B,1,P)
        if true.dim() == 3 and true.shape[2] == 1:  # (B,P,1)
            true = true.transpose(1, 2).contiguous()  # (B,1,P)
        elif true.dim() == 2:  # (B,P)
            true = true.unsqueeze(1)
    return float(torch.mean((pred - true) ** 2).item())


def _try_sklearn_prob_metrics(y_true, y_prob, num_classes):
    if y_prob is None:
        return {}

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except Exception:
        return {}

    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    if y_prob.ndim != 2 or y_prob.shape[0] != y_true.shape[0]:
        return {}

    metrics = {}
    try:
        if num_classes == 2:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
            metrics["auprc"] = float(average_precision_score(y_true, y_prob[:, 1]))
        else:
            y_true_oh = np.eye(num_classes, dtype=np.float32)[y_true]
            metrics["auroc_macro_ovr"] = float(roc_auc_score(y_true_oh, y_prob, average="macro", multi_class="ovr"))
            metrics["auprc_macro"] = float(average_precision_score(y_true_oh, y_prob, average="macro"))
    except Exception:
        return {}
    return metrics


def classification_metrics(y_true, y_pred, y_prob=None, num_classes=None):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.size == 0:
        raise ValueError("classification metrics require at least one sample.")

    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)

    cm = np.zeros((int(num_classes), int(num_classes)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    support = cm.sum(axis=1).astype(np.float32)
    predicted = cm.sum(axis=0).astype(np.float32)
    tp = np.diag(cm).astype(np.float32)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    metrics = {
        "accuracy": float(tp.sum() / max(1.0, float(cm.sum()))),
        "balanced_accuracy": float(recall.mean()) if recall.size > 0 else 0.0,
        "precision_macro": float(precision.mean()) if precision.size > 0 else 0.0,
        "recall_macro": float(recall.mean()) if recall.size > 0 else 0.0,
        "f1_macro": float(f1.mean()) if f1.size > 0 else 0.0,
        "support": support.astype(np.int64),
        "confusion_matrix": cm,
    }
    metrics.update(_try_sklearn_prob_metrics(y_true, y_prob, int(num_classes)))
    return metrics
