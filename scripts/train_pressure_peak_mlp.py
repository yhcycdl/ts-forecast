#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        return list(reader.fieldnames), list(reader)


def _columns_by_prefix(fieldnames: list[str], prefix: str) -> list[str]:
    cols = [name for name in fieldnames if name.startswith(prefix)]
    return sorted(cols, key=lambda name: int(name.rsplit("_", 1)[-1]))


def _matrix(rows: list[dict[str, str]], cols: list[str]) -> np.ndarray:
    return np.asarray([[float(row[col]) for col in cols] for row in rows], dtype=np.float32)


def _split_arrays(rows: list[dict[str, str]], fieldnames: list[str]) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    h_cols = _columns_by_prefix(fieldnames, "in_h_")
    dt_cols = _columns_by_prefix(fieldnames, "in_dt_")
    target_h_cols = _columns_by_prefix(fieldnames, "target_h_")
    target_dt_cols = _columns_by_prefix(fieldnames, "target_dt_")
    if not h_cols or not dt_cols or not target_h_cols or not target_dt_cols:
        raise ValueError("Peak sample CSV must contain in_h_*, in_dt_*, target_h_*, and target_dt_* columns.")
    x_cols = h_cols + dt_cols
    y_cols = target_h_cols + target_dt_cols
    split = np.asarray([row["split"] for row in rows], dtype=object)
    return x_cols, y_cols, {"x": _matrix(rows, x_cols), "y": _matrix(rows, y_cols), "split": split}


def _standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std, mean, std


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _make_loader(torch, x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    tensor_x = torch.as_tensor(x, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_mlp(torch, in_dim: int, out_dim: int, hidden_dim: int, layers: int, dropout: float):
    import torch.nn as nn

    blocks: list[nn.Module] = []
    dim = in_dim
    for _ in range(max(1, layers)):
        blocks.append(nn.Linear(dim, hidden_dim))
        blocks.append(nn.GELU())
        if dropout > 0:
            blocks.append(nn.Dropout(dropout))
        dim = hidden_dim
    blocks.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*blocks)


def _eval(torch, model, loader, criterion, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total += float(loss.item()) * int(x.shape[0])
            count += int(x.shape[0])
    return total / max(1, count)


def _predict(torch, model, x: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    loader = _make_loader(torch, x, np.zeros((x.shape[0], 1), dtype=np.float32), batch_size, False)
    with torch.no_grad():
        for batch_x, _ in loader:
            pred = model(batch_x.to(device)).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds, axis=0) if preds else np.empty((0, 0), dtype=np.float32)


def _write_prediction_csv(path: Path, y_true: np.ndarray, y_pred: np.ndarray, y_cols: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [f"true_{c}" for c in y_cols] + [f"pred_{c}" for c in y_cols]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for true_row, pred_row in zip(y_true, y_pred):
            row = {}
            for col, val in zip(y_cols, true_row):
                row[f"true_{col}"] = float(val)
            for col, val in zip(y_cols, pred_row):
                row[f"pred_{col}"] = float(val)
            writer.writerow(row)


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_cols: list[str], output_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    h_idx = [i for i, col in enumerate(y_cols) if col.startswith("target_h_")]
    dt_idx = [i for i, col in enumerate(y_cols) if col.startswith("target_dt_")]
    saved_paths: list[Path] = []

    for name, idxs, ylabel in [("height", h_idx, "peak height"), ("dt", dt_idx, "peak interval / s")]:
        if not idxs:
            continue
        fig, axes = plt.subplots(len(idxs), 1, figsize=(14, max(3, 2.5 * len(idxs))), sharex=True)
        if len(idxs) == 1:
            axes = [axes]
        for ax, idx in zip(axes, idxs):
            ax.plot(y_true[:, idx], label="true", linewidth=1.2)
            ax.plot(y_pred[:, idx], label="pred", linewidth=1.0, linestyle="--")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.2)
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("test sample")
        plt.tight_layout()
        fig_path = output_dir / f"test_{name}_prediction.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        saved_paths.append(fig_path)
    return saved_paths


def _plot_peak_forecast(
    test_rows: list[dict[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_cols: list[str],
    output_dir: Path,
) -> Path | None:
    import matplotlib.pyplot as plt

    h_idx = [i for i, col in enumerate(y_cols) if col.startswith("target_h_")]
    dt_idx = [i for i, col in enumerate(y_cols) if col.startswith("target_dt_")]
    if not h_idx or len(h_idx) != len(dt_idx) or len(test_rows) != y_true.shape[0]:
        return None

    peak_records: dict[int, dict[str, list[float] | float]] = {}
    for row, true_row, pred_row in zip(test_rows, y_true, y_pred):
        target_start_peak = int(float(row["target_start_peak"]))
        true_dt = true_row[dt_idx]
        pred_dt = pred_row[dt_idx]
        anchor_time = float(row["target_start_time"]) - float(true_dt[0])
        true_times = anchor_time + np.cumsum(true_dt)
        pred_times = anchor_time + np.cumsum(pred_dt)
        for j, (h_col_idx, peak_id_offset) in enumerate(zip(h_idx, range(len(h_idx)))):
            peak_id = target_start_peak + peak_id_offset
            record = peak_records.setdefault(
                peak_id,
                {"true_time": float(true_times[j]), "true_h": float(true_row[h_col_idx]), "pred_time": [], "pred_h": []},
            )
            record["pred_time"].append(float(pred_times[j]))  # type: ignore[index]
            record["pred_h"].append(float(pred_row[h_col_idx]))  # type: ignore[index]

    if not peak_records:
        return None

    ordered_ids = sorted(peak_records)
    true_time = np.asarray([float(peak_records[i]["true_time"]) for i in ordered_ids], dtype=np.float64)
    true_h = np.asarray([float(peak_records[i]["true_h"]) for i in ordered_ids], dtype=np.float64)
    pred_time = np.asarray([float(np.mean(peak_records[i]["pred_time"])) for i in ordered_ids], dtype=np.float64)
    pred_h = np.asarray([float(np.mean(peak_records[i]["pred_h"])) for i in ordered_ids], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(true_time, true_h, marker="o", linewidth=1.8, label="True Peaks")
    ax.plot(pred_time, pred_h, marker="o", linewidth=1.5, linestyle="--", label="Predicted Peaks")
    for t0, t1, h0, h1 in zip(true_time, pred_time, true_h, pred_h):
        ax.plot([t0, t1], [h0, h1], color="gray", alpha=0.25, linewidth=0.8)
    ax.set_title("Peak-Level Forecast on Test Split")
    ax.set_xlabel("time / s")
    ax.set_ylabel("pressure peak")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    plt.tight_layout()
    fig_path = output_dir / "test_peak_forecast.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    return fig_path


def _persistence_baseline(x_raw: np.ndarray, x_cols: list[str], y_cols: list[str]) -> np.ndarray:
    h_cols = [c for c in x_cols if c.startswith("in_h_")]
    dt_cols = [c for c in x_cols if c.startswith("in_dt_")]
    if not h_cols or not dt_cols:
        return np.zeros((x_raw.shape[0], len(y_cols)), dtype=np.float32)
    last_h_col = h_cols[-1]
    last_dt_col = dt_cols[-1]
    last_h = x_raw[:, x_cols.index(last_h_col)]
    last_dt = x_raw[:, x_cols.index(last_dt_col)]
    pred = np.zeros((x_raw.shape[0], len(y_cols)), dtype=np.float32)
    for i, col in enumerate(y_cols):
        if col.startswith("target_h_"):
            pred[:, i] = last_h
        elif col.startswith("target_dt_"):
            pred[:, i] = last_dt
    return pred


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an MLP baseline for pressure peak-sequence forecasting.")
    parser.add_argument("--csv", required=True, type=str, help="Peak sample CSV from prepare_pressure_peak_target.py.")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--hidden-dim", default=128, type=int)
    parser.add_argument("--layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=500, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=60, type=int)
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--gpu", default=-1, type=int, help="-1 uses CPU.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    import torch
    import torch.nn as nn

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames, rows = _read_rows(csv_path)
    x_cols, y_cols, arrays = _split_arrays(rows, fieldnames)
    x = arrays["x"]
    y = arrays["y"]
    split = arrays["split"]

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"
    if not np.any(train_mask) or not np.any(val_mask) or not np.any(test_mask):
        raise ValueError("Need non-empty train/val/test splits.")

    x_norm, x_mean, x_std = _standardize(x[train_mask], x)
    y_norm, y_mean, y_std = _standardize(y[train_mask], y)

    device = "cpu"
    if int(args.gpu) >= 0 and torch.cuda.is_available():
        device = f"cuda:{int(args.gpu)}"

    train_loader = _make_loader(torch, x_norm[train_mask], y_norm[train_mask], args.batch_size, True)
    val_loader = _make_loader(torch, x_norm[val_mask], y_norm[val_mask], args.batch_size, False)
    model = _build_mlp(torch, x.shape[1], y.shape[1], args.hidden_dim, args.layers, args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(batch_x.shape[0])
            count += int(batch_x.shape[0])
        train_loss = total / max(1, count)
        val_loss = _eval(torch, model, val_loader, criterion, device)
        history.append({"epoch": float(epoch), "train_mse_norm": train_loss, "val_mse_norm": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"Epoch {epoch:04d} | train_mse_norm={train_loss:.6f} | val_mse_norm={val_loss:.6f}")
        if int(args.patience) > 0 and bad_epochs >= int(args.patience):
            print(f"Early stop at epoch {epoch}, best_val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_pred_norm = _predict(torch, model, x_norm[test_mask], device, args.batch_size)
    test_true = y[test_mask]
    test_pred = test_pred_norm * y_std + y_mean
    baseline_pred = _persistence_baseline(x[test_mask], x_cols, y_cols)

    h_idx = [i for i, col in enumerate(y_cols) if col.startswith("target_h_")]
    dt_idx = [i for i, col in enumerate(y_cols) if col.startswith("target_dt_")]
    metrics = {
        "csv": str(csv_path.resolve()),
        "samples": {"train": int(np.sum(train_mask)), "val": int(np.sum(val_mask)), "test": int(np.sum(test_mask))},
        "input_columns": x_cols,
        "target_columns": y_cols,
        "best_val_mse_norm": float(best_val),
        "test_mse_raw": float(np.mean((test_pred - test_true) ** 2)),
        "test_mae_raw": float(np.mean(np.abs(test_pred - test_true))),
        "test_pearson_raw": _pearson(test_true, test_pred),
        "baseline_mse_raw": float(np.mean((baseline_pred - test_true) ** 2)),
        "baseline_mae_raw": float(np.mean(np.abs(baseline_pred - test_true))),
        "baseline_pearson_raw": _pearson(test_true, baseline_pred),
    }
    if h_idx:
        metrics["height_mse_raw"] = float(np.mean((test_pred[:, h_idx] - test_true[:, h_idx]) ** 2))
        metrics["height_mae_raw"] = float(np.mean(np.abs(test_pred[:, h_idx] - test_true[:, h_idx])))
        metrics["height_pearson_raw"] = _pearson(test_true[:, h_idx], test_pred[:, h_idx])
    if dt_idx:
        metrics["dt_mse_raw"] = float(np.mean((test_pred[:, dt_idx] - test_true[:, dt_idx]) ** 2))
        metrics["dt_mae_raw"] = float(np.mean(np.abs(test_pred[:, dt_idx] - test_true[:, dt_idx])))
        metrics["dt_pearson_raw"] = _pearson(test_true[:, dt_idx], test_pred[:, dt_idx])

    torch.save({"model_state": model.state_dict(), "metrics": metrics, "args": vars(args)}, output_dir / "best.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "scaler.json").write_text(
        json.dumps(
            {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean.tolist(), "y_std": y_std.tolist()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_prediction_csv(output_dir / "test_predictions.csv", test_true, test_pred, y_cols)

    plot_error = None
    plot_paths: list[Path] = []
    try:
        plot_paths = _plot_predictions(test_true, test_pred, y_cols, output_dir)
        peak_forecast_path = _plot_peak_forecast([row for row, is_test in zip(rows, test_mask) if is_test], test_true, test_pred, y_cols, output_dir)
        if peak_forecast_path is not None:
            plot_paths.append(peak_forecast_path)
    except ModuleNotFoundError as exc:
        plot_error = str(exc)

    print(f"Saved model: {output_dir / 'best.pt'}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    for path in plot_paths:
        print(f"Saved plot: {path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if plot_error is not None:
        print(f"Plots skipped: {plot_error}")


if __name__ == "__main__":
    main()
