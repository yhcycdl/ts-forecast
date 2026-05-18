import json
import os

import numpy as np
import torch

from data_provider.data_factory import _forecast_data_provider, _load_values
from data_provider.data_joint import joint_data_provider
from data_provider.processing import time_split
from data_provider.risk_labels import (
    assign_risk_labels,
    compute_risk_stats_from_future_windows,
    fit_risk_label_config,
    risk_probabilities_from_stats,
)
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.losses import _align_pred_target, build_criterion
from utils.metrics import classification_metrics, mse
from utils.tools import plot_confusion_matrix


def _load_state_dict_forgiving(module, state_dict, *, context: str):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    missing = list(missing)
    unexpected = list(unexpected)

    missing_non_cls = [k for k in missing if not k.startswith("cls_head.")]
    unexpected_non_cls = [k for k in unexpected if not k.startswith("cls_head.")]
    if missing_non_cls or unexpected_non_cls:
        raise RuntimeError(
            f"{context}: incompatible checkpoint. "
            f"missing={missing_non_cls}, unexpected={unexpected_non_cls}"
        )

    if missing or unexpected:
        print(
            f"{context}: loaded with relaxed matching. "
            f"missing={missing}, unexpected={unexpected}"
        )


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _to_bpc(tensor: torch.Tensor, pred_len: int, out_channels: int) -> torch.Tensor:
    """
    Normalize common forecast layouts to (B, P, C).
    Supports:
      - (B, P)
      - (B, 1, P)
      - (B, P, C)
      - (B, C, P)
    """
    if tensor.dim() == 2:
        return tensor.unsqueeze(-1)

    if tensor.dim() != 3:
        raise ValueError(f"Expected forecast tensor dim 2 or 3, got shape {tuple(tensor.shape)}")

    if tensor.shape[1] == int(pred_len):
        return tensor.contiguous()

    if tensor.shape[-1] == int(pred_len):
        return tensor.transpose(1, 2).contiguous()

    if tensor.shape[1] == int(out_channels):
        return tensor.transpose(1, 2).contiguous()

    return tensor.contiguous()


class Exp_Forecast_Post_Risk(Exp_Long_Term_Forecast):
    """
    M3 baseline:
      1. train a forecast-only model (same as M1)
      2. postprocess predicted future windows into risk labels via RMS/BER thresholds
    """

    def __init__(self, args):
        super().__init__(args)

    def _require_generated_labels(self):
        label_mode = str(getattr(self.args, "label_mode", "generated")).lower()
        if label_mode != "generated":
            raise NotImplementedError(
                "M3 requires label_mode='generated', because it must map predicted future waveforms "
                "back to risk labels through the same RMS/BER postprocessing rule."
            )

    def _get_data(self, flag):
        return _forecast_data_provider(self.args, flag)

    def _fit_label_config(self):
        self._require_generated_labels()
        _, values = _load_values(self.args)
        train_raw, _, _ = time_split(
            values,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            split_mode=getattr(self.args, "split_mode", "total"),
        )
        label_config, _ = fit_risk_label_config(train_raw, self.args)
        return label_config

    def _inverse_future_windows(self, scaler, windows: np.ndarray, label_config: dict) -> np.ndarray:
        windows = np.asarray(windows, dtype=np.float32)
        if scaler is None:
            return windows

        try:
            return scaler.inverse_transform(windows)
        except Exception:
            pass

        if hasattr(scaler, "mean") and hasattr(scaler, "std"):
            mean = np.asarray(scaler.mean)
            std = np.asarray(scaler.std)
            if mean.ndim == 2 and std.ndim == 2 and windows.shape[-1] == 1 and mean.shape[-1] >= 1:
                ch = int(label_config.get("target_channel", 0))
                ch = min(max(ch, 0), mean.shape[-1] - 1)
                return windows * float(std[0, ch]) + float(mean[0, ch])

        return windows

    def train(self, setting):
        self._require_generated_labels()
        super().train(setting)

        ckpt_dir = self._make_ckpt_dir(setting)
        label_cfg_path = os.path.join(ckpt_dir, "risk_label_config.json")
        label_config = self._fit_label_config()
        with open(label_cfg_path, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(label_config), f, ensure_ascii=False, indent=2)
        print(f"Saved label config: {label_cfg_path}")

    @torch.no_grad()
    def _eval_postprocess_risk(self, loader, scaler, label_config):
        self.model.eval()

        forecast_criterion = build_criterion(self.args).to(self.device)
        forecast_loss_name = str(getattr(self.args, "loss", "mse")).lower()
        out_channels = int(getattr(self.args, "c_out", getattr(self.args, "out_in", 1)))

        total_forecast_loss = 0.0
        total_forecast_mse = 0.0
        total_rms_mae = 0.0
        total_ber_mae = 0.0
        n = 0

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        for x, y_forecast, y_cls in loader:
            x = x.to(self.device, non_blocking=True)
            y_forecast = y_forecast.to(self.device, non_blocking=True)
            y_cls = y_cls.to(self.device, non_blocking=True).long()

            pred = self.model(x)
            pred, y_forecast = _align_pred_target(pred, y_forecast)

            if forecast_loss_name == "hybrid":
                forecast_loss = forecast_criterion(pred, y_forecast, x)
            else:
                forecast_loss = forecast_criterion(pred, y_forecast)

            pred_bpc = _to_bpc(pred, int(self.args.pred_len), out_channels)
            true_bpc = _to_bpc(y_forecast, int(self.args.pred_len), out_channels)

            pred_np = pred_bpc.detach().cpu().numpy()
            true_np = true_bpc.detach().cpu().numpy()
            pred_raw = self._inverse_future_windows(scaler, pred_np, label_config)
            true_raw = self._inverse_future_windows(scaler, true_np, label_config)
            target_channel = int(label_config.get("target_channel", 0))
            if pred_raw.ndim == 3 and pred_raw.shape[2] == 1:
                target_channel = 0

            pred_stats = compute_risk_stats_from_future_windows(
                pred_raw,
                target_channel=target_channel,
                sample_rate=label_config.get("sample_rate", 1.0),
                band_low=label_config.get("band_low", 0.0),
                band_high=label_config.get("band_high", 0.0),
            )
            true_stats = compute_risk_stats_from_future_windows(
                true_raw,
                target_channel=target_channel,
                sample_rate=label_config.get("sample_rate", 1.0),
                band_low=label_config.get("band_low", 0.0),
                band_high=label_config.get("band_high", 0.0),
            )

            y_pred = assign_risk_labels(pred_stats, label_config)
            y_prob = risk_probabilities_from_stats(pred_stats, label_config)

            total_forecast_loss += float(forecast_loss.item())
            total_forecast_mse += mse(pred, y_forecast)
            total_rms_mae += float(np.mean(np.abs(pred_stats["rms"] - true_stats["rms"])))
            total_ber_mae += float(np.mean(np.abs(pred_stats["ber"] - true_stats["ber"])))
            n += 1

            y_true_all.append(y_cls.detach().cpu().numpy())
            y_pred_all.append(y_pred)
            y_prob_all.append(y_prob)

        y_true = np.concatenate(y_true_all, axis=0)
        y_pred = np.concatenate(y_pred_all, axis=0)
        y_prob = np.concatenate(y_prob_all, axis=0)

        metrics = classification_metrics(
            y_true,
            y_pred,
            y_prob=y_prob,
            num_classes=int(label_config.get("num_classes", self.args.num_classes)),
        )
        metrics["forecast_loss"] = total_forecast_loss / max(1, n)
        metrics["forecast_mse"] = total_forecast_mse / max(1, n)
        metrics["post_rms_mae"] = total_rms_mae / max(1, n)
        metrics["post_ber_mae"] = total_ber_mae / max(1, n)
        return metrics

    def test(self, setting, test=0):
        self._require_generated_labels()
        ckpt_dir = self._make_ckpt_dir(setting)

        source_setting = getattr(self.args, "forecast_ckpt_setting", None)
        source_dir = os.path.join(self.args.checkpoints, source_setting) if source_setting else ckpt_dir
        best_path = os.path.join(source_dir, "best.pt")
        scaler_path = os.path.join(source_dir, "scaler.npz")

        state = torch.load(best_path, map_location=self.device)
        _load_state_dict_forgiving(self.model, state, context="M3 forecast checkpoint")
        self.model.eval()

        test_set, test_loader, scaler = joint_data_provider(self.args, "test")
        if os.path.exists(scaler_path):
            scaler.load(scaler_path)

        label_config = dict(getattr(test_set, "label_config", {}) or self._fit_label_config())
        metrics = self._eval_postprocess_risk(test_loader, scaler, label_config)
        metrics["source_checkpoint"] = best_path

        print(
            f"M3 Test | forecast_mse={metrics['forecast_mse']:.6f} "
            f"| acc={metrics['accuracy']:.6f} "
            f"| f1_macro={metrics['f1_macro']:.6f} "
            f"| bal_acc={metrics['balanced_accuracy']:.6f}"
        )
        if "auprc" in metrics:
            print(f"M3 Test | auprc={metrics['auprc']:.6f}")
        if "auroc" in metrics:
            print(f"M3 Test | auroc={metrics['auroc']:.6f}")
        if "auprc_macro" in metrics:
            print(f"M3 Test | auprc_macro={metrics['auprc_macro']:.6f}")
        if "auroc_macro_ovr" in metrics:
            print(f"M3 Test | auroc_macro_ovr={metrics['auroc_macro_ovr']:.6f}")
        print(f"Loaded forecast checkpoint: {best_path}")

        label_cfg_path = os.path.join(ckpt_dir, "risk_label_config.json")
        with open(label_cfg_path, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(label_config), f, ensure_ascii=False, indent=2)

        metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(metrics), f, ensure_ascii=False, indent=2)

        cm_path = os.path.join(ckpt_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            cm_path,
            metrics["confusion_matrix"],
            class_names=label_config.get("class_names"),
            title=f"{setting}\nM3 F1-macro={metrics['f1_macro']:.4f} | Forecast MSE={metrics['forecast_mse']:.6f}",
        )

        print(f"Saved label config: {label_cfg_path}")
        print(f"Saved metrics: {metrics_path}")
        print(f"Saved confusion matrix: {cm_path}")
