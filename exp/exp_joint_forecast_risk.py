import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_provider.data_joint import joint_data_provider
from exp.exp_basic_joint import Exp_Basic_Joint
from utils.losses import _align_pred_target, build_criterion
from utils.metrics import classification_metrics, mse
from utils.tools import EarlyStopping, plot_confusion_matrix, save_model


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


def _split_joint_outputs(outputs):
    if isinstance(outputs, dict):
        if "forecast" not in outputs or "classification" not in outputs:
            raise KeyError("Joint model output dict must contain 'forecast' and 'classification'.")
        return outputs["forecast"], outputs["classification"]

    if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
        return outputs[0], outputs[1]

    raise TypeError("Joint model must return dict {'forecast', 'classification'} or a 2-tuple.")


class Exp_Joint_Forecast_Risk(Exp_Basic_Joint):
    def __init__(self, args):
        super().__init__(args)
        self.forecast_weight = float(getattr(args, "joint_forecast_weight", 1.0))
        self.cls_weight = float(getattr(args, "joint_cls_weight", 1.0))

    def _get_data(self, flag):
        return joint_data_provider(self.args, flag)

    def _select_optimizer(self):
        name = self.args.optimizer.lower()
        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=float(getattr(self.args, "momentum", 0.9)),
                weight_decay=self.args.weight_decay,
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def _select_forecast_criterion(self):
        self._forecast_loss_name = self.args.loss.lower()
        return build_criterion(self.args).to(self.device)

    def _select_cls_criterion(self, train_set=None):
        weight = None
        if bool(int(getattr(self.args, "use_class_weights", 1))) and train_set is not None:
            if hasattr(train_set, "class_weights"):
                weight = torch.tensor(train_set.class_weights, dtype=torch.float32, device=self.device)
        return nn.CrossEntropyLoss(weight=weight)

    def _selection_score(self, val_result):
        metric_name = str(getattr(self.args, "joint_selection_metric", "joint_loss")).lower()
        if metric_name in {"joint_loss", "forecast_loss", "forecast_mse", "cls_loss"}:
            return float(val_result[metric_name]), False, metric_name

        metric_map = {
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
            "bal_acc": "balanced_accuracy",
            "accuracy": "accuracy",
            "auprc": "auprc",
            "auroc": "auroc",
        }
        key = metric_map.get(metric_name)
        if key is None:
            raise ValueError(f"Unknown joint_selection_metric: {metric_name}")
        if key not in val_result:
            raise ValueError(f"Selection metric '{key}' is unavailable. Available keys: {list(val_result.keys())}")
        return float(val_result[key]), True, key

    def _joint_loss(self, forecast_loss, cls_loss):
        return self.forecast_weight * forecast_loss + self.cls_weight * cls_loss

    def train(self, setting):
        ckpt_dir = self._make_ckpt_dir(setting)

        train_set, train_loader, scaler = self._get_data("train")
        val_loader = None
        if self.args.val_ratio and self.args.val_ratio > 0:
            _, val_loader, _ = self._get_data("val")

        forecast_criterion = self._select_forecast_criterion()
        cls_criterion = self._select_cls_criterion(train_set)
        optimizer = self._select_optimizer()

        best_path = os.path.join(ckpt_dir, "best.pt")
        best_joint_path = os.path.join(ckpt_dir, "best_joint_loss.pt")
        best_f1_path = os.path.join(ckpt_dir, "best_f1_macro.pt")
        best_forecast_path = os.path.join(ckpt_dir, "best_forecast_mse.pt")
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if self.args.patience > 0 else None

        use_amp = bool(self.args.use_amp) and self.device.type == "cuda"
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
        best_metric = float("inf")
        best_joint_loss = float("inf")
        best_f1_macro = float("-inf")
        best_forecast_mse = float("inf")

        print("Start Joint Training...")
        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            loss_sum = 0.0

            loop = tqdm(train_loader, leave=False)
            for x, y_forecast, y_cls in loop:
                x = x.to(self.device, non_blocking=True)
                y_forecast = y_forecast.to(self.device, non_blocking=True)
                y_cls = y_cls.to(self.device, non_blocking=True).long()

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = self.model(x)
                    pred, logits = _split_joint_outputs(outputs)
                    pred, y_forecast = _align_pred_target(pred, y_forecast)

                    if self._forecast_loss_name == "hybrid":
                        forecast_loss = forecast_criterion(pred, y_forecast, x)
                    else:
                        forecast_loss = forecast_criterion(pred, y_forecast)
                    cls_loss = cls_criterion(logits, y_cls)
                    loss = self._joint_loss(forecast_loss, cls_loss)

                scaler_amp.scale(loss).backward()

                if self.args.max_grad_norm and self.args.max_grad_norm > 0:
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                scaler_amp.step(optimizer)
                scaler_amp.update()

                loss_val = float(loss.item())
                loss_sum += loss_val
                loop.set_postfix(loss=loss_val)

            epoch_loss = loss_sum / max(1, len(train_loader))
            print(f"Epoch {epoch:03d} | train_joint_loss={epoch_loss:.6f}")

            metric = epoch_loss
            if val_loader is not None:
                val_result = self._eval_joint(val_loader, forecast_criterion, cls_criterion)
                score_value, maximize, score_name = self._selection_score(val_result)
                metric = -score_value if maximize else score_value

                if val_result["joint_loss"] < best_joint_loss:
                    best_joint_loss = float(val_result["joint_loss"])
                    save_model(self.model, best_joint_path)
                if val_result["f1_macro"] > best_f1_macro:
                    best_f1_macro = float(val_result["f1_macro"])
                    save_model(self.model, best_f1_path)
                if val_result["forecast_mse"] < best_forecast_mse:
                    best_forecast_mse = float(val_result["forecast_mse"])
                    save_model(self.model, best_forecast_path)

                print(
                    f"Epoch {epoch:03d} | val_joint_loss={val_result['joint_loss']:.6f} "
                    f"| val_forecast_loss={val_result['forecast_loss']:.6f} "
                    f"| val_cls_loss={val_result['cls_loss']:.6f} "
                    f"| val_forecast_mse={val_result['forecast_mse']:.6f} "
                    f"| val_f1_macro={val_result['f1_macro']:.6f} "
                    f"| val_bal_acc={val_result['balanced_accuracy']:.6f} "
                    f"| select_{score_name}={score_value:.6f}"
                )

            if early_stopping is not None:
                early_stopping(metric, self.model, best_path)
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break
            else:
                if metric < best_metric:
                    best_metric = metric
                    save_model(self.model, best_path)

        scaler_path = os.path.join(ckpt_dir, "scaler.npz")
        scaler.save(scaler_path)

        label_cfg_path = os.path.join(ckpt_dir, "risk_label_config.json")
        with open(label_cfg_path, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(train_set.label_config), f, ensure_ascii=False, indent=2)

        print(f"Saved best: {best_path}")
        if val_loader is not None:
            print(f"Saved best joint loss: {best_joint_path}")
            print(f"Saved best f1 macro: {best_f1_path}")
            print(f"Saved best forecast mse: {best_forecast_path}")
        print(f"Saved scaler: {scaler_path}")
        print(f"Saved label config: {label_cfg_path}")

    @torch.no_grad()
    def _eval_joint(self, loader, forecast_criterion, cls_criterion):
        self.model.eval()
        total_joint = 0.0
        total_forecast = 0.0
        total_cls = 0.0
        total_mse = 0.0
        n = 0
        y_true, y_pred, y_prob = [], [], []

        for x, y_forecast, y_cls in loader:
            x = x.to(self.device, non_blocking=True)
            y_forecast = y_forecast.to(self.device, non_blocking=True)
            y_cls = y_cls.to(self.device, non_blocking=True).long()

            outputs = self.model(x)
            pred, logits = _split_joint_outputs(outputs)
            pred, y_forecast = _align_pred_target(pred, y_forecast)

            if self._forecast_loss_name == "hybrid":
                forecast_loss = forecast_criterion(pred, y_forecast, x)
            else:
                forecast_loss = forecast_criterion(pred, y_forecast)
            cls_loss = cls_criterion(logits, y_cls)
            joint_loss = self._joint_loss(forecast_loss, cls_loss)

            prob = torch.softmax(logits, dim=-1)
            pred_cls = torch.argmax(prob, dim=-1)

            total_joint += float(joint_loss.item())
            total_forecast += float(forecast_loss.item())
            total_cls += float(cls_loss.item())
            total_mse += mse(pred, y_forecast)
            n += 1

            y_true.append(y_cls.detach().cpu().numpy())
            y_pred.append(pred_cls.detach().cpu().numpy())
            y_prob.append(prob.detach().cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        num_classes = int(getattr(loader.dataset, "label_config", {}).get("num_classes", self.args.num_classes))
        cls_metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, num_classes=num_classes)

        result = {
            "joint_loss": total_joint / max(1, n),
            "forecast_loss": total_forecast / max(1, n),
            "cls_loss": total_cls / max(1, n),
            "forecast_mse": total_mse / max(1, n),
        }
        result.update(cls_metrics)
        return result

    def test(self, setting, test=0):
        ckpt_dir = self._make_ckpt_dir(setting)
        ckpt_name = str(getattr(self.args, "joint_test_checkpoint", "selected")).lower()
        ckpt_map = {
            "selected": "best.pt",
            "best_joint_loss": "best_joint_loss.pt",
            "best_f1_macro": "best_f1_macro.pt",
            "best_forecast_mse": "best_forecast_mse.pt",
        }
        if ckpt_name not in ckpt_map:
            raise ValueError(f"Unknown joint_test_checkpoint: {ckpt_name}")
        best_path = os.path.join(ckpt_dir, ckpt_map[ckpt_name])

        state = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        test_set, test_loader, _ = self._get_data("test")
        forecast_criterion = self._select_forecast_criterion()
        cls_criterion = self._select_cls_criterion(test_set)
        metrics = self._eval_joint(test_loader, forecast_criterion, cls_criterion)

        print(
            f"Test | joint_loss={metrics['joint_loss']:.6f} "
            f"| forecast_mse={metrics['forecast_mse']:.6f} "
            f"| cls_loss={metrics['cls_loss']:.6f} "
            f"| f1_macro={metrics['f1_macro']:.6f} "
            f"| bal_acc={metrics['balanced_accuracy']:.6f}"
        )
        print(f"Loaded checkpoint: {best_path}")

        metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(metrics), f, ensure_ascii=False, indent=2)

        cm_path = os.path.join(ckpt_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            cm_path,
            metrics["confusion_matrix"],
            class_names=test_set.label_config.get("class_names"),
            title=f"{setting}\nF1-macro={metrics['f1_macro']:.4f} | Forecast MSE={metrics['forecast_mse']:.6f}",
        )

        print(f"Saved metrics: {metrics_path}")
        print(f"Saved confusion matrix: {cm_path}")
