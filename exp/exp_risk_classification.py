import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import classification_metrics
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


class Exp_Risk_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        data_set, data_loader, scaler = data_provider(self.args, flag)
        return data_set, data_loader, scaler

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

    def _select_criterion(self, train_set=None):
        weight = None
        if bool(int(getattr(self.args, "use_class_weights", 1))) and train_set is not None:
            if hasattr(train_set, "class_weights"):
                weight = torch.tensor(train_set.class_weights, dtype=torch.float32, device=self.device)
        return nn.CrossEntropyLoss(weight=weight)

    def _selection_score(self, val_loss, val_metrics):
        metric_name = str(getattr(self.args, "cls_selection_metric", "f1_macro")).lower()
        if metric_name == "loss":
            return float(val_loss), False, metric_name

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
            raise ValueError(f"Unknown cls_selection_metric: {metric_name}")
        if key not in val_metrics:
            raise ValueError(
                f"Selection metric '{key}' is unavailable. Available metrics: {list(val_metrics.keys())}"
            )
        return float(val_metrics[key]), True, key

    def _check_finite(self, tensor, name, *, epoch=None, batch_idx=None):
        if bool(torch.isfinite(tensor).all().item()):
            return

        detail = []
        if epoch is not None:
            detail.append(f"epoch={epoch}")
        if batch_idx is not None:
            detail.append(f"batch={batch_idx}")
        detail.append(f"model={self.args.model}")
        detail.append(f"use_amp={bool(self.args.use_amp)}")
        if hasattr(self.args, "cls_use_input_norm"):
            detail.append(f"cls_use_input_norm={getattr(self.args, 'cls_use_input_norm')}")
        if hasattr(self.args, "cls_use_pre_enc"):
            detail.append(f"cls_use_pre_enc={getattr(self.args, 'cls_use_pre_enc')}")

        raise FloatingPointError(
            f"Detected non-finite {name} during risk classification training "
            f"({', '.join(detail)}). If this is TimeMixer, rerun with `--no_use_amp`."
        )

    def train(self, setting):
        ckpt_dir = self._make_ckpt_dir(setting)

        train_set, train_loader, scaler = self._get_data("train")
        val_loader = None
        if self.args.val_ratio and self.args.val_ratio > 0:
            _, val_loader, _ = self._get_data("val")

        criterion = self._select_criterion(train_set)
        optimizer = self._select_optimizer()

        best_path = os.path.join(ckpt_dir, "best.pt")
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if self.args.patience > 0 else None

        use_amp = bool(self.args.use_amp) and self.device.type == "cuda"
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)
        best_metric = float("inf")

        print("Start Classification Training...")
        if use_amp and str(getattr(self.args, "model", "")).lower() == "timemixer":
            print("Warning: TimeMixer classification can overflow under AMP on long sequences. Use --no_use_amp if you see non-finite loss.")

        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            loss_sum = 0.0

            loop = tqdm(train_loader, leave=False)
            for batch_idx, (x, y) in enumerate(loop, start=1):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).long()

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(x)
                    loss = criterion(logits, y)

                self._check_finite(logits, "logits", epoch=epoch, batch_idx=batch_idx)
                self._check_finite(loss, "loss", epoch=epoch, batch_idx=batch_idx)
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
            print(f"Epoch {epoch:03d} | train_loss={epoch_loss:.6f}")

            metric = epoch_loss
            score_name = "train_loss"
            if val_loader is not None:
                val_loss, val_metrics = self._eval_classification(val_loader, criterion)
                score_value, maximize, score_name = self._selection_score(val_loss, val_metrics)
                metric = -score_value if maximize else score_value
                print(
                    f"Epoch {epoch:03d} | val_loss={val_loss:.6f} "
                    f"| val_f1_macro={val_metrics['f1_macro']:.6f} "
                    f"| val_bal_acc={val_metrics['balanced_accuracy']:.6f} "
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
        print(f"Saved scaler: {scaler_path}")
        print(f"Saved label config: {label_cfg_path}")

    @torch.no_grad()
    def _eval_classification(self, loader, criterion):
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred, y_prob = [], [], []

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True).long()

            logits = self.model(x)
            loss = criterion(logits, y)
            self._check_finite(logits, "eval logits")
            self._check_finite(loss, "eval loss")
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(prob, dim=-1)

            total_loss += float(loss.item())
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            y_prob.append(prob.detach().cpu().numpy())

        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)
        y_prob = np.concatenate(y_prob, axis=0)
        num_classes = int(getattr(loader.dataset, "label_config", {}).get("num_classes", self.args.num_classes))
        metrics = classification_metrics(y_true, y_pred, y_prob=y_prob, num_classes=num_classes)
        return total_loss / max(1, len(loader)), metrics

    def test(self, setting, test=0):
        ckpt_dir = self._make_ckpt_dir(setting)
        best_path = os.path.join(ckpt_dir, "best.pt")

        state = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        test_set, test_loader, _ = self._get_data("test")
        criterion = self._select_criterion(test_set)
        test_loss, metrics = self._eval_classification(test_loader, criterion)

        metrics["test_loss"] = test_loss
        print(
            f"Test | loss={metrics['test_loss']:.6f} "
            f"| acc={metrics['accuracy']:.6f} "
            f"| f1_macro={metrics['f1_macro']:.6f} "
            f"| bal_acc={metrics['balanced_accuracy']:.6f}"
        )
        if "auprc" in metrics:
            print(f"Test | auprc={metrics['auprc']:.6f}")
        if "auroc" in metrics:
            print(f"Test | auroc={metrics['auroc']:.6f}")

        metrics_path = os.path.join(ckpt_dir, "test_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_to_builtin(metrics), f, ensure_ascii=False, indent=2)

        cm_path = os.path.join(ckpt_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            cm_path,
            metrics["confusion_matrix"],
            class_names=test_set.label_config.get("class_names"),
            title=f"{setting}\nF1-macro={metrics['f1_macro']:.4f}",
        )

        print(f"Saved metrics: {metrics_path}")
        print(f"Saved confusion matrix: {cm_path}")
