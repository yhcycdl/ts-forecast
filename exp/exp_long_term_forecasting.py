# exp/exp_long_term_forecasting.py
import json
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from exp.exp_basic import Exp_Basic
from data_provider.data_factory import data_provider
from data_provider.data_factory import _split_column_arrays
from data_provider.processing import load_dataframe, time_split
from utils.tools import EarlyStopping, save_model
from utils.metrics import mse
from utils.losses import build_criterion,_align_pred_target

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    def _get_data(self, flag):
        data_set, data_loader, scaler = data_provider(self.args, flag)
        return data_set, data_loader, scaler

    def _select_criterion(self):
        self._loss_name = self.args.loss.lower()
        return build_criterion(self.args)

    def _compute_loss(self, criterion, pred, y, x=None, sample_weight=None):
        pred, y = _align_pred_target(pred, y)
        if self._loss_name == "hybrid":
            if sample_weight is not None:
                raise ValueError("sample weights are not supported together with hybrid loss.")
            return criterion(pred, y, x)

        if sample_weight is None:
            return criterion(pred, y)

        _, sample_weight = _align_pred_target(pred, sample_weight, strict=False)
        sample_weight = sample_weight.to(pred.dtype)
        if self._loss_name == "mse":
            loss_tensor = (pred - y) ** 2
        elif self._loss_name == "mae":
            loss_tensor = torch.abs(pred - y)
        elif self._loss_name == "huber":
            beta = float(getattr(self.args, "huber_beta", 0.3))
            diff = torch.abs(pred - y)
            loss_tensor = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
        elif self._loss_name == "wmse":
            base_w = 1.0 + float(getattr(self.args, "wmse_alpha", 1.0)) * torch.abs(y)
            loss_tensor = base_w * (pred - y) ** 2
        else:
            return criterion(pred, y)

        weighted = loss_tensor * sample_weight
        denom = torch.clamp(sample_weight.sum(), min=1e-6)
        return weighted.sum() / denom


    def _select_optimizer(self):
        """
        根据 args 选择优化器
        """
        name = self.args.optimizer.lower()

        if name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )

        if name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.learning_rate,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay
            )
        else:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay
            )


        raise ValueError(f"Unknown optimizer: {self.args.optimizer}")

    def train(self, setting):
        ckpt_dir = self._make_ckpt_dir(setting)
        args_path = os.path.join(ckpt_dir, "run_args.json")
        with open(args_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.args), f, ensure_ascii=False, indent=2, sort_keys=True, default=str)

        _, train_loader, scaler = self._get_data("train")
        val_loader = None
        if self.args.val_ratio and self.args.val_ratio > 0:
            _, val_loader, _ = self._get_data("val")

        pretrained_path = getattr(self.args, "pretrained_path", None)
        if pretrained_path:
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"pretrained_path not found: {pretrained_path}")
            state = torch.load(pretrained_path, map_location=self.device)
            self.model.load_state_dict(state, strict=bool(getattr(self.args, "pretrained_strict", 1)))
            print(f"Loaded pretrained weights: {pretrained_path}")

        criterion = self._select_criterion().to(self.device)
        optimizer = self._select_optimizer()

        best_path = os.path.join(ckpt_dir, "best.pt")
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True) if self.args.patience > 0 else None

        use_amp = bool(self.args.use_amp) and self.device.type == "cuda"
        scaler_amp = torch.cuda.amp.GradScaler(enabled=use_amp)

        print("Start Training...")
        best_metric = float("inf")

        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            loss_sum = 0.0

            loop = tqdm(train_loader, leave=False)
            for batch in loop:
                if len(batch) == 3:
                    x, y, sample_weight = batch
                else:
                    x, y = batch
                    sample_weight = None
                x = x.to(self.device, non_blocking=True)  # (B,C,L)
                y = y.to(self.device, non_blocking=True)  # (B,P,C) 常见
                if sample_weight is not None:
                    sample_weight = sample_weight.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = self.model(x)
                    loss = self._compute_loss(criterion, pred, y, x=x, sample_weight=sample_weight)

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
            if val_loader is not None:
                metric = self._eval_mse(val_loader)
                print(f"Epoch {epoch:03d} | val_mse={metric:.6f}")

            if early_stopping is not None:
                early_stopping(metric, self.model, best_path)
                if early_stopping.early_stop:
                    print("Early stopping.")
                    break
            else:
                if metric < best_metric:
                    best_metric = metric
                    save_model(self.model, best_path)

        # 保存 scaler（推理需要）
        scaler_path = os.path.join(ckpt_dir, "scaler.npz")
        scaler.save(scaler_path)
        print(f"Saved best: {best_path}")
        print(f"Saved scaler: {scaler_path}")

    @torch.no_grad()
    def _eval_mse(self, loader):
        self.model.eval()
        total = 0.0
        n = 0
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x)
            total += mse(pred, y)
            n += 1
        if n == 0:
            raise ValueError(
                "Validation loader produced 0 batches. "
                "Your val split is too short for seq_len + target_shift + pred_len. "
                "Reduce seq_len/pred_len or increase val_ratio."
            )
        return total / n

    def test(self, setting, test=0):
        ckpt_dir = self._make_ckpt_dir(setting)
        best_path = os.path.join(ckpt_dir, "best.pt")
        scaler_path = os.path.join(ckpt_dir, "scaler.npz")

        # load model
        load_path = best_path
        if not os.path.exists(load_path):
            pretrained_path = getattr(self.args, "pretrained_path", None)
            if test and pretrained_path:
                if not os.path.exists(pretrained_path):
                    raise FileNotFoundError(f"pretrained_path not found: {pretrained_path}")
                load_path = pretrained_path
                print(f"Using --pretrained_path for test: {load_path}")
                print(
                    "Note: test data normalization is still fitted from the current CSV train split. "
                    "For strict zero-shot evaluation, build the test CSV with source-domain train rows "
                    "and target-domain test rows, or keep the scaler source consistent."
                )
            else:
                raise FileNotFoundError(
                    f"best checkpoint not found: {best_path}. "
                    "Use the same run_stamp/model_id as the training run, or pass "
                    "--pretrained_path with --is_training 0 for direct evaluation."
                )

        state = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

        # load dataset/scaler
        test_set, _, scaler = self._get_data("test")
        if os.path.exists(scaler_path):
            scaler.load(scaler_path)

        # series 必须存在于 dataset
        if not hasattr(test_set, "series"):
            raise AttributeError("test_set must have attribute `series` (T,C). Please store raw series inside Dataset.")

        from utils.tools import run_rolling_inference_and_plot
        raw_overlay = self._load_plot_raw_overlay()

        mse_raw, mse_norm, fig_path, _, _, diagnostics = run_rolling_inference_and_plot(
            model=self.model,
            device=self.device,
            series=test_set.series,
            scaler=scaler,
            seq_len=int(self.args.seq_len),
            pred_len=int(self.args.pred_len),
            save_dir=ckpt_dir,
            setting=setting,
            horizon=int(getattr(self.args, "horizon", 2000)),
            start_idx=int(getattr(self.args, "start_idx", 0)),
            in_indices=getattr(test_set, "in_indices", None),
            out_indices=getattr(test_set, "out_indices", None),
            target_shift=int(getattr(self.args, "target_shift", 0)),
            target_name=getattr(self.args, "target", None),
            plot_filename="rolling_forecast.png",
            raw_overlay=raw_overlay,
            start_indices=getattr(test_set, "start_indices", None),
        )

        print(f"Rolling MSE(raw): {mse_raw:.9f}")
        print(f"Rolling MSE(norm): {mse_norm:.9f}")
        print(f"Saved rolling forecast plot: {fig_path}")
        print(f"Saved scatter plot: {diagnostics['scatter_plot']}")
        print(f"Saved zoom plot: {diagnostics['zoom_plot']}")
        print(f"Pearson(raw): {diagnostics['pearson_raw']:.6f}")
        print(f"MAE(raw): {diagnostics['mae_raw']:.9f}")
        print(f"Saved metrics json: {diagnostics['metrics_json']}")

    def _load_plot_raw_overlay(self):
        raw_col = getattr(self.args, "plot_raw_col", None)
        if raw_col is None:
            return None
        csv_path = os.path.join(self.args.root_path, self.args.data_path)
        df = load_dataframe(csv_path, max_rows=getattr(self.args, "max_rows", None))
        if raw_col not in df.columns:
            print(f"Raw overlay skipped: column '{raw_col}' not found in {csv_path}")
            return None

        raw = df[raw_col].to_numpy(dtype=np.float32)
        split_col = getattr(self.args, "split_col", None)
        if split_col is not None:
            if split_col not in df.columns:
                print(f"Raw overlay skipped: split_col '{split_col}' not found in {csv_path}")
                return None
            segment_col = getattr(self.args, "segment_col", None)
            if segment_col is not None:
                if segment_col not in df.columns:
                    print(f"Raw overlay skipped: segment_col '{segment_col}' not found in {csv_path}")
                    return None
                segment_labels = df[segment_col].to_numpy(dtype=object)
            else:
                segment_labels = np.zeros(len(df), dtype=object)
            test_raw, _, _ = _split_column_arrays(
                self.args,
                raw,
                df[split_col].to_numpy(dtype=object),
                segment_labels,
                "test",
            )
            return test_raw

        _, _, test_raw = time_split(
            raw,
            train_ratio=self.args.train_ratio,
            val_ratio=self.args.val_ratio,
            split_mode=getattr(self.args, "split_mode", "total"),
        )
        return test_raw
