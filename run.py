# run.py
import argparse
import hashlib
import os
import random
import re
from datetime import datetime
import numpy as np

from models.registry import MODEL_NAMES

def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_tag(value: str, max_len: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    if not text:
        text = "na"
    return text[:max_len]


def _parse_cols(value):
    if value is None:
        return None
    cols = [c.strip() for c in str(value).split(",") if c.strip()]
    return cols if cols else None


def _require_positive(args, names):
    for name in names:
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}.")


def _looks_like_same_waveform(input_col: str, output_col: str) -> bool:
    a = input_col.lower()
    b = output_col.lower()
    if a == b:
        return True
    if ("raw" in a) != ("raw" in b):
        return False
    smooth_tokens = ("smooth", "ma", "cma")
    if any(t in a for t in smooth_tokens) and any(t in b for t in smooth_tokens):
        return True
    return False


def _validate_experiment_args(args) -> None:
    if args.model is None:
        args.model = "tcn_claude"
    if args.loss.lower() == "hubrid":
        args.loss = "hybrid"

    _require_positive(args, ["itr", "seq_len", "pred_len", "stride", "enc_in", "c_out", "batch_size"])
    if args.eval_stride < -1:
        raise ValueError("--eval_stride must be -1, 0, or a positive integer.")
    if args.target_shift < 0 and args.window_mode == "past":
        raise ValueError("--target_shift must be non-negative when --window_mode=past.")
    if args.window_mode == "center":
        center_left = args.seq_len // 2 if args.center_left < 0 else args.center_left
        if center_left < 0 or center_left + args.pred_len > args.seq_len:
            raise ValueError("center mode requires 0 <= center_left and center_left + pred_len <= seq_len.")

    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be in (0, 1).")
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0, 1).")
    if args.split_col is None and args.split_mode == "total" and args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train_ratio + --val_ratio must be < 1 when not using --split_col.")

    input_cols = _parse_cols(getattr(args, "input_cols", None))
    output_cols = _parse_cols(getattr(args, "output_cols", None))
    if args.input_cols is not None and not input_cols:
        raise ValueError("--input_cols was provided but no valid column names were parsed.")
    if args.output_cols is not None:
        if not output_cols:
            raise ValueError("--output_cols was provided but no valid column names were parsed.")
        if len(output_cols) == 1:
            args.target = output_cols[0]

    if input_cols is not None or output_cols is not None:
        if not input_cols or not output_cols:
            raise ValueError("Explicit IO requires both --input_cols and --output_cols.")
        if int(args.enc_in) != len(input_cols):
            raise ValueError(
                f"--enc_in={args.enc_in} does not match --input_cols count={len(input_cols)} "
                f"({input_cols})."
            )
        if int(args.c_out) != len(output_cols):
            raise ValueError(
                f"--c_out={args.c_out} does not match --output_cols count={len(output_cols)} "
                f"({output_cols})."
            )

        same_first = _looks_like_same_waveform(input_cols[0], output_cols[0])
        if args.model == "tcn_claude" and not same_first:
            if int(getattr(args, "residual_output", 1)):
                print(
                    "[WARN] tcn_claude residual_output assumes the first input channel and target "
                    "are the same waveform quantity. For raw->smooth or other transformed targets, "
                    "prefer --residual_output 0 or use smooth_pecnet with a smooth first branch."
                )
            if int(getattr(args, "use_revin", 1)):
                print(
                    "[WARN] tcn_claude RevIN denormalizes with input-window statistics. "
                    "This is safest for smooth->smooth / same-quantity forecasting; "
                    "for raw->smooth, consider --use_revin 0 together with --residual_output 0."
                )

        if args.model in {"DLinear", "PatchTST"} and len(input_cols) != len(output_cols):
            print(
                f"[WARN] {args.model} is a baseline with weak semantic mapping for "
                "multi-input -> fewer-output explicit IO. Use single-input experiments or "
                "keep input/output column order one-to-one for clean baseline comparisons."
            )

        if str(args.loss).lower() == "hybrid" and float(getattr(args, "cont_weight", 0.0)) > 0 and not same_first:
            print(
                "[WARN] hybrid cont_weight > 0 enforces boundary continuity between the first "
                "input channel and target. For raw->smooth/event targets, set --cont_weight 0."
            )

    if args.sample_weight_col and args.loss.lower() == "hybrid":
        raise ValueError("--sample_weight_col is not supported with --loss hybrid.")

    if args.model in {"tcn_claude", "smooth_pecnet"}:
        _require_positive(args, ["kernel_size", "num_layers", "base_ch", "max_ch", "top_k_freq", "freq_dim"])
        if args.kernel_size < 2:
            raise ValueError("--kernel_size must be >= 2 for TCN models.")
        if args.kernel_size >= args.seq_len:
            raise ValueError("--kernel_size must be smaller than --seq_len for TCN models.")
        if args.max_ch < args.base_ch:
            raise ValueError("--max_ch must be >= --base_ch.")
        if args.smoothpec_window < 1:
            raise ValueError("--smoothpec_window must be >= 1.")

    if args.model == "PatchTST":
        _require_positive(args, ["patch_len", "patch_stride", "n_heads", "factor", "d_model", "d_ff"])
        if args.patch_len > args.seq_len:
            raise ValueError("--patch_len must be <= --seq_len.")
        if args.d_model % args.n_heads != 0:
            raise ValueError("--d_model must be divisible by --n_heads for PatchTST.")

    if args.model == "DLinear":
        if args.moving_avg <= 1:
            raise ValueError("--moving_avg must be > 1 for DLinear.")


def _build_setting(args, ii: int) -> str:
    data_tag = _sanitize_tag(os.path.splitext(os.path.basename(args.data_path))[0], max_len=32)
    output_cols = getattr(args, "output_cols", None)
    if output_cols:
        target_value = output_cols.split(",")[0].strip()
    else:
        target_value = getattr(args, "target", "na")
    target_tag = _sanitize_tag(target_value, max_len=24)

    input_cols = getattr(args, "input_cols", None)
    col_names = getattr(args, "col_names", None)
    if input_cols is not None or output_cols is not None:
        io_cols = []
        if input_cols:
            io_cols.extend([c.strip() for c in input_cols.split(",") if c.strip()])
        if output_cols:
            io_cols.extend([c.strip() for c in output_cols.split(",") if c.strip()])
        col_list = []
        seen = set()
        for col in io_cols:
            if col not in seen:
                col_list.append(col)
                seen.add(col)
        col_count = len(col_list)
        col_sig = hashlib.md5(",".join(col_list).encode("utf-8")).hexdigest()[:8]
    elif col_names is not None:
        col_list = [c.strip() for c in col_names.split(",") if c.strip()]
        col_count = len(col_list)
        col_sig = hashlib.md5(",".join(col_list).encode("utf-8")).hexdigest()[:8]
    else:
        col_count = int(getattr(args, "enc_in", 0))
        col_sig = "auto"

    base = (
        f"{args.task_name}_{args.model}_{args.features}_"
        f"d{data_tag}_t{target_tag}_"
        f"cin{args.enc_in}_cout{args.c_out}_cols{col_count}_{col_sig}_"
        f"sl{args.seq_len}_pl{args.pred_len}_bs{args.batch_size}_lr{args.learning_rate}"
    )
    target_shift = int(getattr(args, "target_shift", 0))
    if target_shift > 0:
        base += f"_ts{target_shift}"
    window_mode = str(getattr(args, "window_mode", "past")).lower()
    if window_mode != "past":
        base += f"_wm{window_mode}_cl{int(getattr(args, 'center_left', -1))}"

    model_id = getattr(args, "model_id", None)
    if model_id:
        base += f"_id{_sanitize_tag(model_id, max_len=32)}"

    run_tag = getattr(args, "run_tag", None)
    if run_tag:
        base += f"_tag{_sanitize_tag(run_tag, max_len=32)}"

    run_stamp = getattr(args, "run_stamp", None)
    if run_stamp:
        base += f"_t{_sanitize_tag(run_stamp, max_len=16)}"

    return f"{base}_{ii}"

def main():
    parser = argparse.ArgumentParser(
        description="Quasi-periodic main-waveform forecasting runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ===== basic =====
    parser.add_argument("--task_name", type=str, default="long_term_forecast",
                        choices=["long_term_forecast"])
    parser.add_argument("--is_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run_tag", type=str, default=None,
                        help="optional human-readable tag appended to the checkpoint directory")
    parser.add_argument("--run_stamp", type=str, default=None,
                        help="optional fixed timestamp/run id appended to the checkpoint directory")
    parser.add_argument("--timestamp_setting", type=int, default=1,
                        help="1 appends YYYYmmdd-HHMMSS to training checkpoint directories when --run_stamp is not set")

    # ===== data =====
    parser.add_argument("--root_path", type=str, default="./data/zenodo_timeseries_csv/")
    parser.add_argument("--data_path", type=str, default="op00_PH2_0p1_Lc_50.csv")
    parser.add_argument("--features", type=str, default="S", choices=["S", "M", "MS"],
                        help="S: univariate->univariate, M: multivariate->multivariate, MS: multivariate->univariate")
    parser.add_argument("--target", type=str, default="P1")
    parser.add_argument("--col_names", type=str, default=None,
                        help="optional, comma-separated column names; if None, infer by features/target")
    parser.add_argument("--input_cols", type=str, default=None,
                        help="optional, comma-separated input columns; when set, overrides implicit input selection")
    parser.add_argument("--output_cols", type=str, default=None,
                        help="optional, comma-separated output columns; when set, overrides implicit output selection")
    parser.add_argument("--sample_weight_col", type=str, default=None,
                        help="optional, pointwise sample-weight column from the CSV used only during forecast training")
    parser.add_argument("--split_col", type=str, default=None,
                        help="optional split label column with train/val/test values; useful for merged multi-segment CSVs")
    parser.add_argument("--segment_col", type=str, default=None,
                        help="optional segment/condition column; windows will not cross segment boundaries when split_col is set")
    parser.add_argument("--plot_raw_col", type=str, default=None,
                        help="optional raw column to overlay on rolling forecast plots, e.g. p00")
    parser.add_argument("--sample_weight_scale", type=float, default=1.0,
                        help="sample weight becomes sample_weight_bias + sample_weight_scale * weight_col")
    parser.add_argument("--sample_weight_bias", type=float, default=1.0,
                        help="base sample weight added to every forecast target point")
    parser.add_argument("--max_rows", type=int, default=10_000_000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="split_mode=total 时表示全数据比例；0 means no val split")
    parser.add_argument("--split_mode", type=str, default="total", choices=["total", "legacy_rest"],
                        help="total: 按全数据比例切 train/val/test；legacy_rest: 兼容旧逻辑，在剩余集上再切 val")

    parser.add_argument("--stride", type=int, default=67)
    parser.add_argument("--eval_stride", type=int, default=-1,
                        help="stride for val/test windows when split_col is used; -1 means pred_len, 0 means use --stride")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--pred_len", type=int, default=128)
    parser.add_argument("--target_shift", type=int, default=0,
                        help="forecast target shift relative to the history window end, in samples")
    parser.add_argument("--window_mode", type=str, default="past", choices=["past", "center"],
                        help="past: use a history window before the target; center: use pressure context around the target.")
    parser.add_argument("--center_left", type=int, default=-1,
                        help="for window_mode=center, number of input samples before the target; <0 uses seq_len//2")
    parser.add_argument("--horizon", type=int, default=2000,
                        help="number of forecast points to render/evaluate in rolling test plots; <=0 means plot the full test split")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="rolling test start index within the chosen split")
    parser.add_argument("--enc_in", type=int, default=1, help="number of input channels")
    parser.add_argument("--c_out", type=int, default=1, help="number of output channels")
    parser.add_argument("--out_in", type=int, default=1, help=argparse.SUPPRESS)
    # scaler: channel(多变量推荐) / global(单变量)
    parser.add_argument("--scaler", type=str, default="global", choices=["global", "channel"])

    # ===== loader =====
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1, choices=[0, 1])
    parser.add_argument("--drop_last", type=int, default=0, choices=[0, 1])

    # ===== model =====
    parser.add_argument("--model", type=str, default=None, choices=MODEL_NAMES)
    parser.add_argument("--model_id", type=str, default=None,
                        help="optional experiment suffix to distinguish runs with different channel selections or configs")


    # ===== loss =====
    parser.add_argument('--loss',type=str, default='MSE',choices=['MSE','hybrid','hubrid','mae','huber','wmse'])
    parser.add_argument("--fft_weight", type=float, default=0.1)
    parser.add_argument("--deriv_weight", type=float, default=1.0)
    parser.add_argument("--cont_weight", type=float, default=5.0)
    parser.add_argument("--hybrid_phase_weight", type=float, default=0.0,
                        help="optional wrapped FFT phase penalty inside hybrid loss; default 0 keeps frequency loss magnitude-only")
    parser.add_argument("--huber_beta", type=float, default=0.3)
    parser.add_argument("--wmse_alpha", type=float, default=1.0)

    # ===== optim/train =====
    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", "--no_use_amp", "--no-use-amp", dest="use_amp",
                        action="store_false", help="disable automatic mixed precision even on CUDA")
    parser.add_argument("--optimizer", type=str, default="AdamW",
                        choices=["Adam", "AdamW", "SGD", "adam", "adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)

    # ===== checkpoint =====
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--patience", type=int, default=0, help="0 disables early stopping")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="optional model state_dict path to initialize training, used for fine-tuning")
    parser.add_argument("--pretrained_strict", type=int, default=1,
                        help="1: require exact key match when loading --pretrained_path")

    # ===== gpu =====
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")
    # ===== model hyperparameters shared by current baselines =====
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--e_layers", type=int, default=2)

    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--individual", type=int, default=0, choices=[0, 1],
                        help="DLinear: use one linear head per output channel")
    parser.add_argument("--patch_len", type=int, default=16,
                        help="PatchTST patch length")
    parser.add_argument("--patch_stride", type=int, default=8,
                        help="PatchTST patch stride")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="PatchTST attention heads")
    parser.add_argument("--factor", type=int, default=1,
                        help="PatchTST attention factor")
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"],
                        help="PatchTST encoder activation")
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="tcn_claude temporal kernel size; larger values increase receptive field")
    parser.add_argument("--num_layers", type=int, default=11,
                        help="tcn_claude requested dilated TCN layers")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="tcn_claude base channel width")
    parser.add_argument("--max_ch", type=int, default=256,
                        help="tcn_claude max channel width")
    parser.add_argument("--top_k_freq", type=int, default=128,
                        help="tcn_claude frequency branch low-frequency bins")
    parser.add_argument("--freq_dim", type=int, default=128,
                        help="tcn_claude frequency branch embedding dimension")
    parser.add_argument("--use_revin", type=int, default=1,
                        help="tcn_claude: 1 enables RevIN instance normalization")
    parser.add_argument("--residual_output", type=int, default=1,
                        help="tcn_claude: 1 predicts last input value + delta; 0 predicts direct output")
    parser.add_argument("--smoothpec_window", type=int, default=1,
                        help="smooth_pecnet causal moving-average window in samples; 1 disables smoothing")
    parser.add_argument("--smoothpec_mode", type=str, default="smooth_raw",
                        choices=["smooth_raw", "raw_smooth", "smooth_residual", "smooth_only"],
                        help="smooth_pecnet input arrangement before the TCN backbone")

    args = parser.parse_args()
    _validate_experiment_args(args)

    import torch
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    set_seed(args.seed)

    if (
        not getattr(args, "run_stamp", None)
        and int(getattr(args, "timestamp_setting", 1))
        and int(getattr(args, "is_training", 1))
    ):
        args.run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.is_training:
        for ii in range(args.itr):
            setting = _build_setting(args, ii)
            exp = Exp_Long_Term_Forecast(args)
            print(f">>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = _build_setting(args, ii)
        exp = Exp_Long_Term_Forecast(args)
        print(f">>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
