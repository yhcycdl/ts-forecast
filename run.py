# run.py
import argparse
import hashlib
import os
import random
import re
import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_risk_classification import Exp_Risk_Classification

def set_seed(seed):
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

    if args.task_name != "long_term_forecast":
        base += f"_nc{args.num_classes}"

    model_id = getattr(args, "model_id", None)
    if model_id:
        base += f"_id{_sanitize_tag(model_id, max_len=32)}"

    return f"{base}_{ii}"

def main():
    parser = argparse.ArgumentParser(description="Project")

    # ===== basic =====
    parser.add_argument("--task_name", type=str, default="risk_classification",
                        choices=["long_term_forecast", "risk_classification"])
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)

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
    parser.add_argument('--enc_in', type=int , default=1,help='input')
    parser.add_argument('--out_in',type=int, default=1,help='output')
    # scaler: channel(多变量推荐) / global(单变量)
    parser.add_argument("--scaler", type=str, default="global", choices=["global", "channel"])

    # ===== loader =====
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--drop_last", type=int, default=0)

    # ===== model =====
    parser.add_argument("--model", type=str, default='timemixer')
    parser.add_argument("--model_id", type=str, default=None,
                        help="optional experiment suffix to distinguish runs with different channel selections or configs")


    # ===== loss =====
    parser.add_argument('--loss',type=str, default='MSE',choices=['MSE','hybrid','hubrid','mae','huber','wmse'])
    parser.add_argument("--fft_weight", type=float, default=0.1)
    parser.add_argument("--deriv_weight", type=float, default=1.0)
    parser.add_argument("--cont_weight", type=float, default=5.0)

    # ===== optim/train =====
    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument('--optimizer',type=str, default='Adamw',help='optimizer')
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
    # ===== timemixer =====
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--e_layers", type=int, default=2)

    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_layers", type=int, default=2)
    parser.add_argument("--down_sampling_method", type=str, default="max", choices=["avg", "max", "conv"])

    parser.add_argument("--channel_independence", type=int, default=0, choices=[0, 1])
    parser.add_argument("--decomp_method", type=str, default="moving_avg", choices=["moving_avg", "dft_decomp"])
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--c_out", type=int, default=1)
    parser.add_argument("--kernel_size", type=int, default=3,
                        help="tcn_claude/FullResTCN temporal kernel size; larger values increase receptive field")
    parser.add_argument("--num_layers", type=int, default=11,
                        help="tcn_claude/FullResTCN requested dilated TCN layers")
    parser.add_argument("--base_ch", type=int, default=32,
                        help="tcn_claude/FullResTCN base channel width")
    parser.add_argument("--max_ch", type=int, default=256,
                        help="tcn_claude/FullResTCN max channel width")
    parser.add_argument("--top_k_freq", type=int, default=128,
                        help="tcn_claude frequency branch low-frequency bins")
    parser.add_argument("--freq_dim", type=int, default=128,
                        help="tcn_claude frequency branch embedding dimension")
    parser.add_argument("--use_revin", type=int, default=1,
                        help="tcn_claude: 1 enables RevIN instance normalization")

    # ===== risk classification =====
    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--label_mode", type=str, default="generated", choices=["generated", "column", "file"])
    parser.add_argument("--label_col", type=str, default=None, help="When label_mode=column, use this column from the input CSV")
    parser.add_argument("--label_path", type=str, default=None, help="When label_mode=file, path to point-label or window-label file")
    parser.add_argument("--label_file_col", type=str, default=None, help="Optional label column name for external CSV label files")
    parser.add_argument("--label_start_col", type=str, default="start_idx", help="Window-label file start index column name")
    parser.add_argument("--label_granularity", type=str, default="auto", choices=["auto", "point", "window"],
                        help="point: one label per timestamp; window: one label per window record with start_idx")
    parser.add_argument("--window_label_strategy", type=str, default="last", choices=["last", "max", "majority"],
                        help="How to aggregate pointwise labels into a future-window label")
    parser.add_argument("--risk_label_channel", type=int, default=0, help="标签使用的目标通道索引")
    parser.add_argument("--risk_use_ber", type=int, default=1, help="1: 使用未来窗口频带能量占比辅助打标签")
    parser.add_argument("--sample_rate", type=float, default=1.0, help="采样率，用于 BER 计算")
    parser.add_argument("--risk_band_low", type=float, default=0.0, help="目标频带下界")
    parser.add_argument("--risk_band_high", type=float, default=0.0, help="目标频带上界")
    parser.add_argument("--risk_rms_low_quantile", type=float, default=0.5)
    parser.add_argument("--risk_rms_high_quantile", type=float, default=0.85)
    parser.add_argument("--risk_ber_low_quantile", type=float, default=0.5)
    parser.add_argument("--risk_ber_high_quantile", type=float, default=0.85)
    parser.add_argument("--use_class_weights", type=int, default=1)
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--cls_hidden_dim", type=int, default=256)
    parser.add_argument("--cls_pool_bins", type=int, default=16)
    parser.add_argument("--cls_use_input_norm", type=int, default=0,
                        help="Classification path: 1 enables sample-wise input normalization before backbone encoding")
    parser.add_argument("--cls_use_pre_enc", type=int, default=0,
                        help="Classification path (TimeMixer family only): 1 applies pre-encoder decomposition before backbone")
    parser.add_argument("--cls_selection_metric", type=str, default="f1_macro",
                        choices=["loss", "f1_macro", "balanced_accuracy", "bal_acc", "accuracy", "auprc", "auroc"])

    args = parser.parse_args()

    if args.model is None:
        args.model = "timemixer" if args.task_name == "long_term_forecast" else "risk_cnn"
    if args.loss.lower() == "hubrid":
        args.loss = "hybrid"

    if args.output_cols is not None:
        out_cols = [c.strip() for c in args.output_cols.split(",") if c.strip()]
        if not out_cols:
            raise ValueError("--output_cols was provided but no valid column names were parsed.")
        if len(out_cols) == 1:
            args.target = out_cols[0]

    set_seed(args.seed)

    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Long_Term_Forecast if args.task_name == "long_term_forecast" else Exp_Risk_Classification

    if args.is_training:
        for ii in range(args.itr):
            setting = _build_setting(args, ii)
            exp = Exp(args)
            print(f">>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = _build_setting(args, ii)
        exp = Exp(args)
        print(f">>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
