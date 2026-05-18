import argparse
import random

import numpy as np
import torch

from exp.exp_cascade_forecast_risk import Exp_Cascade_Forecast_Risk


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Cascade Forecast -> Risk Classification")

    parser.add_argument("--task_name", type=str, default="cascade_forecast_risk", choices=["cascade_forecast_risk"])
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--root_path", type=str, default="./data/zenodo_timeseries_csv/")
    parser.add_argument("--data_path", type=str, default="op00_PH2_0p1_Lc_50.csv")
    parser.add_argument("--features", type=str, default="S", choices=["S", "M", "MS"])
    parser.add_argument("--target", type=str, default="P1")
    parser.add_argument("--col_names", type=str, default=None)
    parser.add_argument("--max_rows", type=int, default=10_000_000)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--split_mode", type=str, default="total", choices=["total", "legacy_rest"])

    parser.add_argument("--stride", type=int, default=67)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--pred_len", type=int, default=128)
    parser.add_argument("--enc_in", type=int, default=1)
    parser.add_argument("--out_in", type=int, default=1)
    parser.add_argument("--scaler", type=str, default="global", choices=["global", "channel"])

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=1)
    parser.add_argument("--drop_last", type=int, default=0)

    parser.add_argument(
        "--forecast_model",
        type=str,
        default="timemixer",
        choices=[
            "CNNLSTM",
            "CRNN",
            "GRU",
            "fast_tcn",
            "Fullrestcn",
            "spetical",
            "inceptiontime",
            "DLinear",
            "PatchTST",
            "mamba",
            "timemixer",
            "tcn_claude",
            "timemixer_claude",
        ],
    )

    parser.add_argument("--loss", type=str, default="MSE", choices=["MSE", "hybrid", "hubrid", "mae", "huber", "wmse"])
    parser.add_argument("--fft_weight", type=float, default=0.1)
    parser.add_argument("--deriv_weight", type=float, default=1.0)
    parser.add_argument("--cont_weight", type=float, default=5.0)

    parser.add_argument("--train_epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", dest="use_amp", action="store_true")
    parser.add_argument("--no_use_amp", dest="use_amp", action="store_false")
    parser.set_defaults(use_amp=True)
    parser.add_argument("--optimizer", type=str, default="Adamw")
    parser.add_argument("--momentum", type=float, default=0.9)

    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--forecast_ckpt_setting", type=str, default=None)

    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0")

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

    parser.add_argument("--num_classes", type=int, default=2, choices=[2, 3])
    parser.add_argument("--label_mode", type=str, default="generated", choices=["generated", "column", "file"])
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--label_path", type=str, default=None)
    parser.add_argument("--label_file_col", type=str, default=None)
    parser.add_argument("--label_start_col", type=str, default="start_idx")
    parser.add_argument("--label_granularity", type=str, default="auto", choices=["auto", "point", "window"])
    parser.add_argument("--window_label_strategy", type=str, default="last", choices=["last", "max", "majority"])
    parser.add_argument("--risk_label_channel", type=int, default=0)
    parser.add_argument("--risk_use_ber", type=int, default=1)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--risk_band_low", type=float, default=0.0)
    parser.add_argument("--risk_band_high", type=float, default=0.0)
    parser.add_argument("--risk_rms_low_quantile", type=float, default=0.5)
    parser.add_argument("--risk_rms_high_quantile", type=float, default=0.85)
    parser.add_argument("--risk_ber_low_quantile", type=float, default=0.5)
    parser.add_argument("--risk_ber_high_quantile", type=float, default=0.85)
    parser.add_argument("--use_class_weights", type=int, default=1)
    parser.add_argument("--class_weight_power", type=float, default=1.0)
    parser.add_argument("--cls_hidden_dim", type=int, default=256)
    parser.add_argument("--cls_pool_bins", type=int, default=16)
    parser.add_argument("--cls_selection_metric", type=str, default="f1_macro",
                        choices=["loss", "f1_macro", "balanced_accuracy", "bal_acc", "accuracy", "auprc", "auroc"])

    parser.add_argument("--cascade_freeze_forecast", type=int, default=1)
    parser.add_argument("--cascade_detach_forecast", type=int, default=None)
    parser.add_argument("--cascade_cls_ch1", type=int, default=32)
    parser.add_argument("--cascade_cls_ch2", type=int, default=64)
    parser.add_argument("--cascade_cls_ch3", type=int, default=128)
    parser.add_argument("--joint_use_future_band_feature", type=int, default=1)

    args = parser.parse_args()
    if args.loss.lower() == "hubrid":
        args.loss = "hybrid"

    if args.cascade_detach_forecast is None:
        args.cascade_detach_forecast = 1 if int(args.cascade_freeze_forecast) else 0

    set_seed(args.seed)

    args.use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Cascade_Forecast_Risk

    if args.is_training:
        for ii in range(args.itr):
            setting = (
                f"{args.task_name}_{args.forecast_model}_{args.features}_sl{args.seq_len}_pl{args.pred_len}_"
                f"nc{args.num_classes}_cf{int(args.cascade_freeze_forecast)}_"
                f"bs{args.batch_size}_lr{args.learning_rate}_{ii}"
            )
            exp = Exp(args)
            print(f">>>>>>> start training : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)
            print(f">>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = (
            f"{args.task_name}_{args.forecast_model}_{args.features}_sl{args.seq_len}_pl{args.pred_len}_"
            f"nc{args.num_classes}_cf{int(args.cascade_freeze_forecast)}_"
            f"bs{args.batch_size}_lr{args.learning_rate}_{ii}"
        )
        exp = Exp(args)
        print(f">>>>>>> testing : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
