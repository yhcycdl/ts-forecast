#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path

import numpy as np
import pandas as pd


TYPE_DEFAULT_CYCLES = {
    "stable_single_freq": (10, 4),
    "noisy_single_freq": (10, 3),
    "am_fm_modulated": (12, 3),
    "spike_event": (10, 3),
    "multi_freq": (12, 2),
    "weak_periodic": (6, 1),
}

TYPE_MODULES = {
    "stable_single_freq": "cycle_adaptive_window",
    "noisy_single_freq": "main_residual_decomposition",
    "am_fm_modulated": "envelope_frequency_conditioning",
    "spike_event": "event_skeleton_constraint",
    "multi_freq": "frequency_band_decomposition",
    "weak_periodic": "predictability_rejection_or_target_switch",
}

TYPE_ENHANCED_INPUTS = {
    "stable_single_freq": "qp_main_input",
    "noisy_single_freq": "qp_main_input,qp_residual,qp_abs_residual",
    "am_fm_modulated": "qp_main_input,qp_envelope,qp_local_freq_ratio,qp_phase_sin,qp_phase_cos",
    "spike_event": "qp_main_input,qp_residual,qp_abs_residual",
    "multi_freq": "qp_main_input,qp_band0_rms,qp_band1_rms,qp_band2_rms",
    "weak_periodic": "qp_main_input,qp_predictability_score,qp_weak_periodic_flag",
}

TYPE_QP_LOSS_WEIGHTS = {
    "stable_single_freq": {"deriv": 0.5, "env": 0.1, "band": 0.02, "event": 0.0, "corr": 0.2, "multi": 0.1, "peak": 0.0},
    "noisy_single_freq": {"deriv": 0.5, "env": 0.3, "band": 0.05, "event": 0.2, "corr": 0.2, "multi": 0.3, "peak": 0.0},
    "am_fm_modulated": {"deriv": 0.4, "env": 1.0, "band": 0.05, "event": 0.0, "corr": 0.2, "multi": 0.3, "peak": 0.0},
    "spike_event": {"deriv": 1.0, "env": 0.2, "band": 0.05, "event": 2.0, "corr": 0.2, "multi": 0.1, "peak": 1.0},
    "multi_freq": {"deriv": 0.4, "env": 0.3, "band": 0.2, "event": 0.0, "corr": 0.1, "multi": 0.2, "peak": 0.0},
    "weak_periodic": {"deriv": 0.2, "env": 0.2, "band": 0.05, "event": 0.0, "corr": 0.1, "multi": 0.2, "peak": 0.0},
}


def _finite_positive(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr) & (arr > 0)]


def _median_positive(df: pd.DataFrame, col: str, default: float) -> float:
    if col not in df.columns:
        return float(default)
    arr = _finite_positive(df[col])
    return float(np.median(arr)) if arr.size else float(default)


def _mode_string(df: pd.DataFrame, col: str, default: str) -> str:
    if col not in df.columns or df.empty:
        return default
    counts = df[col].astype(str).value_counts()
    if counts.empty:
        return default
    return str(counts.index[0])


def _round_to(value: float, quantum: int, minimum: int = 1) -> int:
    quantum = max(1, int(quantum))
    rounded = int(round(float(value) / quantum) * quantum)
    return max(int(minimum), rounded)


def _recommend_tcn(seq_len: int, pred_len: int, period_samples: float) -> dict:
    if seq_len >= 8192:
        kernel_size = 7
    elif seq_len >= 2048:
        kernel_size = 5
    else:
        kernel_size = 3

    desired_rf = min(float(seq_len), max(float(pred_len) * 2.0, float(period_samples) * 6.0))
    layers = int(math.ceil(math.log2(max((desired_rf - 1.0) / max(kernel_size - 1, 1) + 1.0, 2.0))))
    layers = max(4, min(12, layers))
    return {"kernel_size": kernel_size, "num_layers": layers}


def _recommend_patch(seq_len: int, period_samples: float) -> dict:
    raw_patch = max(8, min(seq_len // 4, int(round(period_samples / 4.0))))
    patch_len = _round_to(raw_patch, 4, minimum=4)
    patch_len = min(patch_len, seq_len)
    patch_stride = max(1, patch_len // 2)
    return {"patch_len": patch_len, "patch_stride": patch_stride}


def _aggregate_config(
    profile: pd.DataFrame,
    label: str,
    input_cycles: int | None,
    output_cycles: int | None,
    stride_cycles: float,
    round_to: int,
    signal_type_override: str | None = None,
) -> dict:
    signal_type = signal_type_override or _mode_string(profile, "signal_type", "stable_single_freq")
    if signal_type not in TYPE_DEFAULT_CYCLES:
        raise ValueError(
            f"Unsupported signal type {signal_type!r}. "
            f"Expected one of: {', '.join(sorted(TYPE_DEFAULT_CYCLES))}."
        )
    default_in, default_out = TYPE_DEFAULT_CYCLES.get(signal_type, (10, 3))
    if signal_type_override:
        in_cycles = int(input_cycles if input_cycles is not None else default_in)
        out_cycles = int(output_cycles if output_cycles is not None else default_out)
    else:
        in_cycles = int(input_cycles if input_cycles is not None else _median_positive(profile, "input_cycles", default_in))
        out_cycles = int(output_cycles if output_cycles is not None else _median_positive(profile, "output_cycles", default_out))

    period = _median_positive(profile, "dominant_period_samples", 128.0)
    fs = _median_positive(profile, "sample_rate_hz", 1.0)
    smooth = _median_positive(profile, "recommended_smooth_window", max(1.0, period / 10.0))

    seq_len = _round_to(period * in_cycles, round_to, minimum=16)
    pred_len = _round_to(period * out_cycles, round_to, minimum=1)
    stride = _round_to(period * stride_cycles, round_to, minimum=1)
    eval_stride = pred_len
    tcn = _recommend_tcn(seq_len, pred_len, period)
    patch = _recommend_patch(seq_len, period)

    score = _median_positive(profile, "predictability_score", 0.0)
    return {
        "label": label,
        "signal_type": signal_type,
        "recommended_module": TYPE_MODULES.get(signal_type, "cycle_adaptive_window"),
        "segments": int(len(profile)),
        "sample_rate_hz": float(fs),
        "dominant_period_samples": float(period),
        "dominant_period_sec": float(period / max(fs, 1e-12)),
        "input_cycles": int(in_cycles),
        "output_cycles": int(out_cycles),
        "seq_len": int(seq_len),
        "pred_len": int(pred_len),
        "stride": int(stride),
        "eval_stride": int(eval_stride),
        "smooth_window_samples": int(round(smooth)),
        "smooth_window_sec": float(smooth / max(fs, 1e-12)),
        "predictability_score": float(score),
        "loss": "huber",
        **tcn,
        **patch,
    }


def _q(value) -> str:
    return shlex.quote(str(value))


def _split_cols(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _format_command(parts: list[str]) -> str:
    return " \\\n  ".join(parts)


def _command_common(
    args: argparse.Namespace,
    cfg: dict,
    model: str,
    model_id: str,
    input_col: str,
    output_col: str,
    csv_override: str | None = None,
    loss_override: str | None = None,
) -> list[str]:
    root_path = args.root_path
    data_path = args.data_path
    csv_source = csv_override or args.prepared_csv
    if csv_source:
        csv_path = Path(csv_source)
        root_path = str(csv_path.parent) + "/"
        data_path = csv_path.name
    enc_in = len(_split_cols(input_col))
    c_out = len(_split_cols(output_col))

    return [
        "python run.py",
        "--is_training 1",
        f"--model {_q(model)}",
        f"--model_id {_q(model_id)}",
        f"--root_path {_q(root_path)}",
        f"--data_path {_q(data_path)}",
        "--features MS",
        f"--input_cols {_q(input_col)}",
        f"--output_cols {_q(output_col)}",
        f"--enc_in {enc_in}",
        f"--c_out {c_out}",
        "--scaler channel",
        f"--seq_len {cfg['seq_len']}",
        f"--pred_len {cfg['pred_len']}",
        f"--stride {cfg['stride']}",
        f"--eval_stride {cfg['eval_stride']}",
        f"--batch_size {args.batch_size}",
        f"--learning_rate {args.learning_rate}",
        f"--train_epochs {args.train_epochs}",
        f"--loss {loss_override or cfg['loss']}",
        f"--split_col {_q(args.split_col)}",
        f"--segment_col {_q(args.segment_col)}",
        f"--plot_raw_col {_q(args.raw_col)}",
        f"--horizon {max(cfg['pred_len'] * 6, cfg['pred_len'])}",
        f"--gpu {args.gpu}",
    ]


def _build_commands(args: argparse.Namespace, cfg: dict) -> list[dict]:
    prefix = args.model_id_prefix or cfg["label"]
    commands: list[dict] = []

    tcn_id = f"{prefix}_{cfg['signal_type']}_tcn_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
    tcn = _command_common(args, cfg, "tcn_claude", tcn_id, args.input_col, args.output_col)
    tcn.extend([
        f"--kernel_size {cfg['kernel_size']}",
        f"--num_layers {cfg['num_layers']}",
        "--d_model 128",
        "--d_ff 256",
        "--dropout 0.1",
        "--residual_output 1",
    ])
    commands.append({"name": "QPWave-TCN smooth->smooth", "command": _format_command(tcn)})

    if cfg["signal_type"] != "weak_periodic":
        cycle_id = f"{prefix}_{cfg['signal_type']}_cycleres_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
        cycle = _command_common(args, cfg, "cycle_residual_tcn", cycle_id, args.input_col, args.output_col)
        cycle_mode = "mean" if cfg["signal_type"] in {"stable_single_freq", "noisy_single_freq", "multi_freq"} else "last"
        cycle_count = 3 if cycle_mode == "mean" else 1
        cycle.extend([
            f"--kernel_size {cfg['kernel_size']}",
            f"--num_layers {cfg['num_layers']}",
            "--d_model 128",
            "--d_ff 256",
            "--dropout 0.1",
            f"--period_len {max(1, int(round(cfg['dominant_period_samples'])))}",
            f"--cycle_base_cycles {cycle_count}",
            f"--cycle_base_mode {cycle_mode}",
            "--cycle_backbone_revin 0",
        ])
        commands.append({"name": "CycleResidual-TCN cycle-template prior", "command": _format_command(cycle)})

    dlinear_id = f"{prefix}_{cfg['signal_type']}_dlinear_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
    dlinear = _command_common(args, cfg, "DLinear", dlinear_id, args.input_col, args.output_col)
    dlinear.extend([f"--moving_avg {max(3, cfg['smooth_window_samples'])}", "--individual 0"])
    commands.append({"name": "DLinear baseline", "command": _format_command(dlinear)})

    patch_id = f"{prefix}_{cfg['signal_type']}_patchtst_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
    patch = _command_common(args, cfg, "PatchTST", patch_id, args.input_col, args.output_col)
    patch.extend([
        f"--patch_len {cfg['patch_len']}",
        f"--patch_stride {cfg['patch_stride']}",
        "--d_model 128",
        "--d_ff 256",
        "--n_heads 8",
        "--e_layers 3",
        "--dropout 0.1",
    ])
    commands.append({"name": "PatchTST baseline", "command": _format_command(patch)})

    if args.include_legacy_baselines:
        legacy_models = [
            ("GRU", "GRU recurrent baseline", []),
            ("CNNLSTM", "CNN-LSTM attention baseline", []),
            ("CRNN", "CRNN residual-conv recurrent baseline", []),
            ("InceptionTime", "InceptionTime convolutional baseline", ["--d_model 64", "--e_layers 6"]),
            ("FastTCN", "FastTCN causal-conv baseline", []),
            ("SpectralCNN", "SpectralCNN residual baseline", []),
            (
                "TimeMixer",
                "TimeMixer decomposition baseline",
                [
                    "--d_model 128",
                    "--d_ff 256",
                    "--e_layers 2",
                    f"--moving_avg {max(3, cfg['smooth_window_samples'])}",
                    "--down_sampling_window 2",
                    "--down_sampling_layers 2",
                    "--down_sampling_method avg",
                    "--channel_independence 0",
                ],
            ),
        ]
        for model_name, label, extra_args in legacy_models:
            legacy_id = f"{prefix}_{cfg['signal_type']}_{model_name.lower()}_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
            legacy = _command_common(args, cfg, model_name, legacy_id, args.input_col, args.output_col)
            legacy.extend(extra_args)
            commands.append({"name": label, "command": _format_command(legacy)})

    use_smoothpec = args.include_smoothpec or cfg["signal_type"] in {"noisy_single_freq", "am_fm_modulated", "spike_event"}
    if use_smoothpec:
        pec_id = f"{prefix}_{cfg['signal_type']}_smoothpec_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
        pec = _command_common(args, cfg, "smooth_pecnet", pec_id, args.raw_col, args.output_col)
        pec.extend([
            f"--kernel_size {cfg['kernel_size']}",
            f"--num_layers {cfg['num_layers']}",
            "--d_model 128",
            "--d_ff 256",
            "--dropout 0.1",
            f"--smoothpec_window {max(1, cfg['smooth_window_samples'])}",
            "--smoothpec_mode smooth_raw",
            "--residual_output 0",
            "--use_revin 0",
            "--cont_weight 0",
        ])
        commands.append({"name": "SmoothPECNet raw->smooth", "command": _format_command(pec)})

    if args.enhanced_csv:
        enhanced_input_cols = args.enhanced_input_cols or TYPE_ENHANCED_INPUTS.get(cfg["signal_type"], "qp_main_input")
        weights = TYPE_QP_LOSS_WEIGHTS.get(
            cfg["signal_type"],
            TYPE_QP_LOSS_WEIGHTS["noisy_single_freq"],
        )
        enhanced_id = f"{prefix}_{cfg['signal_type']}_qpenhanced_sl{cfg['seq_len']}_pl{cfg['pred_len']}"
        enhanced = _command_common(
            args,
            cfg,
            "qpenhanced_tcn",
            enhanced_id,
            enhanced_input_cols,
            args.enhanced_target_col,
            csv_override=args.enhanced_csv,
            loss_override="qp_hybrid",
        )
        enhanced.extend([
            f"--kernel_size {cfg['kernel_size']}",
            f"--num_layers {cfg['num_layers']}",
            "--d_model 128",
            "--d_ff 256",
            "--dropout 0.1",
            "--residual_output 1",
            "--qpenhance_gate 1",
            "--qpenhance_gate_hidden 32",
            f"--qp_deriv_weight {weights['deriv']}",
            f"--qp_envelope_weight {weights['env']}",
            f"--qp_band_weight {weights['band']}",
            f"--qp_event_weight {weights['event']}",
            f"--qp_corr_weight {weights['corr']}",
            f"--qp_multiscale_weight {weights['multi']}",
            f"--qp_peak_weight {weights['peak']}",
            f"--qp_peak_pool {max(3, int(round(cfg['dominant_period_samples'] / 8.0)))}",
        ])
        commands.append({"name": "QPEnhanced-TCN feature-aware", "command": _format_command(enhanced)})

    return commands


def _write_shell(commands: list[dict], path: Path) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for item in commands:
        lines.append(f"# {item['name']}")
        lines.append(item["command"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommend cycle-adaptive forecasting settings from profile_by_segment.csv.")
    parser.add_argument("--profile-csv", required=True, help="CSV produced by analyze_quasiperiodic_profile.py.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prepared-csv", default=None, help="Prepared waveform CSV; used to build runnable commands.")
    parser.add_argument("--enhanced-csv", default=None, help="CSV produced by augment_quasiperiodic_dataset.py.")
    parser.add_argument(
        "--enhanced-input-cols",
        default=None,
        help="Comma-separated enhanced input columns for qpenhanced_tcn; default is chosen from signal_type.",
    )
    parser.add_argument("--enhanced-target-col", default="qp_main_target")
    parser.add_argument("--root-path", default="./outputs/")
    parser.add_argument("--data-path", default="data.csv")
    parser.add_argument("--input-col", default="input_smooth")
    parser.add_argument("--output-col", default="target_smooth")
    parser.add_argument("--raw-col", default="raw")
    parser.add_argument("--split-col", default="split")
    parser.add_argument("--segment-col", default="segment_id")
    parser.add_argument("--model-id-prefix", default=None)
    parser.add_argument("--input-cycles", type=int, default=None, help="Override profile/type input cycles.")
    parser.add_argument("--output-cycles", type=int, default=None, help="Override profile/type output cycles.")
    parser.add_argument("--stride-cycles", type=float, default=0.5, help="Training stride as a fraction of dominant period.")
    parser.add_argument("--round-to", type=int, default=8, help="Round seq_len/pred_len/stride to this sample multiple.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-epochs", type=int, default=40)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--signal-type-override",
        default=None,
        help="Use this known signal type instead of the profile's automatic signal_type mode.",
    )
    parser.add_argument("--include-smoothpec", action="store_true", help="Always include raw->smooth SmoothPECNet command.")
    parser.add_argument("--include-legacy-baselines", action="store_true",
                        help="Include restored GRU/CNNLSTM/CRNN/InceptionTime/FastTCN/SpectralCNN/TimeMixer baselines.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = pd.read_csv(args.profile_csv)
    if profile.empty:
        raise ValueError("Profile CSV is empty.")

    dataset_cfg = _aggregate_config(
        profile,
        label="dataset",
        input_cycles=args.input_cycles,
        output_cycles=args.output_cycles,
        stride_cycles=args.stride_cycles,
        round_to=args.round_to,
        signal_type_override=args.signal_type_override,
    )

    type_configs = []
    if args.signal_type_override:
        type_configs.append(
            _aggregate_config(
                profile,
                label=str(args.signal_type_override),
                input_cycles=args.input_cycles,
                output_cycles=args.output_cycles,
                stride_cycles=args.stride_cycles,
                round_to=args.round_to,
                signal_type_override=args.signal_type_override,
            )
        )
    elif "signal_type" in profile.columns:
        for signal_type, group in profile.groupby("signal_type", sort=False):
            type_configs.append(
                _aggregate_config(
                    group,
                    label=str(signal_type),
                    input_cycles=args.input_cycles,
                    output_cycles=args.output_cycles,
                    stride_cycles=args.stride_cycles,
                    round_to=args.round_to,
                )
            )

    commands = _build_commands(args, dataset_cfg)
    config = {
        "profile_csv": str(Path(args.profile_csv)),
        "prepared_csv": args.prepared_csv,
        "dataset_config": dataset_cfg,
        "type_configs": type_configs,
        "commands": commands,
        "notes": [
            "Use the dataset_config commands when one signal type dominates.",
            "If type_configs contain mixed signal types with very different periods, split the dataset by type before training.",
            "The default policy is roughly 10 past cycles -> 3-4 future cycles; weak-periodic signals are intentionally shorter.",
        ],
    }

    config_path = output_dir / "recommended_qp_config.json"
    shell_path = output_dir / "recommended_train_commands.sh"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_shell(commands, shell_path)
    pd.DataFrame([dataset_cfg] + type_configs).to_csv(output_dir / "recommended_qp_config.csv", index=False)

    print(f"Saved recommended config: {config_path}")
    print(f"Saved command script: {shell_path}")
    print(f"Dataset type: {dataset_cfg['signal_type']}")
    print(
        "Recommended window: "
        f"seq_len={dataset_cfg['seq_len']} ({dataset_cfg['input_cycles']} cycles), "
        f"pred_len={dataset_cfg['pred_len']} ({dataset_cfg['output_cycles']} cycles), "
        f"period={dataset_cfg['dominant_period_samples']:.2f} samples"
    )


if __name__ == "__main__":
    main()
