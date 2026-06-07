# Quasi-Periodic Forecasting Experiment Runbook

This runbook is the concrete experiment workflow for the paper line:

**Feature-aware main-waveform forecasting for quasi-periodic time series.**

The goal is not to prove that every raw spike can be predicted. The goal is to
show that different quasi-periodic signal types need different prediction
targets, windows, priors, features, and losses.

Run all commands from the repository root on the server:

```bash
cd ~/data/ts-forecast
conda activate timesnet
```

If the server path is different, replace `~/data/ts-forecast` with the actual
repository path.

## 0. Update Code And Check Environment

```bash
cd ~/data/ts-forecast
git pull origin main

conda activate timesnet

python - <<'PY'
import torch, numpy, pandas, scipy, wfdb
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("wfdb:", wfdb.__version__)
PY
```

If any package is missing:

```bash
pip install -r requirements.txt
```

For 4090/CUDA servers, install PyTorch first if needed:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run a model-interface smoke test before long jobs:

```bash
python scripts/smoke_forecast_models.py
```

## 1. Experiment Logic

Use the same pipeline for every dataset:

1. Prepare long-format CSV: `raw`, `input_smooth`, `target_smooth`, `split`,
   `segment_id`, `fs`.
2. Analyze signal profile from train split only.
3. Split by signal type when needed.
4. Build causal enhanced `qp_*` features.
5. Generate recommended cycle-adaptive commands.
6. Train model matrix.
7. Summarize point and structure metrics.

Model matrix:

- `DLinear`: decomposition-linear baseline.
- `PatchTST`: patch Transformer baseline.
- `tcn_claude`: QPWave-TCN main model.
- `cycle_residual_tcn`: cycle-template prior plus TCN residual.
- `smooth_pecnet`: raw-to-smooth main waveform module.
- `qpenhanced_tcn`: feature-gated TCN with `qp_hybrid` structure-aware loss.

Optional restored legacy baselines are available but not included by default:
`GRU`, `CNNLSTM`, `CRNN`, `InceptionTime`, `FastTCN`, `SpectralCNN`, and
`TimeMixer`. Add `--include-legacy-baselines` to the plan/recommend commands
when you need a wider comparison table.

Important default policy:

- Input: roughly `10` past cycles.
- Output: roughly `3-4` future cycles.
- Weak-periodic data: shorter horizon or target switch, not long raw-waveform
  prediction.

## 2. Controlled Synthetic First Pass

This is the first run. It validates the whole code path on known signal types
before spending GPU time on public or teacher-provided data.

### 2.1 Generate Six Synthetic Signal Types

```bash
mkdir -p ./outputs/synthetic_qp6

python scripts/generate_synthetic_quasiperiodic_dataset.py \
  --types stable_single_freq,noisy_single_freq,am_fm_modulated,spike_event,multi_freq,weak_periodic \
  --records-per-type 8 \
  --cycles-per-record 400 \
  --sample-rate 100 \
  --period-sec 1.0 \
  --input-smooth-sec 0.12 \
  --input-smooth-mode causal \
  --target-smooth-sec 0.12 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --seed 2026 \
  --output ./outputs/synthetic_qp6/synthetic_qp6.csv
```

### 2.2 Profile Synthetic Data

```bash
python scripts/analyze_quasiperiodic_profile.py \
  --csv ./outputs/synthetic_qp6/synthetic_qp6.csv \
  --signal-cols target_smooth \
  --time-col time \
  --fs-col fs \
  --segment-col segment_id \
  --split-col split \
  --split-values train \
  --output-dir ./outputs/synthetic_qp6/profile
```

Check:

```bash
cat ./outputs/synthetic_qp6/profile/profile_report.md
```

### 2.3 Split Synthetic Data By True Type

```bash
python scripts/split_quasiperiodic_dataset_by_type.py \
  --csv ./outputs/synthetic_qp6/synthetic_qp6.csv \
  --type-col synthetic_type \
  --output-dir ./outputs/synthetic_qp6/by_type
```

### 2.4 Build Prepare/Train Plan

```bash
python scripts/build_qp_experiment_plan.py \
  --split-metadata ./outputs/synthetic_qp6/by_type/split_by_type_metadata.json \
  --output-dir ./outputs/synthetic_qp6/plan \
  --model-id-prefix synthetic_qp6 \
  --profile-signal-cols target_smooth \
  --input-col input_smooth \
  --output-col target_smooth \
  --raw-col raw \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --train-epochs 40 \
  --gpu 0
```

This writes:

- `./outputs/synthetic_qp6/plan/prepare_qp_experiments.sh`
- `./outputs/synthetic_qp6/plan/run_all_train_commands.sh`

To include restored legacy baselines in every generated training script, add
`--include-legacy-baselines` to the `build_qp_experiment_plan.py` command.

Run preparation:

```bash
bash ./outputs/synthetic_qp6/plan/prepare_qp_experiments.sh
```

Train all synthetic experiments:

```bash
bash ./outputs/synthetic_qp6/plan/run_all_train_commands.sh
```

If you want to run only one type first:

```bash
python scripts/build_qp_experiment_plan.py \
  --split-metadata ./outputs/synthetic_qp6/by_type/split_by_type_metadata.json \
  --include-types stable_single_freq \
  --output-dir ./outputs/synthetic_qp6/plan_stable_only \
  --model-id-prefix synthetic_qp6_stable \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --train-epochs 20 \
  --gpu 0

bash ./outputs/synthetic_qp6/plan_stable_only/prepare_qp_experiments.sh
bash ./outputs/synthetic_qp6/plan_stable_only/run_all_train_commands.sh
```

### 2.5 Summarize Synthetic Results

```bash
python scripts/summarize_forecast_metrics.py \
  --root ./checkpoints \
  --output ./outputs/synthetic_qp6/synthetic_qp6_metrics_summary.csv
```

Look at these columns first:

- `mse_raw`, `mae_raw`, `pearson_raw`
- `dominant_period_relative_error`
- `spectral_energy_l1`
- `envelope_relative_mae`
- `peak_time_mae_samples`, `peak_hit_rate` for spike data

## 3. Download Public Datasets

Download WFDB public datasets into the data disk:

```bash
mkdir -p ./data/extracted/bidmc ./data/extracted/fantasia ./data/extracted/mitdb

python - <<'PY'
import wfdb

datasets = {
    "bidmc": "./data/extracted/bidmc",
    "fantasia": "./data/extracted/fantasia",
    "mitdb": "./data/extracted/mitdb",
}

for name, path in datasets.items():
    print(f"Downloading {name} -> {path}")
    wfdb.dl_database(name, dl_dir=path)
PY
```

List signals if needed:

```bash
python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset bidmc \
  --sources ./data/extracted/bidmc \
  --list-signals \
  --output /tmp/bidmc_dummy.csv

python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset fantasia \
  --sources ./data/extracted/fantasia \
  --list-signals \
  --output /tmp/fantasia_dummy.csv

python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset mitdb \
  --sources ./data/extracted/mitdb \
  --list-signals \
  --output /tmp/mitdb_dummy.csv
```

## 4. Public Dataset A: BIDMC RESP

Expected role: relatively regular respiratory quasi-periodic signal. Use this
as stable or mildly noisy dominant-period data.

### 4.1 Prepare

```bash
mkdir -p ./outputs/quasi_bidmc_resp_ma2s

python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset bidmc \
  --sources ./data/extracted/bidmc \
  --signal-names RESP \
  --resample-to 25 \
  --input-smooth-sec 2.0 \
  --input-smooth-mode causal \
  --target-smooth-sec 2.0 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv
```

### 4.2 Profile

```bash
python scripts/analyze_quasiperiodic_profile.py \
  --csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --signal-cols target_smooth \
  --time-col time \
  --fs-col fs \
  --segment-col segment_id \
  --split-col split \
  --split-values train \
  --output-dir ./outputs/quasi_bidmc_resp_ma2s/profile
```

### 4.3 Split By Profile Type

```bash
python scripts/split_quasiperiodic_dataset_by_type.py \
  --csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --profile-csv ./outputs/quasi_bidmc_resp_ma2s/profile/profile_by_segment.csv \
  --segment-col segment_id \
  --output-dir ./outputs/quasi_bidmc_resp_ma2s/by_type \
  --drop-unknown
```

### 4.4 Build And Run Experiments

```bash
python scripts/build_qp_experiment_plan.py \
  --split-metadata ./outputs/quasi_bidmc_resp_ma2s/by_type/split_by_type_metadata.json \
  --output-dir ./outputs/quasi_bidmc_resp_ma2s/plan \
  --model-id-prefix bidmc_resp \
  --profile-signal-cols target_smooth \
  --input-col input_smooth \
  --output-col target_smooth \
  --raw-col raw \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --train-epochs 40 \
  --gpu 0

bash ./outputs/quasi_bidmc_resp_ma2s/plan/prepare_qp_experiments.sh
bash ./outputs/quasi_bidmc_resp_ma2s/plan/run_all_train_commands.sh
```

## 5. Public Dataset B: Fantasia RESP

Expected role: respiratory signal with subject variation and possible
amplitude/frequency modulation. Use this for AM/FM conditioning.

### 5.1 Prepare

```bash
mkdir -p ./outputs/quasi_fantasia_resp_ma2s

python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset fantasia \
  --sources ./data/extracted/fantasia \
  --signal-names RESP \
  --resample-to 25 \
  --input-smooth-sec 2.0 \
  --input-smooth-mode causal \
  --target-smooth-sec 2.0 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output ./outputs/quasi_fantasia_resp_ma2s/fantasia_resp_ma2s.csv
```

### 5.2 Profile, Split, Plan, Train

```bash
python scripts/analyze_quasiperiodic_profile.py \
  --csv ./outputs/quasi_fantasia_resp_ma2s/fantasia_resp_ma2s.csv \
  --signal-cols target_smooth \
  --time-col time \
  --fs-col fs \
  --segment-col segment_id \
  --split-col split \
  --split-values train \
  --output-dir ./outputs/quasi_fantasia_resp_ma2s/profile

python scripts/split_quasiperiodic_dataset_by_type.py \
  --csv ./outputs/quasi_fantasia_resp_ma2s/fantasia_resp_ma2s.csv \
  --profile-csv ./outputs/quasi_fantasia_resp_ma2s/profile/profile_by_segment.csv \
  --segment-col segment_id \
  --output-dir ./outputs/quasi_fantasia_resp_ma2s/by_type \
  --drop-unknown

python scripts/build_qp_experiment_plan.py \
  --split-metadata ./outputs/quasi_fantasia_resp_ma2s/by_type/split_by_type_metadata.json \
  --output-dir ./outputs/quasi_fantasia_resp_ma2s/plan \
  --model-id-prefix fantasia_resp \
  --profile-signal-cols target_smooth \
  --input-col input_smooth \
  --output-col target_smooth \
  --raw-col raw \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --train-epochs 40 \
  --gpu 1

bash ./outputs/quasi_fantasia_resp_ma2s/plan/prepare_qp_experiments.sh
bash ./outputs/quasi_fantasia_resp_ma2s/plan/run_all_train_commands.sh
```

## 6. Public Dataset C: MITDB MLII ECG

Expected role: spike-event quasi-periodic signal. Do not expect long raw spike
waveform prediction to be perfect. The useful question is whether event-aware
loss/features improve peak timing and structure metrics.

### 6.1 Prepare

```bash
mkdir -p ./outputs/quasi_mitdb_mlii_ma008s

python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset mitdb \
  --sources ./data/extracted/mitdb \
  --signal-names MLII \
  --resample-to 125 \
  --input-smooth-sec 0.08 \
  --input-smooth-mode causal \
  --target-smooth-sec 0.08 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output ./outputs/quasi_mitdb_mlii_ma008s/mitdb_mlii_ma008s.csv
```

If too many records are mixed and training looks unstable, run a smaller MLII
subset first:

```bash
python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset mitdb \
  --sources ./data/extracted/mitdb \
  --signal-names MLII \
  --record-limit 20 \
  --resample-to 125 \
  --input-smooth-sec 0.08 \
  --input-smooth-mode causal \
  --target-smooth-sec 0.08 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output ./outputs/quasi_mitdb_mlii_ma008s/mitdb_mlii_ma008s_limit20.csv
```

### 6.2 Profile, Split, Plan, Train

```bash
python scripts/analyze_quasiperiodic_profile.py \
  --csv ./outputs/quasi_mitdb_mlii_ma008s/mitdb_mlii_ma008s.csv \
  --signal-cols target_smooth \
  --time-col time \
  --fs-col fs \
  --segment-col segment_id \
  --split-col split \
  --split-values train \
  --output-dir ./outputs/quasi_mitdb_mlii_ma008s/profile

python scripts/split_quasiperiodic_dataset_by_type.py \
  --csv ./outputs/quasi_mitdb_mlii_ma008s/mitdb_mlii_ma008s.csv \
  --profile-csv ./outputs/quasi_mitdb_mlii_ma008s/profile/profile_by_segment.csv \
  --segment-col segment_id \
  --output-dir ./outputs/quasi_mitdb_mlii_ma008s/by_type \
  --drop-unknown

python scripts/build_qp_experiment_plan.py \
  --split-metadata ./outputs/quasi_mitdb_mlii_ma008s/by_type/split_by_type_metadata.json \
  --output-dir ./outputs/quasi_mitdb_mlii_ma008s/plan \
  --model-id-prefix mitdb_mlii \
  --profile-signal-cols target_smooth \
  --input-col input_smooth \
  --output-col target_smooth \
  --raw-col raw \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --train-epochs 40 \
  --gpu 2

bash ./outputs/quasi_mitdb_mlii_ma008s/plan/prepare_qp_experiments.sh
bash ./outputs/quasi_mitdb_mlii_ma008s/plan/run_all_train_commands.sh
```

## 7. Teacher-Provided Or New Domain Data

Preferred raw CSV format for each record:

```text
time, signal_raw, ch00, ch01, ...
```

Minimum required columns:

- `time`
- one numeric signal column, for example `signal_raw`

If each file is one independent record:

```bash
mkdir -p ./data/custom/my_qp_dataset
# Put one or more CSV files in ./data/custom/my_qp_dataset/
```

Prepare generic CSV data:

```bash
mkdir -p ./outputs/custom_my_qp

python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset generic_csv \
  --sources ./data/custom/my_qp_dataset \
  --signal-names signal_raw \
  --sample-rate 1000 \
  --resample-to 200 \
  --input-smooth-sec 0.05 \
  --input-smooth-mode causal \
  --target-smooth-sec 0.05 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output ./outputs/custom_my_qp/custom_my_qp.csv
```

Then use the same profile/split/plan commands:

```bash
python scripts/analyze_quasiperiodic_profile.py \
  --csv ./outputs/custom_my_qp/custom_my_qp.csv \
  --signal-cols target_smooth \
  --time-col time \
  --fs-col fs \
  --segment-col segment_id \
  --split-col split \
  --split-values train \
  --output-dir ./outputs/custom_my_qp/profile

python scripts/split_quasiperiodic_dataset_by_type.py \
  --csv ./outputs/custom_my_qp/custom_my_qp.csv \
  --profile-csv ./outputs/custom_my_qp/profile/profile_by_segment.csv \
  --segment-col segment_id \
  --output-dir ./outputs/custom_my_qp/by_type \
  --drop-unknown

python scripts/build_qp_experiment_plan.py \
  --split-metadata ./outputs/custom_my_qp/by_type/split_by_type_metadata.json \
  --output-dir ./outputs/custom_my_qp/plan \
  --model-id-prefix custom_my_qp \
  --profile-signal-cols target_smooth \
  --input-col input_smooth \
  --output-col target_smooth \
  --raw-col raw \
  --batch-size 32 \
  --learning-rate 1e-4 \
  --train-epochs 40 \
  --gpu 3

bash ./outputs/custom_my_qp/plan/prepare_qp_experiments.sh
bash ./outputs/custom_my_qp/plan/run_all_train_commands.sh
```

For MATLAB/CWRU-like `.mat` files:

```bash
python scripts/prepare_quasiperiodic_wave_dataset.py \
  --dataset mat \
  --sources ./data/custom/my_mat_dataset \
  --sample-rate 12000 \
  --resample-to 1000 \
  --input-smooth-sec 0.01 \
  --input-smooth-mode causal \
  --target-smooth-sec 0.01 \
  --target-smooth-mode centered \
  --split-policy by_record \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --output ./outputs/custom_mat/custom_mat.csv
```

## 8. Running Commands On Multiple GPUs

Generated scripts run sequentially. For early experiments, keep it sequential
so failures are easy to inspect. Later, split type folders across GPUs manually.

Example:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 bash ./outputs/quasi_bidmc_resp_ma2s/plan/run_all_train_commands.sh

# Terminal 2
CUDA_VISIBLE_DEVICES=1 bash ./outputs/quasi_fantasia_resp_ma2s/plan/run_all_train_commands.sh

# Terminal 3
CUDA_VISIBLE_DEVICES=2 bash ./outputs/quasi_mitdb_mlii_ma008s/plan/run_all_train_commands.sh
```

The generated commands also include `--gpu N`. If using
`CUDA_VISIBLE_DEVICES=0`, the visible card is usually `--gpu 0`.

## 9. Manual Single-Model Commands

Use these when debugging one dataset before running the full generated matrix.

### 9.1 QPWave-TCN

Replace `seq_len`, `pred_len`, `stride`, `kernel_size`, and `num_layers` with
values from `recommended_qp_config.json`.

```bash
python run.py \
  --is_training 1 \
  --model tcn_claude \
  --model_id debug_qpwave \
  --root_path ./outputs/quasi_bidmc_resp_ma2s/ \
  --data_path bidmc_resp_ma2s.csv \
  --features MS \
  --input_cols input_smooth \
  --output_cols target_smooth \
  --enc_in 1 \
  --c_out 1 \
  --scaler channel \
  --seq_len 2500 \
  --pred_len 750 \
  --stride 128 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20 \
  --loss huber \
  --kernel_size 5 \
  --num_layers 10 \
  --split_col split \
  --segment_col segment_id \
  --plot_raw_col raw \
  --horizon 4500 \
  --gpu 0
```

### 9.2 CycleResidual-TCN

Use when the profile has a clear `dominant_period_samples`.

```bash
python run.py \
  --is_training 1 \
  --model cycle_residual_tcn \
  --model_id debug_cycle_residual \
  --root_path ./outputs/quasi_bidmc_resp_ma2s/ \
  --data_path bidmc_resp_ma2s.csv \
  --features MS \
  --input_cols input_smooth \
  --output_cols target_smooth \
  --enc_in 1 \
  --c_out 1 \
  --scaler channel \
  --seq_len 2500 \
  --pred_len 750 \
  --stride 128 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20 \
  --loss huber \
  --kernel_size 5 \
  --num_layers 10 \
  --period_len 250 \
  --cycle_base_cycles 3 \
  --cycle_base_mode mean \
  --cycle_backbone_revin 0 \
  --split_col split \
  --segment_col segment_id \
  --plot_raw_col raw \
  --horizon 4500 \
  --gpu 0
```

### 9.3 SmoothPECNet Raw To Smooth

Use for noisy dominant-period data.

```bash
python run.py \
  --is_training 1 \
  --model smooth_pecnet \
  --model_id debug_smoothpec_raw_to_smooth \
  --root_path ./outputs/quasi_bidmc_resp_ma2s/ \
  --data_path bidmc_resp_ma2s.csv \
  --features MS \
  --input_cols raw \
  --output_cols target_smooth \
  --enc_in 1 \
  --c_out 1 \
  --scaler channel \
  --seq_len 2500 \
  --pred_len 750 \
  --stride 128 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20 \
  --loss huber \
  --kernel_size 5 \
  --num_layers 10 \
  --smoothpec_window 50 \
  --smoothpec_mode smooth_raw \
  --residual_output 0 \
  --use_revin 0 \
  --cont_weight 0 \
  --split_col split \
  --segment_col segment_id \
  --plot_raw_col raw \
  --horizon 4500 \
  --gpu 0
```

### 9.4 QPEnhanced-TCN

First create augmented data:

```bash
python scripts/augment_quasiperiodic_dataset.py \
  --csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --profile-csv ./outputs/quasi_bidmc_resp_ma2s/profile/profile_by_segment.csv \
  --output ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s_aug.csv \
  --raw-col raw \
  --time-col time \
  --fs-col fs \
  --segment-col segment_id \
  --split-col split \
  --feature-mode causal
```

Then train enhanced model:

```bash
python run.py \
  --is_training 1 \
  --model qpenhanced_tcn \
  --model_id debug_qpenhanced \
  --root_path ./outputs/quasi_bidmc_resp_ma2s/ \
  --data_path bidmc_resp_ma2s_aug.csv \
  --features MS \
  --input_cols qp_main_input,qp_envelope,qp_local_freq_ratio,qp_phase_sin,qp_phase_cos \
  --output_cols qp_main_target \
  --enc_in 5 \
  --c_out 1 \
  --scaler channel \
  --seq_len 2500 \
  --pred_len 750 \
  --stride 128 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20 \
  --loss qp_hybrid \
  --kernel_size 5 \
  --num_layers 10 \
  --residual_output 1 \
  --qpenhance_gate 1 \
  --qpenhance_gate_hidden 32 \
  --qp_deriv_weight 0.4 \
  --qp_envelope_weight 1.0 \
  --qp_band_weight 0.05 \
  --qp_event_weight 0.0 \
  --qp_corr_weight 0.2 \
  --qp_multiscale_weight 0.3 \
  --qp_peak_weight 0.0 \
  --split_col split \
  --segment_col segment_id \
  --plot_raw_col raw \
  --horizon 4500 \
  --gpu 0
```

For spike-event data, use residual/event-focused inputs:

```bash
--input_cols qp_main_input,qp_residual,qp_abs_residual
--enc_in 3
--qp_event_weight 2.0
--qp_peak_weight 1.0
```

For multi-frequency data, use band RMS inputs:

```bash
--input_cols qp_main_input,qp_band0_rms,qp_band1_rms,qp_band2_rms
--enc_in 4
--qp_band_weight 0.2
```

## 10. Final Metric Summary For Paper Tables

After all runs:

```bash
python scripts/summarize_forecast_metrics.py \
  --root ./checkpoints \
  --output ./outputs/all_forecast_metrics_summary.csv
```

Useful filtered summaries:

```bash
python scripts/summarize_forecast_metrics.py \
  --root ./checkpoints \
  --min-pearson 0.7 \
  --output ./outputs/good_corr_forecast_metrics_summary.csv

python scripts/summarize_forecast_metrics.py \
  --root ./checkpoints \
  --max-mse 0.1 \
  --output ./outputs/low_mse_forecast_metrics_summary.csv
```

Each finished checkpoint contains:

- `rolling_forecast.png`
- `prediction_zoom.png`
- `prediction_scatter.png`
- `rolling_forecast_values.csv`
- `point_metrics.json`
- `run_args.json`

## 11. What To Compare In The Paper

For each signal type, compare:

| Model | Purpose |
|---|---|
| DLinear | decomposition-linear strong baseline |
| PatchTST | patch Transformer strong baseline |
| GRU/CNNLSTM/CRNN/InceptionTime/FastTCN/SpectralCNN/TimeMixer | optional restored legacy baselines for wider comparison |
| QPWave-TCN | current neural main-waveform backbone |
| CycleResidual-TCN | tests whether explicit cycle prior helps |
| SmoothPECNet | tests raw-to-main decomposition for noisy signals |
| QPEnhanced-TCN | tests causal feature conditioning and structure-aware loss |

For each dataset/type, report:

- MSE / MAE
- Pearson correlation
- dominant-period relative error
- spectral-energy L1
- envelope relative MAE
- peak timing MAE and peak hit rate for spike-event data

Expected claims:

- Stable single-frequency: cycle-adaptive windows and cycle residual should
  help.
- Noisy single-frequency: smooth/main waveform target should beat raw target.
- AM/FM modulation: envelope/frequency conditioning should improve structure
  tracking.
- Spike-event: event/peak-aware loss should improve peak metrics even if raw
  waveform MSE is not perfect.
- Multi-frequency: band conditioning should beat a single-cycle assumption.
- Weak-periodic: long waveform prediction should fail or become low-confidence;
  this supports target switching/rejection rather than overclaiming.

## 12. Failure Checks

If prediction becomes a straight line:

1. Check profile: high spectral entropy or weak autocorrelation may mean the
   data is not suitable for long point forecasting.
2. Check `seq_len/pred_len`: use period-scaled recommendation, not fixed point
   counts.
3. Check target: raw spikes/noise should not be the first long-horizon target.
4. Check model type: try `cycle_residual_tcn` for stable periods,
   `smooth_pecnet` for noisy raw data, and `qpenhanced_tcn` for modulated or
   spike-like data.
5. Check plots in `prediction_zoom.png`; overview plots can look dense and hide
   local behavior.

If a command errors with shape mismatch:

1. Verify `--input_cols` count equals `--enc_in`.
2. Verify `--output_cols` count equals `--c_out`.
3. For `qpenhanced_tcn`, keep `qp_main_input` as the first input column.
4. For `cycle_residual_tcn`, use same-quantity input/output and set
   `--period_len`.
