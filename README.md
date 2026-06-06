# Quasi-periodic Main Waveform Forecasting

This repository is now focused on feature-aware forecasting for quasi-periodic
time series. The main task is long-horizon smooth/main-waveform forecasting, not
raw spike-perfect reconstruction.

## Main Workflow

1. Prepare public or domain data into long-format CSV files.
2. Profile the signal type with `scripts/analyze_quasiperiodic_profile.py`.
3. Generate cycle-adaptive `seq_len/pred_len/smooth_window` settings with
   `scripts/recommend_qp_config.py`.
4. Train forecasting baselines and the main TCN model with `run.py`.
5. Compare predictions with point metrics and rolling forecast plots.
6. Summarize experiment results into paper-ready metric tables.

## Active Code Surface

- Training/evaluation: `run.py`, `exp/`, `data_provider/`, `models/`, `utils/`.
- Public quasi-periodic data: `scripts/prepare_quasiperiodic_wave_dataset.py`.
- Combustion pressure waveform data: `scripts/prepare_pressure_channel_wave_dataset.py`.
- Signal profiling: `scripts/analyze_quasiperiodic_profile.py`.
- Cycle-adaptive experiment recommendation: `scripts/recommend_qp_config.py`.
- Feature-aware augmentation: `scripts/augment_quasiperiodic_dataset.py`.
- Experiment metric summary: `scripts/summarize_forecast_metrics.py`.
- Optional analysis/plotting helpers remain under `scripts/`, but old risk-label
  and broken `src.data`-based scripts have been removed from the active branch.

## Core Signal Types

- `stable_single_freq`: clean dominant period, low noise.
- `noisy_single_freq`: dominant period with strong residual/noise.
- `am_fm_modulated`: slowly changing amplitude or frequency.
- `spike_event`: quasi-periodic sharp events where peak timing matters.

The profile script also reports `multi_freq` and `weak_periodic` as boundary
cases for later experiments.

## Installation

```bash
conda create -n timesnet python=3.10 -y
conda activate timesnet
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Public Data Download

```bash
mkdir -p ./data/extracted/bidmc ./data/extracted/fantasia ./data/extracted/mitdb

python - <<'PY'
import wfdb
for name in ["bidmc", "fantasia", "mitdb"]:
    path = f"./data/extracted/{name}"
    print(f"Downloading {name} -> {path}")
    wfdb.dl_database(name, dl_dir=path)
PY
```

## Prepare Example

```bash
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

## Profile Example

```bash
python scripts/analyze_quasiperiodic_profile.py \
  --csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --signal-cols target_smooth \
  --time-col time \
  --segment-col segment_id \
  --split-col split \
  --split-values train \
  --output-dir ./outputs/profile_bidmc_resp
```

The profile outputs are intended to support the paper's "predictability
portrait" table:

- `profile_by_segment.csv`: one row per segment/signal with dominant period,
  spectral entropy, autocorrelation peak, residual energy ratio, spike
  prominence, predictability score, signal type, and recommended module.
- `profile_summary.csv`: median feature summary by signal type.
- `profile_report.md`: compact human-readable report.

## Cycle-Adaptive Config Example

```bash
python scripts/recommend_qp_config.py \
  --profile-csv ./outputs/profile_bidmc_resp/profile_by_segment.csv \
  --prepared-csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --output-dir ./outputs/profile_bidmc_resp/recommend \
  --model-id-prefix bidmc_resp \
  --gpu 1
```

## Feature-Aware Augmentation Example

Use this before the improved modules for noisy, modulated, spike-event, or
multi-frequency signals:

```bash
python scripts/augment_quasiperiodic_dataset.py \
  --csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --profile-csv ./outputs/profile_bidmc_resp/profile_by_segment.csv \
  --output ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s_aug.csv \
  --raw-col raw \
  --segment-col segment_id \
  --split-col split \
  --feature-mode causal
```

The augmented CSV adds reusable `qp_*` columns:

- `qp_main_input`, `qp_main_target`, `qp_residual`: main-wave/residual
  decomposition for noisy dominant-period signals.
- `qp_envelope`, `qp_local_freq_ratio`, `qp_phase_sin`, `qp_phase_cos`:
  conditioning features for amplitude/frequency modulation.
- `qp_event_mask`, `qp_event_prominence`, `qp_event_proximity`,
  `qp_event_weight`: offline event skeleton labels/features for analysis and
  event-focused ablations. They are not used as default forecasting inputs
  because peak detection can use future samples.
- `qp_band0_rms`, `qp_band1_rms`, `qp_band2_rms`: relative band energy
  features for multi-frequency or mode-switching signals.
- `qp_predictability_score`, `qp_weak_periodic_flag`: rejection/target-switch
  indicators for weak-periodic boundary cases.

`--feature-mode causal` is the default and avoids future leakage in model input
features. Use `--feature-mode offline` only for diagnostic analysis or an
explicit offline ablation, because Hilbert/zero-phase/event features can inspect
future samples within the split.
When `--profile-csv` is provided, segments missing from the profile use the
profile-level median period/type instead of estimating from the full val/test
chunk.

Then generate commands including the enhanced model:

```bash
python scripts/recommend_qp_config.py \
  --profile-csv ./outputs/profile_bidmc_resp/profile_by_segment.csv \
  --prepared-csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv \
  --enhanced-csv ./outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s_aug.csv \
  --output-dir ./outputs/profile_bidmc_resp/recommend \
  --model-id-prefix bidmc_resp \
  --gpu 1
```

This writes:

- `recommended_qp_config.json`: dataset-level and type-level recommended
  periods, `seq_len`, `pred_len`, stride, smoothing window, and module.
- `recommended_train_commands.sh`: runnable commands for QPWave-TCN, DLinear,
  PatchTST, SmoothPECNet, and `qpenhanced_tcn` when an enhanced CSV is provided.

Default policy is roughly `10` past cycles as input and `3-4` future cycles as
output. Weak-periodic boundary cases are intentionally shortened because they
are not good long-horizon point-forecast targets.

## Train Example

```bash
python run.py \
  --is_training 1 \
  --model tcn_claude \
  --model_id bidmc_resp_qpwave \
  --root_path ./outputs/quasi_bidmc_resp_ma2s/ \
  --data_path bidmc_resp_ma2s.csv \
  --features MS \
  --input_cols input_smooth \
  --output_cols target_smooth \
  --enc_in 1 \
  --out_in 1 \
  --c_out 1 \
  --scaler channel \
  --seq_len 1000 \
  --pred_len 300 \
  --stride 64 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 40 \
  --loss huber \
  --kernel_size 3 \
  --num_layers 9 \
  --d_model 128 \
  --d_ff 256 \
  --dropout 0.1 \
  --e_layers 3 \
  --horizon 3000 \
  --split_col split \
  --segment_col segment_id \
  --plot_raw_col raw \
  --gpu 1
```

## Correctness Notes

- `tcn_claude`/QPWave-TCN is safest for same-quantity main-wave forecasting,
  for example `input_smooth -> target_smooth` or `p_input_ma1024 -> p_target_cma1024`.
- For `raw -> smooth`, avoid forcing the model to continue from the raw last
  value. Use `--residual_output 0 --use_revin 0`, or use `smooth_pecnet`
  with `--smoothpec_mode smooth_raw` so the smooth branch is first.
- For feature-aware runs, use `--model qpenhanced_tcn` with enhanced `qp_*`
  input columns and `--loss qp_hybrid`. This keeps the same QPWave-TCN backbone
  but adds channel gating and event/envelope/frequency-aware loss terms.
- `hybrid` loss keeps the FFT term magnitude-only by default. If input and
  target are not the same waveform quantity, set `--cont_weight 0`.
- `DLinear` and `PatchTST` are baselines. For clean comparisons, run them with
  single-input/single-output or one-to-one input/output columns.
- Training now rejects prediction/target shape mismatches instead of allowing
  PyTorch broadcasting. If a run errors there, the old metric was not reliable.

## Result Summary

Each test run saves:

- `rolling_forecast.png`, `prediction_scatter.png`, `prediction_zoom.png`
- `rolling_forecast_values.csv`: raw-scale and normalized prediction/target
  sequences for later plotting or re-analysis.
- `point_metrics.json`: point metrics plus quasi-periodic structure metrics.

Structure metrics include dominant-period error, spectral-energy L1 distance,
envelope relative MAE, peak count error, peak timing MAE, and peak hit rate.
These are the metrics to use when arguing that a model preserves quasi-periodic
structure instead of only lowering MSE.

To collect all finished checkpoint results:

```bash
python scripts/summarize_forecast_metrics.py \
  --root ./checkpoints \
  --output ./outputs/forecast_metrics_summary.csv
```
