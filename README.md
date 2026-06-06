# Quasi-periodic Main Waveform Forecasting

This repository is now focused on feature-aware forecasting for quasi-periodic
time series. The main task is long-horizon smooth/main-waveform forecasting, not
raw spike-perfect reconstruction.

## Main Workflow

1. Prepare public or domain data into long-format CSV files.
2. Profile the signal type with `scripts/analyze_quasiperiodic_profile.py`.
3. Train forecasting baselines and the main TCN model with `run.py`.
4. Compare predictions with point metrics and rolling forecast plots.

## Active Code Surface

- Training/evaluation: `run.py`, `exp/`, `data_provider/`, `models/`, `utils/`.
- Public quasi-periodic data: `scripts/prepare_quasiperiodic_wave_dataset.py`.
- Combustion pressure waveform data: `scripts/prepare_pressure_channel_wave_dataset.py`.
- Signal profiling: `scripts/analyze_quasiperiodic_profile.py`.
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
- `hybrid` loss keeps the FFT term magnitude-only by default. If input and
  target are not the same waveform quantity, set `--cont_weight 0`.
- `DLinear` and `PatchTST` are baselines. For clean comparisons, run them with
  single-input/single-output or one-to-one input/output columns.
- Training now rejects prediction/target shape mismatches instead of allowing
  PyTorch broadcasting. If a run errors there, the old metric was not reliable.
