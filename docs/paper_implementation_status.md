# Paper Implementation Status

This note tracks what is implemented strongly enough for the first paper and
what should be described as future/second-stage work. It is intentionally
conservative.

## Ready For First-Stage Experiments

| Paper Component | Code Surface | Status |
|---|---|---|
| Long-format quasi-periodic data interface | `scripts/prepare_quasiperiodic_wave_dataset.py`, `scripts/prepare_pressure_channel_wave_dataset.py` | Ready |
| Controlled synthetic signal types | `scripts/generate_synthetic_quasiperiodic_dataset.py` | Ready for validation/ablation |
| Predictability portrait | `scripts/analyze_quasiperiodic_profile.py` | Ready, heuristic thresholds should be reported |
| Cycle-adaptive window recommendation | `scripts/recommend_qp_config.py` | Ready |
| Type-specific dataset splitting | `scripts/split_quasiperiodic_dataset_by_type.py` | Ready |
| Main waveform / residual augmentation | `scripts/augment_quasiperiodic_dataset.py`, `utils/qperiod_enhance.py` | Ready |
| Causal envelope/frequency/band features | `scripts/augment_quasiperiodic_dataset.py`, `utils/qperiod_enhance.py` | Ready; offline mode is diagnostic only |
| Baselines | `DLinear`, `PatchTST`, `tcn_claude`, `smooth_pecnet` | Ready |
| Feature-aware TCN wrapper | `models/qpenhanced_tcn.py` | Ready as a lightweight feature-gated TCN |
| Structure-aware loss | `utils/losses.py::QPHybridLoss` | Ready as differentiable waveform/envelope/spectrum/event-salience loss |
| Structure metrics and result table | `utils/tools.py`, `scripts/summarize_forecast_metrics.py` | Ready |

## Should Be Described Carefully

| Claimed Module | Current Implementation | Safe Wording |
|---|---|---|
| Event skeleton constraint | Target-driven event salience in loss plus peak metrics; no explicit future peak head | "event-aware weighting/metrics" |
| Frequency-band gated branch | Causal band RMS features plus spectral loss; no separate gated multi-branch predictor | "frequency-band conditioning" |
| Predictability rejection | Profile score and weak-periodic flag; no trained rejection classifier/calibration | "predictability portrait and target-switch indicator" |
| AM/FM conditioning | Envelope, local frequency ratio, cycle phase as input features; no dedicated condition branch | "feature-conditioned forecasting" |
| Cross-dataset transfer | Runner supports pretrained loading/fine-tuning, but no automated transfer protocol script | "supported, experimental" |

## Recommended First Paper Scope

Use the following claims:

1. A feature-aware main-waveform forecasting protocol for quasi-periodic time
   series.
2. A predictability portrait that classifies signal segments and recommends
   period-scaled windows.
3. Causal main/residual, envelope/frequency, and band-energy features.
4. A lightweight feature-gated TCN plus structure-aware loss.
5. Evaluation with point metrics and quasi-periodic structure metrics.

Avoid claiming:

1. A complete event sequence predictor.
2. A full mixture-of-experts frequency-gated network.
3. A calibrated rejector that guarantees forecast failure detection.
4. A single universal pretrained model that zero-shot predicts all domains.

## Minimum Experiment Matrix

For each signal type or dataset subset:

1. DLinear baseline.
2. PatchTST baseline.
3. QPWave-TCN smooth-to-smooth.
4. SmoothPECNet raw-to-smooth for noisy/modulated/spike data.
5. QPEnhanced-TCN with causal `qp_*` features.

Report:

1. MSE / MAE / Pearson.
2. Dominant-period error.
3. Spectral-energy L1 distance.
4. Envelope relative MAE.
5. Peak count/time metrics for spike-like signals.

## Practical Pipeline

```bash
python scripts/analyze_quasiperiodic_profile.py ...
python scripts/split_quasiperiodic_dataset_by_type.py ...
python scripts/augment_quasiperiodic_dataset.py ...
python scripts/recommend_qp_config.py ...
bash recommended_train_commands.sh
python scripts/summarize_forecast_metrics.py ...
```
