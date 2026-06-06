# Literature-To-Module Map

This repository should not claim a universal raw waveform predictor. The safer
paper claim is feature-aware main-waveform forecasting for quasi-periodic time
series. The implemented modules map to the literature as follows.

| Signal issue | Literature idea | Repository module | Safe claim |
|---|---|---|---|
| Stable seasonality / dominant period | Seasonal-trend decomposition and periodic dependency modeling, as in Autoformer and TimesNet | `cycle_residual_tcn`, cycle-adaptive `seq_len/pred_len` | Period-scaled windows and cycle-template residual prediction improve stable quasi-periodic forecasting |
| Noisy dominant-period signals | Decomposition-first forecasting, as in Autoformer/FEDformer/DLinear | `smooth_pecnet`, `qp_main_input`, `qp_residual` | Predicting the main waveform is more reliable than forcing raw residual reconstruction |
| AM/FM modulation | Conditioned forecasting with local structure features | `qp_envelope`, `qp_local_freq_ratio`, `qp_phase_sin`, `qp_phase_cos`, `qpenhanced_tcn` | Envelope/frequency conditioning helps non-stationary quasi-periodic segments |
| Spike-like events | Shape/time-aware losses, related to DILATE-style temporal distortion concerns | `QPHybridLoss` event salience and peak-pool terms, peak metrics | Event-aware loss/metrics reduce peak collapse, without claiming perfect raw spike reconstruction |
| Multi-frequency or mode competition | Frequency-domain enhancement and decomposition, as in FEDformer | `qp_band*_rms`, log-spectrum loss | Band-energy conditioning is a lightweight frequency-aware alternative to a full expert model |
| Distribution shift across records | Instance normalization, as in RevIN | `tcn_claude` RevIN and channel scaler | Per-record normalization reduces amplitude/offset drift for same-quantity forecasting |
| Strong baselines | Linear and patch baselines, as in DLinear and PatchTST | `DLinear`, `PatchTST` | Proposed modules must beat simple decomposition/patch baselines, not only weak neural baselines |

Recommended first-paper framing:

1. Use `analyze_quasiperiodic_profile.py` to classify data difficulty.
2. Use cycle-scaled windows instead of fixed point counts.
3. Compare `QPWave-TCN`, `CycleResidual-TCN`, `SmoothPECNet`, and
   `QPEnhanced-TCN` against `DLinear` and `PatchTST`.
4. Report structure metrics, not just MSE.

References to cite:

- Autoformer: decomposition architecture and autocorrelation for long-term time
  series forecasting. https://arxiv.org/abs/2106.13008
- FEDformer: frequency-enhanced decomposed transformer. https://arxiv.org/abs/2201.12740
- TimesNet: temporal 2D-variation modeling for multiple periodicities. https://arxiv.org/abs/2210.02186
- DLinear: simple decomposition-linear baseline. https://arxiv.org/abs/2205.13504
- PatchTST: patch-based transformer baseline. https://arxiv.org/abs/2211.14730
- RevIN: reversible instance normalization for distribution shift. https://openreview.net/forum?id=cGDAkQo1C0p
- DILATE: shape and temporal distortion loss for time series forecasting. https://arxiv.org/abs/1909.09020
- N-BEATS: interpretable trend/seasonality basis expansion. https://arxiv.org/abs/1905.10437
