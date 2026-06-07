# Codex Server Handoff

本文档给服务器上的 Codex 使用，用来快速理解当前科研任务、代码状态、数据路径和下一步实验。不要把它当成完整论文方案；完整实验流程看 `docs/experiment_runbook.md`，实现状态看 `docs/paper_implementation_status.md`。

## 当前研究主线

原始目标是燃烧不稳定压力/释热率波形预测，但燃烧数据少、采样率高、尖峰和细节非常难长步预测。现在论文主线调整为：

**面向多类型准周期信号的可预测性画像与特征感知主波形预测方法。**

核心思想：

1. 不强行承诺所有 raw 尖峰波形都能长步预测。
2. 先分析信号特征，再选择预测目标、窗口长度、特征增强模块和损失。
3. 容易数据先验证框架，复杂数据再作为模块改进和失败边界。
4. 训练输入优先使用 `outputs/` 里的标准长表 CSV，而不是直接读 `data/` 里的原始文件。

## 统一服务器路径

两台服务器尽量保持同一套路径：

```bash
cd /data/users/yihang/ts-forecast
conda activate timesnet
```

目录约定：

```text
/data/users/yihang/ts-forecast          GitHub 最新代码
/data/users/yihang/ts-forecast/data     原始数据或上游清洗源数据
/data/users/yihang/ts-forecast/outputs  本项目导出的训练 CSV、profile、plan
/data/users/yihang/ts-forecast/checkpoints  训练权重和预测结果
```

如果 `data/outputs/checkpoints` 是软链接到 `/data/users/yihang/ts-forecast-assets/`，这是正常且推荐的。

检查代码版本：

```bash
git remote -v
git log --oneline -5
ls scripts | grep prepare_public
```

正常应能看到 `scripts/prepare_public_quasiperiodic_data.py`。这个脚本来自提交：

```text
3cb0e43 Add public dataset cleaning pipeline
```

## 环境要求

必须用 `timesnet` 环境，不要误用 `efficientad-ad`。

检查：

```bash
python - <<'PY'
import torch, numpy, pandas, scipy, wfdb
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
print("wfdb:", wfdb.__version__)
PY
```

如果处理 Fantasia 或 MITDB 报：

```text
ModuleNotFoundError: No module named 'wfdb'
```

说明环境缺 `wfdb`：

```bash
pip install wfdb
```

## 当前已准备的数据

公开准周期数据已经处理到 `outputs/` 下，训练应读这些文件：

```text
outputs/quasi_bidmc_resp_ma2s/bidmc_resp_ma2s.csv
outputs/quasi_fantasia_resp_ma2s/fantasia_resp_ma2s.csv
outputs/quasi_mitdb_mlii_ma008s/mitdb_mlii_ma008s.csv
```

每个 CSV 都包含：

```text
time, raw, input_smooth, target_smooth, split, segment_id, record_id, signal_name, dataset, fs
```

含义：

- `raw`: 清洗后的原始信号。
- `input_smooth`: 因果平滑输入，不看未来。
- `target_smooth`: 中心平滑目标，作为离线训练标签。
- `split`: train/val/test，公开数据当前用 by-record 划分。

当前平滑窗口：

```text
BIDMC RESP:     2.0 s
Fantasia RESP:  2.0 s
MITDB MLII:     0.08 s
```

质量检查结果已经确认：

```text
BIDMC:    53 records, all ok
Fantasia: 40 records, all ok
MITDB:    69 MLII records, all ok
```

共同特点：

- `cropped_finite_ratio = 1.0`
- `repaired_nonfinite_rows = 0`
- 没有坏记录被跳过

注意：MITDB 是 ECG 尖峰事件型数据，`robust_clip_z=12` 版本可能裁掉部分极高 R 峰。当前版本可以用于平滑主波形实验；如果要专门做尖峰事件实验，建议额外导出 no-clip 对照版本。

## 原始数据位置

公开数据压缩包放在：

```text
data/bidmc-ppg-and-respiration-dataset-1.0.0.zip
data/fantasia-database-1.0.0.zip
data/mit-bih-arrhythmia-database-1.0.0.zip
```

燃烧压力/释热率源数据应放在：

```text
data/pressure_qdot_csv_final_1us/
```

注意：`data/` 里的燃烧数据不是最终训练输入。需要再用项目脚本导出到 `outputs/combustion_*/*.csv` 后训练。

## 数据准备命令

如果服务器只有压缩包，还没有 `outputs/quasi_*`，运行：

```bash
python scripts/prepare_public_quasiperiodic_data.py \
  --data-root ./data \
  --extract-root ./data/extracted \
  --output-root ./outputs \
  --bad-record-policy error \
  --min-finite-ratio 0.995 \
  --robust-clip-z 12
```

如果已经有 BIDMC，只需要继续 Fantasia 和 MITDB：

```bash
python scripts/prepare_public_quasiperiodic_data.py \
  --datasets fantasia,mitdb \
  --fantasia-source ./data/fantasia-database-1.0.0.zip \
  --mitdb-source ./data/mit-bih-arrhythmia-database-1.0.0.zip \
  --data-root ./data \
  --extract-root ./data/extracted \
  --output-root ./outputs \
  --bad-record-policy error \
  --min-finite-ratio 0.995 \
  --robust-clip-z 12
```

清洗质量检查：

```bash
for f in outputs/quasi_*/*.quality.csv; do
  echo "==== $f ===="
  python - <<PY
import pandas as pd
f="$f"
df=pd.read_csv(f)
print(df["status"].value_counts(dropna=False))
print(df[["record_id","signal_name","cropped_finite_ratio","repaired_nonfinite_rows","clipped_low_rows","clipped_high_rows","clipped_std"]].head())
PY
done
```

## 需要先看的报告

正式训练前先看 profile：

```bash
cat outputs/quasi_bidmc_resp_ma2s/profile/profile_report.md
cat outputs/quasi_fantasia_resp_ma2s/profile/profile_report.md
cat outputs/quasi_mitdb_mlii_ma008s/profile/profile_report.md
```

profile 用来确定主周期、信号类型、推荐窗口和预测难度。原则是：

- 输入约过去 `10` 个周期。
- 输出约未来 `3-4` 个周期。
- 弱周期或高噪声数据不要硬做长步 raw 波形预测。

## 当前第一轮实验顺序

建议顺序：

1. `BIDMC RESP`: 最容易，先验证训练和画图流程。
2. `Fantasia RESP`: 呼吸但跨个体、调制更强，验证 AM/FM 条件特征。
3. `MITDB MLII`: ECG 尖峰事件型，验证纯 MSE 长步尖峰预测的困难，以及 event-aware/structure metrics。
4. 燃烧压力/释热率：作为高难度边界或领域案例，不要先拿它证明框架有效。

第一轮不要直接全量扫所有模型。先每个数据集跑一个小实验确认 loss 下降、预测图能生成，再跑完整矩阵。

## 训练输入列选择

主线第一阶段默认：

```text
input_smooth -> target_smooth
```

训练参数里使用：

```bash
--input_cols input_smooth
--output_cols target_smooth
--target target_smooth
```

后续对照可以做：

```text
raw -> raw
raw -> target_smooth
input_smooth -> target_smooth
```

预期：raw 长步预测容易变平滑或均值，尤其是 MITDB 和燃烧压力尖峰型数据。

## 模型和模块现状

核心模型/基线：

- `DLinear`
- `PatchTST`
- `tcn_claude`
- `cycle_residual_tcn`
- `smooth_pecnet`
- `qpenhanced_tcn`

可选旧基线：

- `GRU`
- `CNNLSTM`
- `CRNN`
- `InceptionTime`
- `FastTCN`
- `SpectralCNN`
- `TimeMixer`

重要脚本：

```text
scripts/prepare_quasiperiodic_wave_dataset.py
scripts/prepare_public_quasiperiodic_data.py
scripts/analyze_quasiperiodic_profile.py
scripts/split_quasiperiodic_dataset_by_type.py
scripts/augment_quasiperiodic_dataset.py
scripts/recommend_qp_config.py
scripts/build_qp_experiment_plan.py
scripts/summarize_forecast_metrics.py
```

实现边界要说清楚：

- Event skeleton 目前更准确地说是 event-aware loss/metrics，不是完整的峰事件序列预测器。
- Frequency module 目前是频带特征和频谱损失，不是完整 MoE 频带门控网络。
- Predictability rejection 目前是画像分数和目标切换建议，不是训练好的拒判分类器。

这些措辞已经写在 `docs/paper_implementation_status.md`。

## 先跑一个 BIDMC smoke 训练

下面只是示例，真正 `seq_len/pred_len` 应按 profile 主周期调整：

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id smoke_bidmc_tcn \
  --model tcn_claude \
  --root_path ./outputs/quasi_bidmc_resp_ma2s \
  --data_path bidmc_resp_ma2s.csv \
  --features MS \
  --target target_smooth \
  --input_cols input_smooth \
  --output_cols target_smooth \
  --seq_len 250 \
  --pred_len 100 \
  --enc_in 1 \
  --c_out 1 \
  --out_in 1 \
  --scaler channel \
  --train_epochs 3 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --split_col split \
  --segment_col segment_id \
  --plot_raw_col raw \
  --horizon 1000 \
  --itr 1 \
  --gpu 0
```

如果用户终端里 `nvidia-smi` 正常，但 Codex 默认命令里看不到 GPU，
优先判断是 Codex 命令沙箱没有暴露 `/dev/nvidia*`。正式训练要从普通
shell 或非沙箱执行上下文启动；`timesnet` 环境里的 PyTorch CUDA 版本本身
通常不是问题。

如果 smoke 失败，优先检查：

1. `conda activate timesnet`
2. `pip install -r requirements.txt`
3. `python scripts/smoke_forecast_models.py`
4. `find ./outputs/quasi_bidmc_resp_ma2s -maxdepth 1 -type f | sort`

## 不要做的事

1. 不要把 `data/` 里的压缩包或原始 CSV 直接当训练输入。
2. 不要在 `efficientad-ad` 环境里跑本项目。
3. 不要因为 raw 尖峰预测不好就判断整个方向失败；主线是 feature-aware main waveform forecasting。
4. 不要把服务器旧本地 Git 仓库当 GitHub。remote 应该指向 `git@github.com:yhcycdl/ts-forecast.git`。
5. 不要静默删除坏记录。最终实验如果用了 `--bad-record-policy skip`，必须报告哪些记录被跳过。

## 下一步建议

1. 两台服务器都确认代码和 `outputs/quasi_*` 路径一致。
2. 分别读取三个 profile，记录主周期和信号类型。
3. 每个数据集先跑一个 3 epoch smoke。
4. 再用 `build_qp_experiment_plan.py` 生成标准实验矩阵。
5. 训练后用 `scripts/summarize_forecast_metrics.py` 汇总指标。
6. 先比较 `DLinear / PatchTST / tcn_claude / cycle_residual_tcn / smooth_pecnet / qpenhanced_tcn`，旧基线等需要大表时再加。
