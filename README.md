# Channel-Adaptive Semantic-Drift Controlled Diffusion-JSCC

## 项目状态：方法开发已终止并冻结（2026-08-03）

本仓库曾研究在不同信道条件下用 diffusion 增强 DeepJSCC 图像恢复，并把 semantic drift 作为核心评估目标。基于现有证据，**原始完整联合优势主张未建立，项目停止继续投入**；停止类型为 `ENGINEERING_STOP`，不等同于对所有相关科学方向的普遍反证。

最终解释层为 [`reports/METHOD_TERMINATION_REPORT_2026-08-03.md`](reports/METHOD_TERMINATION_REPORT_2026-08-03.md)，逐 claim 证据与允许/禁止措辞见 [`audit/CLAIM_REGISTRY.csv`](audit/CLAIM_REGISTRY.csv)。本文件下方保留历史实验、环境和复现命令；其中“当前主线”“下一步”“待授权”“继续训练”等旧时态均已被终止报告 supersede，只能用于理解历史，**不构成新的运行授权**。禁止启动 S35R-P1、新 diffusion refiner、semantic gate/controller/fusion 搜索、CVaR 模型训练、Swin extension、A2/S36 或大量旧实验复跑；既有正式输出不得覆盖。

基础设施仍可复用：冻结 checkpoint、canonical noise、信道与 exact-rate 合同、rate/side-information ledger、外部 baseline adapter、指标与 bootstrap、系统代价测量、manifest 和审计注册表均予保留。复用必须作为独立新任务重新定义 scope、预算、预注册和 ID，不能表述为本方法项目自然继续。

## 文档

- `reports/METHOD_TERMINATION_REPORT_2026-08-03.md`：最终终止决策、H1–H5 分项结论、禁推论和可复用资产。
- `audit/CLAIM_REGISTRY.csv`：逐 claim verdict、证据等级、有效范围及允许/禁止措辞。
- `PROJECT.md`：项目定义、核心问题、假设和方法边界。
- `METHOD_CURRENT.md`：S33 历史技术快照及 2026-08-03 冻结入口。
- `MILESTONES.md`：历史里程碑、阶段门槛及顶部终止覆盖说明。
- `AGENTS.md`：AI agent 和贡献者的协作规则。
- `PROGRESS.md`：当前冻结状态和历史进度。
- `EXPERIMENTS.md`：实验记录、结果索引及终止记录。
- `LITERATURE.md`：相关工作、撞车风险和检索关键词。
- `README.md`：冻结状态、环境安装、历史运行命令和代码结构。

## CVaR 候选方向二：条件信道尾部风险诊断（2026-07-31 完成，判定 NO-GO）

只读诊断，回答"均值训练模型对同一图像重复采样信道时，最差 10% 是否明显差于中位数"。不训练、不下载、不访问 official validation。完整结论见 `reports/cvar_p0_tail_risk_result_2026-07-31.md`，预注册见 `reports/cvar_p0_tail_risk_preregistration_2026-07-31.md`。

结论摘要：**AWGN 下不存在条件尾部风险**（`median−p10 ≤ 0.11 dB`，信道方差占比 `≤0.001`）；Rayleigh block fading 下尾部很大（1 dB 处 `median−p10 = 10.06 dB`）但主因是"纯 AWGN 训练模型 + 从未见过的衰落"与有效 SNR 跌出条件嵌入训练范围，而非均值目标掩盖风险。判定 `NO-GO`，未启动 CVaR 训练。

代码：`src/cadsd_jscc/tail_risk.py`（经验 CVaR + block-fading Rayleigh + ZF 均衡）、三个 `scripts/cvar_p0_*.py`、`configs/cvar_p0_tail_risk_diagnostic.yaml`、`tests/test_tail_risk.py`。既有信道/训练/评测代码零改动。

```bash
# 环境检查
python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"

# 单元测试（新增 18 项 / 全仓 140 项）
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_tail_risk.py' -v
PYTHONPATH=src python3 -m unittest discover -s tests

# dry-run：4 图 × 4 realization × 2 SNR × 4 arm = 128 行
python3 scripts/cvar_p0_diagnose_tail_risk.py --dry-run

# 正式诊断：200 图 × 64 realization × 5 SNR × 4 arm = 256,000 行，约 10 分钟（RTX 4090 D）
python3 scripts/cvar_p0_diagnose_tail_risk.py

# 尾部统计、7 张图与 GO/NO-GO 判定
python3 scripts/cvar_p0_analyze_tail_risk.py

# 40 组最差重建案例（含重放校验）
python3 scripts/cvar_p0_export_worst_cases.py
```

输出目录 `outputs/analysis/ANALYSIS-CVAR-P0-TAIL-RISK-001/`：`diagnostic_samples.csv`（逐样本）、`diagnostic_summary.csv`（逐 arm×SNR）、`per_image_tail_stats.csv`（逐图条件尾部）、`variance_decomposition.csv`、`verdict.json`、`plots/`、`worst_examples/`。默认 `overwrite_forbidden: true`，重跑前需先移走旧目录。

## CVaR 候选方向二（2026-07-31 正式结束，判定 END-CVAR）

两阶段完成后方向终结。**P0 诊断判定 `NO-GO`**，**P1 matched mean-training 归因闭环判定 `END-CVAR`**，未训练任何 CVaR 模型。报告：[P0 预注册](reports/cvar_p0_tail_risk_preregistration_2026-07-31.md) / [P0 结果](reports/cvar_p0_tail_risk_result_2026-07-31.md) / [P1 预注册](reports/cvar_p1_rayleigh_matched_preregistration_2026-07-31.md) / [P1 结果](reports/cvar_p1_rayleigh_matched_result_2026-07-31.md)。

结论链：**AWGN 下不存在条件尾部风险**（`median−p10 ≤ 0.11 dB`，信道方差占比 `≤0.001`）→ Rayleigh 下尾部很大但 S33B 存在两重分布外（没见过衰落、有效 SNR 跌出条件嵌入范围）→ **匹配均值训练后平均 `+3.8~5.6 dB`、尾部差最多缩小 `3.80 dB`、outage 降 `2.4~16×`，且残余尾部的信道方差占比降到 `0.55/0.50/0.44/0.29`，图像内容难度已追平信道随机性** → 逐图 CVaR 无从发力，方向结束。两个独立 seed 一致。

副产物 `EXP-CVAR-P1-RAYLEIGH-MATCHED-MEAN-001`（聚合 `28.47 dB`，SHA `4a520284…`）是合格的 Rayleigh block-fading 基线，即任务书要求的 `Repeated-fading mean control`；但 Rayleigh 按 `MILESTONES.md` 仍属 AWGN 最小闭环之后的扩展项，**不自动进入主线**，未做语义评估。

代码：`src/cadsd_jscc/tail_risk.py`（经验 CVaR + block-fading Rayleigh + ZF 均衡，支持逐样本 SNR）、`scripts/cvar_p0_*.py`、`scripts/cvar_p1_*.py`、`configs/cvar_p0_*.yaml`、`configs/cvar_p1_*.yaml`、`tests/test_tail_risk.py`。既有信道/训练/评测代码零改动。

```bash
# 环境检查
python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"

# 单元测试（tail_risk 20 项 / 全仓 142 项）
PYTHONPATH=src python3 -m unittest discover -s tests

# --- P0：冻结 S33B 上的条件尾部诊断 ---
python3 scripts/cvar_p0_diagnose_tail_risk.py --dry-run        # 128 行
python3 scripts/cvar_p0_diagnose_tail_risk.py                  # 256,000 行，约 10 分钟
python3 scripts/cvar_p0_analyze_tail_risk.py                   # 统计 + 7 图 + verdict
python3 scripts/cvar_p0_export_worst_cases.py                  # 40 组最差案例

# --- P1：Rayleigh matched mean-training 归因闭环 ---
python3 scripts/cvar_p1_train_rayleigh_matched.py --dry-run    # FakeData smoke
python3 scripts/cvar_p1_train_rayleigh_matched.py              # 6 epoch，约 92 分钟
python3 scripts/cvar_p0_diagnose_tail_risk.py --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml
python3 scripts/cvar_p0_analyze_tail_risk.py  --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml
python3 scripts/cvar_p1_attribution_verdict.py                 # INCONCLUSIVE / END-CVAR / ENTER-CVAR

# 独立 seed 复现
python3 scripts/cvar_p0_diagnose_tail_risk.py --config configs/cvar_p1_matched_tail_risk_seed_replication.yaml
python3 scripts/cvar_p0_analyze_tail_risk.py  --config configs/cvar_p1_matched_tail_risk_seed_replication.yaml
python3 scripts/cvar_p1_attribution_verdict.py \
  --matched-directory outputs/analysis/ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-002-SEED20260802
```

输出目录：`outputs/analysis/ANALYSIS-CVAR-P0-TAIL-RISK-001/`、`ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001/`、`-002-SEED20260802/`，各含 `diagnostic_samples.csv`、`diagnostic_summary.csv`、`per_image_tail_stats.csv`、`variance_decomposition.csv`、`verdict.json`、`plots/`、`worst_examples/`；训练输出 `outputs/train/EXP-CVAR-P1-RAYLEIGH-MATCHED-MEAN-001/`。默认 `overwrite_forbidden: true`。

注意两点复现细节：诊断的 `realization_chunk` 影响**行顺序**且带 `~1e-3 dB` 的 GPU kernel 非确定性（P0 用 32、P1 用 8），跨运行比较必须按 `(arm, image_id, snr_db, realization_id)` 键而非按行位置；显存紧张时下调 `realization_chunk` 是安全的。

## RDD-P0：重建分布偏移分析（2026-07-30 完成）

纯分析实验，回答"现有方法的重建分布是否可识别地偏离源分布、并偏向各自的生成先验"。不训练、不下载、不访问 official validation；SGD 只做分布分析，不做质量胜负。

分五个阶段顺序执行，`outputs/analysis/ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001/` 为唯一输出目录（stage 1 对非空目录 fail-closed，不覆盖既有实验）：

```bash
# 1. 从既有 montage 无损裁出 author_jscc / diffjscc / sgd_jscc 三臂 + 共享源图（带 PSNR 验证门）
python3 scripts/rdd_p0_build_arms.py

# 1b. 冻结 S33 精确重放（复用既有 canonical-noise 契约，逐图核对历史 PSNR，误差>1e-5 即中止）
python3 scripts/rdd_p0_replay_s33.py

# 2. 参考分布：blur 四档 / resample_512 / jpeg 两档
python3 scripts/rdd_p0_build_references.py

# 2b. 生成先验代理（SD 2.1 与 SGD 各自 VAE 往返）。主环境缺 pytorch_lightning，必须用 .venv-sgdjscc
.venv-sgdjscc/bin/python scripts/rdd_p0_build_vae_references.py

# 3. FID/KID 矩阵（4 臂 × 5 SNR × 10 参考）+ 参考三角 + criterion-2 命中
python3 scripts/rdd_p0_distribution_metrics.py

# 4. 指纹分类器 + C1/C2/C3 伪影控制 + 频段归因
python3 scripts/rdd_p0_fingerprint.py

# 5. CLIC-428 判别式补充（高功效 criterion-2 检验）
python3 scripts/rdd_p0_clic_complement.py
```

注意事项：

- `cleanfid` 会尝试联网下载 Inception 权重。本项目使用 A0 已冻结的本地副本，需软链接到 cleanfid 查找路径（Linux 下为 `/tmp`）：
  `ln -sf paper_idea1b/data/metric_weights/inception-2015-12-05.pt /tmp/inception-2015-12-05.pt`
- KID 为主指标、FID 必报：共享总体每 (method,SNR) 单元 n=192，2048 维协方差秩亏使 FID 正偏。
- 预注册：`reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md`；配置：`configs/rdd_p0_distribution_shift.yaml`；结果：`reports/rdd_p0_distribution_shift_result_2026-07-30.md`。

## paper_idea1b：Kodak / CLIC 主 benchmark 工作区

本篇论文的新数据、配置、脚本和输出统一放在 `paper_idea1b/`。它通过 `PYTHONPATH=src` 复用共享实现，并按原路径引用冻结 S33 checkpoint、canonical noise、S34D结果；这些旧资产没有被移动或复制。

Gate A0 与 A1 判别式主表均已完成：Kodak 24张×5 SNR×3 seeds、CLIC2020 test 428张×5 SNR×1 seed，以及PSNR/MS-SSIM/LPIPS/DISTS/CLIP/FID/KID均已闭合。S33/Swin使用相同固定tile、逐图actual CBR和canonical noise；Kodak为`1/24`，CLIC因共同padding为`0.041667–0.063210`。A1结论是S33没有战胜Swin：Kodak仅对Base-SA的PSNR追平/非劣，对CM-SA劣于；CLIC对两条Swin臂五档均劣于。完整结果见`paper_idea1b/A1_DISCRIMINATIVE_RESULT.md`。

大图不强制跨方法共同tile：尚未执行的DiffJSCC A2仍应保持官方整图入口，SGD保持作者patch入口；公平性由实际发送符号、padding/overlap和sender-side side information账本约束。下列为已完成A0的历史命令：

```bash
# 历史A0下载命令；已完成，下载器显式清空proxy并走服务器直连。
bash paper_idea1b/scripts/download_gate_a0_data.sh

PYTHONPATH=src python3 paper_idea1b/scripts/prepare_gate_a0.py \
  --config paper_idea1b/configs/gate_a0_benchmark_setup.yaml

PYTHONPATH=src python3 paper_idea1b/scripts/metric_identity_sanity.py \
  --config paper_idea1b/configs/gate_a0_benchmark_setup.yaml --device cuda:0
```

完成事实和失败判据修订记录见 `paper_idea1b/PROGRESS.md`；A0/A1正式输出分别见 `paper_idea1b/outputs/GATE-A0-BENCHMARK-SETUP-001/` 与 `paper_idea1b/outputs/ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001/`。已完成输出不可覆盖；A2、refiner训练与official validation仍需另行授权。

## S35R 新主线：轻量 receiver-side generative refiner

下一阶段主线已改为“代价—质量—可靠性公平刻画 + 轻量接收端生成式精修”。永久冻结 S33 `16,384-real` backbone，在 decoder RGB 后增加一个只读取重建图与归一化 SNR 的几百万参数 residual U-Net；它不发送额外符号，不使用文本或外部生成大模型。训练采用 LPIPS、轻 MSE、小 PatchGAN 和 S33-anchor L1，是否继续由冻结960键上的 LPIPS/PSNR/semantic-failure 三重 gate决定。

P0 SGD adaptive-cost 已完成：五档 SNR 都执行50次 denoiser，端到端约 `2.044–2.045 s/图`，没有随好信道下降；BLIP2+MuGE 是当前 released pipeline 约 `1.070 s/图` 的固定地板。`alpha_bar_channel` 改的是 continuous trajectory endpoint，而不是 solver evaluation 数。结果见 `reports/s35r_p0_sgd_adaptive_cost_result_2026-07-23.md`。

P1 目前只完成预注册，smoke 与训练都不得启动。合同见 `configs/s35r_p1_light_receiver_refiner_probe.yaml` 与 `reports/s35r_p1_light_receiver_refiner_preregistration_2026-07-23.md`。official Imagenette validation 继续封存。

P0 派生测量命令（正式输出已经存在，禁止覆盖）：

```bash
python3 scripts/s35r_p0_sgd_adaptive_cost.py
```

原 S34C 严格总码率公平生成式重训已由用户在任何执行前暂停：DiffJSCC 用官方两阶段代码重训技术上可行，SGD 因官方未发布 trainer 只能 approximate，但 14–29 天工期、unequal compute 与尽快投稿冲突。保留合同见 `configs/s34c_fair_generative_reproduction_preregistration.yaml`。

S34C-Lite 已完成：只读复用 S33/S30/S20/S28 各 960 行，制作真实码率、发送端 side-info、接收端外部先验、训练/算力和 PSNR/MS-SSIM/LPIPS/failure 的透明表。结论为 S33 与 DiffJSCC 在相同 `16,384 real` 下构成 fidelity–perception Pareto；SGD 最低总量 `21,856 real`（+33.40%）且使用完美 captions，只作 non-ranking paper upper。现有共同结果没有 FID/KID，必须显式写为证据缺口；official Imagenette validation 继续封存。完整结果见 `reports/s34c_lite_rate_transparency_result_2026-07-23.md`，正式输出见 `outputs/analysis/ANALYSIS-S34C-LITE-RATE-TRANSPARENCY-001/`。

历史执行命令如下；正式输出已经存在，禁止覆盖：

```bash
python3 scripts/s34c_lite_rate_transparency.py
```

## S34D 生成式 JSCC 推理代价

同一 RTX 4090D、batch=1、同一 256×256 主存入口的纯测量已完成。模型加载、磁盘 I/O 和指标不计入 steady-state；方法内部 resize/patch、BLIP2、edge/text conditioning、全部 denoiser evaluations、VAE 编解码、color fix 与数据传回均计入。DiffJSCC `100/50/25/10/4` 步延迟为 `5089.7/2676.2/1458.5/726.3/433.6 ms`；最低仍显著保持相对 S33 LPIPS 优势的是 25 步。为排除框架版本偏差，S33 在共同 PyTorch 2.1 runtime 下为 `8.833 ms`，故公平保守 slowdown=`165.1×`；profiler 支持算子的 FLOPs 下界为 `472×`。

25 步的 semantic failure `14/320` 显著高于 S33 的 `4/320`，所以这是感知最低点，不是语义安全点。50 步=`2.676s/303×` 的 failure 差 CI 跨零，但没有预注册非劣 margin。SGD 50-step paper upper=`2.045s/231.5×`，仍不参与质量排名。完整组件、参数与“固有/可优化”边界见 `reports/s34d_generative_inference_cost_result_2026-07-23.md`。

历史执行命令如下；输出均已存在，禁止覆盖：

```bash
python3 scripts/s34d_measure_s33_cost.py
.venv-sgdjscc/bin/python scripts/s34d_measure_diffjscc_cost_quality.py --preflight
.venv-sgdjscc/bin/python scripts/s34d_measure_diffjscc_cost_quality.py
.venv-sgdjscc/bin/python scripts/s34d_measure_sgd_cost.py --preflight
.venv-sgdjscc/bin/python scripts/s34d_measure_sgd_cost.py
.venv-sgdjscc/bin/python scripts/s34d_measure_s33_cost.py \
  --output-dir outputs/analysis/ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001/s33_torch21_sensitivity
python3 scripts/s34d_aggregate_inference_cost.py
python3 scripts/s34d_common_runtime_post_analysis.py
python3 scripts/s34d_semantic_failure_post_analysis.py
```

## S31/S31b/S32 强 JSCC 基座（阶段完成）

S30 暴露的主要系统短板是旧 JSCC 保真端点。S31 已冻结并实现 clean-room 强基座：`31,118,032` 个可训练参数，四级残差编码器/解码器均带 SNR 条件，`256x256` 输入原生输出 `77x16x16=19,712` 个实符号，不使用 mask、padding 或 side information。第一阶段只训练 MSE，禁止用 diffusion、LPIPS 或语义标签挑 checkpoint；完整预注册见 `reports/s31_strong_jscc_preregistration_2026-07-21.md`。

GPU smoke 历史命令如下；输出已经存在，不可覆盖：

```bash
PYTHONPATH=src python3 scripts/s31_train_strong_jscc.py \
  --config configs/s31_strong_jscc_coco256_awgn.yaml \
  --device cuda:0 --dry-run --max-train-batches 1 --max-val-batches 1
```

原 S31 正式训练在 epoch 3 达到五档平均 `28.0448 dB/0.958405 MS-SSIM`，随后 epoch 4 batch 418 触发 AMP gradient overflow，并按 fail-closed 规则停止；失败输出保留。修正 seed 合同后的 S31b `-002` 固定从原 best SHA `8e8f3b7b...fb0156` 仅加载模型权重，以 FP32/fresh AdamW 完成 8/8 epoch，最终 best epoch7 为 `29.360583 dB/0.967330`，checkpoint SHA `2f8972a9...57ca8`。

冻结后的 S32 在 S30 同 64 图×3 seed×5 SNR 上得到 strong `30.419910 dB/0.970266 MS-SSIM/0.122824 LPIPS/14 failures`，author-JSCC 为 `29.986135/0.963092/0.128342/22`。聚合 strong−author PSNR 为 `+0.433774 dB`（95% CI `[+0.328020,+0.554007]`），LPIPS 为 `-0.005518`（`[-0.007775,-0.003147]`）。strong 使用项目完整 19,712 real，author 使用 16,384 real，因此是预算上限内胜出，不是 exact-rate matched 胜出。完整边界和下一步见 `reports/strong_jscc_backbone_stage_result_2026-07-21.md`。

原 S31 正式训练历史命令（输出已经存在，不可覆盖）：

```bash
PYTHONPATH=src python3 scripts/s31_train_strong_jscc.py \
  --config configs/s31_strong_jscc_coco256_awgn.yaml --device cuda:0
```

S31b FP32 smoke 历史命令（输出已经存在，不可覆盖）：

```bash
PYTHONPATH=src python3 scripts/s31_train_strong_jscc.py \
  --config configs/s31b_strong_jscc_fp32_continuation_002.yaml \
  --device cuda:0 --dry-run --max-train-batches 1 --max-val-batches 1
```

S31b 正式续训命令：

```bash
PYTHONPATH=src python3 scripts/s31_train_strong_jscc.py \
  --config configs/s31b_strong_jscc_fp32_continuation_002.yaml --device cuda:0
```

只有同一实验意外中断且 config snapshot/checkpoint SHA 未变化时才可加 `--resume`；model-only 初始化与 `--resume` 互斥。
早期 `EXP-S31B-STRONG-JSCC-FP32-001` 因 seed 会改变 val512 population，在任何 validation 输出前主动中止；不可续跑，修正边界见 `reports/s31b_strong_jscc_fp32_continuation_002_preregistration_2026-07-21.md`。

S32 历史执行命令；输出已经存在，禁止覆盖：

```bash
PYTHONPATH=src python3 scripts/s32_strong_jscc_external_comparison.py \
  --config configs/s32_strong_jscc_external_comparison.yaml --device cuda:0 --preflight
PYTHONPATH=src python3 scripts/s32_strong_jscc_external_comparison.py \
  --config configs/s32_strong_jscc_external_comparison.yaml --device cuda:0
```

## S33 严格 16,384-real 等码率 Strong JSCC（阶段完成）

S33 保持 S31 的四级 SNR-conditioned clean-room 架构，把原生 latent 改为 `64x16x16=16,384 real`，与 DiffJSCC author-JSCC 严格等码率；训练使用随机初始化、FP32 4+8 epochs、`[1,4,7,13,19] dB` 离散逐图均匀采样。不得描述为连续随机 SNR。最终 checkpoint 为 `outputs/train/EXP-S33B-STRONG-JSCC-16384-FP32-001/checkpoints/best.pt`，SHA=`2daad9e7...5bfb`。

历史命令如下；所有输出目录已经存在，禁止覆盖。只有同一实验意外中断且 config snapshot/checkpoint SHA 完全一致时才允许 `--resume`：

```bash
PYTHONPATH=src python3 scripts/s31_train_strong_jscc.py \
  --config configs/s33_strong_jscc_16384_fp32_main.yaml --device cuda:0

PYTHONPATH=src python3 scripts/s31_train_strong_jscc.py \
  --config configs/s33b_strong_jscc_16384_fp32_continuation.yaml --device cuda:0

PYTHONPATH=src python3 scripts/s32_strong_jscc_external_comparison.py \
  --config configs/s33_strong_jscc_16384_external_comparison.yaml \
  --device cuda:0 --preflight

PYTHONPATH=src python3 scripts/s32_strong_jscc_external_comparison.py \
  --config configs/s33_strong_jscc_16384_external_comparison.yaml --device cuda:0

PYTHONPATH=src python3 scripts/s33_equal_rate_post_analysis.py
```

严格等码率 policy-dev 结果为 strong−author PSNR `+0.479929 dB`，source-image cluster 95% CI `[+0.370006,+0.598197]`，按预注册规则显著超过；聚合 MS-SSIM/LPIPS/failure 也显著有利。13/19 dB 感知边界和非最终测试限定见 `reports/strong_jscc_16384_equal_rate_stage_result_2026-07-21.md`。本轮没有启动 S34 消融、S35 diffusion 或 S36 official validation。

## S34A SwinJSCC 公平对比（equal-budget 已完成，extension 待决定）

官方 `semcomm/SwinJSCC@a6d0e6d...90f` 源码已固定，第三方源码不修改；项目侧 `src/cadsd_jscc/swinjscc_adapter.py` 负责逐图 SNR、逐图功率和 canonical paired-real AWGN。已确认 official Base-SA `28.18M` 与 capacity-matched CM-SA `31.35M` 双臂，二者均原生输出 `16,384 real`。真实 COCO microbatch=8 的单步 smoke 已通过，峰值 reserved VRAM 为 `9.75/10.40 GiB`；正式训练使用 microbatch=8、gradient accumulation=4 保持 effective batch=32。

本轮只授权两臂各 12 epochs，并在同一固定 COCO val512 上检查 epoch 9--12 收敛曲线。两臂现均完成 12/12 且都触发未明确收敛 gate；训练器在 epoch12 硬停止，extension 尚未授权，official Imagenette validation 继续封存。

equal-budget policy-dev 结果：S33−Base PSNR=`+0.173947 dB`，95% CI=`[+0.078178,+0.265733]`，显著超过；S33−CM=`−0.065902 dB`，CI=`[−0.168886,+0.025307]`，未通过 `0.10 dB` 非劣 gate。CM 的 MS-SSIM/LPIPS 显著更好，S33 的观测 failure 更低，因此保守总 verdict 为 Pareto。完整中文结果见 `reports/swinjscc_equal_budget_stage_result_2026-07-22.md`；充分训练结论须等待用户另行授权 extension。

历史 smoke 命令如下；输出已经存在，禁止覆盖：

```bash
PYTHONPATH=src python3 scripts/s34a_swinjscc_smoke.py \
  --config configs/s34a_swinjscc_equal_rate_comparison.yaml --device cuda:0
```

当前正式训练入口（先用 `--preflight-only` 无写入审计，再逐臂运行；已有目录只能显式 `--resume`）：

```bash
.venv-sgdjscc/bin/python scripts/s34a_train_swinjscc_equal_budget.py \
  --arm official_base_sa --preflight-only
.venv-sgdjscc/bin/python scripts/s34a_train_swinjscc_equal_budget.py \
  --arm official_base_sa --device cuda:0
.venv-sgdjscc/bin/python scripts/s34a_train_swinjscc_equal_budget.py \
  --arm capacity_matched_sa --device cuda:0
```

冻结 equal-budget 评估历史命令如下；当前输出已经完成，禁止无新 analysis ID 直接重跑或覆盖：

```bash
.venv-sgdjscc/bin/python scripts/s34a_evaluate_swinjscc_equal_budget.py \
  --config configs/s34a_swinjscc_equal_budget_evaluation.yaml \
  --device cuda:0 --preflight

.venv-sgdjscc/bin/python scripts/s34a_evaluate_swinjscc_equal_budget.py \
  --config configs/s34a_swinjscc_equal_budget_evaluation.yaml \
  --device cuda:0 --resume
```

## SGD-JSCC / S33 top-LPIPS 语义人工核查

`ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004` 从两种方法各自已有 960 条 policy-dev 记录中，按 LPIPS 升序且 source 去重选择 15 张，生成 `[原图 | 重建]` 对照图。SGD 直接裁已有正式 montage；S33 只对入选键使用冻结 checkpoint 和 canonical noise 做推理重放，不训练、不调参、不访问 official val。有效结果中 S33 历史 PSNR 重放最大误差为 `0.0 dB`。人工标签采用绿色 faithful、橙色 minor structure/text change、红色 semantic mismatch；本次两种方法均无红色样本。完整边界见 `reports/top_lpips_semantic_visual_audit_2026-07-23.md`。

历史执行方式如下。输出目录已存在，禁止覆盖；如需新审计必须修改 config 的 analysis ID 和输出路径。`prepare` 必须使用 S33 历史所用的系统 Python/PyTorch 环境，不能使用 SGD 的 PyTorch 2.1 虚拟环境：

```bash
python3 scripts/top_lpips_semantic_visual_audit.py \
  --config configs/top_lpips_semantic_visual_audit.yaml \
  --stage prepare --device cuda:0

# 人工填写输出目录中的 manual_review.json 后：
python3 scripts/top_lpips_semantic_visual_audit.py \
  --config configs/top_lpips_semantic_visual_audit.yaml \
  --stage finalize
```

## 低 SNR 语义漂移定向审计

`ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-003` 只在 1 dB 的 192 个冻结键中，以“LPIPS 尚可但 T_cls / 三个跨模型分类器 / CLIP 异常”为召回条件，source 去重选择 15 张，生成 `[原图 | S33 pure JSCC | SGD diffusion]` 三列图。人工结果：S33 `8 faithful / 7 重建失败 / 0 清晰但错`，SGD `15/0/0`。S33 历史 PSNR 重放最大误差 `0.0 dB`，没有训练或 official val 访问。

`ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001` 固定同一 15 张 source，在 −3/−5 dB 和共同 base seed 下重放，不根据压力结果二次选样本。S33 在 −3/−5 dB 分别有 14/15 个显式重建失败，SGD paper-upper 没有观察到 clear-wrong，但部分图有 patch 接缝。该压力结果不作公平排名：SGD 使用作者权重、额外 edge 码率和免费完美 captions。完整解释见 `reports/low_snr_semantic_drift_visual_audit_2026-07-23.md`。

历史执行命令如下；输出目录均已存在，禁止覆盖。1 dB `prepare` 必须使用 S33 的系统 Python/PyTorch 环境：

```bash
python3 scripts/low_snr_semantic_drift_visual_audit.py \
  --config configs/low_snr_semantic_drift_visual_audit.yaml \
  --stage prepare --device cuda:0

# 填写 1 dB 输出目录的 manual_review.json 后：
python3 scripts/low_snr_semantic_drift_visual_audit.py \
  --config configs/low_snr_semantic_drift_visual_audit.yaml \
  --stage finalize

python3 scripts/low_snr_out_of_range_stress.py --stage prepare-s33 --device cuda:0
.venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py \
  --config outputs/analysis/ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001/sgd_configs/sgd_stress_resolved.yaml \
  --run
python3 scripts/low_snr_out_of_range_stress.py --stage assemble

# 填写 stress 输出目录的 manual_review.json 后：
python3 scripts/low_snr_out_of_range_stress.py --stage finalize
```

## 最新 S30 官方 DiffJSCC 完整对比

官方 OpenImage C16 DiffJSCC 已在冻结 S20/S28 64 图×3 AWGN seed×5 SNR 上完成 `960/960` 行严格对比。current 相对 DiffJSCC 最终输出的 PSNR 为 `+0.625280 dB`（source-image cluster 95% CI `[+0.423123,+0.824753]`），但 LPIPS `+0.051861`（显著更差），failure `29 vs 23` 的 CI 跨零，因此 verdict 为 `PARETO_OR_INCONCLUSIVE`。

关键新发现不是“谁单轴第一”，而是 DiffJSCC 的纯 JSCC 前端本身明显强于 current：仅用 `16,384 real`（项目预算的 `83.1169%`）即达到 `29.986135 dB / 0.128342 LPIPS / 22 failures`，current 为 `28.223678 / 0.152084 / 29`。固定 100-step diffusion 把该前端推向感知端：LPIPS 改善 `-0.028119`，但 PSNR 损失 `-2.387737 dB`，并产生 `10 new / 9 repair`；1/4 dB 为净修复，7 dB 转为净风险，13 dB 为 `3 new / 0 repair`。完整中文报告和下一阶段建议见 `reports/diffjscc_external_comparison_stage_result_2026-07-21.md`。

历史执行顺序如下；这些输出已经存在，禁止覆盖：

```bash
python3 scripts/s30_diffjscc_preflight.py
.venv-sgdjscc/bin/python scripts/s30_diffjscc_checkpoint_audit.py
.venv-sgdjscc/bin/python scripts/s30_diffjscc_external_comparison.py --stage preload --device cuda:0
.venv-sgdjscc/bin/python scripts/s30_diffjscc_external_comparison.py --stage smoke --device cuda:0
.venv-sgdjscc/bin/python scripts/s30_diffjscc_external_comparison.py --stage first-seed --device cuda:0
.venv-sgdjscc/bin/python scripts/s30_diffjscc_external_comparison.py --stage full --device cuda:0 --resume
python3 scripts/s30_diffjscc_post_analysis.py
```

`--resume` 只用于续接已有、带 config snapshot 的 formal output；新复现必须先更换配置中的 analysis ID/output path。S30 完整输出：`outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/`。

## 最新 S28/S29 外部定位

冻结主方法已在 S20 相同的 64 张 Imagenette policy-dev 图像、3 个 AWGN seed、5 个 SNR 上与 B1、等容量 control 和 SGD-JSCC 逐样本比较。当前方法相对 B1 为 PSNR `+0.099085 dB`（95% CI `[+0.088053,+0.111284]`）、LPIPS `-0.007314`、T_cls failure `35→29`；相对 matched control 仍为 `+0.059681 dB/-0.002990 LPIPS`。相对 SGD 免费完美文本论文上界，我们 PSNR 高 `+0.483309 dB`，但 SGD 的 MS-SSIM/LPIPS 更好，形成明确 Pareto；SGD 文本若最低计费会使 19,712-real 预算超出 `10.88%`。

S28 的 batch=16 B1 重算曾因 `0.0004768 dB` 最大浮点差触发严格技术负判定；S29 恢复 S20 原 batch=64 后，960 行全部指标、预测、failure 和 noise SHA 零误差复现，确认不是合同错位。完整中文判断、通俗数据流和论文边界见 `reports/current_method_external_positioning_stage_result_2026-07-21.md`。

历史命令如下；现有输出目录不可覆盖：

```bash
PYTHONPATH=src python3 scripts/s28_external_sgd_positioning.py --device cuda:0
PYTHONPATH=src python3 scripts/s29_s28_b1_exact_batch_audit.py --device cuda:0
```

## 最新综合指标与通俗数据流

S24 已把冻结的 S19、S20、S23 结果统一复核，并生成面向非专业读者的中文说明。当前结论是：S19 的融合质量增益最大，但高 SNR 有负迁移；S23 的额外增益很小，却首次同时做到非零 diffusion 注入、独立 holdout 三项质量 CI 有利、13/19 dB 精确回退 B1。完整指标表、外部 SGD-JSCC 边界、术语解释和一张图的端到端数据流程见 `reports/recent_progress_metrics_and_data_flow_2026-07-20.md`。

以下是本次派生汇总的历史命令。现有 analysis 目录禁止覆盖；复跑时必须先改配置中的 analysis id/output directory：

```bash
MPLBACKEND=Agg PYTHONPATH=src python3 scripts/s24_recent_progress_metrics.py
```

脚本会先校验冻结输入 SHA，再重算同 population 指标、source-image cluster bootstrap、参数量与限定范围的接收端后处理延迟。该延迟不包含 6-step diffusion，因此不能解释为端到端系统延迟。

## 最新 S27 pristine replication

S26 主方法已在与 S16/S18/S19/S21 path/SHA 全部去重的 512 张新 COCO 图片上复现。相对 B1：PSNR `+0.092662 dB`（95% CI `[+0.089147,+0.096313]`）、MS-SSIM `+0.002310`、LPIPS `-0.007922`，majority failure `1561→1517`；相对等容量 control 仍为 `+0.065799 dB/-0.003494 LPIPS`。13/19 dB 精确 B1，9/9 checks PASS。完整中文报告：`reports/s19_exact_fallback_fresh_replication_stage_result_2026-07-21.md`。

历史执行顺序如下，已有目录不可覆盖：

```bash
python3 scripts/s19_prepare_fusion_population.py --config configs/s27_s19_exact_fallback_fresh_replication.yaml
python3 scripts/s19_cache_identity_diffusion.py --config configs/s27_s19_exact_fallback_fresh_replication.yaml --device cuda:0
PYTHONPATH=src python3 scripts/s27_s19_exact_fallback_fresh_replication.py --device cuda:0
```

## 最新 S26 强融合 + exact fallback 结果

当前最好的方法已更新为 `S26 = frozen S19 fusion at 1/4/7 dB + exact B1 at 13/19 dB`。在另一批 256 图×5 SNR 上，相对 B1 得到 PSNR `+0.093267 dB`（95% CI `[+0.087945,+0.098806]`）、MS-SSIM `+0.002188`、LPIPS `-0.007661`，majority failure `744→720`；相对等容量 control 仍为 `+0.065486 dB/-0.003100 LPIPS`。13/19 dB 最大逐像素差为 0，9/9 预注册检查通过。完整边界见 `reports/s19_exact_fallback_replication_stage_result_2026-07-20.md`。

历史执行命令如下；现有目录禁止覆盖：

```bash
PYTHONPATH=src python3 scripts/s26_s19_exact_fallback_replication.py --device cuda:0
```

## 最新 S25 幅度上限判定

S25 已证明继续在 S23 one-epoch feature direction 上训练逐图 amplitude controller 没有足够上限：即使不可部署的 semantic-safe oracle 可以读取原图和三分类器结果，相对固定 `alpha=0.15` 也只增加 `+0.001365 dB` PSNR，95% CI `[+0.001186,+0.001562]`，没有达到冻结的 `+0.02 dB` 门槛。该路线正式关闭，S23 仅保留为 exact-fallback 机制基线。完整中文结果见 `reports/b1_feature_amplitude_headroom_stage_result_2026-07-20.md`。

历史命令如下；输出目录已存在，禁止覆盖：

```bash
PYTHONPATH=src python3 scripts/s25_b1_feature_amplitude_headroom.py --device cuda:0
```

## 最新 S21--S23 合并结果

B1 与 matched diffusion 的简单输出层合并已经关闭：learned gate 会塌零，fixed-gate residual 会饱和，120 个单调像素凸融合候选也只选出全零 B1。S22 冻结 B1，仅新增 1,728 参数把 `D-B0` 注入 B1 feature；非零 endpoint 稳定改善 LPIPS 但轻微损失 PSNR。S23 预注册全局 shrink 后选中 `alpha=0.15`，在独立 256×5 holdout 上相对 B1 取得 PSNR `+0.000568 dB`（95% CI `[+0.000378,+0.000771]`）和 LPIPS `-0.001731`（`[-0.001849,-0.001622]`），5/5 检查通过。它证明最小安全合并可行，但 PSNR 效应量很小。完整中文结论与边界见 `reports/b1_merge_stage_result_2026-07-20.md`。

历史执行命令如下；现有输出禁止覆盖，复跑必须更换 experiment/analysis id：

```bash
PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --mode smoke --device cuda:0
PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --mode train --device cuda:0
PYTHONPATH=src python3 scripts/s23_b1_feature_shrink.py --device cuda:0
PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --config configs/s23_b1_feature_shrink.yaml --mode holdout --device cuda:0
PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --config configs/s23_b1_feature_shrink.yaml --mode bootstrap
```

只有 selection 冻结出非零 checkpoint、把 SHA 和 protocol status 写回新配置后，才允许运行 `--mode holdout` 和 `--mode bootstrap`；S23 已按此顺序完成，现有目录禁止覆盖。

## 最新 S20 结果

64 张独立 Imagenette clean-correct 图×5 SNR×3 channel seeds 的扩展判定表明：SGD-JSCC 免费/完美文本论文上界明显强于普通 exact-rate JSCC，但没有全面支配 B1。SGD−B1 的 PSNR 为 `-0.38422 dB`（source-cluster 95% CI `[-0.61529,-0.16026]`），LPIPS 为 `-0.087297`（`[-0.100439,-0.075641]`）；failure `35→25`，但仍有 `11` 个相对 B1 new error。公开 SGD 图像+边缘支路已占满 `19,712 real`，四个 caption 的最低未保护 BPSK 成本还需 `2,144 real`，因此严格同总码率的“全程 SGD”当前不可执行。完整中文报告见 `reports/sgd_b1_decision_stage_result_2026-07-17.md`。

以下是历史可复现执行顺序；现有输出禁止覆盖，复跑必须修改 analysis id 和输出目录：

```bash
PYTHONPATH=src python3 scripts/s20_prepare_sgd_b1_decision.py
PYTHONPATH=src python3 scripts/s20_sgd_b1_decision.py --mode prepare-sgd-configs
PYTHONPATH=src python3 scripts/s20_sgd_b1_decision.py --mode baseline

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  .venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py \
  --config <sgd_seed_resolved.yaml> --run

PYTHONPATH=src python3 scripts/s20_sgd_b1_decision.py --mode aggregate
```

三份 `<sgd_seed_resolved.yaml>` 位于冻结 population 目录，分别对应 seeds `20260748/20260749/20260750`。本轮全部模型和数据均来自本地缓存，无需联网。

## S19 结果

等容量因果消融已证明 S18 identity-controlled diffusion 对强 B1 具有互补信息：全新 256×5 holdout 上，fusion 相对同参数量 B0-only control 为 `+0.05846 dB`，cluster-bootstrap 95% CI `[+0.05198,+0.06423]`，LPIPS 同时改善 `-0.001493`；相对原 B1 为 `+0.10168 dB`。完整边界和高 SNR 负迁移见 `reports/diffusion_fusion_ablation_stage_result_2026-07-16.md`。

历史可复现执行顺序如下；输出目录已存在，禁止直接覆盖，复跑必须使用新 experiment/analysis id：

```bash
python3 scripts/s19_prepare_fusion_population.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s19_cache_identity_diffusion.py
python3 scripts/s19_train_and_evaluate_fusion.py --mode train
python3 scripts/s19_train_and_evaluate_fusion.py --mode holdout
python3 scripts/s19_fusion_bootstrap.py
```

## 代码结构

```text
configs/          实验配置
data/             数据集说明或本地数据指针
references/       文献 PDF、BibTeX、阅读笔记或外部链接索引
src/              训练、推理、信道和评估代码
scripts/          可复现的命令行流程
outputs/          生成结果、指标和可视化样例
tests/            单元测试和 smoke test
third_party/      外部代码仓库或其路径说明
```

## 环境安装

当前已在用户 Python 环境中安装阶段1和后续研究常用依赖，CUDA 版 PyTorch 已可用，CIFAR-10 已下载到 `data/cifar10/`。

CPU 环境推荐安装命令：

```bash
python3 -m pip install --user --no-cache-dir --default-timeout 120 --retries 10 -r requirements-torch-cpu.txt
python3 -m pip install --user --no-cache-dir --default-timeout 120 --retries 10 -r requirements.txt
python3 -m pip install --user --no-cache-dir --default-timeout 120 --retries 10 -r requirements-research.txt
```

GPU 环境推荐安装命令，适用于当前 RTX 4090 D / CUDA driver 可见的机器：

```bash
python3 -m pip install --user --default-timeout 120 --retries 10 -r requirements-torch-cu128.txt
```

当前已验证的关键版本：

- Python：`3.10.12`
- torch：`2.11.0+cu128`
- torchvision：`0.26.0+cu128`
- numpy：`2.2.6`
- pillow：`12.2.0`
- diffusers：`0.38.0`
- transformers：`5.12.1`

GPU 备注：

- `nvidia-smi` 在非沙箱环境可见 RTX 4090 D，显存约 24GB。
- 非沙箱 Python 已验证 `torch.cuda.is_available()` 为 True，设备为 `NVIDIA GeForce RTX 4090 D`。
- 256x256 高分辨率训练脚本已通过 GPU dry-run，输出位于 `outputs/smoke/s2_deepjscc_coco256_train_gpu/`。

已知问题：

- 直接执行 `python3 -m pip install -r requirements.txt` 不会安装 PyTorch。
- 不建议无脑安装第三方仓库原始 `requirements.txt`，其中 `torchvison` 拼写错误，并且默认 PyPI 路线可能拉取很大的 CUDA 依赖。
- 2026-06-29 早前曾遇到 PyTorch 下载超时和 CPU-only 路线 hash mismatch；后续使用 `--user --no-cache-dir --default-timeout 120` 已成功安装。

下载流量规则：

- 大模型、大数据集、CUDA/PyTorch 等大文件下载默认走服务器直连，不走用户本机代理流量。
- 下载前先检查代理：

```bash
env | grep -i proxy
```

- 若存在代理变量，默认用清空代理变量的方式执行大下载：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy <download-command>
```

- 只有用户明确说可以使用代理/本机流量时，才允许通过代理下载大文件。

第一阶段建议先审计并尝试接入普通 DeepJSCC PyTorch baseline，再决定是否训练自己的 checkpoint。

候选 baseline：

1. `chunbaobao/Deep-JSCC-PyTorch`：第一优先候选，用于快速跑通普通 DeepJSCC baseline。
2. `mingyuyng/Dynamic_JSCC`：第二候选，用于理解 adaptive rate control。
3. `aprilbian/deepjscc-lplusplus`：第三候选，用于后续 channel-adaptive JSCC 对照。

## 第一阶段目标

构建 baseline 评估 pipeline：

1. 在一组 SNR 上运行 DeepJSCC 重建。
2. 加入 blind diffusion refinement。
3. 测量重建质量、感知质量和 semantic drift。
4. 加入 semantic guidance 和 SNR-aware diffusion strength。
5. 比较不同方法的 tradeoff 曲线。

## 最小闭环

本项目优先完成 `MILESTONES.md` 中定义的最小论文闭环：

- CIFAR-10 + AWGN 只作为 sanity baseline。
- COCO2017 `256x256` + AWGN 作为 diffusion 主实验闭环。
- CBR 固定为 `0.17`。
- SNR sweep 固定为 `[1, 4, 7, 13, 19]` dB。
- 对比 `M0-DeepJSCC`、`M1-BlindDiffusion`、`M2-SNRAdaptiveDiffusion`、`M3-Ours`。
- 用 semantic drift / final failure 约束 diffusion refinement 的语义可靠性。

完成该闭环前，不扩展到大型 DiT-JSCC 或复杂 adaptive JSCC 主线。

## 当前 smoke test

第三方 baseline 已浅克隆到：

```text
third_party/Deep-JSCC-PyTorch
```

已写好 smoke test：

```bash
python3 scripts/s1_deepjscc_smoke.py --device cpu --batch-size 2
```

说明：

- 该 smoke test 使用随机合成图像，不下载 CIFAR-10。
- 该 smoke test 只验证 checkpoint 加载、SNR 切换、重建输出和 PSNR 计算。
- smoke test 不是正式实验，不写入 `EXPERIMENTS.md`。
- 当前状态：已在 CPU 上通过 smoke test，输出位于 `outputs/smoke/s1_deepjscc/`。

## 当前 baseline

CIFAR-10 已下载到：

```text
data/cifar10/
```

真实 CIFAR-10 test subset mini-eval：

```bash
python3 scripts/s1_deepjscc_mini_eval.py --device cpu
```

首次下载数据集时使用：

```bash
python3 scripts/s1_deepjscc_mini_eval.py --device cpu --download
```

正式阶段1 baseline 已完成：

```bash
python3 scripts/s1_deepjscc_mini_eval.py --device cpu --num-samples 1024 --batch-size 64 --output-dir outputs/EXP-S1-001 --formal
```

输出位于：

```text
outputs/EXP-S1-001/
```

说明：CIFAR-10 图像为 32x32，`pytorch-msssim` 默认 MS-SSIM 要求边长大于 160，因此当前 S1 记录 PSNR/SSIM，MS-SSIM 留到高分辨率数据集或自定义设置后再启用。

## 高分辨率重训路线

CIFAR-10 只作为 sanity baseline。后续 diffusion 主路线需要重新训练或接入高分辨率 DeepJSCC checkpoint。

当前推荐主路线：

- 训练数据：COCO2017 `train2017`
- 验证数据：COCO2017 `val2017`
- 图像尺寸：`256x256`
- 信道：AWGN
- CBR：`0.17`
- 初始训练 SNR：`7` dB，后续扩展到 `[1, 4, 7, 13, 19]` dB

数据目录约定：

```text
data/coco/train2017/
data/coco/val2017/
```

当前 COCO2017 数据已就位：

```text
data/coco/train2017/  # 118287 images
data/coco/val2017/    # 5000 images
data/coco/annotations/ # COCO2017 captions / instances / keypoints JSON
```

COCO2017 官方 annotations 已通过服务器直连下载并验证：

```text
data/coco/annotations_trainval2017.zip  # 252907541 bytes, unzip -t OK
data/coco/annotations/captions_val2017.json
data/coco/annotations/instances_val2017.json
```

训练脚本：

```bash
python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco256_awgn.yaml --device cuda:0
```

数据准备并启动训练的长任务脚本：

```bash
scripts/run_s2_coco256_awgn_train.sh
```

该脚本中的 COCO 下载命令使用 `wget --no-proxy`，只让数据集下载直连，不影响 Codex 或其他命令使用当前代理环境。

如需临时覆盖数据路径，可使用：

```bash
python3 scripts/train_deepjscc_highres.py --train-root data/coco/val2017 --val-root data/coco/val2017 --device cuda:0
```

当前机器已验证可用 GPU。CPU dry-run 和 GPU dry-run 都已通过，GPU dry-run 命令：

```bash
python3 scripts/train_deepjscc_highres.py --dry-run --device cuda:0 --epochs 1 --batch-size 2 --num-workers 0 --max-train-batches 1 --max-val-batches 1 --output-dir outputs/smoke/s2_deepjscc_coco256_train_gpu
```

CPU dry-run 命令：

```bash
python3 scripts/train_deepjscc_highres.py --dry-run --device cpu --epochs 1 --batch-size 2 --num-workers 0 --max-train-batches 1 --max-val-batches 1
```

CPU dry-run 输出位于：

```text
outputs/smoke/s2_deepjscc_coco256_train/
```

GPU dry-run 输出位于：

```text
outputs/smoke/s2_deepjscc_coco256_train_gpu/
```

真实 COCO `val2017` 图像 GPU smoke 已通过，输出位于：

```text
outputs/smoke/s2_deepjscc_coco256_val2017_gpu/
```

在 `train2017` 下载较慢时，可用已完成的 `val2017` 做非正式高分辨率 pilot：

```bash
python3 scripts/prepare_image_symlink_split.py --source-root data/coco/val2017 --output-root data/coco_val_split --train-size 4500 --val-size 500 --seed 42 --overwrite
python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0
```

当前 pilot 训练输出位于：

```text
outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/
```

该 pilot 只用于高分辨率 JSCC checkpoint 和后续 diffusion 接口调试，不能替代正式 COCO `train2017/val2017` 主实验。

pilot checkpoint 的 M0-HR SNR sweep 和 `x_hat` 导出：

```bash
python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 32 --output-dir outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export
```

输出位于：

```text
outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/
```

其中 `exports/original/` 保存原图，`exports/snr_XXdb/reconstruction/` 保存各 SNR 的 DeepJSCC 重建图，可作为 `M1-BlindDiffusion` 的输入。

正式 COCO-256 训练已完成，但后段出现 NaN。可用 checkpoint 是：

```text
outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt
```

该 checkpoint 来自 epoch 73，验证指标约为 PSNR `31.5618` dB、SSIM `0.9054`。

不要使用：

```text
outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/latest.pt
```

`latest.pt` 来自 epoch 99，参数和指标均已 NaN。

正式 `M0-HR` SNR sweep 和 `x_hat` 导出位于：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/
```

该目录每个 SNR 只保存 32 张 PNG，主要用于复现 `EXP-S2-002` 到 `EXP-S4-005`。更大的 residual validation export 位于：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/
```

该目录仍在同一 512 张 COCO val subset 上评估 M0，但每个 SNR 保存前 256 张 PNG。复现命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 256 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256
```

后续 `M1-BlindDiffusion` 应优先读取：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/exports/
```

当前没有需要等待的 COCO 下载或训练 screen 会话。

## M1-BlindDiffusion 最小接口

当前 M1 配置：

```text
configs/s3_m1_blind_diffusion_coco256_awgn.yaml
```

该配置固定读取正式 M0 export：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/
```

并记录正式 DeepJSCC checkpoint：

```text
outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt
```

不要使用 `latest.pt`。

先做输入检查，不加载 diffusion 模型：

```bash
python3 scripts/s3_blind_diffusion_refine.py --dry-run
```

正式小规模 M1 运行命令：

```bash
python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download
```

默认设置：

- SNR：`[1, 7, 19]` dB
- 每个 SNR：16 张图
- diffusion：`runwayml/stable-diffusion-v1-5` img2img
- strength：`0.25`
- steps：`25`
- guidance scale：`1.0`
- 默认输出：`outputs/EXP-S2-002/`

输出包括：

```text
outputs/EXP-S2-002/exports/snr_XXdb/refined/
outputs/EXP-S2-002/samples/
outputs/EXP-S2-002/metrics.json
outputs/EXP-S2-002/source_manifest.json
```

`metrics.json` 会在相同 16 张图上报告 M0 reconstruction 和 M1 refined 相对原图的 PSNR、SSIM、MS-SSIM；LPIPS 若环境能初始化会一并写入，否则记录失败原因。

当前状态：

- `--dry-run` 已通过。
- `EXP-S2-001` 是早先环境阻塞记录，未创建输出目录。
- `EXP-S2-002` 已完成，输出位于 `outputs/EXP-S2-002/`。
- `runwayml/stable-diffusion-v1-5` 已缓存到 `outputs/cache/huggingface/`；LPIPS AlexNet 权重已缓存到 `outputs/cache/torch/`。
- 官方 `huggingface.co` 服务器直连在 2026-07-01 超时；`hf-mirror.com` 服务器直连可用。后续大下载仍必须清空代理变量，不走用户代理流量。

`EXP-S2-002` 结论：当前 `strength=0.25`、空 prompt、`guidance_scale=1.0` 的 blind SD img2img 是明显负结果。M1 在 1/7/19 dB 上 PSNR 和 MS-SSIM 大幅下降，LPIPS 也变差；样例图显示 hallucination 和 semantic drift 风险。后续不要把该设置包装成视觉提升。

## S4 Semantic Drift 初步诊断

当前已完成 `EXP-S3-001`：对 `EXP-S2-002` 的 M0 reconstruction 和 M1 refined 输出做 CLIP image-image consistency 辅助诊断。

配置：

```text
configs/s4_clip_consistency_m1_exp_s2_002.yaml
```

运行命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_clip_consistency_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S3-001/metrics.json
outputs/EXP-S3-001/per_sample.csv
outputs/EXP-S3-001/source_manifest.json
```

从 CLIP drop 指标整理 failure case gallery：

```bash
python3 scripts/s4_make_clip_failure_gallery.py
```

gallery 输出：

```text
outputs/EXP-S3-001/failure_cases/sheets/global_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/sheets/snr_01db_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/sheets/snr_07db_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/sheets/snr_19db_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/triptychs/
outputs/EXP-S3-001/failure_cases/index.json
outputs/EXP-S3-001/failure_cases/global_top_clip_drop.csv
```

CLIP 权重缓存：

```text
outputs/cache/open_clip/ViT-B-32.pt
```

SHA256：

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

若缓存缺失，可用服务器直连下载 OpenAI 官方权重：

```bash
mkdir -p outputs/cache/open_clip
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy curl -L --fail --retry 3 --connect-timeout 20 -C - -o outputs/cache/open_clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

结论：该指标不是最终分类一致性 semantic drift metric，但已经量化确认 M1 负结果。1/7/19 dB 下，M1 相对原图的 CLIP 相似度均显著低于 M0；所有 48 个样本中 M1 都低于 M0。failure gallery 已固化全局 top 12 和每个 SNR top 6 的 original/M0/M1 triptych；后续应补冻结分类器或 object-level 语义一致性指标。

当前也已完成 `EXP-S3-002`：冻结 ImageNet AlexNet 的 pseudo-label consistency 诊断。该实验不使用 COCO GT 标签，只比较 `c(original)`、`c(M0)` 和 `c(M1)`，因此仍是辅助诊断，不是最终 clean-correct 分类指标。

配置：

```text
configs/s4_classifier_consistency_m1_exp_s2_002.yaml
```

运行命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_classifier_consistency_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S3-002/metrics.json
outputs/EXP-S3-002/per_sample.csv
outputs/EXP-S3-002/source_manifest.json
```

从 pseudo-label drift 指标整理 classifier failure gallery：

```bash
python3 scripts/s4_make_classifier_failure_gallery.py
```

gallery 输出：

```text
outputs/EXP-S3-002/failure_cases/sheets/global_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/sheets/snr_01db_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/sheets/snr_07db_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/sheets/snr_19db_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/triptychs/
outputs/EXP-S3-002/failure_cases/index.json
outputs/EXP-S3-002/failure_cases/global_top_classifier_drift.csv
```

该脚本默认读取本地 AlexNet 权重：

```text
outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
```

结论：all-subset 中，M0 与原图 top-1 一致率在 1/7/19 dB 为 `0.5000/0.6875/0.9375`，M1 仅为 `0.1250/0.0625/0.1250`；在原图 top-1 confidence >= 0.30 的 subset 上，M0 为 `0.8889/1.0000/1.0000`，M1 为 `0.2222/0.1111/0.2222`。这进一步确认当前 blind diffusion 设置存在系统性 semantic drift。

将 M1 图像质量、CLIP 诊断和分类器诊断聚合成一个派生报告：

```bash
python3 scripts/s4_summarize_m1_negative_result.py
```

输出：

```text
outputs/analysis/m1_negative_result_summary/REPORT.md
outputs/analysis/m1_negative_result_summary/summary.csv
outputs/analysis/m1_negative_result_summary/summary.json
```

该报告不新增模型运行，只汇总已有实验。当前汇总结论：平均 PSNR delta M1-M0 为 `-14.7485` dB，平均 LPIPS delta 为 `+0.3877`，平均 CLIP drop 为 `0.2672`，分类器 all-subset M1 pseudo drift-origin 为 `0.8958`。

当前也已完成 `EXP-S3-003`：COCO caption CLIP image-text consistency 诊断。该实验使用 COCO `captions_val2017.json`，把导出的 `sample_XXXXXX.png` 反查到 COCO image id 和 5 条人工 captions，然后比较 original/M0/M1 与 captions 的 CLIP image-text 相似度。它仍是辅助语义诊断，不替代最终 clean-correct 冻结分类器指标。

配置：

```text
configs/s4_coco_caption_clip_m1_exp_s2_002.yaml
```

运行命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_coco_caption_clip_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S3-003/metrics.json
outputs/EXP-S3-003/per_sample.csv
outputs/EXP-S3-003/sample_metadata.json
outputs/EXP-S3-003/source_manifest.json
```

从 caption CLIP drop 指标整理 failure case gallery：

```bash
python3 scripts/s4_make_coco_caption_failure_gallery.py
```

gallery 输出：

```text
outputs/EXP-S3-003/failure_cases/sheets/global_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/sheets/snr_01db_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/sheets/snr_07db_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/sheets/snr_19db_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/triptychs/
outputs/EXP-S3-003/failure_cases/index.json
outputs/EXP-S3-003/failure_cases/global_top_caption_clip_drop.csv
```

结论：caption 语义诊断继续确认当前 blind diffusion 负结果。1/7/19 dB 下，M0 caption-max mean 为 `0.3306/0.3305/0.3263`，M1 为 `0.2816/0.2815/0.2877`；M1 caption-max 低于 M0 的比例为 `1.0000/0.8125/0.8125`。

## S5 Semantic Failure Handling Pilot

当前已完成 `EXP-S4-001`：基于 `EXP-S2-002` 的 M1 输出和 `EXP-S3-002` 的冻结分类器 CSV，评估一个最小 receiver-side fallback 规则。

配置：

```text
configs/s5_semantic_fallback_m1_exp_s2_002.yaml
```

规则：

- 若冻结 AlexNet 对 M1 refined 和 M0 reconstruction 的 top-1 预测一致，则接受 M1。
- 否则回退到 M0 reconstruction。
- detector 不使用 original 图像；original pseudo-label 只用于离线评价 Final-Failure。

先检查输入对齐：

```bash
python3 scripts/s5_semantic_fallback_eval.py --dry-run
```

运行 pilot：

```bash
python3 scripts/s5_semantic_fallback_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-001/metrics.json
outputs/EXP-S4-001/per_sample.csv
outputs/EXP-S4-001/REPORT.md
outputs/EXP-S4-001/exports/snr_XXdb/final/
outputs/EXP-S4-001/samples/snr_01db_original_m0_m1_m3final.png
outputs/EXP-S4-001/samples/snr_07db_original_m0_m1_m3final.png
outputs/EXP-S4-001/samples/snr_19db_original_m0_m1_m3final.png
```

结论：该 fallback 把 all-subset pseudo Final-Failure 从 M1 的 `0.8750/0.9375/0.8750` 降回 M0/M3 的 `0.5000/0.3125/0.0625`，false accept 和 false reject 在当前 48 个样本上均为 0。但它不是完整 M3/Ours，因为仍沿用固定 `strength=0.25` 的负结果 M1；少量 accepted M1 会拉低 PSNR 和 LPIPS。下一步应新建实验 ID 做 `strength <= 0.10` 的 SNR-aware validation 网格，再接这个 fallback 规则。

## S5 SNR-Aware Strength Validation

当前已完成 `EXP-S4-002`：在正式 COCO-256 M0 export 上运行低强度 diffusion validation，覆盖 `[1, 4, 7, 13, 19]` dB，每个 SNR 8 张图。

配置：

```text
configs/s5_snr_adaptive_diffusion_strength_validation.yaml
```

候选：

- `fixed_0p05`：所有 SNR 使用 `strength=0.05`。
- `snr_adaptive_0p10_to_0p05`：1/4/7/13/19 dB 使用 `0.10/0.08/0.06/0.05/0.05`，满足 strength 随 SNR 升高不增加。

先检查输入和 schedule：

```bash
python3 scripts/s5_snr_adaptive_diffusion_validation.py --dry-run
```

运行 validation。该命令应使用本地缓存；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_snr_adaptive_diffusion_validation.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-002/metrics.json
outputs/EXP-S4-002/summary.csv
outputs/EXP-S4-002/per_sample.csv
outputs/EXP-S4-002/REPORT.md
outputs/EXP-S4-002/candidates/fixed_0p05/
outputs/EXP-S4-002/candidates/snr_adaptive_0p10_to_0p05/
```

结论：低强度 diffusion 比 `strength=0.25` 语义更稳，但仍明显损伤图像质量。即使 `strength=0.05`，refined PSNR/LPIPS 仍显著差于 M0；fallback 可降低 final failure，但无法弥补 refined 图像本身的质量损伤。下一步应优先做 SD VAE/latent roundtrip 诊断，判断损伤来自 VAE 重编码、最小 denoise step，还是 prompt-free generative prior。

## S5 SD VAE Roundtrip Diagnostic

当前已完成 `EXP-S4-003`：只加载 Stable Diffusion v1.5 的 VAE，对正式 COCO-256 M0 export 做 encode/decode roundtrip，不运行 UNet denoise，不使用 prompt。

配置：

```text
configs/s5_sd_vae_roundtrip_coco256_awgn.yaml
```

先检查输入和样本对齐：

```bash
python3 scripts/s5_sd_vae_roundtrip_eval.py --dry-run
```

运行诊断。该命令应使用本地缓存；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sd_vae_roundtrip_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-003/metrics.json
outputs/EXP-S4-003/summary.csv
outputs/EXP-S4-003/per_sample.csv
outputs/EXP-S4-003/REPORT.md
outputs/EXP-S4-003/exports/snr_XXdb/m0_vae_roundtrip/
outputs/EXP-S4-003/exports/snr_XXdb/original_vae_roundtrip/
outputs/EXP-S4-003/samples/
```

结论：SD VAE roundtrip 本身已经显著损伤高保真 M0。M0-VAE 相对 M0 的 PSNR 损失从 1 dB 的 `-3.4852` dB 扩大到 19 dB 的 `-7.3260` dB，LPIPS 也变差 `+0.0090` 到 `+0.0578`。这说明当前通用 Stable Diffusion img2img 路线不是简单调低 `strength` 就能变成有效视觉增强；后续应优先考虑 restoration-aware 或 latent-free/像素域保守模块，并继续记录 semantic drift。

## S5 Pixel Residual Restoration Pilot

当前已完成 `EXP-S4-005`：避开 Stable Diffusion 和 SD VAE，只在像素域训练一个小型 SNR-conditioned residual refiner。

配置：

```text
configs/s5_residual_refiner_pilot_coco256_awgn.yaml
```

切分：

- train：`sample_000008.png` 到 `sample_000031.png`，每个 SNR 24 张
- eval：`sample_000000.png` 到 `sample_000007.png`，每个 SNR 8 张

先检查输入和切分：

```bash
python3 scripts/s5_residual_refiner_pilot.py --dry-run
```

运行 pilot。该命令不下载模型或数据；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-005/checkpoints/best.pt
outputs/EXP-S4-005/train_history.csv
outputs/EXP-S4-005/metrics.json
outputs/EXP-S4-005/summary.csv
outputs/EXP-S4-005/per_sample.csv
outputs/EXP-S4-005/REPORT.md
outputs/EXP-S4-005/exports/snr_XXdb/refined/
outputs/EXP-S4-005/exports/snr_XXdb/final/
outputs/EXP-S4-005/samples/
```

结论：这是小样本 pilot，不是最终 M2/M3/Ours，但方向明显比通用 SD img2img 更健康。1/4/7/13/19 dB 上 refined PSNR 相比 M0 分别提升 `+0.3866/+0.1868/+0.0905/+0.1248/+0.1682` dB；LPIPS 除 7 dB 基本持平外均改善；pseudo final failure 没有高于 M0。

注意：`EXP-S4-004` 是同一 pilot 的失败尝试，训练完成后因 `train_history.csv` 字段写入 bug 中断，保留在 `outputs/EXP-S4-004/`，不要复用该实验 ID。

## S5 Pixel Residual Restoration Validation

当前已完成 `EXP-S4-006`：使用更大的 M0 export 训练/验证同一个 SNR-conditioned residual refiner。

配置：

```text
configs/s5_residual_refiner_validation_coco256_awgn.yaml
```

切分：

- train：`sample_000032.png` 到 `sample_000191.png`，每个 SNR 160 张
- eval：`sample_000192.png` 到 `sample_000255.png`，每个 SNR 64 张

先检查输入和切分：

```bash
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --dry-run
```

运行 validation。该命令不下载模型或数据；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --device cuda:0
```

输出：

```text
outputs/EXP-S4-006/checkpoints/best.pt
outputs/EXP-S4-006/train_history.csv
outputs/EXP-S4-006/metrics.json
outputs/EXP-S4-006/summary.csv
outputs/EXP-S4-006/per_sample.csv
outputs/EXP-S4-006/REPORT.md
outputs/EXP-S4-006/exports/snr_XXdb/refined/
outputs/EXP-S4-006/exports/snr_XXdb/final/
outputs/EXP-S4-006/samples/
```

结论：pure refined 在 1/4/7/13/19 dB 上 PSNR 分别提升 `+1.1323/+0.7837/+0.5859/+0.5504/+0.5654` dB，LPIPS 全部改善；经过 top-1 agreement fallback 后，M3 final PSNR 仍提升 `+0.3313/+0.3812/+0.3815/+0.4557/+0.4561` dB，且 pseudo final failure 未高于 M0。低 SNR 下 accept rate 较低，后续应做 detector error analysis，而不能把 pure refined 直接当最终方法。

## S5 Pixel Residual Diffusion Pilot

当前已完成 `EXP-S4-007`：避开 Stable Diffusion、text prompt 和 SD VAE，在像素残差空间训练一个小型 SNR-conditioned DDPM，用来回答“diffusion 是否需要换成 residual-domain 设计”。

配置：

```text
configs/s5_residual_diffusion_pilot_coco256_awgn.yaml
```

切分：

- train：`sample_000032.png` 到 `sample_000111.png`，每个 SNR 80 张
- eval：`sample_000192.png` 到 `sample_000207.png`，每个 SNR 16 张

先检查输入和切分：

```bash
python3 scripts/s5_residual_diffusion_pilot.py --dry-run
```

运行 pilot。该命令不下载模型或数据；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_diffusion_pilot.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-007/checkpoints/best.pt
outputs/EXP-S4-007/train_history.csv
outputs/EXP-S4-007/metrics.json
outputs/EXP-S4-007/summary.csv
outputs/EXP-S4-007/per_sample.csv
outputs/EXP-S4-007/REPORT.md
outputs/EXP-S4-007/exports/snr_XXdb/refined/
outputs/EXP-S4-007/exports/snr_XXdb/final/
outputs/EXP-S4-007/samples/
```

结论：这是明确负结果。Naive DDPM 的 refined PSNR 在 1/4/7/13/19 dB 相比 M0 分别下降 `-7.1634/-7.4843/-7.0882/-5.4204/-4.4217` dB，LPIPS 全部变差；top-1 agreement gate 可把 M3 final failure 拉回 M0 水平，但 M3 final PSNR 仍下降 `-1.4156/-1.6618/-2.6019/-2.1567/-2.1002` dB。后续若继续做 diffusion，应改为从 M0 或 residual CNN 输出附近初始化的短链 conditional restoration diffusion，而不是从随机噪声采样完整残差。

## S5 Semantic Gate Error Analysis

当前已完成 `EXP-S4-006` 的派生 gate error analysis。该流程不跑模型、不联网，只读取：

```text
outputs/EXP-S4-006/per_sample.csv
outputs/EXP-S4-006/exports/
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/exports/
```

运行：

```bash
python3 scripts/s5_analyze_residual_gate_errors.py
```

输出：

```text
outputs/analysis/exp_s4_006_gate_error_analysis/summary.csv
outputs/analysis/exp_s4_006_gate_error_analysis/per_sample_with_case_type.csv
outputs/analysis/exp_s4_006_gate_error_analysis/index.json
outputs/analysis/exp_s4_006_gate_error_analysis/REPORT.md
outputs/analysis/exp_s4_006_gate_error_analysis/*/sheets/
outputs/analysis/exp_s4_006_gate_error_analysis/*/quads/
```

核心结论：当前 gate 是 `c(refined) == c(M0)` 的 top-1 agreement，因此在同一个冻结分类器口径下，M3 final failure 不会超过 M0 是结构性保证；这还不是独立语义可靠性证明。分析中 `protective_reject` 有 28/320 个，说明 gate 确实阻止了一批 refined 改坏 pseudo-label 的情况；`missed_semantic_repair` 有 41/320 个，说明 gate 也拒绝了不少 refined 把 M0 pseudo-label 修回原图 pseudo-label 的样本。下一版 gate 应考虑 top-k、confidence margin 或 CLIP/caption 辅助，允许可信修复，同时保留保护性拒绝。

## S5 Semantic Gate Policy Sweep

当前已完成 `EXP-S4-006` 的派生 gate policy sweep。该流程不训练、不下载，只用本地 AlexNet 权重重新计算 original/M0/refined top-5，然后离线比较 receiver-side gate 策略。

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_residual_gate_policies.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_gate_policy_sweep/topk_predictions.csv
outputs/analysis/exp_s4_006_gate_policy_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_gate_policy_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_gate_policy_sweep/metadata.json
outputs/analysis/exp_s4_006_gate_policy_sweep/REPORT.md
```

核心结论：`top1_equal_or_refined_conf_gain_ge_0p05` 是当前最均衡的候选 gate。相对原始 top-1 agreement gate，全局 final failure 从 `0.3750` 降到 `0.3188`，final PSNR 提升 `+0.1153` dB，missed repair 从 `41` 降到 `20`；代价是 accepted new error 从 `0` 增到 `3`。top-5 overlap 类策略虽然 final PSNR 更高，但 accepted new error 明显更多，风险偏大。该 sweep 是 validation 派生分析，不能直接作为最终 M3 结论。

## S5 Confidence-Gain Gate Auxiliary Audit

当前已完成 `EXP-S4-006` 的 confidence-gain gate 辅助语义审计。该流程不训练、不下载，只读取已有 gate sweep 预测、本地 OpenCLIP ViT-B/32 权重和 COCO captions；gate decision 本身仍只使用 receiver-side 的 M0/refined 分类器预测。

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_residual_gate_aux_semantics.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/per_sample_audit.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/summary.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/new_accepts.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/accepted_new_errors.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/galleries/
```

核心结论：`top1_equal_or_refined_conf_gain_ge_0p05` 新增接受 37/320 个样本，其中 21 个是 pseudo-label repair、3 个是 accepted new error。相对 top-1 gate，final failure 从 `0.3750` 降到 `0.3188`，PSNR 提升 `+0.1153` dB，CLIP image-image 均值略升 `+0.0016`，但 caption CLIP 均值略降 `-0.0007`。该结果支持把它作为下一轮候选 gate，但仍不能直接作为最终 M3。

## S5 Confidence-Gain Gate Candidate Outputs

已将 `top1_equal_or_refined_conf_gain_ge_0p05` 的候选 final PNG 从已有 M0/refined 图像中落盘，方便后续人工审查和 held-out 对照。该流程只复制/选择已有 PNG，不重新训练或生成图像。

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_materialize_residual_gate_policy.py
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/per_sample.csv
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/summary.csv
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/exports/snr_XXdb/final/
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/samples/
```

核心结论：候选 gate final failure 为 `0.3188`，top-1 baseline 为 `0.3750`；candidate final PSNR 为 `32.0966` dB，比 top-1 baseline 高 `+0.1153` dB，比 M0 高 `+0.5164` dB。`samples/accepted_new_error_quads.png` 固化了 3 个 accepted new error，后续必须优先复核。

## S5 Held-Out Confidence-Gain Gate Check

当前也已完成 `EXP-S4-006` 的 held-out confidence-gain gate 复核。该流程不重训模型、不下载数据或权重，只加载 `outputs/EXP-S4-006/checkpoints/best.pt`，在没有参与 `EXP-S4-006` train/eval 的 `sample_000000.png` 到 `sample_000031.png` 上重新生成 refined、top-1 final 和 candidate final。

配置：

```text
configs/s5_residual_refiner_heldout_gate_exp_s4_006.yaml
```

先检查输入和 split：

```bash
python3 scripts/s5_residual_refiner_heldout_gate_eval.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_heldout_gate_check/per_sample.csv
outputs/analysis/exp_s4_006_heldout_gate_check/summary.csv
outputs/analysis/exp_s4_006_heldout_gate_check/new_accepts.csv
outputs/analysis/exp_s4_006_heldout_gate_check/accepted_new_errors.csv
outputs/analysis/exp_s4_006_heldout_gate_check/REPORT.md
outputs/analysis/exp_s4_006_heldout_gate_check/exports/
outputs/analysis/exp_s4_006_heldout_gate_check/samples/
```

核心结论：held-out 上 candidate final failure 为 `0.2812`，top-1 baseline 为 `0.3250`；candidate final PSNR 比 top-1 baseline 高 `+0.1007` dB，比 M0 高 `+0.5460` dB。新增接受 19/160 个样本，其中 9 个是 repair，但仍有 2 个 accepted new error。`samples/accepted_new_error_review.png` 已固化这两个风险样本；候选 gate 方向复现，但还不能写成最终 M3。

## S5 Test-Like Confidence-Gain Gate Check

当前也已完成 `EXP-S4-006` 的 test-like confidence-gain gate 复核。该流程先把正式 M0 export 扩展到每个 SNR 384 张 PNG，然后加载同一个 `outputs/EXP-S4-006/checkpoints/best.pt`，在没有参与 `EXP-S4-006` train/eval/gate sweep 的 `sample_000256.png` 到 `sample_000319.png` 上重新生成 refined、top-1 final 和 candidate final。

先生成不覆盖旧目录的新 M0 export：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 384 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384
```

配置：

```text
configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml
```

先检查输入和 split：

```bash
python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv
outputs/analysis/exp_s4_006_testlike_gate_check/summary.csv
outputs/analysis/exp_s4_006_testlike_gate_check/new_accepts.csv
outputs/analysis/exp_s4_006_testlike_gate_check/accepted_new_errors.csv
outputs/analysis/exp_s4_006_testlike_gate_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_gate_check/exports/
outputs/analysis/exp_s4_006_testlike_gate_check/samples/
```

核心结论：test-like 上 candidate final failure 为 `0.4313`，top-1 baseline 为 `0.4719`；candidate final PSNR 比 top-1 baseline 高 `+0.0814` dB，比 M0 高 `+0.4927` dB。新增接受 26/320 个样本，其中 17 个是 repair，但仍有 4 个 accepted new error。`samples/accepted_new_error_review.png` 已固化这些风险样本；raw confidence-gain gate 的收益复现，但语义风险更明确，不能写成最终 M3。

## S5 Confidence-Gain CLIP Veto Sweep

当前已完成 `EXP-S4-006` confidence-gain gate 的 receiver-side CLIP 二级 veto 扫描。该流程不训练、不下载，只读取 validation/held-out CSV、已有 M0/refined PNG 和本地 OpenCLIP ViT-B/32 权重；veto 决策只使用 `CLIP(M0, refined)`，不看 original 或 caption。

配置：

```text
configs/s5_conf_gain_clip_veto_sweep_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s5_sweep_conf_gain_clip_veto.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_conf_gain_clip_veto.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/per_sample_with_clip.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/joint_policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/galleries/
```

核心结论：全局最保守可用阈值为 `CLIP(M0, refined) >= 0.98`。它在 validation 和 held-out 上都把 accepted new error 压到 0，但只保留 2 个 repair，总 PSNR 相比 top-1 gate 仅提升 `+0.0073` dB。该 veto 是安全参考，不是最终 M3；下一步应做 SNR-calibrated threshold、classifier ensemble 或轻量 receiver-side risk predictor。

## S5 SNR-Calibrated CLIP Veto

当前已完成 `EXP-S4-006` confidence-gain CLIP veto 的 SNR 校准派生分析。该流程不训练、不下载、不重算 CLIP，只读取上一节 sweep 生成的 `per_sample_with_clip.csv`，在 validation 上选择阈值 schedule，并在 held-out 上做风险复核。

配置：

```text
configs/s5_conf_gain_clip_veto_snr_calibration_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/calibrated_schedules.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/galleries/
```

核心结论：validation-only independent schedule 为 `1/4/7/13/19 dB = 0.96/no_veto/0.98/no_veto/no_veto`，在 validation 上保留 10 个 repair 且 0 accepted new error，但 held-out 仍漏 1 个 new error。monotonic schedule 为 `0.98/0.98/0.98/no_veto/no_veto`，held-out 安全但只保留 1 个 repair，几乎退回全局 `0.98`。因此单一 `CLIP(M0, refined)` 标量阈值即使按 SNR 校准，也不足以作为最终 M3；下一步应转向 classifier ensemble 或轻量 receiver-side risk predictor。

## S5 Confidence-Gain Risk Rule Sweep

当前已完成 `EXP-S4-006` confidence-gain gate 的 receiver-side risk-rule sweep。该流程不训练、不下载、不重算 CLIP，只读取已有 validation/held-out 的 CLIP sweep CSV 和 M0/refined top-k classifier CSV，在 validation 上搜索透明规则，并在 held-out 上做风险复核。

配置：

```text
configs/s5_conf_gain_risk_rule_sweep_exp_s4_006.yaml
```

先检查输入和网格规模：

```bash
python3 scripts/s5_sweep_conf_gain_risk_rules.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_conf_gain_risk_rules.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/rule_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/selected_rule.json
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/galleries/
```

核心结论：选中的 shadow-margin risk rule 在 validation 上 final failure `0.3156`、PSNR `+0.0953` dB vs top-1、19 repair、0 accepted new error；held-out 上 final failure `0.2812`、PSNR `+0.0748` dB vs top-1、7 repair、0 accepted new error。它挡掉 raw confidence-gain 的 held-out 2 个新错，同时比全局 `CLIP >= 0.98` 多保留 6 个 held-out repair。当前它是最强 M3 gate 候选，但仍需冻结规则并在正式 test split 或更大 held-out 上复核。

## S5 Selected Risk-Rule Candidate Outputs

已将 `selected_risk_rule` 的 final PNG 和 per-sample 决策落盘，便于后续人工复查和正式 split 复核。该流程不训练、不联网、不重算 CLIP/分类器，只读取 risk-rule sweep 的 `policy_decisions.csv` 并复制已有 M0/refined PNG。

配置：

```text
configs/s5_materialize_risk_rule_gate_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s5_materialize_risk_rule_policy.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_materialize_risk_rule_policy.py
```

输出：

```text
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/per_sample.csv
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/summary.csv
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/REPORT.md
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/metadata.json
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/exports/{validation,heldout}/snr_XXdb/final/
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/samples/
```

核心结论：共导出 480 张 final PNG。validation 上 final failure `0.3156`、PSNR `+0.0953` dB vs top-1、19 repair、0 new error；held-out 上 final failure `0.2812`、PSNR `+0.0748` dB vs top-1、7 repair、0 new error。该 artifact 只固化当前候选，不把 pseudo-label validation/held-out 结果包装成最终 M3。

## S5 Selected Risk-Rule Classifier Ensemble Audit

已用多个冻结 ImageNet 分类器对固定 `selected_risk_rule` 决策做离线复核。该流程不重新搜索 gate，ensemble 也不参与 receiver-side decision；它只用于检查 AlexNet-tuned 规则是否跨分类器仍然稳。

配置：

```text
configs/s5_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml
```

先检查输入和本地权重缓存：

```bash
python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --dry-run
```

首次运行若缺 ResNet18/MobileNetV3-Small 权重，需要按项目流量规则清空代理变量，从 PyTorch/torchvision 官方 model zoo 直连下载约 `44.7MB + 9.83MB` 权重：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --device cuda:0 --allow-download
```

输出：

```text
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/per_model_per_sample.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/model_summary.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/vote_summary.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/REPORT.md
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/galleries/
```

核心结论：AlexNet 口径下 `selected_risk_rule` 仍保持 validation/held-out `0/0` new error；但 ResNet18 和 MobileNetV3-Small 暴露了跨模型风险。validation/held-out 分别有 `26/15` 个样本被至少一个分类器标为 selected accepted new error，多数票 new error 为 `2/1` 个。因此当前 gate 仍是候选，不是跨语义模型安全的最终 M3。

## S5 Ensemble-Risk Veto Sweep

已在 `selected_risk_rule` 之上完成 classifier-ensemble 风险驱动的二级 veto 扫描。该流程不训练、不联网、不下载、不重算分类器，只读取已经 materialize 的 selected-risk-rule 决策、classifier ensemble audit 的投票 CSV 和已有 PNG；搜索目标是在 validation 上清零多数票 accepted-new-error，同时尽量保留 repair。

配置：

```text
configs/s5_ensemble_risk_veto_sweep_exp_s4_006.yaml
```

先检查输入和网格规模：

```bash
python3 scripts/s5_sweep_ensemble_risk_veto.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_ensemble_risk_veto.py
```

若重跑并覆盖已有派生输出，追加 `--overwrite`。

输出：

```text
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/rule_candidates.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/selected_rule.json
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/REPORT.md
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/galleries/
```

核心结论：选中的二级 veto 为 `new_accept_refined_margin <= 0.005`，以及 top-1-equal 接受中 `refined_conf_gain_vs_m0 <= 0.05` 且 `m0_top1_margin >= 0.10` 时回退 M0。它把 validation/held-out 的多数票 new error 从 `2/1` 清到 `0/0`，但很保守：额外 veto `96/58` 张，remaining any-new-error 仍为 `16/8`，remaining majority repair 为 `5/4`，PSNR 相对 `selected_risk_rule` 回吐 `-0.1834/-0.2538` dB。因此它是风险收敛证据，不是最终 M3。

## S5 Receiver-Side Risk Score Sweep

已完成 `selected_risk_rule` 之上的透明 receiver-side risk score 扫描。该流程不训练、不联网、不下载、不重算分类器，只读取 selected-risk-rule 决策、classifier ensemble audit 投票和已有 PNG；目标是测试是否能用较少额外 veto 替代上一节很保守的二级 veto。

配置：

```text
configs/s5_receiver_risk_score_sweep_exp_s4_006.yaml
```

先检查输入和候选规模：

```bash
python3 scripts/s5_sweep_receiver_risk_score.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_receiver_risk_score.py
```

若重跑并覆盖已有派生输出，追加 `--overwrite`。

输出：

```text
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/score_candidates.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/selected_score.json
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/REPORT.md
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/galleries/
```

核心结论：repair-pref validation 目标选择了 `low_overlap_rank` 分数，权重为 `low_top5_overlap + refined_top1_not_in_m0_safe_rank + low_clip`，阈值 `0.444446`。它在 validation 上只额外 veto `48` 张并清零多数票 new error，但 held-out 仍漏 `1` 个多数票 new error；validation/held-out 的 majority repair 也只剩 `4/2`，PSNR 相对 `selected_risk_rule` 回吐 `-0.1396/-0.1581` dB。候选表显示，若要求 validation 和 held-out 同时清零多数票 new error，最好的 score 模板需要额外 veto `143/81` 张，甚至比上一节的保守二级 veto 更重。因此轻量 risk score 暂不适合作为最终 gate，应转向更正式 split 或更强的语义风险模型。

## S5 Test-Like Frozen Risk-Rule Check

已把冻结的 `selected_risk_rule` 和保守 ensemble-risk veto 应用到 `sample_000256.png` 到 `sample_000319.png` test-like split。该流程不训练、不联网、不下载、不重新调阈值，只读取 test-like raw confidence-gain 复核 CSV、旧的 `selected_rule.json`、旧的保守 veto rule，并重新计算本地 `CLIP(M0, refined)`。

配置：

```text
configs/s5_testlike_risk_rule_check_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s5_apply_testlike_risk_rules.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_apply_testlike_risk_rules.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_risk_rule_check/per_sample_with_clip.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_decisions.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_by_snr.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_risk_rule_check/exports/
outputs/analysis/exp_s4_006_testlike_risk_rule_check/galleries/
```

核心结论：`selected_risk_rule` 在 test-like 上把 raw confidence-gain 的 accepted new error 从 `4` 降到 `1`，保留 `10` 个 repair，PSNR 比 top-1 gate 高 `+0.0434` dB；但它没有清零风险。保守 ensemble veto 没有减少剩余 new error，且 PSNR 相比 `selected_risk_rule` 回吐 `-0.1902` dB。剩余风险样本是 13 dB `sample_000312.png`，也是 AlexNet pseudo-label 较吵的 case；当前浅层 receiver-side rule 仍不能写成最终 M3。

## S5 Test-Like Classifier-Ensemble Audit

已用 AlexNet、ResNet18 和 MobileNetV3-Small 三个冻结 ImageNet 分类器离线审计 test-like `selected_risk_rule`。该流程不训练、不联网、不下载，分类器权重来自本地 cache；ensemble 只用于离线风险审计，不参与 receiver-side decision。

配置：

```text
configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml
```

先检查输入和本地权重：

```bash
python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/per_model_per_sample.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/model_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/vote_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/REPORT.md
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/galleries/
```

核心结论：test-like 上 `selected_risk_rule` 没有 majority-vote accepted new error，优于 validation/held-out 审计中暴露的 `2/1` 多数票新错；但仍有 `23/320` 张被至少一个分类器标为 accepted new error。按分类器看，AlexNet/ResNet18/MobileNetV3-Small 的 selected failure 分别为 `0.4437/0.4344/0.5406`，repair 为 `10/31/32`，new error 为 `1/13/9`。因此该规则在 test-like 上不是多数票语义灾难，但也不是跨模型完全安全；下一步应转向带标签 clean-correct 评估或 semantic-risk-aware 训练，而不是继续只调浅层阈值。

## S5 Test-Like COCO Object CLIP Clean-Correct Eval

已完成 test-like split 上的 COCO object clean-correct 辅助诊断。该流程不训练、不联网、不下载，读取 test-like gate 决策、COCO `instances_val2017.json`、正式 M0 export manifest 和本地 OpenCLIP ViT-B/32 权重；先用 COCO instance 面积找每张图的 dominant object label，再要求 CLIP 对 original 的 80 类 COCO zero-shot top-1 与该 label 一致，形成辅助 clean-correct 子集。

配置：

```text
configs/s5_testlike_coco_object_clip_clean_eval_exp_s4_006.yaml
```

先检查输入和样本规模：

```bash
python3 scripts/s5_coco_object_clip_clean_eval.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_coco_object_clip_clean_eval.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/per_sample.csv
outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/summary.csv
outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/by_snr.csv
outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/label_audit.csv
outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/REPORT.md
outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/galleries/
```

核心结论：64 个 test-like 原图中有 55 个满足 dominant COCO object 面积规则，其中 27 个 original 通过 CLIP clean-correct 过滤，形成 135 行 clean-correct 统计。`selected_risk_rule` 在该子集上 final failure 与 top-1 gate 相同，均为 `0.0815`，PSNR 高 `+0.0257` dB，有 `1` 个 GT-like repair 和 `2` 个 GT-like new error；`selected_risk_rule_plus_ensemble_veto` 把 new error 降到 `0`，final failure 降到 `0.0741`，但 repair 也降到 `0`，PSNR 相比 top-1 低 `-0.1727` dB。这个结果继续支持当前判断：更保守的 veto 能保护语义，但会明显牺牲 restoration 收益；COCO object CLIP 只是辅助 clean-correct 诊断，不替代真正带标签 ImageNet/Imagenette 评估。

## S6 Minimal Closure Report

已生成第一版最小闭环汇总报告，并已刷新纳入 residual shrink、adaptive alpha、two-stage alpha 和 receiver alpha predictor M3 候选/消融。该流程不训练、不推理、不分类、不联网，只读取已有 metrics/CSV，把 M0、M1 负结果、`EXP-S4-006` residual M2/M3、residual strength/alpha policy 和 test-like 语义审计汇总到同一个报告里。

配置：

```text
configs/s6_minimal_closure_report.yaml
```

先检查输入：

```bash
python3 scripts/s6_make_minimal_closure_report.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py
```

刷新同一派生输出目录：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

输出：

```text
outputs/analysis/minimal_closure_report/REPORT.md
outputs/analysis/minimal_closure_report/method_closure_summary.csv
outputs/analysis/minimal_closure_report/residual_per_snr_quality_semantics.csv
outputs/analysis/minimal_closure_report/blind_diffusion_negative_reference.csv
outputs/analysis/minimal_closure_report/residual_shrink_policy_tradeoff.csv
outputs/analysis/minimal_closure_report/adaptive_residual_alpha_policy_tradeoff.csv
outputs/analysis/minimal_closure_report/two_stage_residual_alpha_policy_tradeoff.csv
outputs/analysis/minimal_closure_report/receiver_alpha_predictor_tradeoff.csv
outputs/analysis/minimal_closure_report/testlike_policy_tradeoff.csv
outputs/analysis/minimal_closure_report/coco_object_clean_correct_tradeoff.csv
outputs/analysis/minimal_closure_report/figures/
```

核心结论：`M1-BlindDiffusion-SDImg2Img` 保留为负参考，平均 PSNR 相比其 M0 输入下降 `-14.7485` dB、LPIPS 变差 `+0.3877`；`M2-SNRConditionedPixelResidualRestoration` 是正向 restoration anchor，`EXP-S4-006` 上平均 PSNR `+0.7235` dB、LPIPS `-0.0274`；`M3-ResidualRestorationTop1Fallback` 可作为保守第一版闭环，平均 PSNR `+0.4011` dB、LPIPS `-0.0104`，且同一 pseudo-label 口径下 semantic failure 不高于 M0。`M3-ResidualRestorationTop1ShrinkFallback` 是固定 schedule 保守候选：validation 平均 PSNR delta `+0.4584` dB，frozen held-out/test-like 平均 PSNR delta `+0.4689/+0.4552` dB，held-out/test-like accepted new error 均为 0。`M3-AdaptiveResidualAlphaTop1Fallback` 是当前最强保守候选：validation/held-out/test-like PSNR delta 为 `+0.5584/+0.5664/+0.5691` dB，accepted new error 为 `0/0/0`；但 repair 仍为 0。`M3-TwoStageResidualAlphaTop1Fallback` 是部署性消融：validation/held-out/test-like PSNR delta 为 `+0.4831/+0.5009/+0.4875` dB，new error 仍为 `0/0/0`，但低于 exhaustive adaptive alpha。`M3-ReceiverAlphaPredictorTop1Fallback` 是 learned 部署 pilot：validation/held-out/test-like PSNR delta 为 `+0.5584/+0.5099/+0.4871` dB，new error 为 `0/0/0`，接近 two-stage 但仍低于 exhaustive adaptive alpha。`selected_risk_rule` 仍只能作为候选/消融，因为 test-like 和 COCO-object clean-correct 诊断还留有 new-error 风险。

## S6 Residual Shrink Selection

已完成 `EXP-S4-006` 的 residual-strength alpha shrink 派生分析。该流程不训练、不运行 diffusion、不下载，只读取已有 original/M0/refined PNG，构造：

```text
x_alpha = clamp(m0 + alpha * (refined - m0), 0, 1)
```

配置：

```text
configs/s6_residual_shrink_selection_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s6_residual_shrink_selection.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --device cuda:0
```

如需覆盖同一派生输出目录：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --device cuda:0 --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_residual_shrink_selection/REPORT.md
outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv
outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv
outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json
outputs/analysis/exp_s4_006_residual_shrink_selection/alpha_tradeoff.png
outputs/analysis/exp_s4_006_residual_shrink_selection/samples/
```

核心结论：validation-only top-1 fallback shrink schedule 选择 `1 dB alpha=0.5`、其余 SNR `alpha=0.75`，平均 PSNR delta 从 full-strength top-1 fallback 的 `+0.4011` dB 提升到 `+0.4584` dB，LPIPS delta 从 `-0.0104` 改到 `-0.0153`，pseudo final failure 仍不高于 M0。always-accept 虽然 PSNR 更高且平均 failure 可低于 M0，但仍包含 19-28 个 accepted new error，不能作为最终 M3。

## S6 Held-Out Frozen Residual Shrink Schedule Check

已完成 frozen residual shrink schedule 的 held-out 复核。该流程读取 validation shrink selection 的 `selected_schedule.json` 和 held-out gate check 已有 refined PNG，不训练、不运行 diffusion、不下载，也不在 held-out 上重新选择 alpha。

配置：

```text
configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml
```

先检查 frozen schedule 与输入：

```bash
python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/REPORT.md
outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/summary.csv
outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/per_sample.csv
outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/metadata.json
outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/samples/
```

核心结论：frozen validation top-1 shrink schedule 在 held-out 上平均 PSNR delta 为 `+0.4689` dB，比 full-strength top-1 fallback 的 `+0.4454` dB 高 `+0.0236` dB；LPIPS delta 为 `-0.0150`，pseudo final failure 仍等于 M0，accepted new error 为 0。always-accept full strength / validation always-constrained schedule 分别仍有 10/3 个 accepted new error，不能作为最终 M3。

## S6 Test-Like Frozen Residual Shrink Schedule Check

已完成 frozen residual shrink schedule 的 test-like 复核。该流程读取 validation shrink selection 的 `selected_schedule.json` 和 test-like gate check 已有 refined PNG，不训练、不运行 diffusion、不下载，也不在 test-like 上重新选择 alpha。

配置：

```text
configs/s6_testlike_residual_shrink_schedule_check_exp_s4_006.yaml
```

先检查 frozen schedule 与输入：

```bash
python3 scripts/s6_apply_residual_shrink_schedule.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv
outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv
outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/metadata.json
outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/samples/
```

核心结论：frozen validation top-1 shrink schedule 在 test-like 上平均 PSNR delta 为 `+0.4552` dB，比 full-strength top-1 fallback 的 `+0.4113` dB 高 `+0.0439` dB；LPIPS delta 为 `-0.0152`，pseudo final failure 仍等于 M0，accepted new error 为 0。always-accept full strength / validation always-constrained schedule 分别仍有 25/12 个 accepted new error，不能作为最终 M3。

## S6 Residual Shrink M3 Artifact Gallery

已完成 residual shrink M3 的统一 artifact gallery。该流程只读取 validation、held-out、test-like 的 shrink summary/per-sample CSV 和已有 PNG，不训练、不运行 diffusion、不重算分类器、不下载，也不重新选择 alpha。

配置：

```text
configs/s6_residual_shrink_artifact_gallery_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s6_make_residual_shrink_gallery.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_residual_shrink_gallery.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/REPORT.md
outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/policy_summary.csv
outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/case_counts.csv
outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/case_index.csv
outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/samples/
```

核心结论：selected shrink M3 在 validation/held-out/test-like 上 PSNR delta 为 `+0.4584/+0.4689/+0.4552` dB，accepted new error 为 `0/0/0`；always-accept full strength 的 accepted new error 为 `28/10/25`，validation-constrained always-accept 仍有 `19/3/12` 个 new error。该目录提供 safe accept、protective reject、rejected good candidate 和 unsafe new-error 的样例 sheet，适合作为第一版 failure-case / reliability 小节素材。

## S6 Adaptive Residual Alpha Policy

已完成 per-sample adaptive residual alpha policy 派生分析。该流程只读取 validation、held-out、test-like 已有 alpha candidate PNG、本地 AlexNet 和 LPIPS 权重，不训练、不运行 diffusion、不重新生成 residual、不下载，也不在 held-out/test-like 上调参。

核心规则：

```text
adaptive_max_top1_consistent_alpha:
  choose largest alpha in [1.0, 0.75, 0.5, 0.25]
  if candidate top-1 == M0 top-1
  else fallback to M0
```

配置：

```text
configs/s6_adaptive_residual_alpha_policy_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/REPORT.md
outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv
outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv
outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/metadata.json
outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/samples/
```

核心结论：adaptive max top-1-consistent alpha 在 validation/held-out/test-like 上 PSNR delta 为 `+0.5584/+0.5664/+0.5691` dB，accepted new error 为 `0/0/0`，强于 fixed shrink schedule 的 `+0.4584/+0.4689/+0.4552` dB。它仍没有 repair，missed repair 为 `45/31/70`，因此当前定位是更强的保守质量增强候选，而不是语义修复方法。

## S6 Two-Stage Residual Alpha Policy

已完成 two-stage residual alpha policy 派生分析。该流程只读取 adaptive alpha 的已有 `summary.csv` / `per_sample.csv` 和 final 图像路径，不重算分类器、不训练、不运行 diffusion、不加载 LPIPS、不下载。设计目标是把 exhaustive adaptive alpha 的四候选枚举压缩成更接近部署的两阶段接收端策略。

核心规则：

```text
full_then_fixed_schedule:
  try top1_full_strength
  if alpha=1.0 candidate top-1 == M0 top-1, accept full strength
  else try frozen fixed_validation_top1_shrink_schedule
  if fixed candidate top-1 == M0 top-1, accept fixed candidate
  else fallback to M0
```

配置：

```text
configs/s6_two_stage_residual_alpha_policy_exp_s4_006.yaml
```

先检查输入：

```bash
python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/REPORT.md
outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv
outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/per_sample.csv
outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/metadata.json
outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/figures/two_stage_policy_tradeoff.png
```

核心结论：two-stage policy 在 validation/held-out/test-like 上 PSNR delta 为 `+0.4831/+0.5009/+0.4875` dB，accepted new error 为 `0/0/0`，比 fixed schedule 的 `+0.4584/+0.4689/+0.4552` dB 略好，但低于 exhaustive adaptive alpha 的 `+0.5584/+0.5664/+0.5691` dB。它适合作为“少候选检查的部署折中”消融，不替代当前最强的 adaptive alpha。

## S6 Receiver Alpha Predictor

已完成 receiver-side alpha predictor pilot。该流程读取 adaptive alpha 决策表和候选图，在 validation 上训练一个很小的 tabular predictor，特征只包含接收端可见信息：SNR、M0/full candidate 的 classifier confidence、full candidate 是否与 M0 top-1 一致，以及 M0 到 full candidate 的 residual 图像统计。评估时仍对预测 alpha 的候选图执行 top-1 fallback，因此它是 learned deployability pilot，不是语义修复方法。

配置：

```text
configs/s6_receiver_alpha_predictor_exp_s4_006.yaml
```

benefit-aware follow-up 配置：

```text
configs/s6_benefit_alpha_predictor_exp_s4_006.yaml
```

先检查输入和本地权重：

```bash
python3 scripts/s6_train_receiver_alpha_predictor.py --dry-run
```

运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --device cuda:0
```

benefit-aware 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --config configs/s6_benefit_alpha_predictor_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_receiver_alpha_predictor/REPORT.md
outputs/analysis/exp_s4_006_receiver_alpha_predictor/summary.csv
outputs/analysis/exp_s4_006_receiver_alpha_predictor/per_sample.csv
outputs/analysis/exp_s4_006_receiver_alpha_predictor/features.csv
outputs/analysis/exp_s4_006_receiver_alpha_predictor/model_metadata.json
outputs/analysis/exp_s4_006_receiver_alpha_predictor/training_history.csv
outputs/analysis/exp_s4_006_receiver_alpha_predictor/figures/receiver_alpha_predictor_tradeoff.png
outputs/analysis/exp_s4_006_benefit_alpha_predictor/REPORT.md
outputs/analysis/exp_s4_006_benefit_alpha_predictor/summary.csv
outputs/analysis/exp_s4_006_benefit_alpha_predictor/per_sample.csv
outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv
outputs/analysis/exp_s4_006_benefit_alpha_predictor/model_metadata.json
```

核心结论：receiver predictor 在 validation/held-out/test-like 上 PSNR delta 为 `+0.5584/+0.5099/+0.4871` dB，accepted new error 为 `0/0/0`。它在 validation 上完全拟合 adaptive alpha pseudo target，held-out 比 two-stage 略高，test-like 与 two-stage 基本持平，但仍低于 exhaustive adaptive alpha。benefit-aware follow-up 把训练目标换成 validation-derived safe-PSNR utility soft labels，在 validation 上几乎追上 adaptive alpha（`+0.5538` dB），但 held-out/test-like 只有 `+0.4474/+0.4627` dB，低于 two-stage 和原 receiver predictor。这说明收益/风险目标更贴近问题，但当前 tabular receiver 特征泛化不足；下一步应把 alpha/risk 控制放进 residual CNN joint fine-tune 或模型内部特征，而不是继续只换浅层 predictor loss。

## S6 Alpha-Head Residual Refiner Pilot

已完成第一版训练侧 alpha head 探索。该流程加载 `EXP-S4-006` residual refiner checkpoint，默认冻结 residual CNN，只训练一个附着在 refiner feature 上的 alpha head；训练目标可以来自 validation split 的 `adaptive_max_top1_consistent_alpha` pseudo target，也可以来自 benefit-aware predictor feature table 中的 safe-PSNR utility alpha。评估时仍使用 AlexNet top-1 fallback，因此这是训练侧探索，不是新的 M3 闭环。

普通 CE 配置：

```text
configs/s6_alpha_head_residual_refiner_pilot_exp_s4_006.yaml
```

class-weighted follow-up 配置：

```text
configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml
```

benefit-aware follow-up 配置：

```text
configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml
```

joint fine-tune follow-up 配置：

```text
configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml
```

tail-only partial fine-tune follow-up 配置：

```text
configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml
```

tail-only continuous-alpha follow-up 配置：

```text
configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml
```

continuous-alpha LPIPS / classifier-ensemble 审计配置：

```text
configs/s6_continuous_alpha_tail_refiner_audit_exp_s4_006.yaml
```

先检查输入和本地权重：

```bash
python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
```

普通 CE 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --device cuda:0
```

weighted 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml --device cuda:0
```

benefit-aware 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml --device cuda:0
```

joint fine-tune 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --device cuda:0
```

tail-only partial fine-tune 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --device cuda:0
```

tail-only continuous-alpha 运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml --device cuda:0
```

continuous-alpha LPIPS / classifier-ensemble 审计：

```bash
python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/REPORT.md
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/summary.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/per_sample.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/train_history.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/metadata.json
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/figures/alpha_head_tradeoff.png
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/REPORT.md
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/summary.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/per_sample.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/train_history.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/metadata.json
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/REPORT.md
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/summary.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/per_sample.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/train_history.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/metadata.json
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/REPORT.md
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/summary.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/per_sample.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/train_history.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/metadata.json
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/REPORT.md
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/summary.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/per_sample.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/train_history.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/metadata.json
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/REPORT.md
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/summary.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/per_sample.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/train_history.csv
outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/metadata.json
outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/REPORT.md
outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/quality_summary.csv
outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/model_summary.csv
outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/vote_summary.csv
outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/metadata.json
```

核心结论：alpha-head pilot 在 validation/held-out/test-like 上 PSNR delta 为 `+0.3846/+0.3808/+0.3623` dB，accepted new error 为 `0/0/0`，但低于 full-strength top-1 fallback、two-stage policy 和 receiver predictor。weighted follow-up 使用 tempered inverse-frequency CE 后，PSNR delta 变为 `+0.3851/+0.3506/+0.3166` dB，说明类别不均衡不是主因。benefit-aware follow-up 把目标换成 safe-PSNR utility alpha 后，PSNR delta 变为 `+0.4251/+0.4192/+0.3530` dB，new error 仍为 `0/0/0`，相对普通/weighted alpha-head 有部分进展，但仍低于 receiver predictor、two-stage policy 和 exhaustive adaptive alpha；预测分布仍几乎不使用 `alpha=0.25`。

joint fine-tune follow-up 解冻 residual CNN，并让 soft-alpha / target-alpha MSE 反传到 refiner。它把 validation target accuracy 提到 `0.7719`，预测分布也开始覆盖 `alpha=0.25/0.5`，但 restoration anchor 被破坏，PSNR delta 只有 `+0.3294/+0.2303/+0.1869` dB。结论是 benefit/risk 目标能改善 alpha 分类，但全量 unfreeze 且 CE 主导会损伤 residual restoration；下一步应 partial fine-tune（只调 tail/amplitude/head）或 reconstruction-dominant loss，而不是直接全量 joint CE。

tail-only partial fine-tune follow-up 只训练 residual tail 和 alpha head，冻结 head/body，并把 loss 改成 reconstruction-dominant。它在 validation/held-out/test-like 上 PSNR delta 为 `+0.4749/+0.4552/+0.4061` dB，accepted new error 为 `0/0/0`，明显好于冻结 benefit alpha-head 和全量 joint；full-strength top-1 fallback 也恢复到 `+0.4454/+0.4820/+0.4259` dB。结论是 partial/reconstruction-dominant 方向成立，但仍低于 receiver predictor、two-stage policy 和后验 adaptive alpha，暂不升级为最终 M3。

tail-only continuous-alpha follow-up 把 5 类 alpha 分类改为单个连续 alpha regression，仍只训练 residual tail 和 alpha head。它在 validation/held-out/test-like 上 PSNR delta 达到 `+0.5010/+0.5049/+0.5012` dB，accepted new error 为 `0/0/0`，超过离散 tail-only alpha head，并在 held-out/test-like 上达到或超过 two-stage policy 与 receiver predictor。补充审计显示 continuous-alpha 的 LPIPS delta 为 `-0.0149/-0.0149/-0.0162`，优于同 checkpoint full-strength top-1 fallback；但 classifier ensemble 下 any-classifier new error 为 `17/9/14`，majority-vote new error 为 `1/0/0`。该结果是当前训练侧 amplitude-control 最明确的正向突破，但仍低于后验 adaptive alpha upper bound，且不能声明跨模型完全安全，因此暂不直接升级最终 M3。

## SGD-inspired edge-conditioned residual refiner

`EXP-S4-008` 在 residual CNN 输入中加入 receiver-visible 结构条件：从 M0 重建图计算 `sobel_magnitude` 和 `laplacian_abs`，再与 RGB M0、SNR map 一起输入 residual refiner。该设计借鉴 SGD-JSCC 的 edge/structure guidance，但不使用原图 edge，因此不引入接收端不可见信息。

validation 训练与评估：

```bash
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips
```

capacity/training-budget matched controls：

```bash
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_capacity_matched_no_edge_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips --dry-run
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_small_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_capacity_matched_no_edge_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_small_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips
python3 scripts/s6_compare_edge_capacity_ablation.py --config configs/s6_edge_capacity_ablation_exp_s4_006_008_009_010.yaml
```

满足 `gate×alpha` 随 SNR 非增约束的 validation selection：

```bash
python3 scripts/s6_residual_shrink_selection.py --config configs/s6_edge_monotonic_residual_shrink_selection_exp_s4_008.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --config configs/s6_edge_monotonic_residual_shrink_selection_exp_s4_008.yaml --device cuda:0
```

冻结到 held-out / test-like / fresh-holdout，并做独立分类器审计：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_edge_residual_refiner_heldout_gate_exp_s4_008.yaml --device cuda:0 --skip-lpips
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_edge_residual_refiner_testlike_gate_exp_s4_008.yaml --device cuda:0 --skip-lpips
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_edge_residual_refiner_fresh_holdout_gate_exp_s4_008.yaml --device cuda:0 --skip-lpips
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_edge_monotonic_heldout_residual_shrink_schedule_check_exp_s4_008.yaml --device cuda:0
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_edge_monotonic_testlike_residual_shrink_schedule_check_exp_s4_008.yaml --device cuda:0
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_edge_monotonic_fresh_holdout_residual_shrink_schedule_check_exp_s4_008.yaml --device cuda:0
python3 scripts/s6_compare_matched_edge_holdouts.py --config configs/s6_matched_edge_holdout_audit_exp_s4_008_009.yaml
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_audit_residual_policy.py --config configs/s6_edge_monotonic_policy_ensemble_audit_exp_s4_008.yaml --device cuda:0 --skip-lpips --skip-quality-metrics
```

核心结果：2×2 paired bootstrap 证明 edge 的独立 raw PSNR 增益在 small/large 模型上为 `+0.0501/+0.1389` dB，95% CI 均排除 0；large matched pair 在 validation/held-out/test-like/fresh-holdout 上净增 `+0.1389/+0.1565/+0.1585/+0.1411` dB，所有 split 的 5 个 SNR 同向。单调 frozen schedule 在四段上的 PSNR delta 为 `+0.5734/+0.6128/+0.5700/+0.5668` dB，LPIPS delta 为 `-0.0145/-0.0148/-0.0163/-0.0162`。但 ensemble majority new error 为 `1/1/0/3`，因此当前应写成“结构条件带来稳定质量/感知收益，但 raw/跨模型语义风险仍需控制”，不能写成跨模型完全安全。

可直接用于组会/论文讨论的受控结果总结见 `reports/edge_conditioning_significant_result_2026-07-10.md`。

## 项目进度可视化汇总

可从已有 metrics、CSV 和 failure gallery 生成一套派生总览报告；该流程不跑训练、不跑 diffusion、不重新计算模型指标：

```bash
python3 scripts/s4_make_project_progress_visual_summary.py
```

输出：

```text
outputs/analysis/project_progress_visual_summary/REPORT.md
outputs/analysis/project_progress_visual_summary/summary.json
outputs/analysis/project_progress_visual_summary/coco256_m0_snr_sweep.csv
outputs/analysis/project_progress_visual_summary/m1_blind_diffusion_summary.csv
outputs/analysis/project_progress_visual_summary/figures/stage_progress.png
outputs/analysis/project_progress_visual_summary/figures/m0_snr_curves.png
outputs/analysis/project_progress_visual_summary/figures/m1_quality_metrics.png
outputs/analysis/project_progress_visual_summary/figures/m1_semantic_diagnostics.png
outputs/analysis/project_progress_visual_summary/figures/m1_negative_deltas.png
outputs/analysis/project_progress_visual_summary/figures/representative_visual_outputs.png
```

该报告适合快速查看当前项目进度、正式 M0 COCO-256 baseline、M1 负结果和已有 semantic drift failure case。

## Imagenette 严格监督语义审计（2026-07-10）

主实验入口：

```bash
python3 scripts/s6_train_imagenette_scratch_classifiers.py --config configs/s6_imagenette_supervised_clean_eval.yaml --device cuda:0
python3 scripts/s6_imagenette_supervised_clean_eval.py --config configs/s6_imagenette_supervised_clean_eval.yaml --split policy_dev --device cuda:0
```

该 protocol 使用官方 WNID 真值、随机初始化的独立 `G_gate`/`T_cls`、严格 `cls_train/cls_cal/policy_dev/official-val` 隔离，并模拟 PNG 量化。policy-dev 结果为：M2 edge scheduled 相对 M0 的 clean-correct failure 下降 `2.02 pp`、PSNR `+0.7434 dB`；当前 M3 top-1 fallback 相对 M2 failure 上升 `0.7857 pp`，accepted-new-error 保守上界 `1.0795%`，超过预注册 `0.5%`，所以 official val 保持封存，M3 不称为 supervised-safe。完整结果见 `outputs/analysis/imagenette_supervised_policy_dev/REPORT.md` 和 `reports/imagenette_supervised_preregistration_2026-07-10.md`。

## SGD-inspired sender semantic description 与 source-edge oracle（2026-07-10）

coarse source-description 的嵌套开发/审计：

```bash
python3 scripts/s6_imagenette_source_semantic_description_eval.py --config configs/s6_imagenette_source_semantic_description_eval.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_imagenette_source_semantic_description_eval.py --config configs/s6_imagenette_source_semantic_description_eval.yaml --device cuda:0
```

该诊断发送 scratch `G_gate(original)` 的 4-bit top-1 或 80-bit uint8 probability vector，并假设无噪声。policy-dev 内按 WNID+SHA256 固定拆为 `945/949` 张 select/audit；连续匹配规则在 audit 上 failure 比 M2 高 `+1.6078 pp`（95% CI `[+0.8627,+2.3922] pp`），且只保留 `3.26%` M2 PSNR。因此 coarse description 只用于末端 gate 是负结果，official val 仍封存。

fine source-edge feasibility oracle：

```bash
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_source_edge_oracle_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_source_edge_oracle_residual_refiner_validation_coco256_awgn.yaml --device cuda:0
python3 scripts/s6_compare_source_edge_oracle.py --config configs/s6_source_edge_oracle_comparison_exp_s4_008_011.yaml
```

`EXP-S4-011` 与 receiver-edge `EXP-S4-008` 匹配容量、训练预算、split、seed、loss 和 gates，仅把 Sobel/Laplacian 来源换成 sender original。paired bootstrap 得到 source-edge 相对 receiver-edge raw PSNR `+3.5149 dB`，95% CI `[+3.2602,+3.7652]`，五个 SNR 全为正。这只是 perfect-edge feasibility upper bound：edge rate/channel error 未计、总 CBR 未定义，不能作为公平通信增益。下一步必须训练 main≈`1/8` + edge≈`1/24` 的 matched-total-CBR 系统。完整正负边界见 `reports/source_semantic_guidance_significant_result_2026-07-10.md`。

## 精确等总码率 main + decoded-structure 结果（2026-07-11）

当前已经补齐 source-edge oracle 缺失的码率/信道闭环：reference 使用 `c=8` RGB，proposed 使用 `c=6` RGB main 加 `c=2` Sobel/Laplacian structure，total CBR 都严格为 `8/48=1/6`。结构描述经过独立 AWGN DeepJSCC，不向接收端泄漏 perfect edge。

复现入口：

```bash
python3 scripts/s7_train_matched_rate_jscc.py --config configs/s7_matched_rate_jscc_pilot_coco256_awgn.yaml --arm main --device cuda:0 --dry-run
python3 scripts/s7_train_matched_rate_jscc.py --config configs/s7_matched_rate_jscc_pilot_coco256_awgn.yaml --arm structure --device cuda:0 --dry-run
python3 scripts/s7_export_matched_rate_jscc.py --config configs/s7_matched_rate_jscc_export_coco256_awgn.yaml --device cuda:0 --dry-run
python3 scripts/s7_compare_matched_rate_system.py --config configs/s7_matched_rate_system_cross_split_comparison.yaml
python3 scripts/s7_imagenette_matched_rate_eval.py --config configs/s7_imagenette_matched_rate_supervised_eval.yaml --device cuda:0 --dry-run
```

COCO frozen downstream 三段合并的 raw PSNR 增益为 `+0.3772 dB`，95% CI `[+0.3274,+0.4253]`；四个 split、五个 SNR 的 20 个点估计全部为正。预注册 Imagenette policy-dev 上，matched raw 相对 `c=8` 的 PSNR 为 `+1.8341 dB`、LPIPS 为 `-0.0305`，主 SNR supervised failure 从 `3.3785%` 降至 `1.2375%`。但 new-error 保守上界为 `2.4764%`，未过 `0.5%` 安全门槛，因此 official val 仍封存，当前结论是“等码率质量和净语义失败显著改善”，不是“语义无损”。

完整结果、允许表述和下一方向见 `reports/matched_rate_significant_result_2026-07-11.md`。下一版不再扫描浅层 fallback 阈值，而是把 `c=2` 从纯 edge packet 升级为 evaluator-independent 的语义描述/校验通道，并在 restoration 内部融合。

## Hybrid structure + semantic sketch（2026-07-11）

S8 在不增加总码率的前提下，把 frozen AlexNet probability 的 32-D 固定投影塞入已有 `c=2` latent。最终 repetition-4 payload 只占 `128/16384=0.78125%` 的结构 latent；1 dB 恢复 cosine `0.9552`，19 dB `0.9992`，结构前两通道 MSE 只增加 `3.24%-5.84%`。

主要入口：

```bash
python3 scripts/s8_export_hybrid_semantic_structure.py --config configs/s8_hybrid_structure_semantic_export_r4_coco256_awgn.yaml --device cuda:0 --dry-run
python3 scripts/s5_residual_refiner_pilot.py --config configs/s8_per_sample_counterfactual_semantic_refiner_validation.yaml --device cuda:0 --dry-run
python3 scripts/s8_semantic_sketch_ablation.py --config configs/s8_per_sample_semantic_sketch_validation_ablation.yaml --device cuda:0 --dry-run
python3 scripts/s8_semantic_sketch_ablation.py --config configs/s8_per_sample_semantic_sketch_downstream_ablation.yaml --device cuda:0 --dry-run
```

冻结 downstream 160 图上，S8 raw 相对 `c=8` 为 `+0.4691 dB`（95% CI `[+0.4231,+0.5159]`），相对 S7 为 `+0.0919 dB`。正确 received sketch 相对 zero 为 `+0.0849 dB`（CI `[+0.0728,+0.0982]`），但相对 shuffled 只有 `+0.0072 dB`（CI `[-0.0023,+0.0170]`）。因此当前可声称 side signal 有用，不能声称随机投影已提供可靠的样本特异 semantic grounding；S8 不单独送审，只在下节主线 M3 整体协议中审计。详见 `reports/hybrid_semantic_sketch_result_2026-07-11.md`。

## 主线 M3 semantic-sketch controller（2026-07-11）

S8 side signal 已正式合并回 `M3-Ours`：对 `main + alpha*(hybrid_raw-main)` 的五个 alpha 候选，接收端选择与 received source sketch 最一致的输出。它仍是严格等总码率、SNR-aware residual strength + semantic consistency control，不是独立新主线。

```bash
python3 scripts/s7_imagenette_matched_rate_eval.py --config configs/s9_imagenette_hybrid_semantic_controller_eval.yaml --device cuda:0 --dry-run
```

预注册 policy-dev 上，M3 failure 为 `1.2178%`，hybrid raw 为 `1.2571%`，reference c8 为 `3.6142%`；M3 将 raw new-error image clusters 从 23 降到 18，同时保留 `+1.4234 dB` PSNR、`-0.0265` LPIPS 和 74.8% raw PSNR gain。由于 raw-minus-M3 failure CI 跨 0，且 new-error 上界 `1.5875% > 0.5%`，它保留为主线 semantic-control 候选/消融，不升级为 supervised-safe M3，official val 继续封存。完整报告见 `reports/mainline_hybrid_semantic_controller_result_2026-07-11.md`。

## Short-chain residual-shift diffusion pilot（2026-07-12）

本项目没有放弃 diffusion。`EXP-S10-001` 将 diffusion 限定为严格等码率系统的接收端 correction backend：冻结 `c=6 main + c=2 decoded structure` 与 `EXP-S7-002` residual CNN 作为 anchor，在 pixel domain 从 anchor 附近运行 6-step residual-shift bridge。它不依赖 Stable Diffusion、VAE、文本 prompt 或额外传输码率。

```bash
python3 scripts/s10_short_chain_residual_shift_diffusion.py --config configs/s10_short_chain_residual_shift_diffusion_pilot.yaml --device cuda:0 --dry-run
python3 scripts/s10_short_chain_residual_shift_diffusion.py --config configs/s10_short_chain_residual_shift_diffusion_pilot.yaml --device cuda:0
python3 tests/test_short_chain_residual_shift_diffusion.py
```

正式 160/64 split、五 SNR 上，相对 anchor 的 mean ΔPSNR 为 `-0.1548 dB`，mean ΔLPIPS 为 `-0.000195`，且 5/5 SNR 的 LPIPS 都微幅改善；但 raw candidate 新增 12 个 AlexNet pseudo error、只修复 7 个，未通过预注册 semantic-risk gate。因此该精确版本不晋级，但结果支持继续研究“强 anchor 附近的短链 conditional diffusion”，不支持回到 blind img2img 或 pure-Gaussian residual DDPM。详见 `reports/short_chain_residual_shift_diffusion_preregistration_2026-07-12.md` 和 `outputs/EXP-S10-001/REPORT.md`。

## P0 `c8 + same refiner` 公平对照（2026-07-12）

此前 `c6+c2 decoded structure + refiner` 主要与裸 `c8` 比较，无法排除收益只是来自额外后端。`EXP-S11-001` 给 `c8` 配置了与 `EXP-S7-002` 完全匹配的 `64×6`、60 epoch receiver-only refiner，并冻结 seed、split、loss、gates 和模型选择协议。

```bash
python3 scripts/s5_residual_refiner_pilot.py --config configs/s11_p0_c8_same_refiner_validation.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s11_p0_c8_same_refiner_validation.yaml --device cuda:0
python3 scripts/s11_compare_p0_b1_b3.py --config configs/s11_p0_b1_b3_paired_comparison.yaml --dry-run
python3 scripts/s11_compare_p0_b1_b3.py --config configs/s11_p0_b1_b3_paired_comparison.yaml
```

结果改变了归因判断：B1 `c8 + refiner` 相对 bare B0 为 `+1.0192 dB`，B3 相对 B0 仅 `+0.3974 dB`；B3 − B1 为 `-0.6217 dB`，95% image-cluster CI `[-0.6654,-0.5839]`，5/5 SNR 全负，LPIPS 也更差。双方 refiner 均为 448,387 参数、约 2.5 ms/image。因此当前 decoded-structure side path 不再被视为主要贡献，后续 diffusion 使用 B1 作为更强且公平的 deterministic anchor。详见 `reports/p0_c8_same_refiner_result_2026-07-12.md`。

## B1-anchored semantic-preserving diffusion v2（2026-07-12）

`EXP-S12-001` 把 6-step residual-shift diffusion 改接到更强的 B1 anchor，并从 anchor 自身计算 receiver-visible Sobel/Laplacian。训练保持 reconstruction-dominant，同时加入 edge L1 和本地冻结 ResNet18 target KL；最终 pseudo semantic diagnostic 使用不同架构的 AlexNet。

```bash
python3 scripts/s10_short_chain_residual_shift_diffusion.py --config configs/s12_b1_anchored_semantic_preserving_diffusion.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s10_short_chain_residual_shift_diffusion.py --config configs/s12_b1_anchored_semantic_preserving_diffusion.yaml --device cuda:0
```

正式结果为 mean raw ΔPSNR `-0.0775 dB`、mean raw ΔLPIPS `-0.000652`，5/5 SNR LPIPS 改善；相对 S10，PSNR 回吐减半、LPIPS 改善扩大。但 raw new-error/repair 为 `8/4`，仍未通过 semantic-risk gate。best epoch 2 后出现明显小数据过拟合，因此不再继续调整该 160-image bridge；后续 diffusion 只允许转向 COCO train2017-scale、独立 validation 和直接 risk calibration。详见 `reports/b1_anchored_diffusion_result_2026-07-12.md`。

## COCO train2017 scale-up B1 anchor（2026-07-13）

为解决 160-image 过拟合，新增 train2017 内部 10k train + 1k validation 的确定性 scale-up protocol。样本由 `SHA256(seed:path)` 排序冻结，并逐 SHA 排除 local val2017 重复。

```bash
python3 scripts/s13_export_coco_train2017_c8_scaleup.py --config configs/s13_coco_train2017_c8_scaleup_export.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s13_export_coco_train2017_c8_scaleup.py --config configs/s13_coco_train2017_c8_scaleup_export.yaml --device cuda:0
python3 scripts/s5_residual_refiner_pilot.py --config configs/s13_scaleup_b1_anchor_train.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s13_scaleup_b1_anchor_train.yaml --device cuda:0
```

`EXPORT-S13-001` 生成 55k c8 reconstruction，cache 约 6.9GB；`EXP-S13-001` 在独立 1k×5 validation 上得到 mean raw ΔPSNR `+1.3632 dB`、ΔLPIPS `-0.03272`，5/5 SNR 同向，pseudo new-error/repair `339/951`。全部 anchor gate 通过，epoch-9 checkpoint SHA-256 `80133f9d...65562` 已冻结为下一阶段 diffusion anchor。详见 `reports/scaleup_b1_anchor_result_2026-07-13.md`。

## Train2017-scale B1-anchored diffusion（2026-07-13）

```bash
python3 scripts/s10_short_chain_residual_shift_diffusion.py --config configs/s14_scaleup_b1_anchored_diffusion.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s10_short_chain_residual_shift_diffusion.py --config configs/s14_scaleup_b1_anchored_diffusion.yaml --device cuda:0
```

S14 在 10k/1k scale-up 后得到 raw new-error/repair `63/76`，通过净风险 gate；但 mean ΔPSNR `-0.0736 dB`、ΔLPIPS `+0.000081`，感知指标无增益，总判定 NEGATIVE。停止继续调该 residual-shift bridge；详见 `reports/scaleup_b1_anchored_diffusion_result_2026-07-13.md`。

## Received-latent posterior consistency 接口（2026-07-13）

`src/cadsd_jscc/deepjscc_adapter.py` 现可显式返回 transmitted/received channel latent，并对 candidate 计算可微 normalized measurement-consistency loss。formal checkpoint smoke 的 split-forward 最大误差为 `1.788e-7`，received latent shape 为 `(B,16,64,64)`，一致性梯度有限且非零。该接口用于下一版 posterior/data-consistency diffusion，不是 S14 调参。详见 `reports/received_latent_posterior_feasibility_2026-07-13.md`。

## Received-latent posterior correction pilot（2026-07-13）

```bash
python3 scripts/pc_posterior_consistency_pilot.py --config configs/pc001_posterior_consistency_pilot.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_posterior_consistency_pilot.py --config configs/pc001_posterior_consistency_pilot.yaml --device cuda:0
```

在未被 S13/S14 使用的 64 张 train2017 图像上，从冻结 S14 raw 出发做 3 次 received-latent proximal correction。5/5 SNR 的 consistency loss、PSNR、LPIPS 均改善；mean posterior-minus-raw PSNR `+0.2124 dB`、LPIPS `-0.00991`，latent loss 相对下降约 `20.1%`。B1-anchor-relative pseudo new error 保持 `5→5`，repair `2→17`，全部预注册 gate 通过。

这是一项阶段性正结果：diffusion 主线保留，但后续应改成内生 received-latent posterior/data-consistency sampler，而不再调 S14 的无约束 residual-shift bridge。本 pilot 不是最终 semantic-safety 证据。详见 `reports/posterior_consistency_pilot_result_2026-07-13.md`。

## Posterior correction 独立复现与 failure handling（2026-07-13）

```bash
python3 scripts/pc_posterior_consistency_replication.py --config configs/pc002_posterior_consistency_independent_replication.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_posterior_consistency_replication.py --config configs/pc002_posterior_consistency_independent_replication.yaml --device cuda:0
python3 scripts/pc_posterior_consistency_replication.py --config configs/pc003_posterior_consistency_failure_handling.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_posterior_consistency_replication.py --config configs/pc003_posterior_consistency_failure_handling.yaml --device cuda:0
```

PC-002 在新 256 图×5 SNR 上复现 posterior correction：PSNR `+0.2125 dB`、LPIPS `-0.01078`、latent loss 相对约 `-20.4%`，全部 5/5 SNR 同向；但三分类器 semantic gate 失败。PC-003 的 receiver-only AlexNet agreement fallback 以 `87.66%` coverage 保留 PSNR `+0.2062 dB`、LPIPS `-0.00910`，并把 majority new error `4→1`，但仍未达到 raw 的 `0`，也未迁移到另外两套分类器。

因此阶段结论是：posterior-consistent diffusion restoration 已可复现，单模型 failure handling 仍不是跨模型可靠的最终 M3。报告见 `reports/posterior_consistency_independent_replication_result_2026-07-13.md` 和 `reports/posterior_consistency_failure_handling_result_2026-07-13.md`。

## PC consensus controller holdout audit（2026-07-13）

```bash
python3 scripts/pc_posterior_consistency_replication.py --config configs/pc_controller_holdout_audit.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_posterior_consistency_replication.py --config configs/pc_controller_holdout_audit.yaml --device cuda:0
```

AlexNet+ResNet18 consensus controller 的 coverage 为 `78.05%`，final 相对 S14 raw 保留 PSNR `+0.1927 dB`、LPIPS `-0.00791`，controller ensemble 内 majority new error 为 0；但完全未参与控制的 MobileNet holdout new error 从 `12` 增到 `34`，因此总判定 NEGATIVE。停止继续堆 top-1 consensus 规则；详见 `reports/posterior_consensus_controller_holdout_result_2026-07-13.md`。

## PC 独立标注与 Imagenette 监督审计（2026-07-13）

```bash
python3 scripts/pc_posterior_consistency_replication.py --config configs/pc_coco_object_clip_audit.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_posterior_consistency_replication.py --config configs/pc_coco_object_clip_audit.yaml --device cuda:0
python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_supervised_audit.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_supervised_audit.yaml --device cuda:0
```

COCO-object clean-correct 审计得到 final failure `36→32`、new error `2→4`，说明存在真实对象级风险。更严格的 Imagenette policy-dev 监督审计中，1697 张 clean 图的 primary raw/posterior/final failure 为 `69/56/62`，new error 总数为 `4/4/4`；final 保留 `+0.2543 dB/-0.00531 LPIPS`。但 7 dB 出现 final/raw new error `1/0`，逐 SNR gate 失败，official validation 继续封存。详见 `reports/posterior_imagenette_supervised_audit_result_2026-07-13.md`。

## PC task-matched scratch-gate follow-up（2026-07-13）

```bash
python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_scratch_gate_audit.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_scratch_gate_audit.yaml --device cuda:0
```

该 follow-up 只把 PC-SUP 的 ImageNet consensus controller 替换为既有 scratch MobileNetV3-Small `G_gate`，scratch ResNet18 `T_cls` 仍仅用于监督审计。9470 行 raw/posterior 与 PC-SUP 逐值一致；scratch gate 的 clean-row coverage 为 `99.33%`，primary failure `69→57`、new error `4→3`，final 相对 raw 保留 `+0.26394 dB/-0.005966 LPIPS`，均优于旧 controller。

严格判定仍为 NEGATIVE：7 dB 的 new error 是 `1 vs raw 0`，所以逐 SNR gate 未过。不得在已查看的 policy-dev 上继续扫 threshold；official validation 仍封存。当前阶段成果是保留 posterior-consistent diffusion + scratch gate 作为 supervised development candidate，而不是宣称 semantic-safe。详见 `reports/posterior_imagenette_scratch_gate_result_2026-07-13.md`。

## PC scratch-gate multi-channel-seed replication（2026-07-13）

```bash
python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_scratch_gate_multiseed_replication.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_scratch_gate_multiseed_replication.yaml --device cuda:0
```

冻结方法在三个全新 AWGN seed、28,410 行上得到稳定 restoration 收益：15/15 seed×SNR consistency 下降，final 相对 raw mean PSNR/LPIPS `+0.26334 dB/-0.005937`，primary failure `196→163`，且每个 seed 都改善。

严格 semantic-tail 结果仍失败：new-error rows `13→14`，image clusters `10→11`；final `11/1691` 的单侧 95% Clopper-Pearson upper 为 `1.0744% > 0.5%`。1 dB 和 seed 20260722 分别恶化 `8→10`、`5→7`，旧 failure image 也在新 seed 再现。因此当前应该继续保留 posterior-consistent diffusion，但必须淘汰简单 scratch top-1 agreement 作为最终 controller；official validation 继续封存。完整报告见 `reports/posterior_imagenette_scratch_gate_multiseed_result_2026-07-13.md`。

## PC continuous receiver-risk controller 与新 seed 审计（2026-07-14）

以下命令对应预注册顺序。正式输出目录均 fail-if-exists，已有结果不会被覆盖；重跑时必须改用新的输出目录。

```bash
# 1. 独立 scratch G_aux；本地数据、weights=None，不下载
python3 scripts/s6_train_imagenette_scratch_classifiers.py --config configs/pc_imagenette_scratch_aux_classifier.yaml --roles G_aux --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/s6_train_imagenette_scratch_classifiers.py --config configs/pc_imagenette_scratch_aux_classifier.yaml --roles G_aux --device cuda:0

# 2. 已暴露三 seed 的 receiver_risk_v1 development table
python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_receiver_risk_features_multiseed.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_receiver_risk_features_multiseed.yaml --device cuda:0

# 3. 透明六特征 empirical-percentile controller development fit
python3 scripts/pc_fit_receiver_risk_controller.py --config configs/pc_imagenette_receiver_risk_controller_dev.yaml --dry-run
python3 scripts/pc_fit_receiver_risk_controller.py --config configs/pc_imagenette_receiver_risk_controller_dev.yaml

# 4. 冻结 controller 后的新 channel-seed feature generation 与一次性 audit
python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_receiver_risk_seed_20260725_features.yaml --device cuda:0 --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_imagenette_supervised_audit.py --config configs/pc_imagenette_receiver_risk_seed_20260725_features.yaml --device cuda:0
python3 scripts/pc_apply_receiver_risk_controller.py --config configs/pc_imagenette_receiver_risk_seed_20260725_audit.yaml --dry-run
python3 scripts/pc_apply_receiver_risk_controller.py --config configs/pc_imagenette_receiver_risk_seed_20260725_audit.yaml
```

`G_aux` cal macro top-1 为 `0.90270`。43 维 feature table 共 28,410 行，与旧三 seed 审计逐行复现；controller 只输入四个 `G_gate/G_aux` JS 变化和两个 posterior confidence percentile，不输入 `T_cls`/原图/标签/类别/样本 ID。开发集上冻结 10% reject-rate threshold 后，new-error `15→3`、cluster upper `0.4579%`、PSNR/LPIPS `+0.23834/-0.004799`，因此只记 development pass。

预注册新 seed `20260725` 给出相反的独立结论：posterior restoration 仍稳定改善 `+0.26535 dB/-0.006064 LPIPS`，primary failure `50→45`；但冻结 risk controller 两个 new-error 均漏过、误拒 11 个 repair，final failure `56>raw 50`、new-error `2>raw 0`，正式 verdict `NEGATIVE`。这说明 receiver-only uncertainty 会遇到高置信共享盲点。diffusion 不退出，但下一步不能再扫 receiver threshold；应开发任务相关、可纠错、严格计码率的 sender semantic checksum，并在新的 labeled development population 上训练。详见 `reports/posterior_receiver_risk_controller_stage_result_2026-07-14.md`。

## PC 固定码率 sender semantic payload（2026-07-14）

入口 `pc_imagenette_sender_inbudget_awgn_audit.py` 把 sender description 嵌入原 `c=8` latent，和图像主载荷共用 AWGN；receiver 擦除 payload 后解码，posterior consistency 用 mask 排除保留位置。输出目录均 fail-if-exists，下面命令用于 dry-run 或在新输出路径复现，不能覆盖已有实验。

```bash
# 模拟 10 维 probability × R16 开发配置
python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py --config configs/pc_imagenette_sender_aux_inbudget_awgn_dev.yaml --dry-run

# UInt4(10 类，共 40 bit) + BPSK × R4 开发配置
python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py --config configs/pc_imagenette_sender_aux_uint4_bpsk_inbudget_awgn_dev.yaml --dry-run

# 冻结到新 channel seed 20260726 的审计配置
python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py --config configs/pc_imagenette_sender_aux_uint4_bpsk_seed20260726_audit.yaml --dry-run

# 正式运行只使用本地 checkpoint/data，不联网；已有 output_dir 不可覆盖
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py --config <new-config.yaml> --device cuda:0
```

严格率合同为 65,536 个总实符号，其中 160 个 payload symbols、65,376 个 image symbols，总 CBR 仍为 `1/6`。模拟 payload 因连续分数噪声判为 `NEGATIVE`。UInt4+BPSK×4 在 seed `20260725` 开发上通过，但冻结 seed `20260726` 审计中 final new-error `5>in-budget raw 3`，所以最终仍为 `NEGATIVE`；编码层本身已稳定迁移，瓶颈是单一 `G_aux`/JS semantic decision。完整中文报告见 `reports/posterior_sender_inbudget_semantic_payload_stage_result_2026-07-14.md`。

## Cross-model triplet sender controller（2026-07-14，历史 UInt4 版本）

该历史候选使用固定 40-bit `UInt4+BPSK×4` payload、160 个保留符号和同一 AWGN。它不增加发送开销；只把已恢复的 `G_aux(source)` top-1 与独立 scratch `G_gate` 的 anchor/posterior top-1 组成三方自然一致性 gate：

```text
source-JS <= 0
AND recovered G_aux(source).top1 == G_gate(anchor).top1
AND G_gate(anchor).top1 == G_gate(posterior).top1
```

原记录按 in-budget raw/anchor-relative endpoint 给出 `2→0` 和 upper95 `0.1771%`，曾误判为 POSITIVE。后续严格统计改用 paired unpunctured M2 的 system endpoint 后，得到 new/repair clusters `7/8`、upper95 `0.7766%>0.5%`，且 1 dB failure `32→34`，因此正式结论已更正为 **NEGATIVE**。mean final-minus-M2 `+0.01158 dB/-0.002566 LPIPS` 仍成立，但不能抵消 semantic new-error tail；official Imagenette validation 未访问。

```bash
# 输出目录不可复用；下面命令只用于新路径的可复现运行
python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py \
  --config configs/pc_imagenette_sender_crossmodel_triplet_seed20260727_audit.yaml --dry-run

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py \
  --config <new-crossmodel-config.yaml> --device cuda:0
```

完整中文报告：`reports/posterior_sender_crossmodel_triplet_stage_result_2026-07-14.md`。
## 2026-07-14 UInt2 预留感知阶段状态

当前仍保留 diffusion。UInt2 BPSK×4 sender payload 只占 80/65536 个实符号，预留感知 B1 在相同 reserved inputs 上相对旧 B1 得到 `+0.1028 dB` paired PSNR（95% CI `[+0.0934,+0.1140]`）。接回冻结 S14 diffusion 与 received-latent posterior correction 后，新 channel seed 的 final−M2 PSNR/LPIPS 为 `+0.0658 dB/-0.00254`，五个 SNR 的 PSNR 都为正。

严格 semantic verdict 仍为 NEGATIVE：seed20260728 的 M2/final primary failure `62→62`，system new-error cluster upper95 `0.5408%>0.5%`，1 dB failure 增加。official Imagenette validation 未访问。完整结果和下一步边界见 `reports/uint2_reservation_aware_diffusion_stage_result_2026-07-14.md`。

关键复现命令：

```bash
python3 scripts/s15_compare_reservation_aware_b1.py \
  --output-dir outputs/analysis/s15_reservation_aware_b1_paired_comparison_reproduction
python3 scripts/pc_analyze_mismatch_raw_routing.py \
  --input-csv outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_full_dev/per_sample.csv \
  --output-dir outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_offline_seed20260727_reproduction
python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py \
  --config configs/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_seed20260728.yaml \
  --output-dir outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_seed20260728_reproduction
```

所有输出目录均拒绝覆盖；重跑前应指定新的输出路径。大任务按 `AGENTS.md` 清空 proxy，以上命令不需要联网下载。

## 外部方法公平对比轨道（2026-07-14）

外部方法已经进入正式排期：`SGD-JSCC author → SING-Zero-style → DiffJSCC author → DiT-JSCC watch-only`。作者原生复现和本项目 common contract 分开报告；只有同图、同 AWGN realization、同 `[1,4,7,13,19]` SNR、同总 CBR `1/6` 且完整计入 text/edge/pilot side information 后，才允许直接比较优劣。

SGD-JSCC 源码已只读固定在 `third_party/SGDJSCC`，commit 为 `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`。作者 4 个 checkpoint、BLIP2 safetensors、OpenAI CLIP ViT-L/14 和 scheduler 已在清空全部代理变量后通过服务器直连下载并逐项校验；隔离运行环境为 `.venv-sgdjscc`，依赖记录在 `requirements-sgdjscc.txt`。

无下载协议检查：

```bash
python3 scripts/check_external_baseline_contract.py
python3 -m unittest discover -s tests -p 'test_external_baseline_contract.py' -v
.venv-sgdjscc/bin/python scripts/external_sgdjscc_native_smoke.py
```

`SMOKE-EXT-SGDJSCC-001` 已完成一次作者完整链单图运行，输出位于 `outputs/smoke/external_sgdjscc_native_snr1_seed2025_20260714/`。该 run 的 main/edge-active 为 `4096/832` 个实符号，caption 为 488 UTF-8 bits，但作者协议没有 caption channel-symbol mapping；因此结果只能进入 author-native 表，仍禁止与本项目直接排名。默认输出目录拒绝覆盖，重跑必须复制配置并更换 `analysis_id` 与 `output_dir`。完整排期见 `reports/external_method_comparison_schedule_2026-07-14.md`，本次中文阶段结果见 `reports/sgdjscc_author_native_smoke_stage_result_2026-07-14.md`。

## SGD-JSCC 共同协议闭环（2026-07-15）

共同协议适配器已在一张 frozen COCO-256 图上真实跑通。它保留作者四 patch、main JSCC、edge-JSCC、ControlNet 和 50-step diffusion，但把每块 caption 编为固定 UTF-8+CRC16 packet，经 BPSK×21 过同一 AWGN；确定性 edge mask 只发送 active coordinates。总账本为 main `16,384` + edge `3,328` + text `45,024` + padding `800` = `65,536` 个实坐标，即 `32,768` complex uses、CBR `1/6`。

```bash
# 无下载 dry-run
.venv-sgdjscc/bin/python scripts/external_sgdjscc_common_smoke.py

# 新 output_dir 的真实运行；全部资产本地离线，仍清空代理
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    .venv-sgdjscc/bin/python scripts/external_sgdjscc_common_smoke.py \
    --config <new-common-smoke-config.yaml> --run
```

`SMOKE-EXT-SGDJSCC-COMMON-001` 在 1 dB/seed 20260729 得到 CRC `4/4`、caption packet bit error `0/2144`、finite 256×256 output 和 smoke-only PSNR `24.785109 dB`。rate gate 已通过，但单图出现 patch seam 与疑似 text-driven enlarged-player hallucination，因此效果/semantic gate 仍未通过，不能据此排名。完整中文报告：`reports/sgdjscc_common_contract_smoke_stage_result_2026-07-15.md`。

码率术语现统一为：两个实信道坐标组成一个 complex channel use；`65,536/(2×3×256×256)=1/6`。旧 native 报告中 `0.08333` 等数字是 real-coordinate/source-dimension ratio，不是同口径 complex-use CBR。

> 信道口径更正：上述 `24.785109 dB` 单图使用作者每实坐标方差 `P/SNR`，比项目复信道 `P/(2×SNR)` 严苛 3 dB，只保留作接入证据。`configs/external_sgdjscc_common_complex_awgn_smoke.yaml` 的同图复信道结果为 `26.128782 dB`。

## 外部共同协议首轮真实对比（2026-07-15）

`ANALYSIS-EXT-COMMON-PILOT-001` 已在 8 张 frozen Imagenette policy-dev clean 图、五个 SNR `[1,4,7,13,19]` 上跑完当前 M3、SGD-JSCC common adapter 和 SING-Zero-style。每个 sample/SNR 使用相同 65,536-D canonical standard-normal vector，复 AWGN 每实坐标方差为 `P/(2×SNR)`；三方法均严格使用 65,536 real coordinates = 32,768 complex uses = CBR `1/6`。

| 方法 | PSNR | MS-SSIM | LPIPS | final failure / new error |
|---|---:|---:|---:|---:|
| DeepJSCC reference | `31.7438` | `0.97298` | `0.07861` | `0 / —` |
| 当前 M3 | **`33.0594`** | **`0.98203`** | **`0.03532`** | `0 / 0` |
| SGD-JSCC common adapter | `26.8882` | `0.94862` | `0.07763` | `0 / 0` |
| SING-Zero-style final-only | `24.6593` | `0.96118` | `0.31725` | `1 / 1` |

聚合器已验证 120 行的 sample/SNR key、noise SHA、DeepJSCC reference、rate 和 AWGN 口径完全一致。当前 M3 相对 SGD 的配对均值为 `+6.1712 dB/-0.04231 LPIPS`；这是小规模 development pilot 的方向性结果，不授权强于 SGD-JSCC/SING 论文的结论。SGD 是项目侧 common adapter；SING-style 只做最终一步 range/null projection，不是论文逐 reverse-step DDNM。

已存在的输出拒绝覆盖。下列命令用于协议检查；要真实复现必须复制配置、更换 `analysis_id`/所有 output paths，并保持 official val 封存：

```bash
# M3 / SING-style 无下载 dry-run
python3 scripts/external_common_project_pilot.py --method ours
python3 scripts/external_common_project_pilot.py --method sing-zero-style

# SGD 无下载 dry-run；pinned 环境必须在 import 前指向本地 HF cache
HF_HOME="$PWD/third_party/SGDJSCC/runtime_assets/hf_home" \
PYTHONPATH=src:scripts \
.venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py

# 对已有三方法结果做 fail-closed aggregate；已有 aggregate 目录同样拒绝覆盖
python3 scripts/external_common_aggregate.py
```

完整结果、限制与下一步见 `reports/external_common_comparison_pilot_stage_result_2026-07-15.md`。本轮全部资产本地离线，无新增下载；全仓 `99/99` 标准库测试通过。

## 外部方法双工作点码率对齐（2026-07-15）

作者工作点使用精确 19,712 个图像分支实符号；项目工作点继续固定 65,536 个实符号。相关命令如下，所有正式输出目录均拒绝覆盖：

```bash
python3 scripts/external_train_exact_rate_deepjscc.py \
  --config configs/external_author_rate_deepjscc_fullcoco_continue.yaml \
  --device cuda:0

python3 scripts/external_common_project_pilot.py \
  --config configs/external_author_rate_alignment_pilot.yaml \
  --method exact-rate-deepjscc --run

.venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py \
  --config configs/external_author_rate_alignment_pilot.yaml --run

.venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py \
  --config configs/external_project_rate_sgd_reallocation_pilot.yaml --run

python3 scripts/external_rate_alignment_aggregate.py
```

作者工作点的 SGD 数字包含论文假设的免费且无误 caption，只能作为论文协议上界；项目工作点的 R2/R13 只增加抗噪重复，不增加发布模型的表示容量。协议与首轮结果见 `reports/external_two_working_point_alignment_preregistration_2026-07-15.md`，最终训练预算补齐结果另见同阶段中文报告。
## 精确低码率 M3 闭环（2026-07-15）

当前低码率工作点为 19,712 个总实坐标，其中 80 个坐标用于 UInt2+BPSK×4 语义载荷、19,632 个坐标用于图像分支。B1 已成为新的低码率主 anchor；当前短链 diffusion 只在 19 dB 高 SNR 尾部经 posterior consistency 和语义门控后启用。

主要复现入口如下。所有正式输出目录均拒绝覆盖；重跑时必须复制配置并指定新目录。

```bash
# 只检查缓存计划，不产生输出
python3 scripts/s13_export_coco_train2017_c8_scaleup.py \
  --config configs/lowrate_m3_exact19712_cache_export.yaml --dry-run

# B1 / diffusion 输入与协议检查
python3 scripts/s5_residual_refiner_pilot.py \
  --config configs/lowrate_m3_b1_anchor_train.yaml --dry-run
python3 scripts/s10_short_chain_residual_shift_diffusion.py \
  --config configs/lowrate_m3_b1_anchored_diffusion.yaml --dry-run

# 第一组 8×5 严格闭环 dry-run
python3 scripts/lowrate_m3_stage_pilot.py \
  --config configs/lowrate_m3_stage_pilot.yaml

# 独立高 SNR 尾部 holdout dry-run
python3 scripts/lowrate_m3_stage_pilot.py \
  --config configs/lowrate_m3_tail_holdout_pilot.yaml
```

阶段结果：B1 在 1000 图×5 SNR 上平均 `+1.038 dB/-0.114 LPIPS`；原始 diffusion 为负结果；独立尾部 holdout 中，19 dB 门控 final 相对 B1 在全五档平均为 `-0.0099 dB/-0.000389 LPIPS`，failure/new error 保持 0。完整中文解释见 `reports/lowrate_m3_stage_result_2026-07-15.md`。

## Channel-State-Matched Latent Diffusion（2026-07-15）

SGD-JSCC step matching 的项目内最小迁移已经跑通。新实现不在 B1 图像后随机生成残差，而是在 frozen exact-rate DeepJSCC 的 `6×64×64` codeword space 训练 masked epsilon predictor，并按项目 `P/(2×SNR)` 口径使用：

`alpha_channel = 2*gamma/(2*gamma+1)`。

只有 19,632 个图像活动坐标参与 diffusion；80 个语义载荷坐标与 4,864 个未发送稠密坐标始终排除。正式 FP32 训练：

```bash
python3 scripts/s17_channel_matched_latent_diffusion.py \
  --config configs/s17_channel_matched_latent_diffusion.yaml \
  --mode train \
  --device cuda:0
```

一次性 holdout：

```bash
TORCH_HOME="$PWD/outputs/cache/torch" \
python3 scripts/s17_channel_matched_latent_diffusion.py \
  --config configs/s17_channel_matched_latent_diffusion.yaml \
  --mode holdout \
  --device cuda:0
```

256图×5SNR holdout 上，matched DDIM 相对 B0 为 `+0.148715 dB/-0.035305 LPIPS`，相对固定 7 dB 错配为 `+0.233455 dB`；但只在 1/4/7 dB 获得 PSNR 正值，且 naive 接旧 B1 比 B1 低 `-0.231266 dB`。因此当前状态是“step matching 机制成功、最终系统融合未成功”。完整中文报告见 `reports/channel_matched_latent_diffusion_stage_result_2026-07-15.md`。

## Decoder-Aware Latent Diffusion（2026-07-15）

后继实验从 S17-002 best warm-start，使用同三轮预算 control 隔离 frozen-decoder image loss 的贡献。先运行预注册的无更新尺度诊断：

```bash
python3 scripts/s17_channel_matched_latent_diffusion.py \
  --config configs/s17_decoder_aware_latent_diffusion.yaml \
  --mode loss-diagnostic --device cuda:0
```

诊断已冻结 `decoder_image_mse_weight=20`。正式 control 与 decoder-aware 训练：

```bash
python3 scripts/s17_channel_matched_latent_diffusion.py \
  --config configs/s17_decoder_aware_latent_diffusion_control.yaml \
  --mode train --device cuda:0
python3 scripts/s17_channel_matched_latent_diffusion.py \
  --config configs/s17_decoder_aware_latent_diffusion.yaml \
  --mode train --device cuda:0
```

两个 checkpoint 哈希冻结后的一次性 fresh holdout 与 bootstrap：

```bash
TORCH_HOME="$PWD/outputs/cache/torch" \
python3 scripts/s17_channel_matched_latent_diffusion.py \
  --config configs/s17_decoder_aware_latent_diffusion.yaml \
  --mode holdout --device cuda:0
python3 scripts/s17_decoder_aware_latent_bootstrap.py
```

232图×5SNR 上 decoder-aware 相对同预算 control 为 `+0.021605 dB/-0.002502 LPIPS`，95% CI 均不跨零；相对 B0 为 `+0.174221 dB/-0.038540 LPIPS`。但 13/19 dB PSNR 仍为负，且 naive 接旧 B1 仍低于 B1，因此 verdict 为 `NEGATIVE_OR_PARTIAL`。完整中文报告见 `reports/decoder_aware_latent_diffusion_stage_result_2026-07-15.md`。

## SNR-Conditioned Identity Envelope（2026-07-15）

S18 冻结全部网络和码率，只在 codeword 层应用：

`z_final = y + g(SNR)*(z_diff-y)`。

先从未使用 COCO train2017 图像生成与旧 11,000 source 去重的 256/256 selection/holdout：

```bash
python3 scripts/s18_prepare_fresh_coco_population.py
```

selection 冻结 policy 后再运行一次性 holdout 和 bootstrap：

```bash
TORCH_HOME="$PWD/outputs/cache/torch" \
python3 scripts/s18_snr_identity_envelope.py --mode selection --device cuda:0

TORCH_HOME="$PWD/outputs/cache/torch" \
python3 scripts/s18_snr_identity_envelope.py --mode holdout --device cuda:0

python3 scripts/s18_snr_identity_bootstrap.py
```

正式 policy 为 `hard_identity_7db`：1/4/7 dB 使用完整 decoder-aware diffusion，13/19 dB 严格回 B0。fresh holdout 上相对 B0 为 `+0.189717 dB/-0.036284 LPIPS`，五档 PSNR `+0.677172/+0.240940/+0.030472/0/0 dB`；相对 full diffusion PSNR `+0.015642 dB`，95% CI `[+0.014230,+0.016915]`。10/10 checks PASS，但 B1 仍高 `+0.830617 dB`。完整中文报告：`reports/snr_identity_envelope_stage_result_2026-07-15.md`。
