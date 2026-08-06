# RDD-P0 生成式重建分布偏移预注册

日期：2026-07-30
实验ID：`ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001`
状态：预注册已冻结，正式指标未计算

## 背景与动机

考虑借用 rate-distortion-deception (RDD) 理论框架（arXiv 2607.25997, Ulukus/Yener；纯无噪信源编码，定义 deception 为 `D_KL(P_X̂‖P_Y) ≤ P`，即重建分布匹配到某个非源分布的目标 `P_Y`）。

在做任何"实现 deception"的工作之前，必须先验证一个前置事实：**现有生成式 JSCC 的重建，是否已经存在无意的、可识别的分布偏移。** 如果不存在，则 RDD 的 deception 项在本项目语境下没有实证立足点。

本轮是纯分析实验：不训练任何生成模型、不下载、不解封 official validation。指纹分类器是轻量的，只用于分析。

## 关键前提修正（必须记录）

用户原始设计要求"CLIC2020 test 428 图 × 5 SNR，S33/DiffJSCC/SGD"。**该设计在现有输出上不可执行**，理由如下（只读核验）：

- `paper_idea1b/outputs/ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001/reconstructions/clic2020_test/` 只有三臂：`s33_strong`、`swin_official_base_sa`、`swin_capacity_matched_sa`，各 2140 文件 = 428×5。
- `paper_idea1b/A1_DISCRIMINATIVE_RESULT.md` 第 19 行明确写入："DiffJSCC、SGD 和 refiner 未加载"。
- 全仓检索 `outputs/external_baselines/` 无任何 CLIC 引用。

因此 **DiffJSCC 与 SGD 从未在 CLIC2020 上运行过**。唯一三方法共存的总体是 **64 图 Imagenette policy-dev @ 256×256**，5 SNR × 3 seeds = 每方法 960 行。

用户已确认方案：主实验用该共享总体（含 4 臂），另加 CLIC-428 判别式补充。不授权新的 CLIC 生成式推理（A2 仍未授权）。

## 统计功效边界（事前声明）

共享总体每个 `(method, SNR)` 单元只有 **n=192**（64 唯一图 × 3 seeds）。在 2048 维 Inception 特征下协方差秩亏，FID 有明显正偏。因此：

- **KID 为主指标**（无偏、小样本下可用），FID 必报但只作次要参考。沿用本项目 S34C 先例："KID 主、FID 必报"。
- 跨参考集比较必须在**同一 n、同一特征提取器、同一预处理**下进行，只比较相对大小。
- 绝对 FID 值不与 A1 的 CLIC-428 数值直接比较（n 与分辨率均不同）。

## 数据与臂

**主实验总体**：`ANALYSIS-S20-SGD-B1-DECISION-001/population/population_manifest.json` 的 64 个 `sample_ids`，5 SNR `[1,4,7,13,19]` dB × 3 seeds `[20260748,20260749,20260750]`。

四臂（全部来自现有输出，无新生成式推理）：

| 臂 | 先验 | 来源 | 恢复方式 |
|---|---|---|---|
| `s33_strong` | 无（判别式） | 冻结 checkpoint `2daad9e7…5bfb` | 精确重放 |
| `author_jscc` | 无（判别式） | S30 montage 第 2 面板 | 无损裁切 |
| `diffjscc` | SD 2.1 | S30 montage 第 3 面板 | 无损裁切 |
| `sgd_jscc` | MDTv2 (DiT 族) | S20 montage 8×16 tile | 无损裁切 |

`author_jscc` 是免费获得的第二判别式臂，用于区分"生成先验导致的偏移"与"任何 JSCC 都有的偏移"。这是本设计的关键对照：若只有 S33 一个判别式臂，无法排除偏移来自 JSCC 本身。

**SGD 边界**：SGD 仍是 non-ranking paper upper（≥21,856 real vs 16,384，且 captions 免费完美）。本轮只做分布分析，不做质量胜负。

## 参考分布（criterion ② 的候选目标 P_Y）

全部由同一 64 图源图构造，n 与臂一致：

- **(a) `real`**：源图本身（基线）。
- **(b1) `vae_sd21`**：源图过 DiffJSCC 的 `first_stage_model` (SD 2.1 `AutoencoderKL`, z=4) 编解码，不去噪。SD 先验痕迹代理。
- **(b2) `vae_sgd`**：源图过 SGD 的 `AutoencoderKL` (embed_dim=16) 编解码，不去噪。DiT 链先验痕迹代理。
- **(c) `blur_*`**：源图高斯低通，`sigma ∈ {0.5, 1.0, 1.5, 2.0}`。代表判别式 MSE 输出的典型平滑分布。**全部四档都报**，不事后择优。
- **(d1) `resample_512`**：源图 256→512→Lanczos→256 往返。理由：DiffJSCC 已知在 512 内部网格处理并 Lanczos 回 256（PROGRESS 2026-07-21 记录）。这是"非先验的纯重采样"对照，用于把重采样痕迹从"生成先验偏移"中分离出来。
- **(d2) `jpeg_q*`**：源图 JPEG `q ∈ {30, 70}`。理由：常见的"非源但自然"的目标分布，检验 criterion ② 是否对任意退化都成立（若是，则 ② 无区分力，必须如实记录）。

每个 (b) 使用的权重路径与 SHA 在任何指标产生前写入 config 快照。VAE 只做 `encode→decode`，用 posterior mean（不采样），确定性可复现。

## 预注册判据

**"存在可识别偏移"成立需同时满足：**

① 指纹分类准确率显著高于随机（95% CI 下界 > 1/K，K=臂数）；**且**
② 至少一个方法的重建，相对某个非真实参考分布 (b/c/d) 的 KID 低于相对真实图的 KID，即"更像那个目标分布而非源分布"。

**判定分层：**

- 同时满足 ①②：**存在可识别的定向偏移**。
- 只满足 ①：**有指纹但未偏向特定目标分布**。
- 都不满足：**负结果**（现有方法无可识别的定向偏移）。

**② 的必要三角约束（防止平凡成立）**：若参考集 R 自身离 real 很远，则"臂离 R 比离 real 近"可能只反映 R 的偏僻。因此对每个满足 ② 的 (arm, R) 必须同时报告 `KID(R, real)`，并明确标注该 ② 是否可由"R 本身远离 real"解释。仅当 `KID(arm,R) < KID(arm,real)` 且该关系不能由三角不等式平凡推出时，才记为**强 ②**；否则记为**弱 ②**并如实说明。

## 指纹检验设计

**任务**：输入一张重建图，判断来自哪个臂。

- **分组交叉验证**：按 **source image** 分组的 5-fold GroupKFold。同一源图的所有臂/SNR/seed 必须同组，否则分类器可以靠"认出这张图"作弊。这是本设计最重要的防泄漏措施。
- **主设置**：4 臂（chance = 25%）。另报 3 臂 `{s33, diffjscc, sgd}`（chance = 33.3%，对应用户原始要求）。
- **特征**：轻量手工统计特征，不训练深网。包括：分块 DCT 频段能量比、径向功率谱斜率与分段能量、噪声残差（高通）统计、局部方差/梯度分布、通道间相关、饱和/截断像素比例。所有特征在 config 中冻结，不事后增删。
- **分类器**：`sklearn` logistic regression（标准化后）与 gradient boosting 各一，两者都报，不择优。
- **CI**：按 source-image cluster 的 bootstrap（10,000 次）报准确率 95% CI；同时报混淆矩阵与 per-arm recall。
- **归因**：报置换重要性（permutation importance）与按频段聚合的重要性，回答"哪些特征/频段最有区分力"。

**伪影控制（用户已确认必做）**：生成臂带已知实现伪影 —— SGD 有文档记录的 4-patch 接缝，DiffJSCC 有 256→512→Lanczos→256 重采样。这些会被平凡检出。因此除原始准确率外，必须报告以下预注册消融：

- **C1 中心裁切**：只用 128×128 中心区域（避开 SGD patch 接缝，其接缝位于 128 边界）。
- **C2 降采样/模糊**：先降到 128×128 再提特征，压制高频重采样指纹。
- **C3 去先验对照**：`{s33, author_jscc}` 两判别式臂的二分类。若该准确率也远高于 50%，说明"可识别指纹"并非生成先验特有，而是任何 JSCC 实现都有 —— 这会**削弱**把指纹解释为"先验导致的分布偏移"的力度，必须如实写明。

C3 是本轮最可能推翻乐观解释的检验，事前即声明其结论对最终措辞的约束力。

## CLIC 判别式补充

在已有 CLIC-428 重建（`s33_strong`/Base-SA/CM-SA，5 SNR × 1 seed）上，用相同参考集 (a)(b1)(b2)(c)(d) 重算 FID/KID。n=428，功效充足。

用途：在**判别式方法**上做一次高功效的 criterion ② 检验。若判别式重建在 n=428 下明确偏向 `blur_*` 而非 real，这本身就是"无意的定向分布偏移"的干净证据（且与生成先验无关）。该结果只用于分布分析，不改变 A1 已冻结的 S33-vs-Swin 质量胜负结论。

## 执行顺序（严格）

1. preflight（已完成，见下）。
2. 冻结本文件与 config 快照 SHA。
3. 构建四臂图像集 + 验证门（PSNR 复现误差）。
4. 构建参考集。
5. 计算 FID/KID 矩阵 + 三角约束。
6. 指纹分类 + 控制消融。
7. CLIC 补充。
8. 渲染判定、写结果报告、更新 `PROGRESS.md`/`EXPERIMENTS.md`/`LITERATURE.md`。

## 已完成 preflight（feasibility only，未计算任何分布指标）

- DiffJSCC montage 键格式解析为 `sha256(f"{sample_id}|{base_seed}|{float(snr_db)}")[:16]`，**960/960** 命中，0 缺失。
- montage 面板身份用 PSNR 对账：author-JSCC 面板 max|ΔPSNR| = `0.000002 dB`，DiffJSCC 面板 = `0.000003 dB`（n=40）。面板顺序确认为 `[source | author-JSCC | DiffJSCC]`。
- SGD montage 尺寸 `2066×4130`，`crop_sgd_tile` 复用既有已验证实现（8 列、stride 258、重建行偏移 +8）。
- **SGD 源 tile 与 DiffJSCC 源面板逐字节相同：checked=64, mismatch=0。** 证明两条链确实共享同一总体，跨方法比较合法。
- SGD 重建 PSNR 对账 max|Δ| = `0.019832 dB`、median = `0.002016 dB`。该残差与 S34C 记录的 uint8-截断 vs float-张量指标口径差一致，不是裁切错误。
- 环境：RTX 4090 D 23.54 GiB、torch 2.11.0+cu128、`cleanfid`/`lpips`/`sklearn 1.7.2` 就位。
- 两个 VAE 权重均为本地资产，无需下载：DiffJSCC `first_stage_model.*` 在 `model.ckpt` 内；SGD VAE 权重在 `JSCC_model.pth` 内（`scripts/s34d_measure_sgd_cost.py:223-271` 已有先例，作者硬编码路径不被使用）。

## 边界

- 不训练任何生成模型；指纹分类器为轻量分析工具，不进入任何方法主链。
- SGD 永久 non-ranking paper upper，本轮只做分布分析。
- official Imagenette validation 继续封存。
- 不覆盖任何既有实验目录；本轮全部输出写入新 ID。
- 不联网、不下载。
- 本轮结论只针对"是否存在可识别偏移"，**不**声称任何方法实现了 deception，也不声称 RDD 框架适用于本项目。
