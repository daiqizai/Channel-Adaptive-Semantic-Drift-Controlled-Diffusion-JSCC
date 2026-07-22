# 里程碑与收敛标准

本文件用于防止课题无限扩张。CIFAR-10 只作为 JSCC sanity baseline；diffusion 主路线必须使用高分辨率自然图像数据。

## 2026-07-21 主线修订（用户授权）

S32 已证明冻结的 `19,712-real` strong channel-adaptive JSCC 在已知 policy-dev 总体上相对 author-JSCC 五档聚合 PSNR 为 `+0.433774 dB`，95% CI `[+0.328020,+0.554007]`。因此从 S33 起，项目资源优先级修订为：

1. **强 channel-adaptive JSCC backbone 的严格等码率验证与独立论文闭环。**
2. **证明 strong 增益来源的受控消融。**
3. **仅在新 strong 分布上仍有不可替代增益时，保留受控 diffusion 为加分项/第二方向。**
4. **全部方法冻结后，一次性解封 official Imagenette validation。**

本修订不删除既有 M0--M3 diffusion 最小闭环、负结果或 semantic-drift 纪律；它只改变下一篇小论文和后续算力的优先级。任何生成/感知模块仍必须报告 semantic failure。S32 是 policy-development 证据，不能写成 independent final test；所有“追平/超过 author-JSCC”的对外 claim 必须先通过 S33 的 `16,384 real` 等码率 gate。13/19 dB 当前较弱是必须保留的边界，可表述为 low-SNR-regime 优势，不得隐藏。

## 最小论文闭环

第一版论文必须先完成以下闭环：

- Sanity 数据集：CIFAR-10 test set 或固定 test subset，用于验证 JSCC/channel/metric 流程。
- 主数据集：COCO2017 train/val subset，训练和评估统一裁剪到 `256x256`。
- 补充数据集：Kodak，只用于视觉质量补充测试和样例展示，不作为 semantic drift 主统计集。
- 主信道：AWGN。
- SNR：沿用阶段1配置的 `[1, 4, 7, 13, 19]` dB。
- CBR：先固定为 `0.17`。
- JSCC baseline：CIFAR-10 使用已接入 checkpoint；COCO-256 必须训练或接入高分辨率 DeepJSCC checkpoint。
- 方法对比：至少包含 `M0` 到 `M3`。
- 指标：必须同时报告图像质量、感知质量、语义可靠性和推理开销。
- 输出：必须保存指标表、配置副本、样例图、semantic drift failure case。

完成 COCO-256 AWGN 最小闭环前，不引入 Dynamic_JSCC、DeepJSCC-l++、PJSCC、DiT-JSCC 或新的大模型主线。

## 方法分组

正式实验至少包含四组：

- `M0-DeepJSCC`：只使用 DeepJSCC reconstruction，输出 `x_hat`。
- `M1-BlindDiffusion`：对 `x_hat` 做固定强度 diffusion refinement，不使用 SNR-adaptive 或 semantic control。
- `M2-SNRAdaptiveDiffusion`：根据 SNR 调整 diffusion strength，不使用 semantic guidance 或 failure detector。
- `M3-Ours`：SNR-adaptive diffusion strength + semantic consistency control + semantic failure handling。

可选扩展必须放在最小闭环之后：

- Rayleigh 信道。
- ImageNet 子集。
- Dynamic_JSCC / DeepJSCC-l++ 对照。
- CLIP-guided 或 DiT-style diffusion 变体。

## Semantic Drift 定义

第一版必须使用一个冻结语义模型 `T_cls`，例如 CIFAR-10 classifier。设：

- `c(z)`：`T_cls` 对图像 `z` 的 top-1 类别。
- `p(z)`：`T_cls` 对 `c(z)` 的置信度。
- `y`：数据集真实类别。
- `x`：原图。
- `x_hat`：DeepJSCC decoder 输出。
- `x_refined`：diffusion refinement 输出。
- `x_final`：经过 failure handling 后的最终输出。

不同方法中的默认关系：

```text
M0: x_final = x_hat
M1/M2: x_final = x_refined
M3: x_final = accepted x_refined, fallback x_hat, or weaker refinement output
```

正式统计优先在 clean-correct 子集上进行：

```text
A = {i | c(x_i) = y_i and p(x_i) >= tau_clean}
```

第一版主指标：

```text
Drift-Origin = mean_i[ c(x_refined_i) != c(x_i) ], i in A
Drift-GT = mean_i[ c(x_refined_i) != y_i ], i in A
Refinement-Drift = mean_i[ c(x_refined_i) != c(x_hat_i) ], i in A
Final-Failure = mean_i[ c(x_final_i) != c(x_i) ], i in A
Prediction-Consistency = mean_i[ c(x_final_i) = c(x_i) ], i in A
```

若使用 CLIP 或其他语义特征模型，只能作为辅助指标：

```text
CLIP-Drift = mean_i[ sim(T_clip(x_i), T_clip(x_refined_i)) < tau_clip ]
```

不能只用 CLIP similarity 替代分类一致性主指标。

## Semantic Failure Handling

第一版 failure detector 可以简单，但必须可复现：

- 若 `x_refined` 的语义不可信，输出回退到 `x_hat`，或降低 diffusion strength 后重试。
- 必须记录 detector 的接受率、拒绝率和最终 failure rate。
- 若 detector 接受了语义错误结果，记为 false accept。
- 若 detector 拒绝了语义正确且质量更好的结果，记为 false reject，可选统计。

第一版不要求 detector 完美，但必须证明它不是只提高视觉指标、同时放任语义错误。

## Diffusion 第一版边界

第一版 diffusion refinement 只允许作为 DeepJSCC 后处理模块：

- 输入是 `x_hat`，输出是 `x_refined`。
- 不从零训练大型 diffusion 或 DiT-JSCC。
- 不把 diffusion 替换成主 JSCC decoder。
- 不使用需要人工文本 prompt 的流程作为主实验。
- 不在 test set 上调 diffusion strength、guidance weight 或 threshold。

SNR-adaptive strength 必须满足：

- strength 随 SNR 升高而不增加。
- 高 SNR 少修，低 SNR 多修。
- semantic guidance 或 failure handling 在低 SNR 下不能弱于高 SNR。

示例约束：

```text
strength(1 dB) >= strength(4 dB) >= strength(7 dB) >= strength(13 dB) >= strength(19 dB)
semantic_weight(1 dB) >= semantic_weight(4 dB) >= ... >= semantic_weight(19 dB)
```

具体数值必须只在 validation subset 上确定。

## 成功判据

`M3-Ours` 的目标不是在所有指标上绝对最优，而是在感知质量和语义可靠性之间取得更好的 tradeoff。

优先成功判据：

- 相比 `M1-BlindDiffusion`，`M3-Ours` 在低/中 SNR 下有更低 semantic drift 或 final failure。
- 相比 `M0-DeepJSCC`，`M3-Ours` 保留主要感知质量收益，例如 LPIPS 或 FID 改善。
- 相比 `M2-SNRAdaptiveDiffusion`，`M3-Ours` 证明 semantic control 不是多余模块。

以下情况不能算课题成功：

- 只提升 PSNR/MS-SSIM，但 semantic drift 没有统计。
- 只提升 LPIPS/FID，但 drift 或 final failure 明显上升。
- 只在高 SNR 有效，低 SNR 下 diffusion 大量 hallucination。
- 只展示少量好看样例，没有固定 test split 的统计。

## 阶段门槛

### S1 DeepJSCC Baseline

完成标准：

- smoke test 能加载 checkpoint、切换 SNR、输出重建图和 PSNR。
- mini-eval 能在固定 CIFAR-10 subset 上输出 `M0` 指标。
- `EXP-S1-001` 写入 `EXPERIMENTS.md`。

### S2-HR High-Resolution DeepJSCC

完成标准：

- 准备 COCO2017 train/val 或等价自然图像高分辨率数据。
- 训练或接入 `256x256` DeepJSCC checkpoint，至少覆盖 AWGN 和 CBR `0.17`。
- 在固定 COCO val subset 上输出 `M0-HR` 指标和样例图。
- 记录 checkpoint、训练配置、数据 split 和训练日志。

### S3 Blind Diffusion

完成标准：

- `M1` 在相同图像、相同 SNR、相同 CBR 上可复现运行。
- 保存 `x_hat`、`x_refined` 和样例对比。
- 报告视觉指标和初步 semantic drift。

### S4 Semantic Metrics

完成标准：

- 冻结 `T_cls`。
- 固定 clean-correct 子集和阈值。
- Drift-Origin、Drift-GT、Refinement-Drift、Final-Failure 中至少实现前三项；进入 adaptive control 前必须实现 Final-Failure。

### S5 Adaptive Control

完成标准：

- 实现 SNR-adaptive diffusion strength。
- 实现 semantic consistency control 或 failure handling。
- 完成 `M0` 到 `M3` 的同表对比。

### S6 完整实验

完成标准：

- 在所有固定 SNR 上完成正式实验。
- 至少输出一张 tradeoff 图。
- 至少整理一组 semantic drift failure case。
- 明确写出方法成功、部分成功或失败的结论。

### S7 强 JSCC 保真基座（最小闭环完成后的收敛修正）

S30 外部复现证明旧 14 万参数、固定 7 dB、非原生低码率主干已成为主要瓶颈。用户确认后允许在不改变 AWGN、固定总码率和 semantic-drift 主问题的前提下升级保真基座。完成标准：

- clean-room 强主干原生输出严格 `19,712` 个实符号，不允许先生成更密 latent 后固定裁剪或用零填充冒充 exact rate。
- 编码器和解码器均显式接收 SNR，第一版仍只优化重建损失，不引入 diffusion 或语义标签选择 checkpoint。
- 必须先通过功率、码率、有限梯度、固定验证噪声和可恢复 checkpoint 审计。
- 冻结强 JSCC 后，才允许在 S20 相同总体/噪声上与 B1、current 和 author-JSCC 对比；不能用训练期 COCO validation 调外部排名。
- 旧 B1/S19/diffusion checkpoint 只保留为历史证据，不能不经重训直接接到新 latent/重建分布并称为最终方法。
- 历史立项时强基座用于服务“生成收益与 semantic drift 风险控制”；自 2026-07-21 用户授权主线修订后，strong backbone 允许优先形成独立 channel-adaptive JSCC 小论文，但不得退化为只报 PSNR：仍须同时报告 LPIPS、MS-SSIM、语义 failure、码率、分 SNR 边界和统计不确定性。

状态（2026-07-21）：**完成。** S31b best 原生发送 `19,712 real`，31.12M 参数，固定 COCO 五档平均 `29.360583 dB/0.967330 MS-SSIM`；冻结后 S32 在 S30 同 960 键上相对 author-JSCC 聚合 PSNR `+0.433774 dB`、LPIPS `-0.005518`，两项 cluster CI 均显著有利。该结果建立了 backbone 主线的可行性，但因为 strong/author 分别使用 `19,712/16,384 real`，尚未完成严格等码率论文 gate；下一阶段先执行 S33，而不是先重训 diffusion。

### S8 / S33：`16,384-real` 等码率 Strong Backbone（最高优先级）

状态（2026-07-21）：**完成并通过最高优先级 gate。** 最终 `16,384-real` strong checkpoint SHA=`2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`；S32 同 960-key policy-dev 上 strong−author 聚合 PSNR=`+0.479929 dB`，95% CI=`[+0.370006,+0.598197]`，按冻结规则为显著超过。MS-SSIM=`+0.006616`、LPIPS=`-0.008357`、failure-rate=`-0.013542`，CI 也均显著有利。1/4/7 dB 显著领先；13 dB PSNR显著但 LPIPS较差；19 dB 未通过逐档 0.10 dB 非劣 gate且 LPIPS较差。结论仅为 known policy-dev exact-rate positioning；本轮不启动 S34--S36。

冻结输入：

- S31b `19,712-real` strong B0 checkpoint 永久冻结，SHA-256=`2f8972a943599bae016f6f64550ca81ea5f861654d9ace6931aebe6cf9057ca8`，不得覆盖或续训。
- author-JSCC、S32 64 图 population、3 个 canonical channel seed、5 个 SNR、指标实现和语义分类器全部复用冻结版本。
- official Imagenette validation 继续封存。

必须满足的物理与训练合同：

- 新模型原生输出严格 `64x16x16=16,384 real`，对应 `8,192 complex channel uses`；当前实现精确可训练参数为 `31,028,163`，与约 `31.289M` 参数的 author-JSCC 相差不到 1%；mask、padding、固定裁剪、重发和 side information 均为 0。
- AWGN 继续采用项目 paired-real half-variance 口径，逐 key 使用与 author-JSCC 相同的前 `16,384` 个 canonical real noise coordinates。
- 第一版保持 S31b 的四级结构、encoder/decoder SNR conditioning、COCO 数据、MSE-only selection 和五档 `[1,4,7,13,19]` dB 离散逐图均匀训练；不得用 S32 外部排名或 official validation 选 checkpoint。
- 正式训练前必须通过 exact-symbol、归一化功率、finite forward/backward、固定 validation noise、可恢复 checkpoint 和 output-dir 不覆盖审计。

必须输出：

- 64 图×3 channel seeds×5 SNR 的逐样本和 per-SNR PSNR、LPIPS、MS-SSIM、semantic failure/new-error/repair。
- strong−author 的五档聚合差值，以及按 source image 聚类的 95% CI；随机种子、噪声 SHA、checkpoint/config/script SHA 和 symbol ledger 必须落盘。
- 13/19 dB 单列，不得只报五档平均。

判定：

- 聚合 PSNR 差的双侧 95% CI 下界 `>0`：可称严格等码率下显著超过 author-JSCC。
- 聚合 PSNR 差的 95% CI 下界 `>-0.10 dB`、但不大于 0：只可称在预注册 `0.10 dB` margin 下非劣/追平。
- 若 LPIPS、MS-SSIM 或 semantic failure 与 PSNR 结论冲突：只可称 Pareto，不可称全面超过。
- 若 PSNR CI 下界不高于 `-0.10 dB`：backbone 小论文 gate 未通过；允许诊断和按原预注册消融解释，禁止事后改 margin 或使用 S32 `19,712-real` 结果替代等码率 claim。

### S9A / S34A：SwinJSCC 严格等设定外部骨干对比（审稿 gate）

状态（2026-07-22）：**smoke 已通过；Base-SA 已完整完成 5/12 epochs，并在一次前台会话退出后从 epoch5 checkpoint 通过 detached screen 恢复 epoch6；CM-SA 后续串行。epoch 9--12 只做收敛检查，extension 未授权。**

S33 通过 author-JSCC exact-rate gate 后，在内部因果消融前补充官方 SwinJSCC Transformer 骨干。完整预注册见 `reports/swinjscc_equal_rate_comparison_preregistration_2026-07-22.md`，计划配置见 `configs/s34a_swinjscc_equal_rate_comparison.yaml`。必须满足：

- 官方源码固定 `semcomm/SwinJSCC@a6d0e6da53548976acbe9317839a077ef31f190f`；不用不同数据/码率/SNR 的官方 checkpoint 参与初始化或排名。
- fixed-rate 使用 `SwinJSCC_w/_SA`、`C=64`，原生 exact `64x16x16=16,384 real`，无 mask/padding/side information。
- 推荐同时训练未改深度的 official Base（`28,182,512` 参数）和仅把第三 stage 从 6 增至 8 的 capacity-matched official-code control（`31,348,752` 参数，距 S33 `+1.03%`）；总判定取两臂中更保守的结果。
- 与 S33 使用相同 COCO manifest/增强、逐图离散五档 SNR、逐图功率、paired-real half-variance AWGN、FP32 4+8 epoch/equal optimizer-step 和 MSE-only checkpoint selection。
- 冻结后才可在同 64 图×3 seed×5 SNR policy-dev 上使用相同 canonical 16,384-D noise prefix；逐档/聚合报告 PSNR、LPIPS、MS-SSIM、semantic failure/new-error/repair 和 source-cluster 95% CI。
- S33−Swin 的 PSNR CI 下界 `>0` 为显著超过，位于 `(-0.10,0]` 为非劣，`<-0.10` 为劣于；二级指标冲突降为 Pareto。13/19 dB 必须单列。
- official Imagenette validation 继续封存；本轮 formal 输出只允许两个预注册 equal-budget 目录，每臂硬上限 12 epochs。
- 12 epochs 不能预先视为 SwinJSCC 已收敛。双臂必须逐 epoch 保存同一固定 val512 曲线，并按 epoch 9--12 gate 报 `triggered/not_triggered`。即使触发也不得自动 extension；必须先报用户，再由用户决定是否延训和延到多少 epochs。此前讨论的 60-epoch 上限不构成本轮授权。

### S9B / S34B：Strong Backbone 因果消融（必需）

状态：**待 S34A 外部骨干对比冻结后执行。**

在 exact `16,384 real`、同数据、同训练样本/增强、同 optimizer steps、同 checkpoint selection 下做 one-factor-at-a-time control：

- `no-sample-CSI`：encoder/decoder 使用常量 condition；条件分支保留，避免用删参数混淆 CSI 效果。
- `three-level`：原生 `16x32x32=16,384 real` 的三级下采样版本；通过 width/blocks 将总参数控制在 full 的 `±2%`，其余 SNR conditioning 和训练合同不变。
- `single-7dB-train`：固定 7 dB 训练，架构、参数和训练步数不变；用于隔离五档逐图随机训练合同。

每个 control 都必须相对 full 报 per-SNR/aggregate 四类指标和 source-cluster 95% CI。当前 S31/S31b/S33 使用离散五档训练，不得把未来可能增加的连续 `Uniform[1,19] dB` 变体倒写成现有增益来源；连续变体若执行，登记为独立扩展。author-JSCC 与 S33 strong 的总参数约为 `31.289M/31.028M`，本身已近似参数匹配，论文仍须用上述同参数 OAT control 说明增益不是简单扩大模型。

### S10 / S35：新 Strong 分布上的 Matched B1 / M2 / Envelope（条件性第二方向）

状态：**待 S33/S34 冻结后执行；不得抢占 S33/S34。**

- 冻结新 strong B0，从新重建分布重新训练 new-B1、M2/identity envelope、matched residual diffusion/fusion。
- 旧弱 backbone 的 B1/S18/S19/M2/envelope checkpoint 只作历史或 distribution-shift 诊断，不能直接迁移为正式方法。
- 低 SNR 主评估为 1/4/7 dB；13/19 dB 默认 exact-B0 fallback，并保留 high-SNR LPIPS 边界。
- 必须有参数量、训练数据和优化预算匹配、但不读取 diffusion observation 的 control。

继续 gate：diffusion 分支相对 matched control 在低 SNR 的预注册主要质量指标必须有 CI 支持，并且 semantic failure/new-error 不恶化。通过则保留为第二方向或 backbone 论文加分节；不通过则记录为 negative/limitation，并结束该轮 diffusion 扩展，不因视觉个例继续扫模块。

### S11 / S36：Official Imagenette Validation 一次性解封

状态：**封存。**

只有 S33--S35 的方法、checkpoint、route、阈值、统计脚本、claim 和停止规则全部冻结后才能解封。一次性最终运行至少包括 author-JSCC、`16,384-real strong`、`19,712-real strong`，以及仅在 S35 通过继续 gate 时保留的 diffusion 方法。运行后不得据 official validation 调参、改变模型或回选 checkpoint；任何失败同样进入最终报告。

## 复现记录

每个正式实验必须保存：

- 实验 ID。
- 日期。
- 项目 commit；如果当前目录不是 git 仓库，写 `N/A (not a project git repo)`。
- 第三方 baseline commit。
- config 路径和 config 内容副本。
- 运行命令。
- 数据 split 或样本 ID。
- 随机种子。
- checkpoint 路径。
- 输出路径。
- 环境信息，至少包含 Python 版本和核心依赖版本。

如果没有项目 git commit，必须额外记录本次使用的脚本路径、配置路径和关键源码路径，避免实验无法追溯。
