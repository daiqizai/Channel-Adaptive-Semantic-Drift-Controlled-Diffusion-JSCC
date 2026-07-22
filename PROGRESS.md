# 当前进度

## 当前方法说明书（2026-07-22）：S33 主链与旧命名关系已厘清

新增 `METHOD_CURRENT.md`，只描述截至 S33 真正活着的论文主方法，不复述历史实验流水账。核心边界已经明确：当前 inference 是 `256x256 RGB → 四级 SNR-conditioned residual encoder → 原生 64x16x16=16,384 real → 逐图单位功率 + paired-real AWGN → 四级 SNR-conditioned decoder → 256x256 RGB`；decoder 输出就是最终输出，没有 B1、M2、diffusion、envelope 或分类器 gate。S31 是同架构 `77x16x16=19,712 real` 的较宽工作点，S33 是严格与 author-JSCC 等码率的主版本；两者均无 mask/prefix/padding/side information。

名词状态也已冻结：B0 是“纯 JSCC 输出”的角色，因此 S33 可称 new strong-B0；旧 B1 是围绕旧弱 B0 训练的 deterministic residual refiner，不等于 M2；M2 是 SNR-aware diffusion 方法组，D 是其 diffusion 候选输出，envelope 是控制 diffusion correction 的 SNR 强度/恒等规则。旧 B1/M2/D/envelope 的具体 checkpoint 均不在当前主链，不能直接迁移到 strong 分布；diffusion 研究方向只是暂挂，只有将来从 strong-B0 分布合法重训并通过 gate 后才可能作为第二方向恢复。文档同时解释了完整模块维度、`16,384/19,712 real` 的原生 exact-rate 计算、canonical-noise prefix 与 latent 裁剪的区别，以及当前论文可写/不可写的 claim。

## S34A SwinJSCC equal-budget 已获分阶段授权（2026-07-22）：只跑双臂 12 epochs，extension 禁跑

用户已接受 official Base-SA `28,182,512` 参数与 capacity-matched CM-SA `31,348,752` 参数双臂、S33 FP32 4+8 epoch/equal-step 合同，以及总 verdict 取对 S33 更不利一臂。本轮授权边界已进一步收紧为：只执行两臂各 12 epochs，并检查 epoch 9--12 的固定 COCO val512 曲线；若最佳点在 epoch 11/12、OLS slope `>=0.01 dB/epoch` 且 epoch12−epoch9 `>=0.03 dB`，只报告 extension trigger，**不得自动延训**。此前讨论的 60-epoch 上限不构成当前权限；extension 是否执行及其 epoch 数必须等用户查看 equal-budget 结果后另行决定。

正式训练运行中。运行配置 SHA=`a209af08...676d`；Base-SA 已完整完成 epochs 1--5，aggregate val PSNR=`26.9447→27.2621→27.9832→28.1977→28.2061 dB`，epoch5 MS-SSIM=`0.960111`，五档 PSNR=`26.1814/27.5046/28.4294/29.3187/29.5963 dB`。原前台会话在 epoch6 的 6,000/14,786 microbatches 处退出；该 partial epoch 没有写入 history/checkpoint，现从完整 epoch5 checkpoint 恢复，所以不会把半轮结果混入曲线。

首次恢复预检发现 `torch.load(map_location=cuda)` 会把 checkpoint 的 CPU RNG byte state 一并搬到 CUDA，导致 `torch.set_rng_state` 拒绝；这发生在任何新 optimizer step 前，checkpoint 未损坏。恢复脚本已做窄修复：只把 CPU/default、SNR 和 shuffle generator state 显式搬回 CPU，不改变模型、optimizer 或训练数学，并写入 `resume_event_before_epoch_06.json` 记录新旧脚本 SHA。精确恢复测试 PASS。任务现运行在 detached GNU screen `s34a_equal_budget`（screen PID parent=`1`），不再依赖对话会话；2026-07-22 09:27 CST GPU 利用率=`99%`、显存约 `10.95/24.56 GiB`，Base 正在重跑 epoch6。CM-SA 仍只会在 Base 恰好完成 12 epochs 后启动，extension 无入口。

双臂完成后的评估也已按独立配置 `configs/s34a_swinjscc_equal_budget_evaluation.yaml` 排队，但在 checkpoint 冻结前不会访问 policy-dev。评估入口只接受两个训练 `summary.json` 选出的 continuation-best 及其落盘 SHA，拒绝任何 epoch>12/extension checkpoint；随后复用 S33 的 960 keys、完整 19,712-D canonical noise SHA 和相同前 16,384-D prefix，输出两臂 aggregate/per-SNR PSNR、LPIPS、MS-SSIM、failure/new-error/repair、source-image cluster 95% CI 与 `0.10 dB` margin 保守 verdict。official validation 始终封存。

官方源码已从 `semcomm/SwinJSCC@a6d0e6da53548976acbe9317839a077ef31f190f` 的 GitHub codeload tarball 经服务器直连取得；tarball `17,887 bytes`、SHA-256=`3f837eef...21688`，本地逐文件 hash 与此前缓存静态审计完全一致，没有下载或使用官方 checkpoint。项目侧新增 adapter，仅把官方 SA-only Swin/Channel ModNet 拓扑接到逐图 SNR、逐图单位功率和 canonical paired-real AWGN；第三方算法源码没有改动。

`SMOKE-S34A-SWINJSCC-CALIBRATION-001` 在真实 COCO microbatch=8 上对两臂各运行一次 FP32 forward/backward/AdamW step，均通过：latent=`[8,256,64]` 即每图 `16,384 real`，输出=`[8,3,256,256]`，最大单位功率误差均为 `1.1921e-7`，参数量与静态审计精确一致，有限梯度和 checkpoint strict round-trip 通过。另用同一模型、同一 scalar SNR 对项目 adapter 与官方原始 encoder/decoder forward 做数值对照，最大绝对差均为 `0`，确认 vectorization 没有改变 scalar-SA 算法。Base/CM peak reserved VRAM=`9.748/10.396 GiB`，因此正式训练冻结 microbatch=8、gradient accumulation=4，保持 effective batch=32。单 microbatch 计时 Base=`0.537s`、CM=`0.179s`，但 Base 是进程首臂，包含 CUDA cold-start，禁止据此声称 Base 比 CM 慢三倍；按 warm CM 估算单个 12-epoch 臂纯训练约 `9h`，保守计入冷启动、逐 epoch 五档 validation、I/O 后约 `10--14h`，双臂约 `20--28h`。smoke result SHA=`010d9befe9...cacb3`。1-batch 不能判断收敛，当前收敛状态仍为 unknown；必须等 12-epoch val 曲线。official validation 未访问，正式训练输出未创建。

完整更新合同：`reports/swinjscc_equal_rate_comparison_preregistration_2026-07-22.md`；配置当前为 `equal_budget_dual_arm_authorized_extension_forbidden`。允许创建且仅允许创建 Base/CM 两个 equal-budget 目录；每臂训练脚本必须硬拒绝超过 12 epochs，official validation 继续封存。

## S34A SwinJSCC 初始预注册（2026-07-22）：已由上方确认与 smoke 结果更新

S33 已在严格 `16,384 real` 下显著超过 author-JSCC；用户要求下一项补充公认更强的 Transformer JSCC 骨干 SwinJSCC。官方 `semcomm/SwinJSCC` 代码可公开获取，本轮静态审计固定 `main@a6d0e6da53548976acbe9317839a077ef31f190f`；README 提供公开 Google Drive 权重入口，但这些权重使用作者的 DIV2K/CLIC、码率/SNR 合同，不得参与本轮初始化或排名。当前服务器清空代理后直连 Drive 超时，因此权重状态只写“公开入口存在”，不写“本机已下载验证”。官方源码仓库未发现 `LICENSE`，后续须保留许可边界。

公平对比选择 fixed-rate `SwinJSCC_w/_SA` 而非带 Rate ModNet/mask 的 `w/_SAandRA`；`C=64` 原生输出 `64x16x16=16,384 real`、`8,192 complex uses`、CBR=`1/24`，无 mask/padding/side information。建议以双臂堵住两类审稿疑问：原版 Base `[2,2,6,2]` 静态实测 `28,182,512` 参数（比 S33 少 `9.17%`），以及只把第三 stage 深度增到 8 的 capacity-matched official-code control `31,348,752` 参数（比 S33 多 `1.03%`）。总 verdict 取两臂中对 S33 更不利的结论，禁止只挑较弱一臂。

两臂拟完全复用 S33 的 COCO train/val manifest、增强、离散逐图 `[1,4,7,13,19] dB`、逐图单位功率、paired-real half-variance AWGN、随机初始化、FP32 4+8 epochs、equal optimizer-step、MSE-only selection；评估复用同 64 图×3 seed×5 SNR 与 canonical noise prefix，报告 aggregate/per-SNR PSNR、LPIPS、MS-SSIM、semantic failure/new-error/repair 和 source-cluster 95% CI。`0.10 dB` PSNR margin 规则保持不变；secondary metric 冲突即降为 Pareto。official Imagenette validation 继续封存。

该段保留初始审计背景；确认状态、收敛补充和实测校时以上方最新 S34A 记录为准。

## S33 阶段成果（2026-07-21）：严格 `16,384 real` 等码率下显著超过 author-JSCC

用户确认的“随机初始化 + FP32 12 epochs + 离散五档 SNR 训练”已完整执行；连续 `Uniform[1,19]` 未使用，只保留 future work。新 strong 原生输出 `64x16x16=16,384 real`、`8,192 complex uses`，参数 `31,028,163`，与约 `31.289M` 的 author-JSCC 参数差不到 1%；无 mask/padding/裁剪/side information。主阶段 4 epochs 从 `26.0623` 升至 `28.5879 dB`；续训首点回落至 `28.5105` 后持续升至 epoch7 的 `29.415098 dB/0.966782 MS-SSIM`。最终 checkpoint SHA=`2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`，五档 COCO PSNR=`27.3119/28.5714/29.5445/30.6344/31.0134 dB`，最大功率误差 `2.38e-7`。

冻结后在 S32 相同 64 图×3 seeds×5 SNR policy-dev 上，strong/author 的 PSNR=`30.466064/29.986135`、MS-SSIM=`0.969708/0.963092`、LPIPS=`0.119985/0.128342`、failure=`9/22`。strong−author PSNR=`+0.479929 dB`，source-image cluster 95% CI=`[+0.370006,+0.598197]`，按冻结规则为 **显著超过**；MS-SSIM delta=`+0.006616`、LPIPS delta=`-0.008357`、failure-rate delta=`-0.013542`，三个 CI 也均显著有利。artifact audit 验证 `960/960` 唯一键、author 行逐字段一致，以及完整 19,712-D noise SHA 后取前 16,384-D prefix 的 `960/960` 同噪声合同。

分 SNR PSNR delta=`+0.9663/+0.7717/+0.5261/+0.1312/+0.0043 dB`。1/4/7 dB 显著领先；13 dB PSNR CI 下界 `+0.0044`，但 LPIPS 显著更差 `+0.002205`；19 dB PSNR CI `[-0.1615,+0.2027]`，按 `0.10 dB` margin 规则未过非劣且 LPIPS 显著更差 `+0.006905`。因此对外只写聚合严格等码率显著优势与 low-to-mid-SNR 主导，不写每档全面支配。S33 仍是 known policy-dev，不是 independent final test。

最高优先级 backbone 论文经验 gate 已通过。按用户本轮指令在此停止：S34 消融、S35 matched diffusion、S36 official validation 均未启动。完整中文报告：`reports/strong_jscc_16384_equal_rate_stage_result_2026-07-21.md`；S33 `per_sample.csv` SHA=`5e585ca5...7dbe1`，独立 `post_analysis.json` SHA=`d7a89ef2...bdfc`。

## 主线调整与下一轮预注册（2026-07-21；S33 已完成，S34--S36 未执行）：strong backbone 优先，diffusion 改为条件性第二方向

用户基于 S32 的冻结结果正式授权调整项目主线：在项目 `19,712 real` 上，S31b strong 相对 DiffJSCC author-JSCC 的五档聚合 PSNR 为 `+0.433774 dB`，source-image cluster 95% CI `[+0.328020,+0.554007]`，同时 MS-SSIM/LPIPS 聚合也有利。后续第一优先级改为 **强 channel-adaptive JSCC backbone 本身作为独立贡献**，目标是形成一篇面向 IEEE WCL / Communications Letters 或对口会议的小论文，并作为毕业论文基石；原“强 JSCC + 受控 diffusion 可靠性”降为取决于 matched 重训结果的加分项/第二方向。既有 diffusion 最小闭环和负结果继续保留，不回写、不删除，也不再支配下一轮资源顺序。

当前 S31b strong B0 永久冻结为 `outputs/train/EXP-S31B-STRONG-JSCC-FP32-002/checkpoints/best.pt`，SHA-256=`2f8972a943599bae016f6f64550ca81ea5f861654d9ace6931aebe6cf9057ca8`；后续不得覆盖、续训或据新结果回选该 checkpoint。S32 仍只定位为已知 policy-development population，不是 independent final test。13/19 dB 相对 author-JSCC 的已知边界必须逐档报告：PSNR 分别为 `-0.028961/-0.225914 dB`，LPIPS 分别为 `+0.008213/+0.013666`（更差）；允许把结论写成 low-SNR-regime 优势，禁止用聚合值掩盖高 SNR 边界。

下一轮严格按以下优先级顺序执行。用户已于 2026-07-21 确认 S33 使用“随机初始化 + FP32 12 epochs + 离散五档 SNR 训练”及 `0.10 dB` PSNR 非劣 margin；本轮只启动 S33，S34--S36 仍停留在计划态，official Imagenette validation 继续封存：

1. **S33：16,384-real 等码率 strong（最高优先级，论文 gate；已完成）。** strong−author 聚合 PSNR=`+0.479929 dB`，95% CI=`[+0.370006,+0.598197]`，判定为显著超过；聚合 LPIPS/MS-SSIM/failure 也显著有利。分档 high-SNR 边界和 policy-dev 限定见本文件首节；本轮在此停止，不自动启动后续步骤。
2. **S34A：SwinJSCC 严格等设定外部骨干对比（审稿 gate；已预注册、未运行）。** 用官方 fixed-rate `w/_SA`、`C=64` 在 S33 的 COCO/码率/SNR/训练预算下从零重训；推荐同时跑 official Base 与 `31.349M` capacity-matched control，总判定取更保守一臂。用户确认双臂和 equal-step 训练预算前不得启动，official validation 继续封存。
3. **S34B：strong 增益来源消融（必需）。** 以冻结的 S33 full model/训练合同为参照，做 one-factor-at-a-time 且 exact-rate、训练步数匹配的三类 control：`(a)` 将 encoder/decoder 的样本级 SNR condition 替换为常量 condition，保留相同条件分支和参数预算；`(b)` 用原生 `16x32x32=16,384 real` 的三级下采样 control 替换四级结构，并通过宽度/blocks 把参数量控制在 full 的 `±2%`；`(c)` 将 per-image 五档离散随机 SNR 训练替换为固定 `7 dB`，其他训练样本、增强、优化步和 checkpoint selection 不变。注意：当前成功的 S31/S31b 使用的是 `[1,4,7,13,19]` **离散均匀逐图采样**，不是连续随机 SNR；连续 `Uniform[1,19] dB` 如要测试，只能登记为额外训练合同扩展，不能倒写成现有 strong 的增益来源。三项消融均在 S32 policy-dev 上按相同指标/CI 报告，不用 official validation 调结构。
4. **S35：新 strong 分布上的 matched B1/M2/envelope（中优先级，diffusion 去留 gate）。** 只有 S33/S34 主要结论冻结后才开始。冻结新的 strong B0，从其重建分布重新训练 new-B1、M2/identity envelope 和 matched residual diffusion/fusion；旧弱 backbone 上的 B1/S18/S19/M2/envelope checkpoint 只能作历史或 distribution-shift 诊断，不得直接迁移并称新结果。1/4/7 dB 检验低 SNR 增益，13/19 dB 默认保留 exact-B0 fallback 作为可靠性边界；必须加入参数量/训练预算匹配、但不读取 diffusion observation 的 control。只有 diffusion 分支相对该 control 在低 SNR 的主要质量指标有预注册 CI 支持、且 semantic failure/new-error 不恶化，才能称“不可由同容量 control 替代”并保留为第二方向或论文加分节；否则如实关闭为 negative/limitation，主论文纯走 strong backbone。
5. **S36：official Imagenette validation 一次性最终验证（最后解锁）。** 只有 S33--S35 的方法、checkpoint、route、阈值、统计脚本和 claim 全部冻结后才解封；一次性运行后不得再据结果调参或回选方法。最终表必须把 `19,712-real strong`、`16,384-real strong`、author-JSCC、SwinJSCC，以及仅在 S35 通过 gate 时保留的 diffusion 方法分层呈现。

S33 配置已经用户确认并冻结：原生 latent `[64,16,16]`；随机初始化；FP32 共 12 epochs（前 4 epoch 主训练 + 后 8 epoch只加载主阶段 best model、fresh optimizer 的低学习率 continuation）；训练 SNR 为五档离散逐图均匀采样。连续 `Uniform[1,19] dB` 只记 future work，不得倒写为当前反超原因。主阶段配置为 `configs/s33_strong_jscc_16384_fp32_main.yaml`，完整预注册为 `reports/s33_strong_jscc_16384_preregistration_2026-07-21.md`。按 RTX 4090 D 历史实测 FP32 `15.7 min/epoch`，12 epoch 约 `3.1 h`；加合同 smoke、五档 COCO validation、960-key 对比、bootstrap 与报告，单次无故障预计 `4--6 h`，不需要新增下载。

## S31/S31b/S32 阶段成果（2026-07-21）：强基座已在项目预算内超过 author-JSCC

S31 已实现 clean-room 31,118,032 参数、四级下采样、encoder/decoder 全程 SNR 条件的原生 exact-rate JSCC；`256x256` 输入直接生成 `77x16x16=19,712 real`，无 mask、padding 或 side information。单测、GPU smoke、half-variance AWGN 和归一化功率合同通过。原 AMP 实验在 epoch3 达到 `28.0448 dB/0.958405` 后，于 epoch4 batch418 检出非有限梯度并 fail-closed；独立 FP32 审计表明相同 checkpoint 在 batch8--32 的完整 step 均有限。

修正 seed 合同后的 `EXP-S31B-STRONG-JSCC-FP32-002` 只加载上述 epoch3 模型权重，不加载 optimizer/scheduler/scaler；8 个 FP32 epoch 全部 finite，COCO 固定 512 图五档平均从 `28.5686` 单调升到 `29.360583 dB/0.967330`。最终分 SNR PSNR 为 `27.4221/28.6232/29.5118/30.4640/30.7819 dB`，最大功率误差 `2.38e-7`；best epoch7 SHA `2f8972a9...57ca8`。此前 `-001` 因总 seed 会改变 val512，在任何 validation 结果前主动中止并保留为 0-row 合同失败。

S32 在 checkpoint/SHA 冻结后，首次把 strong 放到 S30 同 64 图×3 seed×5 SNR。strong 为 PSNR/MS-SSIM/LPIPS/failure `30.419910/0.970266/0.122824/14`；author-JSCC 为 `29.986135/0.963092/0.128342/22`。strong−author 的 PSNR `+0.433774 dB`，cluster 95% CI `[+0.328020,+0.554007]`；LPIPS `-0.005518`，CI `[-0.007775,-0.003147]`，三项质量指标聚合显著有利。边界是 strong 使用完整 `19,712 real`，author 只用 `16,384 real`，且 S32 是已知旧结果后的 policy-dev 定位，不是 independent final test。分 SNR 上 strong 在 1/4/7 dB 明显领先，author 在 13/19 dB 保留 `0.029/0.226 dB` PSNR 优势。

strong 相对完整 DiffJSCC 为 `+2.821512 dB/+0.029467 MS-SSIM/+0.022600 LPIPS`（LPIPS 更差），仍是保真/感知 Pareto；相对旧 current 则 PSNR `+2.196232 dB`、LPIPS `-0.029260`、failure `14 vs 29`，旧 current 已被纯 strong 基座全面压过，不能继续作为最终方法。下一阶段冻结 strong，重新训练与其分布匹配的 matched residual diffusion 与 semantic-risk controller；旧 B1/S19/diffusion checkpoint 只保留历史证据。完整中文报告：`reports/strong_jscc_backbone_stage_result_2026-07-21.md`；S32 `per_sample.csv` SHA `74997b3c...8714a`。official Imagenette validation 仍封存。

## S30 后续诊断（2026-07-21）：JSCC 差距主要是主干与训练合同差距

对冻结 S30 `per_sample.csv` 的只读分 SNR复核表明，author-JSCC 相对 B1 的 PSNR 优势随 SNR 从 `+0.969695/+1.303401/+1.729425/+2.486298/+2.818847 dB` 单调扩大；这更符合本项目低容量/非原生低码率主干提前进入表示瓶颈，而不是 AWGN 注噪口径错误。当前 exact-rate JSCC 只有 `140,239` 个可训练参数，采用两级下采样、固定 `7 dB` 训练且无 encoder/decoder 内部 CSI modulation；它由原 `c8` checkpoint 选取 6 个实 latent 通道 warm-start 到 `c3`，再以固定 evenly-spaced mask 仅传 `19,712/24,576` 个实坐标。DiffJSCC 的 author-JSCC 则是约 `15.6M+15.6M` 参数的四级 ResNet，encoder/decoder 每个残差阶段都做 SNR modulation，并在 `[0,14] dB` 连续随机 SNR、目标 C16 表示上端到端训练。因此 B1/融合器只能修复弱保真端点，不能补回发送端未有效编码的信息。

该诊断不把全部 `1.762457 dB` 归因于单一因素：S30 还包含作者 `256→512` 处理、最终 Lanczos 回到 `256` 的输入网格差异，以及 OpenImage/COCO 训练域差异。下一步若实施强主干替换，应先做固定缩放往返 control，再以同源图、同噪声、同总码率比较“当前 tiny backbone / clean-room strong ResNet / official author-JSCC”；在这之前不能宣称通过增加训练轮数即可追平。

## 最新完整外部复现（2026-07-21）：S30 官方 DiffJSCC 960 行对比完成

S30 已完成官方 `mingyuyng/DiffJSCC@13aeb624...` OpenImage C16 全链：checkpoint、精确 base BLIP2 两分片、OpenCLIP 2.24 和 Transformers 4.51.1 runtime 全部固定尺寸/SHA；preflight、checkpoint audit、preload、1-row smoke、第一 seed 320 行和完整 960 行均 PASS。总体严格复用 S20/S28 的 64 张 Imagenette policy-dev、3 channel seeds、5 SNR 和 canonical AWGN；DiffJSCC 使用同一 19,712-D 噪声向量前 16,384 个实坐标。完整输出有 960 个唯一键、960 张拼图，功率归一化范围 `0.9999998–1.0000002`，无 NaN/OOM。

current 相对 DiffJSCC 最终输出是 fidelity/perception Pareto：PSNR `+0.625280 dB`，source-image cluster 95% CI `[+0.423123,+0.824753]`；MS-SSIM `+0.008258`；LPIPS `+0.051861`（更差），CI `[+0.041360,+0.063002]`；failure `29 vs 23`，差值 CI `[-0.017708,+0.034375]`。预注册 verdict 为 `PARETO_OR_INCONCLUSIVE`，不能宣布 current 全面胜出。

更重要的新发现是 backbone 差距：author-JSCC 前端只用 `16,384 real`、即项目预算的 `83.1169%`，却达到 PSNR/MS-SSIM/LPIPS `29.986135/0.963092/0.128342`、failure `22`；current 为 `28.223678/0.949057/0.152084`、failure `29`。current−author-JSCC PSNR `-1.762457 dB`，CI `[-1.938592,-1.601835]`；LPIPS `+0.023742`，CI `[+0.019926,+0.027543]`。质量三轴均显著落后，failure 点估计也较差但 CI 跨零。下一阶段不应继续在旧弱主干上微调小融合模块，应先把强 JSCC backbone 纳入同一风险控制合同；报告仅提出该建议，未擅自改写 `PROJECT.md/MILESTONES.md`。

DiffJSCC 相对自身 author-JSCC 平均以 `-2.387737 dB/-0.022293 MS-SSIM` 换取 `-0.028119 LPIPS`，failure `22→23`、`10 new / 9 repair`。分 SNR 的 new/repair 为 `1/3、2/4、3/2、3/0、1/0`：1/4 dB 为净修复，7 dB 转为净风险，13 dB 在 3/3 seed 均出现 `1 new/0 repair`；19 dB 属于作者训练范围外。该结果支持“保留 diffusion，但观测越可靠，离开强 JSCC 保真端点所需证据越强”，而不是机械按 SNR 全开/全关。

完整中文报告：`reports/diffjscc_external_comparison_stage_result_2026-07-21.md`。核心产物为 `outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/`；`per_sample.csv` SHA `549720b8...b6e2`，最终派生 `post_analysis_v3.json` SHA `87c2ffcb...aa1f`。完整 DiffJSCC 平均 `5.238 s/图`、peak allocated VRAM `14,927.42 MiB`；S28 current 的毫秒数不含完整 DeepJSCC/diffusion，禁止直接计算速度倍数。

## 最新外部定位（2026-07-21）：当前方法跨域优于 B1，与 SGD-JSCC 构成明确 Pareto

S28 把冻结的 S19 low-SNR fusion + 13/19 dB exact-B1 fallback 放到 S20 完全相同的 64 张 Imagenette policy-dev 图像、3 个 canonical AWGN seed、5 个 SNR 上，共 960 行/方法。相对冻结 B1，当前方法 PSNR `+0.099085 dB`，source-image cluster 95% CI `[+0.088053,+0.111284]`；MS-SSIM `+0.002360`；LPIPS `-0.007314`；T_cls failure `35→29`，`6 new / 12 repair`。相对等容量 matched control 仍有 PSNR `+0.059681 dB`、LPIPS `-0.002990`，两项 CI 显著，第三次证明增益来自 diffusion observation 而非额外 CNN 容量。

相对 SGD-JSCC 免费完美文本论文协议上界，当前方法 PSNR 高 `+0.483309 dB`（CI `[+0.258829,+0.711956]`），但 MS-SSIM 低 `0.003916`、LPIPS 高 `0.079983`，failure 为 `29 vs 25`，因此是清晰的 fidelity/perception Pareto，不能宣称全面胜出。当前方法严格使用 19,712 real、diffusion side information=0；SGD released main+edge 已占 19,712 real，四个 caption packet 至少再需 2,144 real，若计费则最低超预算 `10.88%`。

S28 因 batch=16 相对 S20 batch=64 的单样本 PSNR 最大浮点差 `0.0004768 dB` 超过预注册 `0.0001 dB` 技术阈值，原 verdict 保留 `NEGATIVE`。S29 不改阈值，按原 batch=64 重放 960 行 B1，PSNR/MS-SSIM/LPIPS、预测、failure 和 noise SHA 全部零误差，6/6 PASS，确认只是 batch-dependent 浮点运算而非合同错位。当前内部方法继续冻结；下一步只做严格总码率生成基线和最终隔离总体。完整中文报告：`reports/current_method_external_positioning_stage_result_2026-07-21.md`。

## 最新正式复现（2026-07-21）：S26 主方法通过 512-image pristine population

S27 在 population 产生前冻结全部方法、checkpoint、route、seed 和 S26 原成功门槛；从本地 COCO train2017 排除 S16/S18/S19/S21 共 22,536 个唯一 source path/SHA，最终 512 张新图与所有旧总体 path/SHA overlap=`0/0`，无 selection。canonical AWGN cache 为 512×5=2,560 行，其中 1/4/7 dB 共 1,536 个 6-step matched-diffusion 输出；全程本地、无下载。

routed fusion 相对 B1：PSNR `+0.092662 dB`，source-image cluster 95% CI `[+0.089147,+0.096313]`；MS-SSIM `+0.002310`，CI `[+0.002149,+0.002482]`；LPIPS `-0.007922`，CI `[-0.008465,-0.007398]`。majority failure `1561→1517`，difference CI `[-0.02813,-0.00664]`，相对 B1 为 `60 new / 104 repair`。相对等容量 routed control 仍有 PSNR `+0.065799 dB`、LPIPS `-0.003494`，两项 CI 显著。

分 SNR fusion−B1 PSNR=`+0.135781/+0.153492/+0.174039/0/0 dB`；13/19 dB 最大逐像素差 0。9/9 checks PASS。S26/S27 aggregate 增益仅差约 `0.00060 dB`，主结果已高稳定复现。当前内部方法冻结，不再训练 controller 或调 route；下一步仅做同 population/码率边界下的 SGD-JSCC 外部定位与论文汇总。中文报告：`reports/s19_exact_fallback_fresh_replication_stage_result_2026-07-21.md`。

## 最新阶段性成果（2026-07-20）：S19 强表示与 S23 exact-fallback 思想已合并

S26 在任何目标 S19 输出产生前冻结：1/4/7 dB 使用 frozen S19 fusion/control，13/19 dB 两者均结构性返回 frozen B1；不访问目标 selection、不训练、不增加参数或 side-information symbols。目标是 S21/S23 的另一批 COCO holdout 256 图×5 SNR。该图片总体的 B1/S23 outcome 已知，但 S19 checkpoint 从未在其上运行，因此结论定位为 frozen cross-population method replication，而不是完全 pristine final test。

routed fusion 相对 B1：PSNR `+0.093267 dB`，source-image cluster 95% CI `[+0.087945,+0.098806]`；MS-SSIM `+0.002188`，CI `[+0.001972,+0.002412]`；LPIPS `-0.007661`，CI `[-0.008438,-0.006915]`。majority failure `744→720`，difference CI `[-0.03203,-0.00547]`，相对 B1 为 `27 new / 51 repair`。相对等容量 routed control 仍有 PSNR `+0.065486 dB`、LPIPS `-0.003100`，两项 CI 均显著，第二次证明 diffusion 信息不是额外 CNN 容量的视觉包装。

分 SNR fusion−B1 PSNR 为 `+0.141105/+0.154616/+0.170612/0/0 dB`；13/19 dB 最大逐像素差为 0。9/9 预注册检查全部通过。当前最好的方法更新为 **S26 = S19 low-SNR fusion + exact-B1 high-SNR fallback**：它保留了接近 S19 的效应量，同时取得 S23 想要但未能放大的结构安全边界。中文报告：`reports/s19_exact_fallback_replication_stage_result_2026-07-20.md`。

## 最新路线判定（2026-07-20）：S23 逐图幅度 controller 上限不足，正式关闭

S25 在已暴露的 S23 selection 256 图×5 SNR 上冻结原有 12 个 alpha，计算不可部署的 PSNR oracle、LPIPS oracle 和“不新增三分类器 majority failure”的 semantic-safe PSNR oracle。即使 oracle 可以读取原图与评估器，semantic-safe oracle 相对固定 `alpha=0.15` 也只有 PSNR `+0.001365 dB`，source-image cluster 95% CI `[+0.001186,+0.001562]`；LPIPS `-0.001817`，majority `0 new / 10 repair`。预注册的最小有意义 PSNR headroom 为 `+0.02 dB`，4 项继续 gate 仅通过 3 项，正式 `continue=false`。

因此不再在 S23 one-epoch feature direction 上训练 receiver-only amplitude head、扫 threshold 或细化 alpha。这是表示上限不足，不是 controller 优化问题。S23 保留为 exact-fallback 机制基线；下一轮转向 S19 的更强 joint-fusion representation，并冻结 1/4/7 dB fusion、13/19 dB exact B1 的结构性策略，在另一 population 上同时复核 frozen S19 control。中文报告：`reports/b1_feature_amplitude_headroom_stage_result_2026-07-20.md`。本轮未访问 holdout、未联网、未下载。

## 最新综合复核（2026-07-20）：S17--S23 指标、外部对照与数据流已统一整理

S24 只读取冻结的 S19、S20、S23 产物，未重新选模型、未用 holdout 调参、未访问 official Imagenette validation。它在 S23 的同一独立 COCO 256 图×5 SNR holdout 上统一重算 PSNR、MS-SSIM、LPIPS、三分类器多数票/AlexNet 辅助语义失败，并以 source image 为 cluster 做 10,000 次 bootstrap。S23 相对 B1 为 PSNR `+0.000567`（95% CI `[+0.000376,+0.000762]`）、MS-SSIM `+0.0000224`（`[+0.0000116,+0.0000338]`）、LPIPS `-0.001731`（`[-0.001844,-0.001619]`）；多数票为 `3 new / 7 repair`，但 failure-rate CI 跨 0，不能声称语义改善显著。

横向结论保持诚实分层：S19 是当前质量增益最大的内部融合版本（相对 B1 `+0.10173 dB/-0.00640 LPIPS`），但有高 SNR 负迁移；S23 是当前结构最安全的机制闭环，13/19 dB 精确回退 B1，但新增 PSNR 太小；SGD-JSCC 免费完美文本上界感知指标强，但 caption 至少使严格预算超出 `10.88%`，且当前外部结果与 S23 不在同一 population，不能直接排绝对名次。RTX 4090 D、batch 16、已缓存 B0/diffusion 的接收端后处理 microbenchmark 中，B1/S23 为 `2.491/2.602 ms/图`；这明确不包含 6-step diffusion，不能当端到端延迟。

面向非专业读者的完整中文报告、流程图和可视化见 `reports/recent_progress_metrics_and_data_flow_2026-07-20.md`；派生输出见 `outputs/analysis/ANALYSIS-S24-RECENT-PROGRESS-SUMMARY-001/`。本轮无联网、无下载。

## 最新阶段性成果（2026-07-20）：B1 + diffusion 首个非零安全合并闭环

S21/S22 在同一份全新 COCO development population 上完成，256×5 holdout 始终封存、未访问。S21 依次排除了三类简单输出合并：带 penalty 的 learned gate 第 1 轮塌零；去掉 penalty 后第 4 轮仍塌零；fixed-gate bounded residual 第 3 轮达到 `0.06` envelope 上限并使 PSNR 崩至 `22.73 dB`。无训练单调凸融合穷举 120 个组合，也只有全零 B1 同时满足低 SNR PSNR 与 aggregate LPIPS 约束。

S22 随后冻结 B1 全部参数，只用 `1,728` 参数的零初始化 `Conv3x3(3→64)` 将 `D-B0` 注入 B1 head feature；control 和 13/19 dB 由结构保证严格等于 B1。真实 cache smoke 中初始差为 0、projection gradient L1=`0.03054`。10 个训练 epoch 均显著改善 selection LPIPS：epoch1 为 `-0.01096`，epoch10 达 `-0.01580`；没有 gate collapse 或饱和。但所有非零 epoch 的 PSNR 都比 B1 低，最接近的是 epoch6 的 `-0.01789 dB`。按预注册 Pareto 规则最终选择 epoch0，checkpoint SHA `b7eac7ec...a0d79`，不解封 holdout。

S23 随后在已知 S22 结果的前提下明确注册 development follow-up：固定重训最早的 epoch1 非零方向，在运行前冻结 12 个全局 shrink alpha，不扫描 epoch 或 per-SNR schedule。selection 选中 `alpha=0.15`（PSNR `+0.000536 dB`、LPIPS `-0.001681`，1/4/7 dB 全正），冻结 checkpoint SHA `53692278...1abbf` 与 policy SHA `54c2639f...8c68f` 后才首次访问独立 256×5 holdout。

S23 holdout 相对 B1 为 PSNR `+0.000568 dB`，source-image cluster 95% CI `[+0.000378,+0.000771]`；LPIPS `-0.001731`，CI `[-0.001849,-0.001622]`。分 SNR PSNR `+0.000701/+0.001158/+0.000979/0/0 dB`，13/19 dB 最大逐像素差为 0；majority pseudo new/repair=`3/7`。五项预注册检查全部通过。

因此已经取得首个“frozen B1 + matched diffusion 非零注入 + exact fallback”的独立 holdout 闭环；但 PSNR 效应量只有 `5.7e-4 dB`，远小于 S19 joint fusion 的 `+0.10168 dB`，只能称机制突破，不能包装为强主方法。下一步主攻可学习/解析的 SNR/sample-adaptive amplitude，并保持 exact-B1 fallback；不再细扫全局 alpha 或 gate 小模块。完整中文报告：`reports/b1_merge_stage_result_2026-07-20.md`。

本轮全部使用既有本地数据、模型与 cache，无联网、无下载；新增/相关脚本 `py_compile` 通过，标准库单测 `122/122` 通过。

## 方向确认（2026-07-20）：B1 与 matched diffusion 合并，而非互相替代

结合 S19 的互补信息因果证据与 S20 的 B1/SGD Pareto 对比，用户确认后续主线采用合并方案。这里的“合并”明确限定为：严格 19,712-real 链路先产生 B1 保真锚点；同一接收观测经解析 `alpha(SNR)` 进入 channel-state-matched diffusion 辅助分支；receiver-visible controller 只把经过 measurement/semantic-risk 检查的 diffusion 增量注入 B1，高风险时严格回退 B1。目标形式为 `x_final = x_B1 + A(y,SNR,x_B1,x_D) ⊙ R(x_B1,x_D)`，其中 `A=0` 必须精确恢复 B1。

这不是把作者 SGD 接在 B1 后面，也不是固定加权平均：公开 SGD 的 VAE/channel latent 与当前 DeepJSCC latent 不同，且免费 caption 不满足严格总码率。S19 已经完成第一版 `[B0,D_identity,SNR,Sobel,Laplacian]` 联合 CNN，并证明 diffusion 信息不可被等容量 B0-only control 替代；下一版应做 B1-anchored auxiliary-only/参数解耦门控，显式输入 `|x_D-x_B1|`、信道/measurement uncertainty 和语义风险证据，解决 S19 在 13/19 dB 的共享权重负迁移。当前只冻结方向，尚未新增配置、训练或 outcome；正式实验前仍需另行预注册 fresh selection/holdout。

## 最新阶段性成果（2026-07-17）

S20 完成了“SGD-JSCC 既然优于普通 JSCC，是否应全程替代 B1”的扩展判定。任何结果产生前冻结了 64 张独立 Imagenette policy-dev clean-correct 图、5 个 SNR、3 个新 channel seed；每个方法 960 个配对观测，全部 `(seed, sample_id, SNR)` canonical noise SHA 一致，official validation 未访问。

SGD 论文协议的免费/完美文本上界相对普通 B0-full 是明确强基线：PSNR `+0.63461 dB`、LPIPS `-0.18332`，95% CI 均不跨零，failure `111→25`。但它没有全面支配 B1：SGD−B1 PSNR `-0.38422 dB`，source-cluster 95% CI `[-0.61529,-0.16026]`；MS-SSIM `+0.006276`，LPIPS `-0.087297`，对应 CI 都显著有利。failure `35→25` 的差值 CI 跨零，且 SGD 同时产生 `11` 个相对 B1 new error、修复 `21` 个。

严格码率审计进一步否决“直接全用公开 SGD”：其 main `16,384` + active edge `3,328` 已占满 `19,712 real`；四个固定 caption 即使只做未保护 BPSK 也至少再需 `2,144 real`，最低超预算 `10.8766%`。SGD 推理约 `2064.7 ms/图`，是 B1 `2.642 ms/图` 的 `781.4×`。因此当前证据支持保留严格同码率保真路径，并把 channel-state-matched diffusion 用作受 measurement/semantic-risk 约束的感知先验；不支持无条件全程 SGD，也不支持放弃 diffusion。

完整中文报告：`reports/sgd_b1_decision_stage_result_2026-07-17.md`；聚合结果 SHA `3023ac91...64d5`。本轮本地离线、无下载；115/115 项单测、`py_compile` 和 `git diff --check` 通过。

术语澄清：本项目当前所称 **B1** 特指 `EXP-S16-B1-001` 的 exact-rate receiver-side residual restoration anchor，不是 diffusion，也不是外部论文名称。它以严格链路得到的 B0 RGB、归一化 SNR、B0 自身的 Sobel magnitude 和 Laplacian absolute map 共 6 通道为输入，预测受 SNR gate 缩放的 RGB residual；S20 中 80-real sender payload 只用于保持预留坐标的严格物理输入合同，B1 网络本身不读取该 payload。

## 上一阶段性成果（2026-07-16）

S19 完成了“diffusion 是否提供 B1 之外信息”的等容量因果消融。新建 5,000 train / 256 selection / 256 holdout 的 COCO train2017 population，与旧 11,000 和 S18 512 的 path/SHA 重叠均为 0；固定 27,560 行精确 19,712-real AWGN cache。control 与 fusion 都是 450,115 参数、从同一个 B1 权重零辅助展开、使用同 batch/crop/flip；唯一信息差异是第二 RGB 输入为 B0 复制还是 S18 identity-controlled diffusion。

一次性 256×5 holdout 上，fusion/control/B1 PSNR 为 `27.40649/27.34803/27.30482 dB`。fusion−control 为 `+0.05846 dB`，image-cluster bootstrap 95% CI `[+0.05198,+0.06423]`；LPIPS `-0.001493`，CI `[-0.002162,-0.000824]`。fusion−B1 为 `+0.10168 dB`，CI `[+0.09431,+0.10915]`，LPIPS `-0.006394`。majority pseudo new/repair 为 fusion `54/280`、control `60/276`。主互补信息判据通过，证明 diffusion 不是可被同容量 B0-only CNN 完全替代的视觉包装。

预注册 7 项通过 6 项。唯一未过的是 fusion−control 只有 1/4/7 dB 为正，13/19 dB 为 `-0.02492/-0.01903 dB`，非负 SNR 数为 3/5，低于预注册 4/5；不过 fusion 在五个 SNR 上均高于原 B1。后续主攻 auxiliary-only SNR-gated adapter/参数解耦，不能事后修改本轮 policy。中文报告：`reports/diffusion_fusion_ablation_stage_result_2026-07-16.md`。

本轮 cache 曾因对话中断留下 1 张未写完 PNG，第一次训练在读取前失败；损坏 cache 与失败训练目录完整保留。cache runner 已改为原子 PNG 写入，修复后全量 49,608 张 PNG 校验坏文件为 0，正式结果使用新 cache manifest SHA `8d88daf7...75e3`。

## 当前阶段

- 阶段0：文献与代码准备
- 阶段1：DeepJSCC baseline
- 阶段2-HR：高分辨率 DeepJSCC 重训
- 阶段3：Blind diffusion refinement
- 阶段4：Semantic drift metric
- 阶段5：Channel-adaptive semantic guidance
- 阶段6：完整实验
- 阶段7：论文整理

当前处于：阶段5 validation，已基于 `EXP-S2-002` 完成 CLIP image-image consistency、冻结 ImageNet 分类器 pseudo-label consistency、COCO caption CLIP image-text consistency 三条辅助语义诊断，并用冻结分类器 top-1 agreement 实现了最小 semantic fallback。`EXP-S4-002` 已完成低强度固定 diffusion 与保守 SNR-aware schedule 的小规模验证，`EXP-S4-003` 进一步确认 SD VAE encode/decode roundtrip 在不运行 UNet denoise、不使用 prompt 的情况下已显著损伤高保真 M0。`EXP-S4-005` 和扩展后的 `EXP-S4-006` 均显示，避开 SD VAE 的 pixel-domain residual restoration 能在 5 个 SNR 上稳定提升 PSNR/LPIPS；`EXP-S4-006` gate error analysis 进一步确认，top-1 agreement gate 确实能保护部分 M0-correct 样本，但也会拒绝不少 refined 修复了 M0 pseudo-label 的样本。gate policy sweep、辅助语义审计、held-out 复核和 test-like 复核显示，`top1_equal_or_refined_conf_gain_ge_0p05` 是有收益但不安全的候选：validation/held-out/test-like 上分别有 3/2/4 个 accepted new error。全局 CLIP veto 和 SNR-calibrated CLIP veto 均确认单一 `CLIP(M0, refined)` 标量阈值不够：安全时过保守，放宽时会漏新错。receiver-side risk rule sweep 使用 validation-only 规则搜索，选出 `shadow-margin` 风险规则：AlexNet 口径 validation 上 final failure `0.3156`、PSNR `+0.0953` dB vs top-1、19 repair、0 new error；held-out 上 final failure `0.2812`、PSNR `+0.0748` dB vs top-1、7 repair、0 new error。该规则的 final PNG、per-sample CSV、summary 和样例 sheet 已落盘到 `outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/`；进一步 classifier ensemble audit 显示它并非跨语义模型完全安全：ResNet18/MobileNetV3-Small 下 validation 有 26/320 个样本被至少一个模型标为 selected accepted new error，held-out 有 15/160 个，且多数票新错分别为 2/1 个。基于该审计完成的 ensemble-risk 二级 veto sweep 选出保守规则，可把 validation/held-out 多数票新错清为 0/0，但额外 veto 96/58 张、PSNR 相比 `selected_risk_rule` 回吐 `-0.1834/-0.2538` dB，且 any-new-error 仍剩 16/8。进一步 receiver-side risk score sweep 尝试用更少 extra veto 保留 repair，但 validation 选出的 `low_overlap_rank` 分数在 held-out 漏 1 个多数票 new error；若强行同时清零 validation/held-out，多数候选比保守二级 veto 更重。最新 frozen risk-rule test-like 复核显示，`selected_risk_rule` 可把 raw confidence-gain 的 4 个 test-like accepted new error 降到 1 个，并保留 10 个 repair 和 `+0.0434` dB PSNR vs top-1；保守 ensemble veto 没有进一步降低 new error，却让 PSNR 相比 selected rule 回吐 `-0.1902` dB。因此浅层接收端标量规则已接近瓶颈，不是最终 M3。`EXP-S4-007` 测试了 latent-free pixel residual DDPM，结果为明确负向：从随机噪声采样残差会大幅降低 refined PSNR，说明 diffusion 若继续做，必须换成更强条件化、近 M0 初始化或以 residual CNN 为先验的设计。

最新 test-like classifier-ensemble 审计显示，`selected_risk_rule` 在 test-like 上没有 majority-vote accepted new error，但仍有 `23/320` 张被至少一个分类器标为 accepted new error；AlexNet/ResNet18/MobileNetV3-Small 的 selected new error 分别为 `1/13/9`，repair 分别为 `10/31/32`。这说明当前 rule 有迁移性，但不能声明跨模型完全安全。

最新 COCO object CLIP clean-correct 辅助诊断显示，64 个 test-like 原图中有 55 个 dominant object label 可用，其中 27 个通过 original CLIP clean-correct 过滤，形成 135 行统计。`selected_risk_rule` 在该子集上 final failure 与 top-1 gate 持平 `0.0815`，PSNR 高 `+0.0257` dB，但仍有 2 个 GT-like new error；保守 ensemble veto 可把 new error 清零，却让 PSNR 比 top-1 低 `-0.1727` dB。该结果是 COCO-object/CLIP zero-shot 辅助诊断，不是最终监督真值。

最新 minimal closure report 已把 residual shrink 结果并入第一版论文闭环口径：`M1-BlindDiffusion-SDImg2Img` 是负参考，平均 PSNR delta `-14.7485` dB；`M2-SNRConditionedPixelResidualRestoration` 是正向 restoration anchor，平均 PSNR delta `+0.7235` dB、LPIPS delta `-0.0274`；`M3-ResidualRestorationTop1Fallback` 是保守第一版闭环，平均 PSNR delta `+0.4011` dB、LPIPS delta `-0.0104`，且 pseudo semantic failure 不高于 M0；`M3-ResidualRestorationTop1ShrinkFallback` 是当前最强保守候选，validation PSNR delta `+0.4584` dB，frozen held-out/test-like PSNR delta `+0.4689/+0.4552` dB，held-out/test-like accepted new error 均为 0。`selected_risk_rule` 继续保留为候选/消融，不写成最终安全方法。

最新 residual shrink selection 派生分析显示，直接缩放 `EXP-S4-006` 残差强度有价值：validation-only per-SNR top-1 fallback shrink schedule 选出 `1 dB alpha=0.5`、其余 SNR `alpha=0.75`，平均 PSNR delta 从 full-strength top-1 fallback 的 `+0.4011` dB 提升到 `+0.4584` dB，LPIPS delta 从 `-0.0104` 改到 `-0.0153`，pseudo final failure 仍不高于 M0。always-accept 路线虽然平均 failure 可低于 M0、PSNR 更高，但仍有 accepted new error（full strength 28 个，M0-failure constrained schedule 19 个），因此不能作为最终 M3；它只说明 residual strength 应进入训练/选择约束，而不是只靠事后阈值。

最新 frozen residual shrink schedule test-like 复核显示，validation 选出的 top-1 shrink schedule 在 `sample_000256`-`sample_000319` 上仍迁移：full-strength top-1 fallback 平均 PSNR delta `+0.4113` dB，frozen shrink schedule 提升到 `+0.4552` dB，LPIPS delta 从 `-0.0116` 改到 `-0.0152`，pseudo final failure 仍等于 M0，accepted new error 为 0。always-accept full strength 和 validation always-constrained schedule 仍分别有 25/12 个 accepted new error，继续不能作为最终 M3。

最新 frozen residual shrink schedule held-out 复核显示，同一 validation top-1 shrink schedule 在 `sample_000000`-`sample_000031` 上也迁移：full-strength top-1 fallback 平均 PSNR delta `+0.4454` dB，frozen shrink schedule 提升到 `+0.4689` dB，LPIPS delta 从 `-0.0113` 改到 `-0.0150`，pseudo final failure 仍等于 M0，accepted new error 为 0。always-accept full strength 和 validation always-constrained schedule 仍分别有 10/3 个 accepted new error，继续不能作为最终 M3。

最新 residual shrink M3 artifact gallery 已把 validation、held-out、test-like 三段 shrink 输出整理到一个可引用目录：selected shrink M3 的 PSNR delta 为 `+0.4584/+0.4689/+0.4552` dB，accepted new error 为 `0/0/0`；safe accept 数为 `183/102/156`，protective reject 数为 `17/6/13`，rejected good candidate 数为 `34/19/44`。对应 unsafe always-accept full strength new error 为 `28/10/25`，validation-constrained always-accept 仍有 `19/3/12` 个 new error。该 artifact 支持第一版论文把 shrink-M3 写成“保守质量增强 + 明确负对照”，而不是把 always-accept 包装成主方法。

最新 adaptive residual alpha policy 派生分析显示，per-sample residual strength control 比固定 per-SNR shrink schedule 更强：`adaptive_max_top1_consistent_alpha` 在 validation/held-out/test-like 上的 PSNR delta 为 `+0.5584/+0.5664/+0.5691` dB，accepted new error 为 `0/0/0`，超过 fixed shrink schedule 的 `+0.4584/+0.4689/+0.4552` dB。该规则只在接收端候选图中选择“最大且 top-1 与 M0 一致”的 alpha，否则回退 M0，不使用原图；但它仍没有 repair，missed repair 为 `45/31/70`，因此当前性质是更强的保守质量增强，不是语义修复方法。

最新 minimal closure report 已纳入 adaptive alpha M3：报告新增 `M3-AdaptiveResidualAlphaTop1Fallback`、`adaptive_residual_alpha_policy_tradeoff.csv` 和 adaptive tradeoff 图。闭环口径更新为：M3 top-1 fallback 是保守第一版闭环，fixed shrink schedule 是固定 schedule 消融/备选，adaptive max top-1-consistent alpha 是当前最强保守候选；它在 validation/held-out/test-like 上 new error 为 `0/0/0`，但 repair 仍为 0，后续必须把 alpha/幅度控制前移到 residual CNN 训练或 model selection。

最新 class-weighted alpha-head residual refiner follow-up 显示，第一版 alpha-head 的问题不只是类别不均衡。普通 CE pilot 中 validation target `alpha=1.0` 为 `205/320`，预测 `alpha=1.0` 达 `280/320`；weighted CE 后预测分布更分散（validation `0.0/0.25/0.5/0.75/1.0 = 46/12/9/30/223`），但 validation/held-out/test-like PSNR delta 仅为 `+0.3851/+0.3506/+0.3166` dB，低于 unweighted alpha-head 的 `+0.3846/+0.3808/+0.3623` 和 full-strength top-1 fallback 的 `+0.4011/+0.4454/+0.4113`。new error 仍为 `0/0/0`，但 weighted 版更常接受弱 alpha 候选，低 SNR 质量收益被压低。结论是冻结 residual 特征 + alpha 分类伪标签不足以学习“质量收益/语义风险”的决策边界；下一步应改成 benefit/risk-aware alpha 目标、联合微调 residual CNN，或设计从 M0/refined 附近初始化的短链 conditional residual diffusion。

最新 benefit-aware alpha predictor follow-up 把训练目标从 hard pseudo alpha 改为 validation-derived safe PSNR utility soft labels。它在 validation 上几乎追上 exhaustive adaptive alpha（`+0.5538` vs `+0.5584` dB，new error `0`），但 held-out/test-like 只有 `+0.4474/+0.4627` dB，低于 two-stage 的 `+0.5009/+0.4875` 和原 receiver predictor 的 `+0.5099/+0.4871`。这说明“收益/风险目标”方向比普通 CE 更贴近问题，但当前 receiver-visible tabular feature + 小 MLP 泛化仍不够；项目状态不是没希望，而是普通 diffusion、随机 residual diffusion、冻结 alpha-head、浅层 tabular predictor 都已暴露瓶颈，正向 anchor 仍是 `EXP-S4-006` residual CNN + adaptive alpha/top-1 fallback。

最新 benefit-aware alpha-head residual refiner follow-up 把同一 safe-PSNR utility alpha 目标接到冻结 residual CNN 的内部 feature 上。它相对普通/weighted alpha-head 有部分进展：validation/held-out/test-like PSNR delta 为 `+0.4251/+0.4192/+0.3530` dB，new error 仍为 `0/0/0`，其中 validation 高于 full-strength top-1 fallback，held-out 高于此前两个 alpha-head 版本。但它仍低于 receiver predictor、two-stage 和 exhaustive adaptive alpha，且 test-like 低于普通 alpha-head。预测分布显示模型几乎不预测 `alpha=0.25`，仍主要在 `0.75/1.0` 和 fallback 间跳，说明冻结 residual feature 上的 alpha classifier 仍读不出细粒度 benefit/risk 边界。下一步不要继续只换 alpha 分类标签，应尝试 joint fine-tune residual CNN 或直接加入 semantic-risk-aware residual amplitude loss。

最新 benefit-aware joint alpha-head residual refiner follow-up 解冻 residual CNN，并让 soft-alpha reconstruction loss 与 target-alpha reconstruction loss 反传到 residual CNN。它明确提升了 validation alpha target accuracy（`0.7719`，预测分布 `0.0/0.25/0.5/0.75/1.0 = 28/24/23/127/118`，不再完全忽略小 alpha），但 restoration anchor 被损伤：validation/held-out/test-like PSNR delta 只有 `+0.3294/+0.2303/+0.1869` dB，低于冻结 benefit alpha-head 的 `+0.4251/+0.4192/+0.3530`，也远低于 full-strength top-1 fallback。new error 仍为 `0/0/0`，但质量收益被明显吃掉。结论是“直接全量 unfreeze + CE 主导”方向不行；下一步应做 partial fine-tune（例如只调 tail/amplitude）或 reconstruction-dominant loss，而不是让分类目标主导 shared residual feature。

最新 benefit-aware tail-only partial fine-tune 验证了上述判断：冻结 residual CNN 的 head/body，只训练 residual tail 与 alpha head，并用 reconstruction-dominant loss 保护 full-strength restoration anchor。它在 validation/held-out/test-like 上取得 PSNR delta `+0.4749/+0.4552/+0.4061` dB，accepted new error 仍为 `0/0/0`；full-strength top-1 fallback 也恢复/提升到 `+0.4454/+0.4820/+0.4259` dB，说明全量 joint 的问题主要来自 shared feature 被分类/target-alpha loss 拉偏。该结果显著好于冻结 benefit alpha-head 和全量 joint，但仍低于 receiver predictor、two-stage 和后验 adaptive alpha；预测分布仍基本不使用 `alpha=0.25`，说明细粒度幅度边界还没学好。因此这是训练侧正向阶段成果，不是最终 M3。

最新 benefit-aware continuous-alpha tail-only follow-up 把离散 alpha 分类改成单值连续 alpha regression，仍只训练 residual tail 与 alpha head。它在 validation/held-out/test-like 上取得 PSNR delta `+0.5010/+0.5049/+0.5012` dB，accepted new error 为 `0/0/0`，明显超过 tail-only classification 的 `+0.4749/+0.4552/+0.4061`，并在 held-out/test-like 上达到或超过 two-stage/receiver predictor 的 learned 部署水平。连续 alpha 的最近离散 target accuracy 较低（`0.4188/0.3625/0.3469`），但输出分布覆盖中间幅度（test-like mean alpha `0.7123`，nearest counts `1/11/78/176/54`），说明它不是在做离散标签复刻，而是学到更平滑的质量/风险折中。该结果是当前训练侧最明确的正向突破；仍低于后验 exhaustive adaptive alpha，其 LPIPS/ensemble 风险边界见下一段补充审计，因此暂不直接升级最终 M3。

最新 continuous-alpha tail refiner 审计已补齐 LPIPS 与跨分类器安全复核。连续 alpha top-1 fallback 在 validation/held-out/test-like 上的 LPIPS delta 为 `-0.0149/-0.0149/-0.0162`，优于同 checkpoint full-strength top-1 fallback 的 `-0.0097/-0.0106/-0.0098`；PSNR delta 仍为 `+0.5010/+0.5049/+0.5012` dB，AlexNet accepted new error 仍为 `0/0/0`。但 ensemble 审计显示它不是跨模型完全安全：any-classifier new error 为 `17/9/14`，majority-vote new error 为 `1/0/0`；唯一 majority case 是 validation 4 dB `sample_000248.png`，由 ResNet18 和 MobileNetV3-Small 同时标出。结论：continuous-alpha 是当前最强 learned training-side amplitude-control 候选，但还不能直接升级最终 M3；下一步需要 semantic-risk-aware loss/listwise utility 或二级安全约束。

最新 edge × capacity/training-budget 受控消融已完成，修正了此前对 `EXP-S4-008` 的过早归因。原 `EXP-S4-008` 相比 `EXP-S4-006` 同时扩大了网络并增加训练轮数；新增 matched large no-edge `EXP-S4-009` 和 matched small edge `EXP-S4-010` 后形成完整 2×2。sample-cluster paired bootstrap 显示，edge 在 small/large 配置上的 raw PSNR 独立增益分别为 `+0.0501` dB（95% CI `[+0.0249,+0.0696]`）和 `+0.1389` dB（`[+0.1031,+0.1805]`）；M3 增益分别为 `+0.0455/+0.0617` dB，CI 也均排除 0。large edge 的 raw semantic failure 相比 matched no-edge 增加 `+0.0438`，new error `26→34`、repair `44→38`，因此结论是“结构条件带来真实质量收益，但不是无条件语义改进”。

同一 large matched pair 的跨 split paired audit 进一步确认 edge raw PSNR 净增益在 validation/held-out/test-like/fresh-holdout 上为 `+0.1389/+0.1565/+0.1585/+0.1411` dB，四个 95% CI 下界均大于 `+0.10` dB，且每个 split 的 5 个 SNR 全部同向。fresh-holdout 使用此前未做 downstream residual 分析的 `sample_000320`-`sample_000383`，冻结后未据此调参。

原 edge shrink schedule 的有效强度在 4→7 dB 非单调，已按 `MILESTONES.md` 约束改成 validation-only 全局单调选择 `{1:0.75,4:0.75,7:0.75,13:1.0,19:0.75}`，对应 `gate×alpha={0.09,0.075,0.06,0.05,0.03}`。冻结策略在 validation/held-out/test-like/fresh-holdout 上的 PSNR delta 为 `+0.5734/+0.6128/+0.5700/+0.5668` dB，LPIPS delta 为 `-0.0145/-0.0148/-0.0163/-0.0162`，所有目标 split/SNR 的 LPIPS 均改善。AlexNet gate 下 new error 为 0 是规则内生保证，不再当作独立安全证据；三分类器离线审计显示 any-model new error 为 `20/7/17/17`，majority new error 为 `1/1/0/3`（validation/held-out/test-like/fresh-holdout），所以 edge monotonic policy 仍是强质量候选而非跨模型完全安全的最终 M3。

## 当前任务

- 状态：阶段1 DeepJSCC sanity baseline 已完成；COCO2017 `train2017/val2017` 和官方 annotations 已完成；COCO-256 正式训练已产生可用 `best.pt`，但 epoch 89 后出现 NaN，`latest.pt` 不可用；`M1-BlindDiffusion` 已在 1/7/19 dB、每个 SNR 16 张图上完成，结果为明显负向；`EXP-S3-001` 已完成 CLIP image-image consistency 辅助诊断和 failure case gallery；`EXP-S3-002` 已完成冻结 AlexNet pseudo-label consistency 诊断；`EXP-S3-003` 已完成 COCO caption CLIP image-text consistency 诊断；`EXP-S4-001` 已完成 receiver-side semantic fallback pilot；`EXP-S4-002` 已完成 `[1,4,7,13,19]` dB、每个 SNR 8 张图的低强度/SNR-aware diffusion validation，结果仍不满足视觉收益要求；`EXP-S4-003` 已完成 SD VAE roundtrip 诊断，确认 VAE 重编码本身会带来约 3.49-7.33 dB 的 M0 PSNR 损失；`EXP-S4-004` 因 CSV 记录 bug 失败并保留；`EXP-S4-005` 完成 SNR-conditioned pixel residual refiner pilot；`EXP-S4-006` 在 160 train / 64 eval images per SNR 上完成 validation，M3 final PSNR 相比 M0 提升约 `+0.33-0.46` dB；`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/` 已完成，用于 `sample_000256`-`sample_000319` test-like 复核；`outputs/analysis/exp_s4_006_gate_error_analysis/`、`outputs/analysis/exp_s4_006_gate_policy_sweep/`、`outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/`、`outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/`、`outputs/analysis/exp_s4_006_heldout_gate_check/`、`outputs/analysis/exp_s4_006_testlike_gate_check/`、`outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/`、`outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/`、`outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/`、`outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/`、`outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/`、`outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/`、`outputs/analysis/exp_s4_006_receiver_risk_score_sweep/` 和 `outputs/analysis/exp_s4_006_testlike_risk_rule_check/` 已完成 detector 分析、候选 gate 扫参、辅助语义审计、候选 final PNG 落盘、held-out gate 复核、test-like gate 复核、CLIP 二级 veto 扫描、SNR 校准、轻量 risk-rule 搜索、selected risk-rule final PNG 固化、跨分类器辅助审计、ensemble-risk 二级 veto 扫描、receiver-side risk score 扫描和 frozen risk-rule test-like 复核；`EXP-S4-007` 完成 naive residual DDPM pilot，refined PSNR 相比 M0 下降 `-7.16/-7.48/-7.09/-5.42/-4.42` dB，M3 final 仍低于 M0，记录为负结果
- 负责人/对话：liulu + Codex
- 开始日期：2026-06-29
- 相关代码：`configs/`, `data/`, `src/`, `scripts/`, `outputs/`, `tests/`, `references/`, `third_party/`
- 日志路径：`outputs/EXP-S1-001/metrics.json`
- checkpoint 路径：`third_party/Deep-JSCC-PyTorch/out/checkpoints/CIFAR10_8_13.0_0.17_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl`

## 已完成

| 日期 | 内容 | 路径 | 验证方式 | 结论 |
|---|---|---|---|---|
| 2026-06-29 | 创建项目中枢文档 | `PROJECT.md`, `AGENTS.md`, `PROGRESS.md`, `EXPERIMENTS.md`, `LITERATURE.md`, `README.md` | 检查项目目录中文件存在 | 项目边界和记录文件已就位 |
| 2026-06-29 | 定义核心问题、核心假设和不做的事情 | `PROJECT.md` | 复核项目定义 | 项目聚焦 diffusion-enhanced JSCC，并显式控制 semantic drift |
| 2026-06-29 | 将中枢文档统一改为中文 | 全部中枢文档 | 人工检查文档语言 | 后续协作以中文为主，必要英文术语保留 |
| 2026-06-29 | 整理相关工作分类和创新边界 | `LITERATURE.md` | 检查三类相关工作和创新边界是否写入 | 文献调研按 Diffusion JSCC、Channel-adaptive JSCC、Semantic reliability 三条线推进 |
| 2026-06-29 | 明确多 AI 协作协议 | `AGENTS.md` | 检查开始必读、结束必更、禁止事项是否写入 | 本地文件作为唯一共享记忆，降低多对话协作丢上下文风险 |
| 2026-06-29 | 完成第一轮文献和 baseline 代码扫描 | `LITERATURE.md` | 记录 DiffJSCC、SGD-JSCC、DiT-JSCC、JSCGC、Dynamic_JSCC、DeepJSCC-l++、PJSCC 等工作 | diffusion/generative JSCC 撞车风险较强，项目应收紧到 SNR-aware diffusion refinement + semantic drift 显式度量 |
| 2026-06-29 | 创建初始代码目录 | `configs/`, `data/`, `src/`, `scripts/`, `outputs/`, `tests/`, `references/`, `third_party/` | 检查目录和 README 文件存在 | 代码、配置、输出、第三方依赖和文献资料有了固定位置 |
| 2026-06-29 | 克隆第一候选 DeepJSCC baseline | `third_party/Deep-JSCC-PyTorch` | `git rev-parse HEAD` 返回 `2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06` | 第三方仓库已本地化，包含 CIFAR-10/AWGN/Rayleigh checkpoint |
| 2026-06-29 | 审计 Deep-JSCC-PyTorch baseline | `third_party/Deep-JSCC-PyTorch` | 阅读 README、`train.py`、`eval.py`、`model.py`、`channel.py`、`dataset.py`、`utils.py` 并运行 `py_compile` | 适合作为阶段1 baseline，但应通过本项目 adapter 包装，不直接修改第三方源码 |
| 2026-06-29 | 写入阶段1配置和 smoke test | `configs/s1_deepjscc_cifar10_awgn.yaml`, `scripts/s1_deepjscc_smoke.py`, `src/cadsd_jscc/` | `py_compile` 通过 | 依赖安装成功后，可用合成图像验证 checkpoint 加载、SNR 切换、重建输出和 PSNR |
| 2026-06-29 | 尝试安装阶段1依赖 | `requirements.txt`, `requirements-torch-cpu.txt`, `README.md` | `pip show torch torchvision tensorboardX` 未找到包 | 默认 PyPI 下载超时；CPU-only PyTorch 下载 hash mismatch，当前 smoke test 受环境依赖阻塞 |
| 2026-06-29 | 添加忽略规则 | `.gitignore` | 检查规则覆盖 `__pycache__`、数据、输出和第三方仓库 | 避免缓存、数据集、实验输出和外部仓库误入版本管理 |
| 2026-06-29 | 补充课题收敛约束 | `MILESTONES.md`, `PROJECT.md`, `EXPERIMENTS.md`, `AGENTS.md`, `README.md` | 人工复核文档是否覆盖最小闭环、semantic drift 定义、实验矩阵和成功/失败判据 | 项目从方向约束升级为可收敛执行约束 |
| 2026-06-29 | 安装阶段1和研究扩展依赖 | `requirements-torch-cpu.txt`, `requirements.txt`, `requirements-research.txt`, `README.md` | import `torch`, `torchvision`, `pytorch_msssim`, `lpips`, `diffusers`, `transformers`, `open_clip`, `cleanfid` 通过 | CPU 环境依赖已就位，尚未下载正式数据集或模型权重 |
| 2026-06-29 | 运行 DeepJSCC smoke test | `scripts/s1_deepjscc_smoke.py`, `outputs/smoke/s1_deepjscc/` | `python3 scripts/s1_deepjscc_smoke.py --device cpu --batch-size 2` 成功，生成 `metrics.json` 和两张样例图 | checkpoint 加载、SNR 切换、重建输出和 PSNR 计算已验证；该结果不是正式实验 |
| 2026-06-29 | 下载 CIFAR-10 并运行 mini-eval | `data/cifar10/`, `outputs/mini/s1_deepjscc_cifar10_awgn/` | `python3 scripts/s1_deepjscc_mini_eval.py --device cpu --download` 成功 | 64 张固定 test subset 上 PSNR/SSIM 随 SNR 升高而提升；MS-SSIM 因 32x32 尺寸限制不可用 |
| 2026-06-29 | 完成 EXP-S1-001 正式 baseline | `outputs/EXP-S1-001/`, `EXPERIMENTS.md` | `python3 scripts/s1_deepjscc_mini_eval.py --device cpu --num-samples 1024 --batch-size 64 --output-dir outputs/EXP-S1-001 --formal` 成功 | M0-DeepJSCC 在 CIFAR-10 test subset/AWGN/CBR 0.17 上有了可复现基线 |
| 2026-06-29 | 新增高分辨率 DeepJSCC 训练入口 | `configs/s2_deepjscc_coco256_awgn.yaml`, `scripts/train_deepjscc_highres.py`, `src/cadsd_jscc/datasets.py` | `py_compile` 通过；dry-run 使用合成 256x256 图像完成 1 个 epoch | COCO-256 重训路线已落地，正式训练需要 GPU 和 COCO2017 数据 |
| 2026-06-29 | 检查 GPU 并尝试安装 CUDA PyTorch | `requirements-torch-cu128.txt`, `README.md` | `nvidia-smi` 在提权环境可见 RTX 4090 D；当前 Python 仍显示 `torch 2.12.1+cpu` | 机器有 GPU，但 CUDA PyTorch 安装未完成；`torch==2.11.0+cu128` 下载速度过慢，后续网络操作被审批额度限制拦截 |
| 2026-06-30 | 验证 CUDA PyTorch 和高分辨率训练 GPU 路径 | `scripts/train_deepjscc_highres.py`, `outputs/smoke/s2_deepjscc_coco256_train_gpu/` | `torch 2.11.0+cu128`，`torch.cuda.is_available()` 为 True，设备为 RTX 4090 D；GPU dry-run 完成 1 个 epoch；`diffusers`、`transformers`、`open_clip`、`lpips`、`cleanfid` import 通过 | CUDA 训练链路和关键研究依赖可用；当前仍缺 COCO2017 `train2017/val2017` 数据 |
| 2026-06-30 | 下载并验证 COCO2017 val，运行真实图像 GPU smoke | `data/coco/val2017/`, `outputs/smoke/s2_deepjscc_coco256_val2017_gpu/` | `val2017.zip` 下载并解压完成，图片数 5000；用 `val2017` 临时覆盖 train/val root 跑通 256x256 GPU smoke | 真实 COCO 图像读取、crop、GPU 前后向、checkpoint 和样例图保存均可用；`train2017.zip` 正在下载 |
| 2026-06-30 | 启动 COCO-256 AWGN DeepJSCC 长任务 pipeline | `scripts/run_s2_coco256_awgn_train.sh`, `outputs/logs/s2_coco256_awgn_train.screen.log` | `screen` 会话 `s2_coco256_awgn_train` 已启动；脚本会续传 `train2017.zip`、解压、检查图片数，然后运行 GPU 训练 | 长任务已开始；当前阶段仍需等待 `train2017` 下载和训练完成后再登记正式 `M0-HR` baseline |
| 2026-06-30 | 暂停 COCO2017 train 下载以确认代理/流量来源 | `data/coco/train2017.zip`, `data/coco/train2017.zip.possibly_corrupt_20260630_2046` | 环境变量显示 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=http://127.0.0.1:17890`；所有 `wget` 和 `screen` 下载进程已停止 | 当前不会继续产生下载流量；新 `train2017.zip` 约 15MB，早先双进程风险 partial 已改名保留为 possibly_corrupt |
| 2026-06-30 | 将 COCO2017 train 下载切换为直连 | `scripts/run_s2_coco256_awgn_train.sh`, `outputs/logs/s2_coco256_awgn_train.direct.screen.log` | `wget --no-proxy --spider` 可直连 COCO 官方源；长任务 screen 会话 `s2_coco256_awgn_train` 已重启，实际 wget 命令包含 `--no-proxy` | Codex 仍可使用当前代理环境，但 COCO 数据下载不再走 `127.0.0.1:17890` 代理 |
| 2026-06-30 | 准备 COCO-val 高分辨率 pilot 训练集并完成训练 | `scripts/prepare_image_symlink_split.py`, `configs/s2_deepjscc_coco_val256_awgn_pilot.yaml`, `data/coco_val_split/`, `outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/` | 从 `data/coco/val2017/` 用 seed 42 生成 4500/500 不重叠符号链接切分；训练 50 epoch 完成；final PSNR 26.6647 dB，SSIM 0.7837 | 已得到非正式 HR pilot checkpoint，可用于 diffusion/refinement 接口调试；该结果不能替代正式 COCO train/val 主实验 |
| 2026-06-30 | 评估其他高分辨率数据集下载路线 | `data/imagenette/imagenette2-320.tgz` | `Imagenette2-320` 官方包可 `wget --no-proxy` 直连，大小约 326MB，但实测直连速度也偏慢，当前仅保留约 1.1MB partial | Imagenette 适合作为带分类标签的高分辨率语义 pilot 备选，但当前优先利用已完成的 COCO val split 训练 |
| 2026-06-30 | 检查后台下载与训练状态 | `data/coco/train2017.zip`, `outputs/logs/s2_coco256_awgn_train.direct.screen.log`, `outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/` | 23:00 检查：screen 仅剩 `s2_coco256_awgn_train`；`wget --no-proxy` 仍在运行；`train2017.zip` 约 46MB；GPU 空闲；pilot checkpoint 和 `metrics.json` 已落盘 | COCO 正式训练尚未开始；当前可继续用 pilot checkpoint 推进 high-res inference/diffusion 接口 |
| 2026-06-30 | 扫描近期 JSCC / diffusion-JSCC 论文数据集设置 | `LITERATURE.md` | 整理 Dynamic_JSCC、DeepJSCC-l++、DiffJSCC、SGD-JSCC、DiT-JSCC、JSCGC 的训练/测试数据集 | 近期 generative/diffusion JSCC 主流转向 OpenImages/ImageNet/COCO/Kodak；本项目 COCO-256 主路线合理，CIFAR-10 只保留为 sanity |
| 2026-06-30 | 细读 DeepJSCC-l++ 可取之处 | `LITERATURE.md` | 梳理 side information、Swin backbone、mask/zero-padding、DWA 和公开代码价值 | 该工作适合作为 channel-adaptive JSCC 相关工作和后续扩展参考；当前不建议改为主 baseline，以免偏离 semantic drift controlled diffusion 主线 |
| 2026-06-30 | 再次检查当前总体进度 | `data/coco/train2017.zip`, `outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/` | 23:50 检查：screen 仍只有 `s2_coco256_awgn_train`；实际进程为 `wget --no-proxy`；`train2017.zip` 约 56MB；GPU 空闲；pilot 输出文件齐全 | 当前不需要等 COCO，可先基于 pilot checkpoint 开始 high-res inference/export 和 diffusion refinement 接口 |
| 2026-06-30 | 完成 COCO-val pilot M0-HR SNR sweep 和 `x_hat` 导出 | `scripts/s2_deepjscc_highres_export.py`, `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/` | 在 500 张 pilot val 图上跑 `[1,4,7,13,19]` dB；每个 SNR 导出 32 张重建 PNG；保存 PSNR/SSIM/MS-SSIM/推理时间 | high-res DeepJSCC export 接口可用；后续 `M1-BlindDiffusion` 可直接读取 `exports/snr_XXdb/reconstruction/` |
| 2026-07-01 | 完成 COCO2017 train 下载并自动解压 | `data/coco/train2017/`, `data/coco/val2017/` | `train2017.zip` 完整存在；`train2017` 图片数 118287，`val2017` 图片数 5000 | 正式 COCO-256 训练数据已就位 |
| 2026-07-01 | COCO-256 正式训练完成但后段 NaN | `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/` | 训练共写入 100 行 history；epoch 0-88 有限，epoch 89-99 为 NaN；`best.pt` 为 epoch 73，val PSNR 31.5618 dB，SSIM 0.9054；`latest.pt` 为 NaN，不可用 | 已有可用正式 high-res DeepJSCC checkpoint，但必须使用 `best.pt`；训练稳定性需记录为风险 |
| 2026-07-01 | 评估正式 COCO-256 best checkpoint 的 M0-HR SNR sweep | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/` | 在 512 张 COCO val subset 上跑 `[1,4,7,13,19]` dB；导出每个 SNR 32 张 `x_hat`；7 dB PSNR 31.5590，19 dB PSNR 33.7264 | 正式 `M0-HR` baseline 可用于接 `M1-BlindDiffusion`；后续不要再用 pilot 或 `latest.pt` 做主输入 |
| 2026-07-01 | 给高分辨率训练脚本增加 NaN 防护 | `scripts/train_deepjscc_highres.py` | `py_compile` 通过；训练中若 loss 或 metrics 非有限会提前停止，且 final metrics 会回到 `best.pt` 评估 | 后续重训可避免 `latest.pt` 被 NaN 覆盖；当前已有结果不被改写 |
| 2026-07-01 | 分析 COCO-val pilot M0-HR 结果 | `outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/`, `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/` | 训练末期 val PSNR 稳定到 26.66 dB；SNR sweep 从 1 到 19 dB 的 PSNR 从 25.13 升到 27.30，SSIM 从 0.7096 升到 0.8125，MS-SSIM 从 0.8991 升到 0.9607 | pilot checkpoint 质量足够调试 diffusion；高 SNR 收益趋于平台，说明瓶颈主要来自 JSCC 压缩/模型容量；低 SNR 仍保留主体语义，适合测试 diffusion semantic drift |
| 2026-07-01 | 接入 M1-BlindDiffusion 最小脚本和配置 | `configs/s3_m1_blind_diffusion_coco256_awgn.yaml`, `scripts/s3_blind_diffusion_refine.py` | `py_compile` 通过；`--dry-run` 验证 1/7/19 dB 每个 SNR 16 张样本能和正式 M0 export 对齐 | 脚本只读取正式 `best.pt` 对应 M0 export，拒绝覆盖已有输出目录，可保存 refined 图、metrics 和三行样例图 |
| 2026-07-01 | 尝试运行 M1-BlindDiffusion 正式小规模实验 | `outputs/EXP-S2-001/` | 提权运行 `python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download` 被审批层拒绝；local-only CPU 运行因 `runwayml/stable-diffusion-v1-5` 不在本地 cache 而失败，未创建 `outputs/EXP-S2-001/` | 当前不能虚构 M1 指标；后续需要用户显式允许下载/使用 GPU，或先把 diffusion 权重放入 `outputs/cache/huggingface` |
| 2026-07-01 | 记录下载流量规则 | `AGENTS.md`, `README.md` | 写入大模型/大数据/ CUDA 等大文件下载默认清空代理变量、走服务器直连；只有用户明确允许时才走代理/本机流量 | 后续执行 Hugging Face、COCO、PyTorch 等大下载前必须检查 `env | grep -i proxy`，必要时使用 `env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy ...` |
| 2026-07-01 | 完成 M1-BlindDiffusion 小规模正式实验 | `outputs/EXP-S2-002/` | 用服务器直连 `hf-mirror.com` 补齐 SD v1.5 UNet 权重；运行 `python3 scripts/s3_blind_diffusion_refine.py --device cuda:0`；生成 48 张 refined 图、3 张样例图和 `metrics.json` | 负结果：M1 在 1/7/19 dB 上 PSNR 从 28.17/31.83/34.14 dB 降到 16.22/16.78/16.89 dB，LPIPS 从 0.1747/0.0542/0.0254 升到 0.5025/0.4600/0.4549；样例显示明显 hallucination 和语义漂移 |
| 2026-07-02 | 完成 M1 结果的 CLIP consistency 初步诊断 | `configs/s4_clip_consistency_m1_exp_s2_002.yaml`, `scripts/s4_clip_consistency_eval.py`, `outputs/EXP-S3-001/` | 使用 OpenAI CLIP ViT-B/32 本地权重评估 1/7/19 dB、每个 SNR 16 张图；保存 `metrics.json` 和 `per_sample.csv` | 辅助语义诊断确认 M1 明显漂移：原图-M0 CLIP mean 为 0.9022/0.9587/0.9848，原图-M1 为 0.6619/0.6867/0.6954；M1 在所有 48 个样本上都低于 M0 |
| 2026-07-02 | 整理 CLIP top failure case gallery | `scripts/s4_make_clip_failure_gallery.py`, `outputs/EXP-S3-001/failure_cases/` | 从 `outputs/EXP-S3-001/per_sample.csv` 选取全局 top 12 和每个 SNR top 6，生成 original/M0/M1 triptych、sheet、CSV 和 JSON 索引 | 已固化 18 个不重复 failure case；全局最大 CLIP drop 为 19 dB `sample_000013.png`，drop 0.4026，图中 M0 接近原图但 M1 明显改写主体结构 |
| 2026-07-02 | 完成冻结分类器 pseudo-label consistency 诊断 | `configs/s4_classifier_consistency_m1_exp_s2_002.yaml`, `scripts/s4_classifier_consistency_eval.py`, `outputs/EXP-S3-002/` | 使用本地缓存 AlexNet ImageNet 权重评估 1/7/19 dB、每个 SNR 16 张图；保存 `metrics.json` 和 `per_sample.csv` | 辅助分类器诊断确认 M1 明显漂移：all-subset 中 M0 top-1 与原图一致率为 0.50/0.6875/0.9375，M1 仅为 0.125/0.0625/0.125；在原图置信度 >=0.3 子集上 M0 为 0.8889/1.0/1.0，M1 为 0.2222/0.1111/0.2222 |
| 2026-07-02 | 整理冻结分类器 top failure case gallery | `scripts/s4_make_classifier_failure_gallery.py`, `outputs/EXP-S3-002/failure_cases/` | 从 `outputs/EXP-S3-002/per_sample.csv` 选取 M0 匹配原图但 M1 不匹配的样本，生成全局 top 12 和每个 SNR top 6 triptych、sheet、CSV 和 JSON 索引 | 已固化 18 个不重复 classifier failure case；典型例子是 19 dB `sample_000002.png`，原图/M0 均为 `Pomeranian`，M1 变为 `gondola` |
| 2026-07-02 | 汇总 M1 负结果跨指标证据 | `scripts/s4_summarize_m1_negative_result.py`, `outputs/analysis/m1_negative_result_summary/` | 聚合 `EXP-S2-002` 图像指标、`EXP-S3-001` CLIP 诊断和 `EXP-S3-002` 分类器诊断，输出 `REPORT.md`、`summary.csv`、`summary.json` | 派生汇总确认固定强度 blind diffusion 是系统性负结果：平均 PSNR delta 为 -14.7485 dB，平均 LPIPS delta 为 +0.3877，平均 CLIP drop 为 0.2672，分类器 all-subset M1 pseudo drift-origin 为 0.8958 |
| 2026-07-02 | 下载并验证 COCO2017 annotations | `data/coco/annotations_trainval2017.zip`, `data/coco/annotations/` | 使用清空代理变量和 `--no-proxy` 的服务器直连方式下载 241MB 官方 zip；`unzip -t` 显示无错误；解压 captions/instances/keypoints JSON | COCO caption/object 语义评估所需标注已就位；未走用户本机代理流量 |
| 2026-07-02 | 完成 COCO caption CLIP image-text consistency 诊断 | `configs/s4_coco_caption_clip_m1_exp_s2_002.yaml`, `scripts/s4_coco_caption_clip_eval.py`, `outputs/EXP-S3-003/` | 用 COCO `captions_val2017.json` 反查 48 个样本的 captions，使用本地 OpenAI CLIP ViT-B/32 评估 image-text 相似度；保存 `metrics.json`、`per_sample.csv`、`sample_metadata.json` | caption 语义诊断继续确认 M1 漂移：1/7/19 dB 下 M0 caption-max mean 为 0.3306/0.3305/0.3263，M1 为 0.2816/0.2815/0.2877；M1 caption-max 低于 M0 的比例为 1.0/0.8125/0.8125 |
| 2026-07-02 | 整理 COCO caption top failure case gallery | `scripts/s4_make_coco_caption_failure_gallery.py`, `outputs/EXP-S3-003/failure_cases/` | 从 `outputs/EXP-S3-003/per_sample.csv` 按 caption CLIP drop 生成全局 top 12 和每个 SNR top 6 triptych、sheet、CSV/JSON/README 索引 | 已固化 caption-based failure case；全局最大 caption drop 为 7 dB `sample_000008.png`，COCO caption 为 car/clock/flowers，M1 明显改写为杂乱纹理 |
| 2026-07-03 | 梳理当前语义评价指标和后续主指标口径 | `PROJECT.md`, `MILESTONES.md`, `EXPERIMENTS.md`, `README.md` | 复核当前三套辅助诊断和里程碑中的冻结语义模型定义 | 当前不应把 CLIP/caption/pseudo-label 诊断包装成最终主指标；后续主线应固定 `T_cls`、clean-correct subset、Drift-Origin/Refinement-Drift/Final-Failure，并用辅助指标解释 failure case |
| 2026-07-03 | 生成项目进度可视化总览 | `scripts/s4_make_project_progress_visual_summary.py`, `outputs/analysis/project_progress_visual_summary/` | 从已有 metrics/CSV/PNG 派生 `REPORT.md`、汇总 CSV、阶段进度图、M0 SNR 曲线、M1 质量对比、M1 语义诊断图和代表性可视化拼图；抽查图像尺寸和关键 PNG 显示正常 | 已得到当前项目全局进度与负结果证据包；本次不新增模型运行，不写入 `EXPERIMENTS.md` |
| 2026-07-03 | 整理 2026-07-04 组会汇报材料 | `reports/group_meeting_2026-07-04.md` | 汇总 `PROJECT.md`、`MILESTONES.md`、`EXPERIMENTS.md`、`outputs/analysis/m1_negative_result_summary/REPORT.md` 和三类 failure case gallery | 已形成可直接搬到 PPT 的 8-10 分钟汇报主线、关键表格、推荐图、讲稿提示和答疑口径；本次未新增实验，不更新 `EXPERIMENTS.md` |
| 2026-07-03 | 完成最小 semantic fallback pilot | `configs/s5_semantic_fallback_m1_exp_s2_002.yaml`, `scripts/s5_semantic_fallback_eval.py`, `outputs/EXP-S4-001/` | `py_compile`、`--dry-run`、`python3 scripts/s5_semantic_fallback_eval.py --device cuda:0` 成功；生成 `metrics.json`、`per_sample.csv`、`REPORT.md`、M3 final 图和 3 张 original/M0/M1/M3 拼图 | receiver-side top-1 agreement detector 不看原图即可拒绝大多数 M1 漂移，M3 pseudo final failure 回到 M0 水平；但少量 accepted M1 仍降低 PSNR/LPIPS，说明固定强度 M1 太激进，下一步必须做更弱的 SNR-aware strength 网格 |
| 2026-07-03 | 完成低强度 SNR-aware diffusion validation | `configs/s5_snr_adaptive_diffusion_strength_validation.yaml`, `scripts/s5_snr_adaptive_diffusion_validation.py`, `outputs/EXP-S4-002/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_snr_adaptive_diffusion_validation.py --device cuda:0` 成功；生成 2 个候选、5 个 SNR、每个 SNR 8 张图的 metrics/CSV/样例拼图 | `fixed_0p05` 和 `snr_adaptive_0p10_to_0p05` 均比 0.25 语义更稳，但 refined PSNR/LPIPS 仍明显差于 M0；fallback 可把 final failure 压回 M0 附近，却不能弥补 SD img2img 对高保真重建的质量损伤 |
| 2026-07-03 | 完成 SD VAE roundtrip 诊断 | `configs/s5_sd_vae_roundtrip_coco256_awgn.yaml`, `scripts/s5_sd_vae_roundtrip_eval.py`, `outputs/EXP-S4-003/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_sd_vae_roundtrip_eval.py --device cuda:0` 成功；生成 5 个 SNR、每个 SNR 8 张图的 M0-VAE/original-VAE 往返图、metrics/CSV/样例拼图 | 不运行 UNet denoise、不使用 prompt 时，M0-VAE 相对 M0 仍损失约 3.49-7.33 dB PSNR，LPIPS 变差约 0.009-0.058；高 SNR 下 VAE 将 M0 质量压到约 27 dB，确认通用 SD VAE 是当前 SD img2img 路线的重要瓶颈 |
| 2026-07-03 | 记录失败的 residual refiner 初跑 | `configs/s5_residual_refiner_pilot_coco256_awgn.yaml`, `scripts/s5_residual_refiner_pilot.py`, `outputs/EXP-S4-004/` | 训练 80 epoch 完成后，写 `train_history.csv` 时因 CSV 字段只取首行而失败；保留 `config.yaml`、`source_manifest.json` 和 checkpoint | 这是失败实验，不能复用 `EXP-S4-004`；已修复 CSV 字段合并逻辑并新建 `EXP-S4-005` |
| 2026-07-03 | 完成 SNR-conditioned pixel residual refiner pilot | `configs/s5_residual_refiner_pilot_coco256_awgn.yaml`, `scripts/s5_residual_refiner_pilot.py`, `outputs/EXP-S4-005/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_residual_refiner_pilot.py --device cuda:0` 成功；训练样本为 `sample_000008`-`sample_000031`，评估样本为 `sample_000000`-`sample_000007` | 避开 SD VAE 后，像素域 residual 在 1/4/7/13/19 dB 上 PSNR 分别提升 `+0.3866/+0.1868/+0.0905/+0.1248/+0.1682` dB；LPIPS 除 7 dB 基本持平外均改善；pseudo final failure 未高于 M0 |
| 2026-07-03 | 扩大正式 M0 export 供 residual validation 使用 | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/` | 清空代理变量后运行 `scripts/s2_deepjscc_highres_export.py`，在同一 512 张 COCO val subset 上重新导出每个 SNR 前 256 张 M0 reconstruction | 该导出不改变 M0 512 张评估指标，只把可用于后处理训练/验证的 PNG 从 32 张/SNR 扩大到 256 张/SNR；不覆盖旧 export |
| 2026-07-03 | 完成 SNR-conditioned pixel residual refiner validation | `configs/s5_residual_refiner_validation_coco256_awgn.yaml`, `scripts/s5_residual_refiner_pilot.py`, `outputs/EXP-S4-006/` | `--dry-run`、清空代理变量后运行 `python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --device cuda:0` 成功；训练样本为 `sample_000032`-`sample_000191`，评估样本为 `sample_000192`-`sample_000255` | Refined PSNR 在 1/4/7/13/19 dB 上提升 `+1.1323/+0.7837/+0.5859/+0.5504/+0.5654` dB，LPIPS 全部改善；经过 top-1 agreement fallback 后，M3 PSNR 仍提升 `+0.3313/+0.3812/+0.3815/+0.4557/+0.4561` dB，M3 final failure 未高于 M0 |
| 2026-07-03 | 完成 `EXP-S4-006` semantic gate error analysis | `scripts/s5_analyze_residual_gate_errors.py`, `outputs/analysis/exp_s4_006_gate_error_analysis/` | 读取 `outputs/EXP-S4-006/per_sample.csv`，生成 `summary.csv`、`REPORT.md`、按 case type 分类的 original/M0/refined/M3 拼图；不跑模型、不联网 | 当前 top-1 agreement gate 有结构性保证：同一分类器下 M3 final failure 不会超过 M0；但它也拒绝了 41/320 个 refined 修复 M0 pseudo-label 的样本，同时保护了 28/320 个 M0-correct/refined-wrong 样本 |
| 2026-07-06 | 完成 `EXP-S4-006` semantic gate policy sweep | `scripts/s5_sweep_residual_gate_policies.py`, `outputs/analysis/exp_s4_006_gate_policy_sweep/` | 清空代理变量后运行 `python3 scripts/s5_sweep_residual_gate_policies.py --device cuda:0`，用本地 AlexNet 重新计算 original/M0/refined top-5，并离线评估 22 个 receiver-side gate policies | `top1_equal_or_refined_conf_gain_ge_0p05` 当前最均衡：final failure `0.3188` vs top-1 gate `0.3750`，final PSNR `+0.1153` dB，missed repair 从 41 降到 20，但 accepted new error 从 0 增到 3；top-5 overlap 类策略 PSNR 更高但语义风险更大 |
| 2026-07-06 | 完成 `EXP-S4-006` confidence-gain gate 辅助审计和候选输出落盘 | `configs/s5_residual_gate_aux_audit_exp_s4_006.yaml`, `configs/s5_materialize_conf_gain_gate_exp_s4_006.yaml`, `scripts/s5_audit_residual_gate_aux_semantics.py`, `scripts/s5_materialize_residual_gate_policy.py`, `outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/`, `outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/` | `py_compile`、`--dry-run`、清空代理变量后运行辅助审计和 materialize 脚本；审计使用本地 OpenCLIP ViT-B/32 与 COCO captions，不下载；materialize 只复制已有 M0/refined PNG | 候选 gate 新增接受 37/320 个样本，其中 21 个是 pseudo-label repair、3 个是 accepted new error；final failure `0.3188`，PSNR 比 top-1 gate 高 `+0.1153` dB；CLIP image-image 均值略升 `+0.0016`，caption CLIP 均值略降 `-0.0007`，因此仍是候选而非最终 M3 |
| 2026-07-06 | 完成 `EXP-S4-006` held-out confidence-gain gate 复核 | `configs/s5_residual_refiner_heldout_gate_exp_s4_006.yaml`, `scripts/s5_residual_refiner_heldout_gate_eval.py`, `outputs/analysis/exp_s4_006_heldout_gate_check/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_residual_refiner_heldout_gate_eval.py --device cuda:0`；加载 `EXP-S4-006/checkpoints/best.pt`，在 `sample_000000`-`sample_000031` 上推理，不重训、不下载 | Held-out 上候选 gate final failure `0.2812` vs top-1 gate `0.3250`，PSNR `+0.1007` dB；新增接受 19/160 个样本，其中 9 个 repair、2 个 accepted new error。方向复现但风险仍在，不能直接定为最终 M3 |
| 2026-07-06 | 完成 `EXP-S4-006` confidence-gain gate 的 receiver-side CLIP veto sweep | `configs/s5_conf_gain_clip_veto_sweep_exp_s4_006.yaml`, `scripts/s5_sweep_conf_gain_clip_veto.py`, `outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_sweep_conf_gain_clip_veto.py --device cuda:0`；读取 validation/held-out CSV 和本地 OpenCLIP ViT-B/32，不下载 | 阈值 `CLIP(M0, refined) >= 0.98` 可使 validation/held-out accepted new error 均为 0，但只保留 2 个 repair，总 PSNR 增益仅 `+0.0073` dB vs top-1 gate；单一 CLIP veto 安全但过保守，不能作为最终 M3 |
| 2026-07-06 | 完成 `EXP-S4-006` confidence-gain CLIP veto 的 SNR 校准分析 | `configs/s5_conf_gain_clip_veto_snr_calibration_exp_s4_006.yaml`, `scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py`, `outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --overwrite`；只读取已有 CLIP sweep CSV，不联网、不训练、不重算 CLIP | validation-only 独立 SNR schedule 在 validation 上 final failure `0.3438`、PSNR `+0.0533` dB vs top-1、10 repair、0 new error，但 held-out 仍有 1 个 accepted new error；monotonic schedule held-out 安全但只保留 1 个 repair。单一 CLIP 阈值即使按 SNR 校准也不足以收敛最终 M3 |
| 2026-07-06 | 完成 `EXP-S4-006` confidence-gain gate 的 receiver-side risk-rule sweep | `configs/s5_conf_gain_risk_rule_sweep_exp_s4_006.yaml`, `scripts/s5_sweep_conf_gain_risk_rules.py`, `outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_sweep_conf_gain_risk_rules.py --overwrite`；只读取已有 CLIP/top-k CSV 和 PNG，不联网、不训练 | 选出的 shadow-margin 风险规则在 validation 上 final failure `0.3156`、PSNR `+0.0953` dB vs top-1、19 repair、0 new error；held-out 上 final failure `0.2812`、PSNR `+0.0748` dB vs top-1、7 repair、0 new error。它挡掉 raw confidence-gain 的 5 个 validation/held-out new error，是当前最强 M3 gate 候选，但仍需正式 split 复核 |
| 2026-07-06 | 完成 `EXP-S4-006` selected risk-rule 候选输出落盘 | `configs/s5_materialize_risk_rule_gate_exp_s4_006.yaml`, `scripts/s5_materialize_risk_rule_policy.py`, `outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_materialize_risk_rule_policy.py`；只读取 risk-rule decision CSV 并复制已有 M0/refined PNG | 已导出 480 张 final PNG、`per_sample.csv`、`summary.csv`、`REPORT.md` 和样例 sheet；validation/held-out 分别保持 19/7 repair、0/0 accepted new error，作为当前最强 M3 gate 候选的可复查 artifact |
| 2026-07-06 | 完成 `EXP-S4-006` selected risk-rule classifier ensemble 审计 | `configs/s5_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml`, `scripts/s5_audit_risk_rule_classifier_ensemble.py`, `outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/` | `py_compile`、`--dry-run`、检查代理变量后清空代理直连下载 torchvision ResNet18/MobileNetV3-Small 权重，并运行 `python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --device cuda:0 --allow-download` | AlexNet 口径仍为 0 new error，但 ResNet18/MobileNetV3-Small 暴露跨模型风险：validation/held-out 分别有 26/15 个样本被至少一个分类器标为 selected accepted new error，多数票新错为 2/1 个；该 gate 不能写成跨模型安全，只能作为需继续收紧的候选 |
| 2026-07-06 | 完成 `EXP-S4-006` ensemble-risk 二级 veto sweep | `configs/s5_ensemble_risk_veto_sweep_exp_s4_006.yaml`, `scripts/s5_sweep_ensemble_risk_veto.py`, `outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_sweep_ensemble_risk_veto.py`；只读取 selected risk-rule CSV、ensemble audit CSV 和已有 PNG，不联网、不重训 | 选中规则把 validation/held-out 多数票 new-error 清到 0/0，但额外 veto 96/58 张、remaining any-new-error 16/8、PSNR 相比 `selected_risk_rule` 回吐 `-0.1834/-0.2538` dB；这是保守风险收紧证据，不是最终 M3 |
| 2026-07-06 | 完成 `EXP-S4-006` receiver-side risk score sweep | `configs/s5_receiver_risk_score_sweep_exp_s4_006.yaml`, `scripts/s5_sweep_receiver_risk_score.py`, `outputs/analysis/exp_s4_006_receiver_risk_score_sweep/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_sweep_receiver_risk_score.py --overwrite`；只读取 selected risk-rule CSV、ensemble audit CSV 和已有 PNG，不联网、不重训 | repair-pref score 只额外 veto 48/26 张但 held-out 漏 1 个多数票 new-error；若要求 validation/held-out 同时清零，多数候选比保守二级 veto 更重。浅层 risk score 暂不适合作最终 gate，记录为负/部分结果 |
| 2026-07-06 | 完成 `EXP-S4-007` latent-free residual diffusion pilot | `configs/s5_residual_diffusion_pilot_coco256_awgn.yaml`, `scripts/s5_residual_diffusion_pilot.py`, `outputs/EXP-S4-007/` | `py_compile`、`--dry-run`、1 epoch smoke 和清空代理变量后正式运行 `python3 scripts/s5_residual_diffusion_pilot.py --device cuda:0` 成功；训练样本为 `sample_000032`-`sample_000111`，评估样本为 `sample_000192`-`sample_000207` | 负结果：naive DDPM refined PSNR 在 1/4/7/13/19 dB 下降 `-7.1634/-7.4843/-7.0882/-5.4204/-4.4217` dB，LPIPS 全部变差；top-1 gate 可把 M3 final failure 拉回 M0，但 M3 PSNR 仍下降 `-1.4156/-1.6618/-2.6019/-2.1567/-2.1002` dB |
| 2026-07-06 | 扩大正式 M0 export 供 test-like 复核使用 | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/` | 清空代理变量后运行 `scripts/s2_deepjscc_highres_export.py --export-count 384`；original、1 dB、19 dB 目录抽查均为 384 张；指标与旧正式 export 一致 | 新增 `sample_000256`-`sample_000383` 的 M0/original PNG，可用于不覆盖旧输出的 test-like gate 复核；该导出不代表新 M0 模型 |
| 2026-07-06 | 完成 `EXP-S4-006` test-like confidence-gain gate 复核 | `configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml`, `scripts/s5_residual_refiner_heldout_gate_eval.py`, `outputs/analysis/exp_s4_006_testlike_gate_check/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --device cuda:0`；加载 `EXP-S4-006/checkpoints/best.pt`，在 `sample_000256`-`sample_000319` 上推理，不重训、不下载 | Test-like 上 raw confidence-gain candidate final failure `0.4313` vs top-1 gate `0.4719`，PSNR `+0.0814` dB；新增接受 26/320 个样本，其中 17 个 repair、4 个 accepted new error。收益复现但风险更明确，不能作为最终 M3 |
| 2026-07-07 | 完成 `EXP-S4-006` frozen risk-rule test-like 复核 | `configs/s5_testlike_risk_rule_check_exp_s4_006.yaml`, `scripts/s5_apply_testlike_risk_rules.py`, `outputs/analysis/exp_s4_006_testlike_risk_rule_check/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_apply_testlike_risk_rules.py --device cuda:0`；只读取 test-like CSV、旧 selected-rule JSON、旧保守 veto JSON 和本地 CLIP 权重，不重训、不下载、不调参 | `selected_risk_rule` 在 test-like 上 final failure `0.4437`、PSNR `+0.0434` dB vs top-1、10 repair、1 accepted new error；保守 ensemble veto 未降低 new error，PSNR 反而相对 selected 回吐 `-0.1902` dB。浅层接收端规则仍不是最终 M3 |
| 2026-07-07 | 完成阶段性方向复盘 | 本地中枢文档与当前实验链路 | 复读 `PROJECT.md`、`MILESTONES.md`、`PROGRESS.md`、`EXPERIMENTS.md`、`LITERATURE.md`、`README.md`，并对照近期 generative/diffusion JSCC 相关工作 | 方向仍值得做，但不应继续作为“普通 diffusion 后处理”推进；应收缩为 semantic-risk-aware residual restoration / failure handling。Diffusion 只作为受控短链或负结果/消融保留，不能再从空 prompt 或随机噪声路线硬推 |
| 2026-07-07 | 完成 `EXP-S4-006` test-like classifier-ensemble 审计 | `configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml`, `scripts/s5_audit_risk_rule_classifier_ensemble.py`, `outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/` | `py_compile`、新旧配置 `--dry-run`、清空代理变量后运行 `python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --device cuda:0`；3 个 torchvision 分类器权重均来自本地 cache，不下载 | Test-like 上 `selected_risk_rule` 无 majority-vote accepted new error，但有 23 个 any-model new-error vote；AlexNet/ResNet18/MobileNetV3-Small selected failure 分别为 `0.4437/0.4344/0.5406`，repair 为 `10/31/32`，new error 为 `1/13/9`。该 rule 有迁移性但仍不是跨模型安全的最终 M3 |
| 2026-07-07 | 完成 `EXP-S4-006` test-like COCO object CLIP clean-correct 辅助诊断 | `configs/s5_testlike_coco_object_clip_clean_eval_exp_s4_006.yaml`, `scripts/s5_coco_object_clip_clean_eval.py`, `outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s5_coco_object_clip_clean_eval.py --device cuda:0`；读取 COCO instances 和本地 OpenCLIP ViT-B/32，不下载 | 64 个 test-like 原图中 55 个 dominant label 可用，27 个进入 CLIP clean-correct，形成 135 行；`selected_risk_rule` final failure 与 top-1 持平 `0.0815`、PSNR `+0.0257` dB、1 repair/2 new error；保守 ensemble veto new error 为 0 但 PSNR 比 top-1 低 `-0.1727` dB。该结果是辅助 GT-like 诊断，不是最终监督真值 |
| 2026-07-07 | 完成第一版 minimal closure report | `configs/s6_minimal_closure_report.yaml`, `scripts/s6_make_minimal_closure_report.py`, `outputs/analysis/minimal_closure_report/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_minimal_closure_report.py`；只读取已有 metrics/CSV 并生成报告、CSV 和 3 张 tradeoff 图 | 当前可用闭环口径：M1 是负参考；M2 residual CNN 平均 PSNR `+0.7235` dB、LPIPS `-0.0274`；M3 top-1 fallback 平均 PSNR `+0.4011` dB、LPIPS `-0.0104` 且 pseudo semantic failure 不增。`selected_risk_rule` 保留为候选/消融 |
| 2026-07-07 | 完成 `EXP-S4-006` residual shrink selection 派生分析 | `configs/s6_residual_shrink_selection_exp_s4_006.yaml`, `scripts/s6_residual_shrink_selection.py`, `outputs/analysis/exp_s4_006_residual_shrink_selection/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_residual_shrink_selection.py --device cuda:0 --overwrite`；只读取已有 M0/refined PNG、本地 AlexNet 和 LPIPS 权重，不训练、不下载 | Validation-only top-1 fallback shrink schedule 把平均 PSNR delta 从 `+0.4011` 提升到 `+0.4584` dB，LPIPS delta 从 `-0.0104` 改到 `-0.0153`，pseudo final failure 仍等于 M0；always-accept 虽更高质量但留下 19-28 个 accepted new error，不能作为最终 M3 |
| 2026-07-07 | 完成 frozen residual shrink schedule test-like 复核 | `configs/s6_testlike_residual_shrink_schedule_check_exp_s4_006.yaml`, `scripts/s6_apply_residual_shrink_schedule.py`, `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_apply_residual_shrink_schedule.py --device cuda:0`；读取 test-like refined PNG 和 validation schedule，不调参、不训练、不下载 | Frozen top-1 shrink schedule 在 test-like 上平均 PSNR delta `+0.4552` dB vs M0，比 full-strength top-1 高 `+0.0439` dB，LPIPS delta `-0.0152`，pseudo final failure 不增且 new error 为 0；always-accept 仍有 25/12 个 new error |
| 2026-07-07 | 刷新 minimal closure report 并纳入 shrink M3 | `configs/s6_minimal_closure_report.yaml`, `scripts/s6_make_minimal_closure_report.py`, `outputs/analysis/minimal_closure_report/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_minimal_closure_report.py --overwrite`；只读取已有 metrics/CSV，不训练、不下载 | 报告新增 `M3-ResidualRestorationTop1ShrinkFallback`、`residual_shrink_policy_tradeoff.csv` 和 shrink tradeoff 图；当前最强保守候选为 frozen top-1 shrink schedule，test-like PSNR delta `+0.4552` dB 且 pseudo new error 为 0 |
| 2026-07-07 | 完成 frozen residual shrink schedule held-out 复核 | `configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml`, `scripts/s6_apply_residual_shrink_schedule.py`, `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/` | `py_compile`、test-like/held-out `--dry-run`、清空代理变量后运行 `python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --device cuda:0`；读取 held-out refined PNG 和 frozen validation schedule，不调参、不训练、不下载 | Frozen top-1 shrink schedule 在 held-out 上平均 PSNR delta `+0.4689` dB vs M0，比 full-strength top-1 高 `+0.0236` dB，LPIPS delta `-0.0150`，pseudo final failure 不增且 new error 为 0；always-accept 仍有 10/3 个 new error |
| 2026-07-07 | 再次刷新 minimal closure report 纳入 held-out shrink 证据 | `configs/s6_minimal_closure_report.yaml`, `scripts/s6_make_minimal_closure_report.py`, `outputs/analysis/minimal_closure_report/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_minimal_closure_report.py --overwrite`；只读取已有 metrics/CSV，不训练、不下载 | 主报告现在同时列出 validation、held-out 和 test-like 三段 shrink tradeoff；`M3-ResidualRestorationTop1ShrinkFallback` 在 held-out/test-like 上分别为 `+0.4689/+0.4552` dB 且 new error 为 `0/0` |
| 2026-07-07 | 完成 residual shrink M3 artifact gallery | `configs/s6_residual_shrink_artifact_gallery_exp_s4_006.yaml`, `scripts/s6_make_residual_shrink_gallery.py`, `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_residual_shrink_gallery.py --overwrite`；只读取已有 shrink CSV/PNG，不训练、不下载、不调参 | 生成统一 `REPORT.md`、policy/case CSV 和 safe accept、protective reject、rejected good、unsafe new-error 样例 sheet；selected shrink M3 三段 new error 为 `0/0/0`，always-accept full strength 为 `28/10/25` |
| 2026-07-07 | 完成 adaptive residual alpha policy 派生分析 | `configs/s6_adaptive_residual_alpha_policy_exp_s4_006.yaml`, `scripts/s6_apply_adaptive_residual_alpha_policy.py`, `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --device cuda:0`；只读取已有 alpha candidate PNG、本地 AlexNet 和 LPIPS 权重，不训练、不运行 diffusion、不下载 | Per-sample 最大 top-1-consistent alpha 在 validation/held-out/test-like 上 PSNR delta 为 `+0.5584/+0.5664/+0.5691` dB，accepted new error 为 `0/0/0`，强于 fixed shrink schedule；但 repair 仍为 0，missed repair 为 `45/31/70` |
| 2026-07-07 | 刷新 minimal closure report 纳入 adaptive alpha M3 | `configs/s6_minimal_closure_report.yaml`, `scripts/s6_make_minimal_closure_report.py`, `outputs/analysis/minimal_closure_report/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_minimal_closure_report.py --overwrite`；只读取已有 metrics/CSV，不训练、不下载、不重算分类器 | 报告新增 `M3-AdaptiveResidualAlphaTop1Fallback`、`adaptive_residual_alpha_policy_tradeoff.csv` 和 adaptive tradeoff 图；当前最强保守候选更新为 adaptive alpha，validation/held-out/test-like PSNR delta `+0.5584/+0.5664/+0.5691` dB，new error `0/0/0` |
| 2026-07-07 | 完成 two-stage residual alpha policy 派生分析 | `configs/s6_two_stage_residual_alpha_policy_exp_s4_006.yaml`, `scripts/s6_apply_two_stage_residual_alpha_policy.py`, `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --device cuda:0`；只读取 adaptive alpha 逐样本表和 final 图，不重算分类器、不训练、不下载、不加载 LPIPS；一次 ad hoc LPIPS 探针误触发临时权重下载，已停止并删除 `/tmp/alpha_twostage_cache`，未使用其结果 | Two-stage `full_then_fixed_schedule` 在 validation/held-out/test-like 上 PSNR delta `+0.4831/+0.5009/+0.4875` dB，new error `0/0/0`；比 fixed schedule 略好但低于 exhaustive adaptive alpha，适合作部署性消融 |
| 2026-07-07 | 刷新 minimal closure report 纳入 two-stage alpha 消融 | `configs/s6_minimal_closure_report.yaml`, `scripts/s6_make_minimal_closure_report.py`, `outputs/analysis/minimal_closure_report/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_minimal_closure_report.py --overwrite`；只读取已有 metrics/CSV，不训练、不下载 | 主报告新增 `M3-TwoStageResidualAlphaTop1Fallback` 和 `two_stage_residual_alpha_policy_tradeoff.csv`；结论保持 adaptive alpha 是最强保守候选，two-stage 是少候选检查的部署折中 |
| 2026-07-09 | 完成 receiver-side alpha predictor pilot | `configs/s6_receiver_alpha_predictor_exp_s4_006.yaml`, `scripts/s6_train_receiver_alpha_predictor.py`, `outputs/analysis/exp_s4_006_receiver_alpha_predictor/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_receiver_alpha_predictor.py --device cuda:0`；validation-only 训练小型 tabular predictor，评估时只分类预测 alpha 候选并用 top-1 fallback，不训练图像模型、不运行 diffusion、不下载、不加载 LPIPS | Predictor 在 validation/held-out/test-like 上 PSNR delta `+0.5584/+0.5099/+0.4871` dB，new error `0/0/0`；held-out 略高于 two-stage，test-like 基本持平，但仍低于 exhaustive adaptive alpha |
| 2026-07-09 | 刷新 minimal closure report 纳入 receiver alpha predictor | `configs/s6_minimal_closure_report.yaml`, `scripts/s6_make_minimal_closure_report.py`, `outputs/analysis/minimal_closure_report/` | `py_compile`、`--dry-run`、清空代理变量后运行 `python3 scripts/s6_make_minimal_closure_report.py --overwrite`；只读取已有 metrics/CSV，不训练、不下载 | 主报告新增 `M3-ReceiverAlphaPredictorTop1Fallback` 和 `receiver_alpha_predictor_tradeoff.csv`；结论保持 adaptive alpha 是最强保守候选，receiver predictor 是 learned deployability pilot |
| 2026-07-09 | 明确最小闭环与探索推进的节奏边界 | `PROGRESS.md` | 复读中枢文档并讨论后续推进方式；未运行实验、不改代码、不新增实验结果 | 后续不要求每个小探索都刷新 minimal closure；小实验可以先用 smoke/validation 快速试错，只有候选进入 M2/M3 命名、跨 split 复核或论文口径时才收敛到最小闭环报告。这样保留 semantic drift 安全底线，同时避免过度保守导致方法不前进 |
| 2026-07-09 | 完成 alpha-head residual refiner pilot | `configs/s6_alpha_head_residual_refiner_pilot_exp_s4_006.yaml`, `scripts/s6_train_alpha_head_residual_refiner.py`, `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/` | `py_compile`、清空代理变量后 `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_alpha_head_residual_refiner.py --device cuda:0`；加载 `EXP-S4-006` residual checkpoint，冻结 residual CNN，只训练 alpha head；不运行 diffusion、不下载、不加载 LPIPS | Alpha head 在 validation/held-out/test-like 上 PSNR delta `+0.3846/+0.3808/+0.3623` dB，new error `0/0/0`，target-alpha accuracy `0.6687/0.6500/0.5844`；低于 full-strength top-1、two-stage 和 receiver predictor，说明冻结 residual 特征 + 普通 CE 不足，下一步需 class-weighting、联合训练或 semantic-risk-aware residual amplitude loss |
| 2026-07-09 | 完成 weighted alpha-head residual refiner follow-up | `configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml`, `scripts/s6_train_alpha_head_residual_refiner.py`, `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/` | `py_compile`、清空代理变量后 weighted config `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml --device cuda:0`；启用 tempered inverse-frequency CE weights；不运行 diffusion、不下载、不加载 LPIPS | Weighted alpha head 在 validation/held-out/test-like 上 PSNR delta `+0.3851/+0.3506/+0.3166` dB，new error `0/0/0`，target-alpha accuracy `0.6375/0.5750/0.4969`；预测分布比普通 CE 更分散，但 held-out/test-like 更差，说明类别不均衡不是唯一瓶颈，alpha 选择需要收益/风险感知目标或联合训练 |
| 2026-07-09 | 完成 benefit-aware alpha predictor follow-up | `configs/s6_benefit_alpha_predictor_exp_s4_006.yaml`, `scripts/s6_train_receiver_alpha_predictor.py`, `outputs/analysis/exp_s4_006_benefit_alpha_predictor/` | `py_compile`、默认 receiver predictor 和 benefit config `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_receiver_alpha_predictor.py --config configs/s6_benefit_alpha_predictor_exp_s4_006.yaml --device cuda:0 --overwrite`；训练目标为 safe PSNR utility soft labels；不运行 diffusion、不下载、不加载 LPIPS | Benefit predictor 在 validation/held-out/test-like 上 PSNR delta `+0.5538/+0.4474/+0.4627` dB，new error `0/0/0`；validation 贴近 adaptive alpha，但 held-out/test-like 低于 two-stage/原 receiver predictor，说明目标设计更合理但 tabular receiver 特征泛化不足 |
| 2026-07-09 | 完成 benefit-aware alpha-head residual refiner follow-up | `configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml`, `scripts/s6_train_alpha_head_residual_refiner.py`, `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/` | `py_compile`、默认 alpha-head 和 benefit config `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml --device cuda:0`；冻结 residual CNN，仅把训练目标替换为 safe-PSNR utility alpha；不运行 diffusion、不下载、不加载 LPIPS | Benefit alpha-head 在 validation/held-out/test-like 上 PSNR delta `+0.4251/+0.4192/+0.3530` dB，new error `0/0/0`，target accuracy `0.5406/0.4313/0.4062`；比普通/weighted alpha-head 有部分改进，但仍低于 receiver predictor、two-stage 和 adaptive alpha，说明冻结内部特征仍不足以学习细粒度 benefit/risk alpha |
| 2026-07-09 | 完成 benefit-aware joint alpha-head residual refiner follow-up | `configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml`, `scripts/s6_train_alpha_head_residual_refiner.py`, `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/` | `py_compile`、默认 alpha-head 和 joint benefit config `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --device cuda:0`；解冻 residual CNN，soft-alpha/target-alpha MSE 反传到 refiner；不运行 diffusion、不下载、不加载 LPIPS | Joint benefit alpha-head 在 validation/held-out/test-like 上 PSNR delta `+0.3294/+0.2303/+0.1869` dB，new error `0/0/0`，target accuracy `0.7719/0.3875/0.3719`；validation alpha 分类大幅改善，但 full/refined PSNR 被损伤，说明全量 unfreeze + CE 主导会破坏 residual restoration anchor |
| 2026-07-09 | 完成 benefit-aware tail-only partial alpha-head residual refiner follow-up | `configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml`, `scripts/s6_train_alpha_head_residual_refiner.py`, `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/` | `py_compile`、默认/全量 joint/tail-only 三个配置 `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --device cuda:0`；只训练 tail 与 alpha head，不运行 diffusion、不下载、不加载 LPIPS | Tail-only partial fine-tune 在 validation/held-out/test-like 上 PSNR delta `+0.4749/+0.4552/+0.4061` dB，new error `0/0/0`，target accuracy `0.5437/0.4313/0.4250`；明显好于冻结 benefit alpha-head 和全量 joint，说明 partial/reconstruction-dominant 方向成立，但仍低于 receiver predictor/two-stage/adaptive alpha |
| 2026-07-09 | 完成 benefit-aware continuous-alpha tail-only residual refiner follow-up | `configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml`, `scripts/s6_train_alpha_head_residual_refiner.py`, `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/` | `py_compile`、默认/tail classification/continuous regression 三个配置 `--dry-run`、清空代理变量后运行 `python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml --device cuda:0`；只训练 tail 与连续 alpha head，不运行 diffusion、不下载、不加载 LPIPS | Continuous-alpha tail-only 在 validation/held-out/test-like 上 PSNR delta `+0.5010/+0.5049/+0.5012` dB，new error `0/0/0`；超过离散 tail-only classification，并在 held-out/test-like 达到或超过 learned 部署 baseline，但仍低于后验 adaptive alpha |
| 2026-07-09 | 完成 continuous-alpha tail refiner LPIPS 与 classifier-ensemble 审计 | `configs/s6_continuous_alpha_tail_refiner_audit_exp_s4_006.yaml`, `scripts/s6_audit_continuous_alpha_tail_refiner.py`, `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/` | `py_compile`、`--dry-run`、修正 LPIPS 缓存路径后清空代理变量运行 `python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --device cuda:0`；只读取已有 continuous-alpha PNG/CSV、本地 LPIPS/AlexNet/ResNet18/MobileNetV3-Small 权重，不训练、不运行 diffusion | Continuous-alpha 的 LPIPS delta 为 `-0.0149/-0.0149/-0.0162`，优于 full-strength fallback；AlexNet new error 仍 `0/0/0`，但 ensemble any new error 为 `17/9/14`，majority new error 为 `1/0/0`，说明它是强候选但还非最终 M3 |
| 2026-07-09 | 复核上次总结后新增进展 | `PROGRESS.md`, `EXPERIMENTS.md`, `README.md`, `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/REPORT.md` | 按 `AGENTS.md` 读取中枢文档、实验索引和最新审计报告；未运行新实验、不改运行方式、不新增文献 | 上次阶段总结后实质新增为 `ANALYSIS-S6-020`：continuous-alpha 补齐 LPIPS 与 classifier-ensemble 审计；结论维持“强候选但非最终 M3” |
| 2026-07-09 | 整理组会可展示结果 | `reports/showcase_results_2026-07-09.md`, `outputs/analysis/minimal_closure_report/`, `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/`, `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/` | 按中枢文档和现有报告筛选展示材料；查看关键 tradeoff 图和 artifact sheet；未运行新实验、不新增指标、不改运行方式 | 形成一份组会展示清单：包含 blind diffusion 负结果、M2 residual CNN 正向 anchor、M3 shrink/adaptive-alpha 保守结果、continuous-alpha learned 候选和 ensemble 风险边界；明确哪些能当正结果、哪些只能当负对照或候选 |
| 2026-07-09 | 整理当前技术路线和 semantic drift 计算口径 | `reports/technical_method_notes_2026-07-09.md`, `PROJECT.md`, `MILESTONES.md`, `scripts/s4_classifier_consistency_eval.py`, `scripts/s5_residual_refiner_pilot.py`, `scripts/s6_residual_shrink_selection.py`, `scripts/s6_apply_adaptive_residual_alpha_policy.py` | 按中枢文档和实际脚本核对指标定义；未运行实验、不新增结果、不改运行方式 | 明确当前 pipeline 为 DeepJSCC + SNR-conditioned residual restoration + semantic fallback/alpha control；semantic drift 当前按冻结分类器 top-1 pseudo-label 一致性统计，`accepted_new_error` 是核心风险指标 |
| 2026-07-10 | 完成阶段性方向审计与继续价值判断 | `reports/phase_summary_2026-07-10.md`, `PROJECT.md`, `MILESTONES.md`, `PROGRESS.md`, `EXPERIMENTS.md`, `LITERATURE.md`, `README.md`, `outputs/analysis/minimal_closure_report/REPORT.md` | 按项目规则复读中枢文档和 closure report；未运行新实验、不新增指标、不改运行方式 | 判断：若主线仍是 blind diffusion 后处理则不值得继续；若收缩为 semantic-risk-controlled residual restoration / alpha control，则值得继续。下一阶段优先补 supervised clean-correct 评估，并围绕 continuous alpha 做 semantic-risk-aware 训练或 model selection |
| 2026-07-10 | 复核 SGD-JSCC 可借鉴点与边界 | `LITERATURE.md`, `PROJECT.md`, `MILESTONES.md`, `PROGRESS.md`, `EXPERIMENTS.md`, `README.md` | 按项目规则复读中枢文档，核对 SGD-JSCC 论文摘要和官方代码仓库说明；未运行实验、不改运行方式 | 结论：应该参考 SGD-JSCC 的语义条件和信道自适应设计，但不能照搬成新主线；当前应把 text/edge/structure guidance 思想转化为受控 residual/diffusion correction，并继续保留 semantic drift / accepted new error 作为本项目核心指标 |
| 2026-07-10 | 完成 SGD-inspired edge-conditioned residual refiner 与 frozen shrink 复核 | `scripts/s5_residual_refiner_pilot.py`, `scripts/s5_residual_refiner_heldout_gate_eval.py`, `scripts/s6_residual_shrink_selection.py`, `scripts/s6_apply_residual_shrink_schedule.py`, `configs/s5_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml`, `configs/s6_edge_residual_shrink_selection_exp_s4_008.yaml`, `outputs/EXP-S4-008/`, `outputs/analysis/exp_s4_008_edge_*` | `py_compile`、dry-run、清空代理变量后运行 `EXP-S4-008` validation、held-out/test-like gate check、validation shrink selection、frozen held-out/test-like shrink schedule check；不下载、不运行 diffusion，LPIPS 省略 | 结构条件来自 M0 的 Sobel/Laplacian，refined PSNR delta `+0.9398` dB；frozen top-1 shrink schedule 在 validation/held-out/test-like 上 PSNR delta `+0.5782/+0.6041/+0.5707` dB，accepted new error `0/0/0`，成为当前最强 AlexNet-pseudo 安全保守候选；confidence-gain gate 和 always-accept 仍会漏 new error，不能作为安全方法 |
| 2026-07-10 | 完成 edge × capacity/training-budget 2×2 受控消融 | `EXP-S4-009`, `EXP-S4-010`, `scripts/s6_compare_edge_capacity_ablation.py`, `outputs/analysis/exp_s4_006_008_009_010_edge_capacity_ablation/` | 两个新增训练均 dry-run 后清空代理运行；逐 PNG 重算；sample-cluster 10,000 次 paired bootstrap；匹配字段、checkpoint/config、参数量和 SHA256 全部校验 | small/large edge raw 净增益 `+0.0501/+0.1389` dB，95% CI 均排除 0；edge 质量收益成立，但 large raw pseudo failure 增加 `+0.0438`，不能称为语义改进 |
| 2026-07-10 | 完成 matched large edge 跨 split/fresh-holdout 审计 | `scripts/s6_compare_matched_edge_holdouts.py`, `outputs/analysis/exp_s4_008_009_matched_edge_holdout_audit/` | validation/held-out/test-like/fresh-holdout 分别按 sample cluster 做 10,000 次 paired bootstrap；fresh-holdout 固定为此前未分析的 `sample_000320`-`sample_000383` | edge raw PSNR 净增益 `+0.1389/+0.1565/+0.1585/+0.1411` dB，所有 CI 下界 > 0、所有 5-SNR 方向一致；pseudo semantic 变化跨 split 不稳定 |
| 2026-07-10 | 修正单调 schedule 并完成 LPIPS / classifier-ensemble 审计 | `scripts/s6_residual_shrink_selection.py`, `scripts/s6_audit_residual_policy.py`, `outputs/analysis/exp_s4_008_edge_monotonic_*` | 全局枚举满足 `gate×alpha` 随 SNR 非增的 validation schedule；冻结到 held-out/test-like/fresh-holdout；本地 LPIPS 与三冻结分类器离线审计；无下载 | 四段 PSNR `+0.5734/+0.6128/+0.5700/+0.5668` dB、LPIPS 全改善；ensemble majority new error `1/1/0/3`，故暂不升级为跨模型安全最终 M3 |
| 2026-07-10 | 整理 edge conditioning 显著成果报告 | `reports/edge_conditioning_significant_result_2026-07-10.md` | 逐项回查 2×2、cross-split bootstrap、单调 frozen schedule、LPIPS 和 ensemble 输出；不新增指标 | 形成可直接用于组会/论文讨论的正结果、风险边界、允许/禁止表述和下一步优先级 |
| 2026-07-10 | 完成 SGD-inspired coarse source-description 嵌套审计 | `configs/s6_imagenette_source_semantic_description_eval.yaml`, `scripts/s6_imagenette_source_semantic_description_eval.py`, `outputs/analysis/imagenette_source_semantic_description_policy_dev/` | 在 policy-dev 内按 WNID+SHA256 固定拆成 945 张 semantic-select / 949 张 semantic-audit；只在 select 选择 80-bit uint8 source-probability 距离规则，audit 一次性检验；逐行复现原 G_gate 输出，official val 保持封存 | 连续描述规则没有满足“选择集零 new-error + 至少保留 50% M2 PSNR”的候选；最保守规则在 audit 仅保留 3.26% M2 PSNR，failure 比 M2 高 `+1.6078 pp`，95% CI `[+0.8627,+2.3922] pp`。结论：source description 只用于末端 gate 不足以解决语义风险 |
| 2026-07-10 | 完成 sender source-edge oracle 与 matched paired bootstrap | `EXP-S4-011`, `scripts/s6_compare_source_edge_oracle.py`, `outputs/analysis/exp_s4_011_source_edge_oracle_vs_receiver_edge/` | 与 EXP-S4-008 匹配容量/epochs/split/seed/loss/gates，只把 Sobel/Laplacian 来源从 M0 换为 sender original；64 图×5 SNR；10,000 次 sample-cluster paired bootstrap | source-edge raw PSNR 相对 receiver-edge 再提升 `+3.5149 dB`，95% CI `[+3.2602,+3.7652]`，五个 SNR 全为正；raw pseudo failure `0.3625→0.2062`。这是 perfect-edge、总 CBR 未定义的 feasibility upper bound，不是可部署通信结果 |

## 下一步

1. 第一版正向主线继续以 `EXP-S4-006` 的 pixel residual CNN 为 anchor；不要把 `EXP-S4-007` 的 naive random-residual DDPM 纳入 M2/M3 正结果。
2. 若继续做 diffusion，必须换设计：从 M0 或 residual CNN 输出附近初始化，做短链 conditional restoration diffusion / residual correction，而不是从高斯噪声生成完整残差；同时考虑直接 `x0`/residual prediction、低噪声 schedule、identity-preserving loss 和 semantic gate 联训。
3. `selected_risk_rule` 的 final PNG、classifier ensemble 审计、ensemble-risk 二级 veto sweep、receiver-side risk score sweep、raw confidence-gain test-like 复核、frozen risk-rule test-like 复核、test-like classifier-ensemble 审计和 COCO object CLIP clean-correct 辅助诊断已完成；浅层接收端标量规则已经显示出“少 veto 会漏 any-model/GT-like 风险，多安全会过保守”的瓶颈。下一步优先补真正带监督标签的 clean-correct 评估，或把 semantic-risk-aware 约束放进 residual CNN 训练/选择流程；不能把 raw confidence-gain、过保守 CLIP veto、SNR-calibrated scalar CLIP veto、当前 AlexNet-tuned selected rule、保守 ensemble-risk veto 或少 veto risk score 写成最终 M3。
4. 已用 minimal closure report、residual shrink artifact gallery、adaptive residual alpha policy、two-stage alpha policy 和 receiver alpha predictor 固定当前 M0/M1/M2/M3 命名与样例证据。`EXP-S4-008` 的 receiver-edge 质量增益已严格成立；`EXP-S4-011` 进一步给出 source-edge feasibility upper bound，但 perfect edge 未计 rate/channel error，不能升级为 M2/M3。下一步的最高优先级是训练 CBR≈`1/8` 的主图 DeepJSCC 与 CBR≈`1/24` 的独立 edge-JSCC，使总 CBR≈`1/6`，再与当前 CBR 0.17 路线公平比较；在完成前禁止把 oracle `+3.5149 dB` 写成通信增益。
5. 执行节奏上区分“探索实验”和“收敛闭环”：探索阶段允许先用小样本、validation、smoke run 或负结果快速推进；只有方法要进入 M2/M3 命名、跨 split 复核、论文表格或 minimal closure report 时，才必须执行完整最小闭环检查。不能因为 closure 口径保守而停止训练侧或 diffusion 设计侧的大胆尝试。Alpha-head residual refiner 的普通 CE、weighted CE 和 benefit target 三版都低于后验 adaptive alpha、receiver predictor 和 two-stage；benefit-aware tabular predictor 虽在 validation 贴近 adaptive，但 held-out/test-like 仍不够。全量 joint fine-tune 虽显著改善 validation alpha 分类，却损伤 residual restoration；tail-only partial fine-tune 已确认 partial/reconstruction-dominant 方向能恢复质量收益；continuous alpha regression 进一步把 learned training-side policy 推到 `+0.5010/+0.5049/+0.5012` dB，LPIPS 也优于 full-strength fallback，但 ensemble 审计仍有 validation majority new error。下一步若继续训练侧，应围绕连续 amplitude head 做 semantic-risk-aware/listwise utility loss、轻量扩容或 ensemble-aware model selection，而不是回到离散 CE 或全量 unfreeze。
6. 继续收敛正式主语义指标：Imagenette 严格监督 policy-dev 已证明 top-1 fallback 和 80-bit coarse source-description gate 都未通过 accepted-new-error/gate-efficacy 门槛，因此不得解锁 official val，也不再优先扫描 receiver-side 标量阈值。下一轮应先完成 matched-rate lossy source-edge restoration，再用独立 `T_cls` 做新的 policy-dev；只有新方法通过才允许另行冻结并讨论 official val。
7. 后续正式流程一律使用 `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt` 和大导出 `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/` / `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/` 做 residual validation 与 test-like 复核；旧 32 张 export 仅保留用于复现实验 `EXP-S2-002` 到 `EXP-S4-005`。
8. 后续下载大模型或数据仍必须清空代理变量，默认走服务器直连；官方 Hugging Face 直连当前超时，`hf-mirror.com` 服务器直连可用。

### 2026-07-10：完成 Imagenette 严格监督 clean-correct policy-dev 审计

- 数据：官方 `Imagenette2-320`，归档大小/MD5/成员内容均通过；train/val 分别 `9469/3925`，跨 split 精确 SHA-256 与 64-bit 感知哈希近重复均为 0。
- 无泄漏分类器：`G_gate`（随机初始化 MobileNetV3-Small）cal macro top-1 `0.8961`；`T_cls`（随机初始化 ResNet18）`0.9046`；二者只看 `cls_train`，按 `cls_cal` 选优并温度标定，official val 未访问。
- policy-dev 输出：`outputs/analysis/imagenette_supervised_policy_dev/`，真实 WNID 标签、1894 图像、9470 行（5 SNR）。
- M2 edge scheduled 相对 M0 的 clean-correct failure `3.30%→1.28%`，M3 fallback 为 `2.06%`；M3-M2 `+0.7857 pp`，95% CI `[+0.4125,+1.1589] pp`，gate efficacy 失败。
- M3-M0 failure `-1.2375 pp`，95% CI `[-1.6893,-0.7858] pp`；PSNR `+0.7434 dB`，95% CI `[+0.7296,+0.7571]`；LPIPS `-0.0307`；保留 M2 PSNR 增益 `92.87%`。
- accepted-new-error 为 `16/4923` 行、`11/1683` 图像；保守上界 `1.0795%`，高于预注册 `0.5%`。因此 `M3_scratch_gate_fallback` 不升级为 supervised-safe，official val 保持锁定。
- 结论：这是当前最重要的监督闭环边界——COCO 训练出的 edge residual quality gain 在外部真实标签上成立，但 source-model top-1 fallback 的安全泛化不成立；后续应改 gate/训练目标或接受 M2-only 结论，不得把 M3 的净 failure 改善包装成无风险语义控制。

### 2026-07-11：完成精确等总码率 main + decoded-structure 闭环

- 码率契约：reference `c=8`，proposed `c=6 RGB + c=2 structure`，两者 total CBR 均为 `8/48=1/6`；`c=2` Sobel/Laplacian 经独立 AWGN DeepJSCC 传输，不再使用未计码率的 perfect source edge。
- 训练：20k COCO warm-start pilot 的 `c=6`/`c=2` best validation PSNR 为 `30.7497/30.4991 dB`；decoded-structure residual refiner 输出为 `outputs/EXP-S7-002/`。
- COCO 四段：validation/held-out/test-like/fresh-holdout 相对 `c=8` raw PSNR 分别 `+0.3974/+0.3261/+0.4198/+0.3600 dB`，每段 paired bootstrap CI 下界均大于 0，20 个 split×SNR point estimate 全为正；三个冻结 downstream split 合并为 `+0.3772 dB`，95% CI `[+0.3274,+0.4253]`。
- 预注册 Imagenette policy-dev：1,894 图、9,470 行；PSNR `+1.8341 dB`，95% CI `[+1.7742,+1.8949]`；LPIPS `-0.0305`；主 SNR supervised failure `3.3785%→1.2375%`，failure delta CI `[-2.8678,-1.3946] pp`。
- 安全边界：matched raw 仍产生 31/1684 个 new-error image clusters，保守上界 `2.4764% > 0.5%`，所以预注册总判定为 FAIL、official val 继续封存。50 个 new-error rows 中有 9 行（7 图）main/raw 都错，证明继续扫描二选一 fallback 阈值不足以解决问题。
- 显著成果报告：`reports/matched_rate_significant_result_2026-07-11.md`；监督审计：`outputs/analysis/imagenette_matched_rate_policy_dev/REPORT.md`；COCO 比较：`outputs/analysis/s7_matched_rate_system_cross_split_comparison/REPORT.md`。
- 下一方向：把 `c=2` 从纯边缘通道升级为 evaluator-independent、rate-accounted 的语义描述/校验通道，并在 restoration 内部融合；不能把当前净 failure 改善写成“语义无损”。

### 2026-07-11：完成 rate-accounted hybrid semantic sketch 探索

- 实现 `src/cadsd_jscc/semantic_sketch.py`：固定 Rademacher 投影、连续 repetition payload、均匀 latent 位置预留、AWGN 后恢复与结构 decoder 前擦除；总码率始终 `c=6+c=2=c=8`。
- `EXPORT-S8-001` repetition-16 按预注册失败：sketch cosine 全通过，但结构 MSE 增加 `16.41%-29.40%`。输出保留、不覆盖。
- `EXPORT-S8-002` repetition-4 通过：payload 只占 `128/16384=0.78125%` 的 `c=2` latent；1/4/7/13/19 dB cosine 为 `0.9552/0.9772/0.9880/0.9970/0.9992`，结构 MSE 增加 `3.24%-5.84%`。
- 新增 zero-initialized semantic FiLM、冻结 AlexNet KL/投影一致性和反事实 ranking；`EXP-S8-001/002/003` 均保留。最终逐样本 ranking checkpoint 为 epoch 5，SHA-256 `64754d2da87984c07d699b7b961b16e40fe1742436504dbb59a27df7d706f50f`。
- 160-image frozen downstream：S8 raw 相对 reference `c=8` `+0.4691 dB`，95% CI `[+0.4231,+0.5159]`；相对 S7 raw `+0.0919 dB`，CI `[+0.0746,+0.1147]`。
- 因果边界：received 相对 zero `+0.0849 dB`，CI `[+0.0728,+0.0982]`，15/15 split×SNR 全正；received 相对 shuffled 仅 `+0.0072 dB`，CI `[-0.0023,+0.0170]`，严格门槛失败。
- 决定：已证明不到 1% 结构 latent 可稳定承载有用连续 side signal，但 32-D 随机投影尚未形成足够样本特异的 semantic grounding；不把 S8 作为独立方法运行 Imagenette 审计，后续只允许在预注册的主线 M3 controller 中整体审计；official val 继续封存。完整报告：`reports/hybrid_semantic_sketch_result_2026-07-11.md`。

### 2026-07-11：将 semantic sketch 合并回主线 M3

- 主线定义：严格等码率 `c=6 main + c=2 hybrid` + SNR-aware residual refiner + transmitted source checksum + threshold-free alpha controller；不再把 S8 写成独立质量支路。
- 控制器：`alpha={0,.25,.5,.75,1}`，选择 candidate projected-AlexNet sketch 与 received source sketch cosine 最大者，平局取较小 alpha；receiver 不看原图。
- 预注册 Imagenette policy-dev：M3 failure `1.2178%`，hybrid raw `1.2571%`，reference c8 `3.6142%`；M3-reference failure CI `[-3.0642,-1.7482] pp`。
- 风险/收益：new-error rows/clusters 从 raw `41/23` 降到 M3 `29/18`，repairs 从 `161/109` 降到 `151/105`；约减少 22% new-error clusters，但 raw-minus-M3 failure CI `[-0.1768,+0.2554] pp` 跨 0。
- 质量：M3-reference PSNR `+1.4234 dB`，CI `[+1.3693,+1.4799]`；LPIPS `-0.0265`；保留 hybrid raw PSNR gain `74.8%`。
- 严格结论：failure noninferiority、PSNR、LPIPS、gain-retention 通过；controller efficacy 和 new-error `1.5875% <=0.5%` 门槛失败。主线 M3 集成成立但不升级为 supervised-safe，official val 继续封存，不再在 policy-dev 调 alpha/threshold。
- 报告：`reports/mainline_hybrid_semantic_controller_result_2026-07-11.md`；输出：`outputs/analysis/imagenette_hybrid_semantic_controller_policy_dev/`。

### 2026-07-12：复核未落盘的 OpenCode 文献调研材料

- 确认 `reports/literature_and_direction_assessment_2026-07-12.md` 从未写入；OpenCode 会话只保留最终摘要、两个调研子任务完整输出和原始 arXiv 查询结果，无法逐字恢复原定正文。
- 核验后确认核心理论线索可用：Perception-Distortion、Classification-Distortion-Perception、Cohen uncertainty-perception、HalluGen/SHAFE 等均与 semantic reliability 叙事相关。
- 发现原摘要的 novelty 判断过强：`RDPC/JSCM` 已研究 rate-distortion-perception-classification 四元权衡；`RDP-JSCC/DPCT` 已研究可控生成式 JSCC；`TOAST` 已按实时信道条件平衡 reconstruction fidelity 与 classification accuracy，并包含 latent diffusion denoiser。
- 方向据此收紧：不能把“四轴曲线”或“channel-adaptive + classification + diffusion”组件组合作为首次贡献；更可守的核心是 matched-total-rate 下的 refinement-induced accepted new error、tail risk、receiver-visible channel-conditioned risk control 和 explicit failure handling。
- 修正 SGD-JSCC 码率表述：text 成本被忽略且假设完美传输；edge 经独立 DeepJSCC 路径传输并有 BCR，不能笼统说全部 side information 未计 rate。
- 本轮未运行实验、未修改运行方式，也尚未重建缺失报告；详细文献核验已写入 `LITERATURE.md`。

### 2026-07-12：文献结论对当前项目的方向启发

- 当前最有价值的问题不再是“diffusion 是否提升 JSCC 质量”，而是“生成式/恢复式后处理在带来平均质量和净语义收益时，如何约束其相对基线新引入的单样本语义错误”。现有 matched-rate 结果正好展示了该矛盾：平均 failure 与 PSNR 均改善，但 new-error 上界未过预注册安全门槛。
- 论文差异化应从一般 RDPC 多轴权衡和 channel-adaptive classification/quality balancing 中退出，收紧为 `refinement-induced new error + tail risk + matched-rate semantic side information + explicit failure handling`。
- 当前 32-D 随机投影 sketch 的 received-vs-shuffled CI 跨 0，说明它主要提供非零条件能量而非可靠的样本身份；下一方法应改用可解释、可纠错、严格计码率的 class/caption/spatial semantic token 或 checksum，并要求 received > shuffled 的因果门槛。
- 末端 alpha/router 不是充分解：当前 proposed system 的 c6 main 与 refined candidate 存在共同错误，二选一无法满足严格 new-error 门槛。语义 token/checksum 必须进入 restoration/decoder 内部，产生新的受约束候选，而不是只用于事后 veto。
- 下一项优先实验应是预注册的 risk-constrained selective controller：只使用 receiver-visible SNR、decoded main/structure、received semantic payload 及候选内部统计，在独立 development split 上最大化质量收益，同时约束 per-SNR new-error 上置信界和 tail risk；冻结后再做新的监督 audit。
- 若最终方法仍是 residual CNN 而没有可用 diffusion，论文题目和贡献应诚实改为 semantic-risk-controlled generative/restoration JSCC；只有短链、近 M0 初始化、强条件化的 diffusion correction 真正跑通后，才保留 `Diffusion-JSCC` 作为主标题。
- 本轮为方向分析，未运行新实验、未改运行方式、未解锁 official Imagenette validation。

### 2026-07-12：复核 OpenCode 第一版 A/B 方向判断

- 第一版“A 质量增强路线撞车严重、B 必须依靠系统化度量与风险控制”的总体判断，比后续“核心命题基本空白”的绝对说法更稳健。
- `SING` 直接覆盖 DeepJSCC reconstruction 后的 diffusion inverse restoration；Rate-Adaptive Generative SemCom 覆盖 conditional diffusion + rate adaptation；因此单独的质量增强 A 不适合作为主贡献。
- `RD-JSCC` 的短链 residual diffusion 与 channel-conditioned switch 确实是重要组件近邻，但其任务是 MIMO CSI reconstruction，不是自然图像 semantic drift，属于方法组件威胁而非同题抢先。
- 第一版 B 的三个窄点仍有价值：refinement-induced accepted new error、matched-total-rate 公平协议、receiver-side failure handling；但“硬语义度量无人做”“侧信息成本无人计”“channel-adaptive semantic risk 基本空白”均不能在未完成系统全文检索前写成绝对 novelty。
- `TOAST` 已根据实时信道条件动态平衡 reconstruction fidelity 与 classification accuracy 并包含 latent diffusion；因此本项目必须把风险变量限定为 baseline-correct 样本被 refinement 新破坏的事件概率/尾部上界，而不是一般 classification accuracy。
- 若走 benchmark/measurement 路线，只有覆盖多种 generative/refinement 方法、多个信道/数据集、统一风险协议并公开可复现资产，才可能形成独立主贡献；仅对当前单一系统做分析不足以支持高档 benchmark 叙事。
- 本轮未运行实验、未修改运行方式、未重建缺失报告。

### 2026-07-12：完成项目深审计、补充文献调研与未来路线规划

- 已形成正式报告 `reports/literature_and_direction_assessment_2026-07-12.md`，覆盖现有证据、技术归因缺口、近期文献、候选方向、分阶段实验和 go/no-go 条件。
- 当前最强资产确认是完整的 matched-rate + supervised clean-correct + refinement-induced new-error/UCB + preregistered failure 证据链，不是某个单独 residual/diffusion 模块。
- 新发现的最高优先级缺口：`c6+c2 structure+refiner` 虽与 `c8` 严格等信道使用量，但尚缺 `c8+同等 refiner`、双分支无结构语义、参数/训练预算/推理开销匹配对照；现有结果证明完整系统有效，尚不能把全部收益归因于 structure/semantic representation。
- 统计缺口：当前 image-cluster bootstrap 主要覆盖跨图像/SNR 相关性，正式版本还需多个 channel seeds，并区分单次传输风险、image susceptibility 和 tail risk。
- Semantic sketch 缺口：当前 received-vs-shuffled PSNR CI 跨 0，payload 为 post-hoc latent overwrite，且因果 gate 未直接要求 hard semantic 改善；下一版应联合学习符号/功率分配，并对质量和 new-error 同时做 received/shuffled/zero 审计。
- 文献边界进一步核验：JSCGC 正确 arXiv ID 为 `2601.12808`；MTGC、ADDPS、SBGSC、Hallucination Index、Selective Classification 和 Conformal Risk Control 已补入 `LITERATURE.md`。
- 推荐路线：P0 先补公平因果基线；P1 最多筛三种 rate-accounted semantic anchor；P2 做 selective risk controller；P3 full-scale/multi-seed/final audit；P4 再决定短链 conditional diffusion；P5 方法冻结后补 Rayleigh。
- 本轮未运行实验、未改运行方式、未访问 official Imagenette validation。

### 2026-07-12：完成 matched-rate short-chain diffusion 正式 pilot

- 新增预注册、配置、实现和单元测试：`reports/short_chain_residual_shift_diffusion_preregistration_2026-07-12.md`、`configs/s10_short_chain_residual_shift_diffusion_pilot.yaml`、`scripts/s10_short_chain_residual_shift_diffusion.py`、`tests/test_short_chain_residual_shift_diffusion.py`。
- `EXP-S10-001` 使用冻结 `c=6 main + c=2 decoded structure` 和 `EXP-S7-002` residual anchor；采用 pixel-domain residual-shift bridge，从 anchor 起步并用 6 个 deterministic steps 回到 clean endpoint，不使用 SD/VAE/prompt。
- 正式 160/64 split、五 SNR 结果：相对 anchor mean ΔPSNR `-0.1548 dB`，mean ΔLPIPS `-0.000195`；LPIPS 在 5/5 SNR 均微幅改善。相比旧 `EXP-S4-007` 的数 dB 崩塌，近 anchor 短链设计显著更稳定。
- 预注册语义 gate 失败：raw candidate 新增 12 个 AlexNet pseudo error，只修复 7 个；因此本版本结论为 NEGATIVE，不晋级为主线 M2/M3。
- 方向判断：不放弃 diffusion；保留“matched-rate anchor + short-chain conditional diffusion”作为有上限潜力的后端，但下一版必须先补更大训练集、感知/语义风险目标和与同 anchor deterministic refiner 的公平对照，不能只扫描 steps/seed。
- 输出：`outputs/EXP-S10-001/`；正式报告：`outputs/EXP-S10-001/REPORT.md`。未下载任何模型/数据，未访问 official Imagenette validation。

### 2026-07-12：完成 P0 `c8 + same refiner` 公平对照

- 预注册并完成 B1 `EXP-S11-001`：给裸 `c=8` reference 配置与 B3 `EXP-S7-002` 完全匹配的 `64×6`、60 epoch receiver-only residual refiner；双方 seed、160/64 split、loss、crop、batch 和 gates 相同，参数量均为 448,387，延迟均约 2.5 ms/image。
- B1 raw 相对 bare B0 平均 `+1.0192 dB`，明显高于 B3 相对 B0 的 `+0.3974 dB`。
- `ANALYSIS-S11-001` 从 PNG 重算并按 64 个 image clusters 做 10,000 次 bootstrap：B3 raw − B1 raw 为 `-0.6217 dB`，95% CI `[-0.6654,-0.5839]`，5/5 SNR 均为负；B3 LPIPS 也比 B1 差 `+0.00664`。
- 伪语义诊断同样未通过 gate：B1 raw new-error/repair 为 `31/45`，B3 为 `37/57`。三个预注册条件全部失败。
- 方向调整：当前 `c6+c2 decoded structure` 不再作为已被因果证明的主要增益；不继续用 post-hoc structure/sketch 调参救该叙事。后续 diffusion 改用更强的 B1 作为 deterministic anchor，核心贡献继续围绕 refinement-induced new error 和 semantic-risk control。
- 报告：`reports/p0_c8_same_refiner_result_2026-07-12.md`；未下载数据/权重，未访问 Imagenette policy-dev 或 official validation。

### 2026-07-12：完成 B1-anchored semantic-preserving diffusion v2

- 将 `scripts/s10_short_chain_residual_shift_diffusion.py` 泛化为兼容 decoded-structure 和 receiver-anchor structural maps 两种条件模式；旧 S10 dry-run 回归通过。
- 冻结并运行 `EXP-S12-001`：formal `c8` + B1 anchor、receiver-only Sobel/Laplacian、6-step residual-shift diffusion，并加入 edge L1 与独立于 AlexNet 诊断器的本地 ResNet18 target KL。
- 正式五 SNR 结果：mean raw ΔPSNR `-0.0775 dB`、mean raw ΔLPIPS `-0.000652`，5/5 SNR LPIPS 改善；相比 S10 的 `-0.1548/-0.000195`，质量/感知 tradeoff 明显改善。
- semantic-risk gate 仍失败：raw new-error/repair `8/4`；top-1 fallback mean ΔPSNR/ΔLPIPS `-0.0747 dB/-0.000613`，但只继承 AlexNet anchor prediction，不能当独立安全证据。
- best checkpoint 为 epoch 2，后续 train loss 继续下降而 eval PSNR 最差回吐超过 1 dB，确认 160-image diffusion 训练强过拟合。按预注册不再调该小数据 bridge 的 weights/steps/seed。
- 第一次 smoke 的训练/推理成功但 `--skip-lpips` 报告遇到 `None` 格式化错误，失败目录保留；修复可选 LPIPS 报告后在新目录重跑通过。正式 run 未受影响。
- 下一步若保留 diffusion，转为 COCO train2017-scale anchor/diffusion dataset + 独立 validation + direct new-error risk calibration；报告：`reports/b1_anchored_diffusion_result_2026-07-12.md`。未下载、未访问 Imagenette policy-dev/official val。

### 2026-07-13：完成 COCO train2017 scale-up cache 与 B1 anchor

- 新增确定性 scale-up exporter：按 `SHA256(seed:path)` 从 118,287 张 local train2017 固定 10k train + 1k validation，并逐 SHA 排除 local val2017 重复。
- `EXPORT-S13-001` 完整输出 11k original + 55k five-SNR c8 reconstructions，manifest/per-sample 唯一性、目录计数和 manifest hash 全通过；cache 约 6.9GB，无下载。
- `EXP-S13-001` 用 50k pairs/epoch 训练 receiver-only B1 10 epochs；best epoch 9 validation PSNR `32.5588 dB`，没有 S12 的早期强过拟合。
- 正式 1k×5 validation：mean raw ΔPSNR `+1.3632 dB`、mean raw ΔLPIPS `-0.03272`，5/5 SNR 均改善；new-error/repair `339/951`，全部预注册 anchor gate 通过。
- top-1 fallback mean ΔPSNR/ΔLPIPS `+0.8384 dB/-0.01529`；raw 虽净修复 612 rows，仍保留 339 个 individual new errors，不能写成无风险。
- 冻结 checkpoint SHA-256 `80133f9d9649c1a5d9514cf2b4f0d04802b6ebe03cc970bfcec86eddfd165562`，作为 scale-up diffusion anchor。未访问 Imagenette policy-dev/official val。
- 用户看到的中断只发生在终端输出回传；后台训练和完整评估已正常结束，不需要续跑或覆盖目录。

### 2026-07-13：完成 train2017-scale B1-anchored diffusion

- `EXP-S14-001` 物化 55k frozen B1 anchor cache，并用 10k/1k×5-SNR 训练/验证原样 S12 6-step residual-shift diffusion。
- best epoch 2；mean raw ΔPSNR `-0.0736 dB`、ΔLPIPS `+0.000081`，LPIPS 仅 2/5 SNR 改善，质量/感知 gate 失败。
- raw pseudo new-error/repair `63/76`，规模扩大后首次通过 net incremental-risk gate；但不能用净风险改善掩盖感知无收益。
- 正式结论 NEGATIVE，停止该 bridge 家族的 validation 超参扫描。若以后继续 diffusion，必须换 posterior/data-consistency 设计和新 development protocol。

### 2026-07-13：打通 received-latent posterior consistency 接口

- 审计第三方 DeepJSCC 后确认可拆分 `encoder→channel→received latent→decoder`；新增五个 adapter 接口和 3 项单元测试。
- 实际 formal checkpoint、7 dB、256×256 smoke：拆分路径与原 forward 最大误差 `1.788e-7`，received latent shape `(B,16,64,64)`，tx/rx power `1.0000/1.0973`。
- B0 candidate 的 normalized received-latent loss `0.06218`，对图像梯度有限且非零，证明可在 posterior sampler 内做真实 measurement-consistency correction。
- 该结果只证明工程可行性，不是新实验正结果；下一步需保存与 cache 对齐的 received latent，并冻结新的 proximal sampler protocol，不能继续调 S14。
- 报告：`reports/received_latent_posterior_feasibility_2026-07-13.md`。

### 2026-07-13：received-latent posterior correction 取得阶段性正结果

- 预注册 `ANALYSIS-PC-001`，使用 S13/S14 从未访问的 train2017 SHA-rank `11000--11063`（64 图×5 SNR），冻结 S13 B1、S14 diffusion、三步 correction 和 `0.001` normalized step，不在该 split 调参。
- 5/5 SNR 的 received-latent consistency loss 均下降；总体从 `0.10363` 降到 `0.08275`，相对约 `-20.1%`。
- 相对未约束 S14 raw，mean PSNR `+0.2124 dB`、LPIPS `-0.00991`，且 PSNR/LPIPS 均为 5/5 SNR 同向改善；这排除了本轮只是用失真换一致性的解释。
- B1-anchor-relative AlexNet pseudo new error 保持 `5→5`，repair 从 `2→17`；全部 feasibility/promotion gates 通过。
- 阶段性结论：保留 diffusion，但主线从无约束 residual-shift bridge 改为 **received-latent posterior/data-consistency diffusion**。下一步在新 frozen split 上训练/验证内生 consistency sampler，并补 classifier ensemble 与监督语义审计；本 pilot 不能写成最终安全证据。
- 验证：`python3 -m unittest discover -s tests -p 'test_*.py' -v` 共 45 项全部通过；`py_compile`、PC-001 dry-run 和 `git diff --check` 通过。
- 报告：`reports/posterior_consistency_pilot_result_2026-07-13.md`。

### 2026-07-13：独立复现 posterior restoration，并定位 cross-model semantic 瓶颈

- `ANALYSIS-PC-002` 在全新 256 图×5 SNR 上冻结复现 PC-001：latent loss 相对下降约 `20.4%`，相对 S14 raw PSNR `+0.2125 dB`、LPIPS `-0.01078`，均为 5/5 SNR 同向；与 PC-001 的 `+0.2124/-0.00991` 高度一致。
- PC-002 完整 gate 仍失败：ensemble-majority new error `0→2`，三分类器 individual new error 均增加。结论是 measurement/posterior consistency 稳定改善 restoration，但本身不保证 semantic consistency。
- `ANALYSIS-PC-003` 在再次全新的 256 图×5 SNR 上加入冻结 receiver-only AlexNet agreement fallback；coverage `87.66%`，final 仍保留 PSNR `+0.2062 dB`、LPIPS `-0.00910`。
- PC-003 将 uncontrolled posterior majority new error `4→1`，但 raw 为 `0`，ResNet18/MobileNetV3-Small new error 仍增加，故不晋级安全 M3。单语义模型 fallback 的跨模型瓶颈被独立确认。
- 阶段成果：diffusion 不退出；其可复现价值已收紧为 received-latent posterior restoration。下一步必须用分离的 controller-development 与 holdout semantic audit，或监督 COCO object/Imagenette 口径，不能继续在 PC-002/003 调 correction 参数。
- 验证：46 项标准库 `unittest` 全通过；PC-002/003 dry-run、`py_compile`、`git diff --check` 通过。无下载、未访问 Imagenette official validation。

### 2026-07-13：posterior-consistency 记录编号收敛

- 将此前误写成新阶段的 `S15/S16/S17` 统一更名为同一阶段5 validation study 下的 `PC-001/002/003`。
- canonical config/entry 改为 `configs/pc*.yaml` 与 `scripts/pc_*.py`；旧 `outputs/analysis/s15*--s17*` 仅作为不可覆盖的 legacy artifact path 保留。
- 后续 posterior-consistency 工作继续使用 `ANALYSIS-PC-*`，不再增加阶段号；只有 `MILESTONES.md` 定义的真实阶段迁移才使用新的 S 编号。

### 2026-07-13：PC-CTRL holdout audit 否定 classifier-consensus 堆叠

- 在新 256 图×5 SNR 上冻结 AlexNet+ResNet18 consensus controller，MobileNetV3-Small 完全不参与决策或调参。
- posterior restoration 再次稳定复现：PSNR `+0.2119 dB`、LPIPS `-0.01061`、latent loss `0.10458→0.08347`。
- controlled final coverage `78.05%`，相对 S14 raw 仍保留 PSNR `+0.1927 dB`、LPIPS `-0.00791`；三模型 majority new error 为 0。
- 但 held-out MobileNet new error `12→34`，故完整判定 NEGATIVE。controller 内两模型的 new error 为 0 是规则构造结果，不能作为跨模型安全证据。
- 停止继续增加 classifier-consensus 规则。下一步转向独立 supervision/calibration 的 risk model，或使用 COCO object / Imagenette 监督语义口径；posterior correction 强度继续冻结。
- 报告：`reports/posterior_consensus_controller_holdout_result_2026-07-13.md`。

### 2026-07-13：PC 独立标注与真实类别监督审计

- PC-GT 在新 512 图×5 SNR 上引入 COCO dominant-object 标注和独立本地 OpenCLIP。195 张 clean-correct 图中，final failure `36→32`，但 object new error `2→4`，严格 gate 失败；确认 semantic risk 不是 ImageNet pseudo-label 模型间分歧。
- PC-SUP 使用既有 Imagenette policy-dev、真实 WNID 和 scratch calibrated `T_cls`；1894 图中 1697 张进入 clean-correct，official validation 保持封存。
- supervised primary `[1,4,7] dB`：raw/posterior/final failure `69/56/62`，new error `4/4/4`；final 相对 raw mean PSNR `+0.2543 dB`、LPIPS `-0.00531`。
- aggregate new error 不增且 failure 净改善，但 7 dB 出现 final/raw new error `1/0`，逐 SNR gate 失败。因此结果是明确的 supervised partial success，不晋级 final-safe M3、不解封 official val。
- 方法决策收敛：保留 frozen posterior restoration，终止 classifier-consensus 规则扩张；下一步只能在独立 development supervision 上训练/校准 risk controller，并保留未使用的审计集。
- 报告：`reports/posterior_coco_object_clip_audit_result_2026-07-13.md`、`reports/posterior_imagenette_supervised_audit_result_2026-07-13.md`。

### 2026-07-13：task-matched scratch gate 明显改善，但严格 tail gate 未过

- 完成 `ANALYSIS-PC-RISK-001`：仅把 PC-SUP 的 ImageNet consensus controller 换成 2026-07-10 已冻结的 scratch MobileNetV3-Small `G_gate`；scratch ResNet18 `T_cls` 继续只作独立 outcome audit，official validation 未访问。
- checkpoint loader 现在 fail-closed 校验 scratch/random-init、角色、架构、质量门槛、`cls_train/cls_cal`、policy-dev separation、official-val lock 和类别顺序；旧 ImageNet consensus 路径保持兼容。
- 与 PC-SUP 的 9470 行 raw/posterior 配对字段逐值比较为 0 mismatch。scratch gate 在 clean rows 上接受 `8428/8485`（`99.33%`），final mean PSNR/LPIPS 相对 S14 raw 为 `+0.26394 dB/-0.005966`。
- primary `[1,4,7] dB` failure 为 raw/posterior/final `69/56/57`，优于旧 consensus final 的 `62`；new error 为 `4/4/3`，优于旧 final 的 `4`。
- 严格总判定仍为 NEGATIVE：7 dB final/raw new error `1/0`，唯一 per-SNR gate 失败。该行 `G_gate` 对 anchor/posterior 预测相同，证明简单 task-matched top-1 agreement 仍覆盖不到 evaluator-disagreement tail。
- 阶段成果：frozen posterior-consistent diffusion + scratch gate 已成为当前最强 supervised development candidate；停止在已查看的 policy-dev 上扫 top-1/threshold。下一步必须先冻结真正独立的 controller-development/final-audit protocol，再讨论 official validation；当前结果不称为 semantic-safe。
- 验证：46 项 `unittest`、`py_compile`、dry-run、checkpoint contract load 和逐行配对检查通过；未下载、未联网。
- 报告：`reports/posterior_imagenette_scratch_gate_result_2026-07-13.md`。

### 2026-07-13：multi-seed 复现确认 restoration 稳定、tail risk 真实存在

- 完成 `ANALYSIS-PC-RISK-REP-001`：冻结 PC-RISK-001 的所有模型、三步 posterior correction、`0.001` step 和 scratch `G_gate`，只换三个全新 AWGN seeds `[20260722,20260723,20260724]`；共 `28,410` 唯一行，official validation 未访问。
- 15/15 seed×SNR received-latent consistency 下降；mean final-minus-raw PSNR/LPIPS `+0.26334 dB/-0.005937`，每个 seed 质量/感知均同向。
- primary failure raw/posterior/final `196/164/163`；三个 seed 分别 `63→56`、`67→53`、`66→54`。posterior diffusion 的净监督收益不是单 seed 偶然。
- primary new-error rows `13/15/14`，raw/final image clusters `10/11`。final `11/1691=0.6505%` 的单侧 95% Clopper-Pearson upper `1.0744% > 0.5%`。
- 1 dB new error `8→10`，seed 20260722 `5→7`；旧 event `n03425413/n03425413_3069.JPEG` 在新 seed 再现。四个 new-error/tail gates 失败，完整 verdict NEGATIVE。
- 阶段判断进一步收敛：保留 received-latent posterior diffusion 作为稳定 restoration mechanism；淘汰 scratch top-1 agreement 作为足够的最终风险控制。不得换 seed、按净 repair 抵消 new error、或继续在 policy-dev 扫 threshold。
- official validation 继续封存。下一控制器必须把这些多 seed 结果显式标为 development data，并在未使用的图像 population 上一次性审计。
- 验证：46 项 `unittest`、新旧 config dry-run、`py_compile`、`git diff --check`、28,410 row/key/clean-membership/artifact hash 检查通过；未联网、未下载。
- 报告：`reports/posterior_imagenette_scratch_gate_multiseed_result_2026-07-13.md`。

### 2026-07-14：连续 receiver-risk controller 完成开发闭环，但新 seed 审计失败

- 完成 `TRAIN-PC-AUX-001`：scratch EfficientNet-B0 `G_aux` 只用 `cls_train/cls_cal` 训练、选点与校准；best epoch 64，cal macro top-1 `0.90270`，checkpoint SHA-256 `8e074be6ec854edbc144d95d9fe5cd7d098c61bca853915108952acfa094b455`；policy-dev 未参与训练选择，official val 未访问。
- 完成 `ANALYSIS-PC-RISK-FEAT-001`：在已暴露三 seed policy-dev 上生成 `28,410` 行、43 维 `receiver_risk_v1`；仅含 SNR、latent consistency、图像扰动、`G_gate/G_aux` confidence/entropy/margin/JS/retention/agreement。`teacher_*` 标签物理分列且禁止进入 controller；与 PC-RISK-REP 的普通审计字段在 `1e-9` 内逐行复现。
- 完成 `ANALYSIS-PC-RISK-CTRL-DEV-001`：冻结四个 JS percentile + 两个 sign-reversed posterior-confidence percentile 的算术均值，阈值 `0.85372653`。开发结果 new-error `15→3`、cluster upper `0.4579%`，final failure `180<raw 196`，PSNR/LPIPS `+0.23834/-0.004799`，保留 posterior PSNR gain `89.83%`；只记为 development pass。
- 在任何 seed-20260725 行生成前冻结 extraction config SHA、controller/CDF SHA、阈值和成功判据；新 seed 共 `9,470` 行，键唯一、score finite、official val 未访问。
- `ANALYSIS-PC-RISK-SEED-AUDIT-001` 正式结果 NEGATIVE：posterior 仍稳定改善质量 `+0.26535 dB/-0.006064 LPIPS` 且 primary failure `50→45`；冻结 risk final 保留 `+0.23827/-0.004800`，但两个 1 dB posterior new-error 均未拦截，并误拒 11 个 posterior repair，final failure `56>raw 50`、new-error `2>raw 0`。
- 失败不是轻微阈值偏差：其中一例 `G_gate/G_aux` posterior confidence 为 `0.973/0.963`、JS 接近 0；事后若要靠同一分数拦住两例，需要拒绝约 `38.3%` 参考行。确认纯 receiver ensemble uncertainty 存在高置信共享盲点。
- 决策：保留 posterior-consistent diffusion restoration，淘汰本连续 receiver-risk controller 的晋级资格；不换 seed、不补阈值、不继续堆接收端分类器。下一方法应使用任务相关、可纠错、严格计码率的 sender semantic checksum/token，并在新的 labeled development population 上训练/校准；现有 random sketch 只能复用通信框架，不能直接当充分语义证据。
- 报告：`reports/posterior_receiver_risk_controller_stage_result_2026-07-14.md`。验证：51 项 unittest、py_compile、dry-run、row/key/finite/hash 与 `git diff --check`；无联网/下载，official validation 继续封存。

### 2026-07-14：固定码率 sender semantic channel 做通，单模型 semantic veto 新 seed 仍失败

- `ANALYSIS-PC-RISK-FAIL-001` 精确重放 seed `20260725` 的 2 个 receiver-risk 漏检和 11 个误拒 repair：`G_gate/G_aux` top-1 checksum 均抓不到两个漏检，但 `G_aux` full-probability JS/CE 增量在该 2-vs-11 事后集合上的 pairwise AUC 为 1.0，促成 source-grounded sender score 开发。
- `ANALYSIS-PC-SENDER-DEV-001` 先验证额外 80-bit 无噪声 source probability 的自然零阈值 veto：primary failure `50→45`、new-error `0→0`（相对旧 unpunctured raw），mean final-minus-raw `+0.19569 dB/-0.004130 LPIPS`。它只证明可达性，不具 matched-rate 资格。
- 新增固定率链：在原 `c=8` 的 65,536 个实符号中保留 160 个，sender payload 与图像 latent 共同经过一次 AWGN；receiver 恢复后擦除载荷位置，posterior consistency 只使用剩余 65,376 个位置。总 CBR 仍为 `1/6`，payload 占 `0.24414%`。
- 模拟 10D probability×R16 的 `ANALYSIS-PC-SENDER-RATE-DEV-001` 正式 verdict `NEGATIVE`：top-1 恢复 `99.84%`、质量 `+0.03890 dB/-0.003242 LPIPS`、primary failure `50→49`，但 1 dB final/raw new-error `2>1`。完美载荷反事实显示实际/完美决策有 `40.50%` 翻转，说明连续概率 top-1 正确不等于零阈值差分稳定。
- 固定 `UInt4+BPSK×4`（40 bits/160 symbols）后，开发 seed 的 BER `0.01452%`、整向量无误率 `99.43%`、决策翻转 `0.0739%`；primary reference-raw / final failure `50/45`，in-budget raw/final new-error `4/2`，mean quality `+0.02665 dB/-0.003165 LPIPS`，全部开发门槛通过。
- 方法原样冻结到新 channel seed `20260726` 后，编码与质量继续迁移：BER `0.01716%`、整向量无误率 `99.32%`、quality `+0.02643 dB/-0.003182 LPIPS`、reference-raw/final failure `58/55`。但 in-budget raw/final new-error `3→5`，总量与逐 SNR gate 失败，正式 verdict `NEGATIVE`；5 行 payload 全部正确，完美载荷反事实仍为 5，确认瓶颈已从通信层收敛到单一 `G_aux`/JS 语义盲区。
- 决策：保留 posterior-consistent diffusion 和 `UInt4+BPSK×4` matched-rate communication layer；淘汰单一 `G_aux` zero-veto 的最终 M3 资格。不得在 seed `20260726` 调 threshold/bit-width/repetitions。下一候选若继续，应只改 semantic decision layer，例如 source-JS 与独立 `G_gate` top-1 的自然交集，并把 seed 20260726 降格为 development，再用全新 seed 审计。
- 完整中文报告：`reports/posterior_sender_inbudget_semantic_payload_stage_result_2026-07-14.md`。无联网/下载，official validation 继续封存。
- 验证：58 项标准库 `unittest`、关键脚本 `py_compile`、三份 strict-rate config dry-run、reference SHA/键完整性和 `git diff --check` 全部通过。
- M2 对比口径复核：当前 UInt4+BPSK×4 M3 相对完整 `c=8` S14 raw（正式 M2）在 seed `20260725/20260726` 的五 SNR 平均 PSNR 分别 `+0.02665/+0.02643 dB`、LPIPS `-0.003165/-0.003182`，primary final failure `50→45/58→55`；但 new-error 为 `0→2/5→5`，且新 seed 的 4 dB failure/new-error `18→19/2→3`。因此只能表述为 aggregate tradeoff 优于或接近 M2，尚不能声明跨 seed、逐 SNR、语义尾部全面强于 M2。

### 2026-07-14：与外部 diffusion / generative JSCC 方法的定位复核（历史状态；已被 S20/S30 更新）

- 当时复核 SGD-JSCC、DiffJSCC、SING、DiT-JSCC、TOAST 与 JSCGC 后，项目尚未在本仓库复现任一外部方法，因此不能声明领先；该句只记录 7 月 14 日的决策背景。后续 S20 已完成 SGD-JSCC、S30 已完成 DiffJSCC 官方权重对比，当前结论以文档顶部 S30 为准。
- 当前可成立的差异化不在“更强生成器”，而在更严格的 reliability protocol：语义载荷计入固定总 CBR 并真实过同一 AWGN；显式统计 refinement-induced hard new error、逐 SNR failure、图像簇尾部置信上界与冻结 seed 负结果；这比只报告 PSNR/LPIPS/FID/CLIP 或分类均值更直接回答 semantic drift 风险，但不能替代同条件性能比较。
- 逐篇判断：SGD-JSCC/DiT-JSCC/TOAST 的语义引导、生成骨干或信道自适应更完整；DiffJSCC/SING 的感知生成或逆问题恢复能力更强；JSCGC 的理论刻画更完整。项目的潜在贡献应收紧为“matched-total-rate sender semantic checksum + received-latent posterior diffusion + refinement-induced semantic tail-risk audit”，而非泛称 channel-adaptive diffusion JSCC 或性能 SOTA。
- 外部对比的最小下一步：先在相同 COCO256/Imagenette、AWGN、总 CBR、SNR、随机种子和统一 evaluator 下复现最接近的 SGD-JSCC 或 SING；联合报告 PSNR、LPIPS、FID/感知指标、监督任务正确率、failure/new-error、语义载荷开销与推理成本。在完成此前，不把跨论文不可比数值写成领先结论。

### 2026-07-14：外部复现与主方法开发的次序决策

- 不采用“等结果足够好后才开始对比”，也不立即铺开多篇完整复现；采用主线约 `80%`、外部基线约 `20%` 的非对称并行策略。理由是当前 `UInt4+BPSK×4` 固定总码率通信层已稳定，但 semantic decision layer 在冻结 seed 上尚未通过，过早围绕未冻结 M3 大规模适配外部代码会重复返工。
- 现在只做低成本外部对比准备：冻结统一的 AWGN/总 CBR/SNR/evaluator/seed/指标合同，并在小 split 上接入一个最邻近方法。若没有作者代码，只能明确标成 `SING-style/DDNM-style` 或 `SGD-JSCC-style` mechanism baseline，不能写成论文的精确复现。
- 完整外部复现的启动条件不是“结果看起来好”，而是 M3 接口冻结且至少在独立新 seed 上满足：aggregate 与逐 SNR new-error 不劣于 M2、final failure 低于 M2、仍保留非零且稳定的 LPIPS/质量收益。达到后优先做最接近贡献边界的 SGD-JSCC，再做逆问题路线 SING；DiT-JSCC 保持为 AWGN 最小闭环后的重型扩展，不提前改成新主线。
- 在上述条件前，主开发只允许收敛 semantic decision layer，不再改变固定码率载荷和 posterior correction 主体；外部小基线用于校准效果量和暴露协议问题，不参与当前 M3 阈值选择。

### 2026-07-14：cross-model triplet sender checksum 旧口径阶段结果（后续更正为 NEGATIVE）

- 先验证 `G_aux source-JS ∩ G_gate(anchor/posterior) top-1`：在 seed `20260726` 额外 veto `0.623%` 行却没有减少 `5` 个 primary new-error，正式 development verdict 为 NEGATIVE；该负结果确认单纯 receiver guard 仍会与 sender model 共享盲点。
- 从该失败的 source/anchor prediction mismatch 导出零额外码率的自然 triplet：`source-JS<=0`、`argmax(q_recovered)=argmax(G_gate(anchor))`、`argmax(G_gate(anchor))=argmax(G_gate(posterior))`。它只复用已有 40-bit `G_aux(source)` payload 与独立 receiver `G_gate`，不使用 `T_cls`、标签、阈值或 SNR 例外。
- 两个已暴露 development seed 均通过：20260725 primary M2 failure/final `50→48`、raw/final new `4→1`；20260726 为 `58→55`、`3→0`。均保留约 `+0.011 dB/-0.00255 LPIPS`，coverage 约 45–46%。两者只能用于冻结规则。
- 新 channel seed `20260727` 先生成 unpunctured reference，再仅写入 CSV SHA 后一次性 strict-rate audit。9470 行、1697 clean 图、五 SNR 的所有 gate 通过：M2 failure/final `61→60`，in-budget raw/final new `2→0`，new-error cluster `0/1690`、upper95 `0.1771%`，mean final-minus-M2 `+0.01158 dB/-0.002566 LPIPS`，payload BER `0.01610%`、40-bit vector exact `99.377%`、coverage `46.01%`。
- **后续统计更正：**上一条的 `0/1690` 是 anchor/in-budget-raw-relative endpoint，不能替代相对 paired unpunctured M2 的系统 new error。正确系统端点为 new/repair clusters `7/8`、upper95 `0.7766%>0.5%`；1 dB M2/final failure `32→34`。因此原“全部 gate 通过/当前最强 M3”结论作废，严格 verdict 改为 NEGATIVE；official-val 协议在产生任何 outcome 前取消。
- 阶段结论：该 UInt4 版本只证明固定码率 payload 与 diffusion quality tradeoff 可运行，不能证明 semantic-tail safe。后续以 UInt2、预留感知 B1 和严格 system endpoint 继续开发。
- 完整中文报告：`reports/posterior_sender_crossmodel_triplet_stage_result_2026-07-14.md`；official Imagenette validation 仍未访问，无下载、无联网。
### 2026-07-14：UInt2 预留感知链路取得稳定质量增益，但独立信道语义尾部仍未过门槛

- sender payload 已从 UInt4 收缩为 UInt2：10 类×2 bit、BPSK×4，共 80 个实符号；总 65536 实符号和 CBR `1/6` 不变。旧 S13 B1 的 seed20260727 full policy-dev 结果相对 paired M2 为 PSNR `+0.071845 dB`（95% CI `[+0.066098,+0.077531]`）、LPIPS `-0.002577`，五 SNR PSNR 全正；但 system new/repair `7/8`、cluster upper95 `0.7766%`，严格 verdict NEGATIVE。
- 新增 reservation-aware COCO cache（2000 train/200 val×5 SNR）与 B1 微调。新旧 B1 在完全相同 reserved inputs 的配对比较为 PSNR `+0.102782 dB`，image-cluster 95% CI `[+0.093375,+0.114000]`；5/5 SNR 均显著为正，LPIPS `-0.001682`，checkpoint SHA-256 `57aa5283...495`。
- 新 B1 接回冻结 S14 diffusion/posterior 后，seed20260727 final−M2 PSNR/LPIPS `+0.073967/-0.002633`；但 M2 自身改善为 59 failures，旧二路 final 为 60，system new/repair `7/6`。7 个新增中 6 个是拒绝后退到错误 anchor，且 raw/posterior 都正确。
- 冻结三路路由：accept→posterior；reject 且 recovered-source/anchor mismatch→raw；其余 reject→anchor。seed20260727 离线为 M2/final `59→56`、new/repair `2/5`、五 SNR PSNR 全正，但 failure CI 上界仍跨 0，只能作 development selection。
- 在读取结果前预注册新 AWGN seed20260728；冻结规则的 paired M2/final failure `62→62`，system new/repair rows `4/4`，new-error cluster upper95 `0.5408%>0.5%`，且 1 dB `34→36`，故 verdict NEGATIVE。质量收益完整复现：PSNR `+0.065798 dB`（95% CI `[+0.060055,+0.071703]`），LPIPS `-0.002540`（95% CI `[-0.002805,-0.002286]`），五 SNR PSNR 均正。
- 阶段判断：**保留 diffusion，冻结 UInt2/预留感知 B1/S14/三步 posterior 物理链；停止在 seed20260728 补布尔规则。** 下一开发对象必须是用 `cls_train/cls_cal` 分离监督训练的 anchor/raw/posterior 三路 semantic decision layer，并换 image population 一次性审计；official val 继续封存。
- 完整中文报告：`reports/uint2_reservation_aware_diffusion_stage_result_2026-07-14.md`。本轮无联网/下载，大任务清空 proxy；`py_compile` 通过，标准库全套 76 项 `unittest` 全部通过。

### 2026-07-14：外部方法对比正式排期，完成 SGD-JSCC 源码与公平性审计

- 冻结外部对比顺序：SGD-JSCC 作者代码 → SING-Zero-style 同底座机制对照 → DiffJSCC 作者代码 → DiT-JSCC watch-only；不再等本方法“看起来足够好”才开始，也不同时铺开多套大型生成系统。
- 清空全部代理变量后浅克隆作者 `MauroZMJ/SGDJSCC`，固定 commit `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`；源码约 4.1 MiB 落盘，仓库未发现 LICENSE/COPYING，保持只读并只从本项目侧接 adapter。
- 作者 checkpoint bundle 约 2.931 GB，推理还依赖 BLIP2/CLIP；本轮未下载任何权重。README 的 batch preprocessing 与 training guideline 仍未发布，配置含作者绝对路径。
- 源码核验确认公平性阻塞：main latent、独立 edge-JSCC 支路和假设完美/免费传输的 text caption 必须统一计入总符号账本；在 exact active symbols 和 text payload 未闭合前，作者原生结果不得直接与本项目 CBR `1/6` 排名。
- 新增 `configs/external_baseline_comparison_contract.yaml`：分离 author-native/common-contract 两张表，固定 COCO-256、AWGN、SNR `[1,4,7,13,19]`、65,536 总实符号、同图同噪声以及 PSNR/LPIPS + supervised failure/new-error/tail-risk + runtime/VRAM 指标。
- 新增 fail-closed checker 和 4 项单测；no-download dry-run `PASS`，确认 SGD commit 匹配、checkpoint 目录为空、official Imagenette validation 未访问、结果声明仍禁止。
- 详细中文报告：`reports/external_method_comparison_schedule_2026-07-14.md`。本轮是协议/源码审计，不是实验，因此未写 `EXPERIMENTS.md`；下一里程碑为 EXT1 no-download adapter 与 symbol counter。

### 2026-07-14：SGD-JSCC 作者完整链单图 smoke 跑通，并闭合实测 rate hooks

- 清空全部代理变量后完成外部资产直连下载和校验：作者 4 checkpoints `2,930,865,634` bytes；BLIP2 仅取两个 safetensors 分片 `15,496,030,352` bytes；OpenAI CLIP ViT-L/14 `932,768,134` bytes；scheduler 固定 SD-v1-4 commit。所有 checkpoint/BLIP2/CLIP 精确尺寸与 SHA-256 匹配。
- 新增隔离 `.venv-sgdjscc` 运行协议与 `requirements-sgdjscc.txt`；PyTorch 2.1/cu121、xFormers 0.0.22.post7 在 RTX 4090 D 上实际 kernel 通过，`pip check` 无冲突。MuGE 用 `encoder_weights=None` 构造后 strict-load 完整发布权重，避免作者代码先下载后被全覆盖的 EfficientNet-B7 初始化。
- 新增项目侧只读 adapter/config/tests：冻结源码 commit、输入、AWGN 1 dB/seed 2025、作者 50-step continuous diffusion 参数、资产 hash、不可覆盖输出和 author-native 禁止 outcome/direct-ranking 边界；BLIP2 caption 后立即释放以控制显存。
- `SMOKE-EXT-SGDJSCC-001` 第一次真实 run 即 PASS：输出 `[1,3,128,128]`、finite，smoke-only PSNR `25.055894 dB`，模型加载加单图前向 `12.6837 s`，peak allocated VRAM `7234.28 MiB`，无 `failure.json`。
- 实测 main latent `4096` real symbols；edge channel 输入 dense `16384`、nonzero-active `832`；active interpretation 的 main+edge CBR 为 `0.1002604`，literal dense-tensor interpretation 为 `0.4166667`。caption 为 488 UTF-8 bits，但作者协议无 text channel-symbol mapping，所以 common-contract 直接排名继续 fail-closed。
- 当前阶段只证明外部作者方法在本仓库可运行、可计量，不证明它或本方法更强；下一步用同图/同 AWGN realization/同 `[1,4,7,13,19]` 和总 65,536 real symbols 构建 SGD-JSCC common-contract 小 split，并显式闭合 caption/edge 物理码率，再推进 SING-Zero-style。
- 验证：标准库全套 84 项测试通过；adapter pytest 4 项、`py_compile`、`pip check`、tracked-source clean 与离线 scheduler 均通过。详细中文报告：`reports/sgdjscc_author_native_smoke_stage_result_2026-07-14.md`。

### 2026-07-15：SGD-JSCC 共同协议单图闭环，rate gate 通过并暴露 patch 语义风险

- 新增 `configs/external_sgdjscc_common_smoke.yaml`、`scripts/external_sgdjscc_common_smoke.py` 和 8 项 fail-closed 单测；作者第三方文件未改，author-native 与 common-adapter 标签严格分离。
- 256×256 输入按作者 `split_image_v2` 无重叠切为四块。实测四块 main/active-edge 为 `16,384/3,328` 实坐标；每块 caption 固定为 536-bit UTF-8+CRC16 packet，经 BPSK×21 占 `11,256` 实坐标，四块文本共 `45,024`；另计 `800` 无信息 padding，总数精确 `65,536` 实坐标=`32,768` 复使用，CBR `1/6`。
- 冻结 channel seed `20260729` 的 65,536 维 canonical AWGN 向量按 main→edge→text→padding 切片，SHA-256 `f8edbfe0...f416`。四个 caption 在 1 dB 下共有 `5,981/45,024=13.2840%` raw hard-symbol errors，但 R21 后 `0/2,144` packet bit errors、CRC `4/4` 通过。
- `SMOKE-EXT-SGDJSCC-COMMON-001` 第一次真实 run PASS：输出 finite `[1,3,256,256]`，smoke-only PSNR `24.785109 dB`，耗时 `13.1588 s`，peak allocated VRAM `7364.35 MiB`。所有模型/asset hash 复核通过，全程 offline、清空代理、official validation 未访问。
- 肉眼检查发现明确需要后续统计的风险：横纵 patch seam 可见，右边缘出现疑似由 patch caption 驱动的放大白衣人物，而原图相应区域无同尺度目标。单图只记作 hallucination/semantic-drift suspect，不包装成定量 new error。
- 术语更正：旧 native JSON 的 `main_real_cbr=0.08333` 是 real-coordinate/source-dimension ratio；按项目复信道口径 main complex-use CBR 为 `1/24`，main+active-edge 为 `0.0501302`。旧输出不覆盖，脚本和报告现同时列两种口径。
- 阶段判断：SGD-JSCC common-adapter 的 rate gate 已通过，但效果/semantic gate 仍未开始，禁止直接优劣结论。按冻结排期下一项为 SING-Zero-style 同底座机制协议，之后统一做 64 图×3 seed 外部 stage。
- 验证：全仓 94 项 `unittest` 通过，external contract checker、common dry-run、`py_compile`、asset hash、tracked-source clean 均通过。
- 详细中文报告：`reports/sgdjscc_common_contract_smoke_stage_result_2026-07-15.md`。

### 2026-07-15：外部共同协议 8×5 pilot 完成，当前 M3 获得方向性优势

- 先纠正 SGD 单图 common smoke 的 AWGN 口径：旧 `24.785109 dB` 使用作者每实坐标 `P/SNR`，比项目复信道 `P/(2×SNR)` 严苛 3 dB，只保留作接入证据；新复信道 smoke 为 `26.128782 dB`。
- 新增 `ANALYSIS-EXT-COMMON-PILOT-001`：从 frozen Imagenette policy-dev clean membership 按预注册 SHA 规则选 8 图，固定 SNR `[1,4,7,13,19]`、base seed `20260729`、每图/SNR 65,536 维 canonical CPU noise 和 CBR `1/6`；official val 未访问。
- 当前 M3、SGD-JSCC common adapter、SING-Zero-style 各完成 `40/40` rows。聚合器验证全部 120 行的 sample/SNR key、noise SHA、DeepJSCC reference、复 AWGN 方差口径与 rate 完全一致。
- DeepJSCC/当前 M3/SGD/SING-style 的 aggregate PSNR 为 `31.7438/33.0594/26.8882/24.6593 dB`，LPIPS 为 `0.07861/0.03532/0.07763/0.31725`。当前 M3 相对 SGD 配对为 `+6.1712 dB/-0.04231 LPIPS`，PSNR `40/40`、LPIPS `39/40` 行更优。
- 当前 M3 与 SGD 均为 `0` final failure/new error；SING-style 在 1 dB 有 `1` 个相对 DeepJSCC new error。该 SING 对照只做 final-only 2×2 mean-pool range/null projection，不是论文逐步 DDNM 复现，负结果不得外推到 SING 论文。
- SGD 的 `160/160` caption packets CRC 全通过，但 sender BLIP2 已在 patch level 把 dog 写成 cat、chainsaw 图写成 snowy-road driver 等；确认 packet reliability 与 semantic reliability 必须分列。视觉上作者四 patch 路径仍有 seam。
- SGD 批跑有两次 pre-inference scheduler-cache 失败，失败目录和 `failure.json` 均保留；把 `HF_HOME`/offline flags 移到所有 transitive hub import 前后，第三次完整 PASS。全程本地离线，无新增下载。
- 阶段判断：当前 M3 在共同协议小 pilot 获得可信方向性正信号，diffusion 不退出；但 8 张已暴露 development 图不能授权论文级领先或 semantic-tail safe。下一外部阶段应预注册 64 图×3 channel seeds，并把 SING 升级为逐步 DDNM-style 后再谈方法级比较。
- 中文报告：`reports/external_common_comparison_pilot_stage_result_2026-07-15.md`。验证：全仓 `99/99` unittest、三个 runner dry-run/real-run、120-row aggregate gate 与 `py_compile` 通过。

### 2026-07-15：补齐 SGD-JSCC 作者/项目双工作点，确认 diffusion 继续保留

- 全文复核确认 SGD-JSCC 所有实验标称 CBR `1/20`，发布路径 main+active-edge 实测 `19,712 real = CBR 0.0501302`；论文明确忽略文本成本并假设无误传输。因此作者工作点结果严格标为“免费/无误文本论文协议上界”，不再与物理 common-contract 混写。
- 新增精确低码率 DeepJSCC：c3 的 24,576 维 latent 固定发送 19,712 个活动坐标，按活动功率归一化和 `P/(2×SNR)` 训练。首轮 AMP 在 epoch1/batch213 非有限失败并保留；FP32 稳定化后继续完整 COCO 12 epoch，COCO-512 达到 `26.6981 dB/0.77855 SSIM`。
- `ANALYSIS-EXT-AUTHOR-RATE-PILOT-002` 验证同 key、同 19,712-D noise SHA。低码率 DeepJSCC/SGD 上界为 PSNR `25.9260/26.8389`、LPIPS `0.28716/0.07856`、failure `3/0`；SGD 配对 `+0.91283 dB`，PSNR 31/40、LPIPS 40/40 行更优，并修复全部 3 个低码率 failure。
- `ANALYSIS-EXT-SGD-REALLOC-PILOT-001` 在项目 CBR `1/6` 冻结 main-R2/edge-R1/text-R13/pad 分配。SGD 从 `26.8882→27.3933 dB`、LPIPS `0.07763→0.07246`，PSNR 40/40 改善，160/160 caption CRC 通过；证明旧差距部分来自预算分配。
- 当前 M3 相对重分配 SGD 仍为 `+5.6661 dB/-0.03714 LPIPS`，40/40 PSNR、38/40 LPIPS 更优，双方 0 hard failure；只记为 8 图方向性 pilot，禁止论文级领先。
- 决策：**不放弃 diffusion。** 下一方法阶段应以 19,712-real DeepJSCC 重新生成 cache 并训练低码率 B1/diffusion/posterior，只在活动坐标做 measurement consistency；随后用 64 图×3 seeds 做严格 semantic-tail 外部审计。现有 c8 M3 权重不得直接冒充低码率 M3。
- 新增双工作点配置、exact-rate/repetition adapter、训练/评估/聚合脚本和 3 项单测；99/99 unittest、3/3 pytest、`py_compile`、`git diff --check` 通过。无新增下载，official val 未访问。中文报告：`reports/external_two_working_point_alignment_stage_result_2026-07-15.md`。
### 2026-07-15：精确低码率 M3 最小闭环取得阶段成果

- 新建 exact-rate COCO cache：`19712` 总实坐标中严格保留 `80` 个 UInt2+BPSK×4 语义载荷坐标、`19632` 个图像坐标；10000/1000×5 SNR 共 55000 行，manifest SHA 仍为 `93ae3f3b...2de9`。B0 的 11000 图均值为 `24.1888/25.5855/26.4744/27.2784/27.5063 dB`。
- `EXP-S16-B1-001` 在低码率缓存上从头训练 B1：1000 图×5 SNR 的 PSNR 全正，平均 `+1.03815 dB`；LPIPS 全负，平均 `-0.11412`。best SHA `7a295976...615a`。旧 B0-top1 一致性路由会误拒大量真实修复，低码率下不再把 B0 当最终裁判。
- `EXP-S16-DIFF-001` 重新训练短链 diffusion 后为明确负结果：五档 PSNR/LPIPS 全部恶化，raw new/repair `318/139`，预注册检查只通过 sampling-step；best SHA `44915d7e...8a`。负结果完整保留，禁止包装为 diffusion 成功。
- `ANALYSIS-S16-LOWRATE-M3-STAGE-001` 完成严格 8×5 链：payload BER 0；B1 把 strict B0 的 `25.8765/0.28872 LPIPS/3 failures` 改为 `26.8461/0.17140/0`。posterior consistency `0.13253→0.11138` 且 40/40 行下降，但原全 SNR 路由 final 比 B1 `-0.2027 dB/+0.00250 LPIPS`，完整判定失败。
- 在任何新输出前冻结 `SNR<19→B1、SNR=19→语义门控 posterior`，用哈希排名第 9–16 的独立 8 图和新 seed20260731 做 `ANALYSIS-S16-LOWRATE-M3-TAIL-HOLDOUT-001`。7 项预注册检查全过：B1 相对 B0 `+1.0273 dB/-0.10969 LPIPS`、failure `3→0`；final 相对 B1 全五档 `-0.00990 dB/-0.000389 LPIPS`、failure/new `0/0`；19 dB 单档 `-0.0495 dB/-0.001945 LPIPS`。
- 阶段结论：不放弃 diffusion，但将当前可复现价值收紧为 **高 SNR posterior-constrained perceptual tail**；低中 SNR 主力冻结为 B1。第一组相同 8×5 上 B1 PSNR `26.8461` 与 SGD 免费 caption 上界 `26.8389` 基本持平，但 LPIPS `0.1714` 明显不如 SGD `0.07856`，不能声明超过论文。
- 完整中文报告：`reports/lowrate_m3_stage_result_2026-07-15.md`。全程本地离线、未下载、未访问 official val。
- 验证：102/102 项标准库 `unittest`、关键脚本 `py_compile`、两份 40-row exact-rate/consistency 完整性检查和 `git diff --check` 通过。当前系统 Python 未安装 `pytest` 模块；`pip check` 仅报告既有环境的 `pynacl 1.5.0` 缺少 `cffi`，本轮未联网修改环境。

### 2026-07-15：完成 SGD-JSCC step matching 机制的项目级复核

- 作者代码确认：归一化 AWGN 潜变量使用 `signal_scale=gamma/(gamma+1)`，离散模式取 scheduler `alphas_cumprod` 的最近时刻，并以接收潜变量作为该时刻的反向起点；text 与 edge/ControlNet 分别约束全局语义与空间结构。
- 修正“严格对应”的表述：该结论在归一化 AWGN 与方差口径一致时是 forward marginal 同形；作者实现还有逐样本 L2 球面归一化和离散最近 step，不是对单条已实现 Markov 轨迹的唯一精确识别。
- 发现与当前负结果直接相关的架构差异：SGD-JSCC 发送的就是 diffusion/VAE latent，而 `EXP-S16-DIFF-001` 是 B1 后的 image-space residual bridge，没有 channel-state/scheduler-state 对齐。因此当前负结果不应外推为 diffusion 上限低。
- 项目复 AWGN 使用每实坐标 `P/(2*gamma)` 方差，若训练实值 latent diffusion，匹配量应为 `alpha_bar_channel=2*gamma/(2*gamma+1)`，不能直接抄作者口径。
- 下一 diffusion 候选方向收紧为 **exact-rate channel-state-matched latent diffusion + 逐步 active-coordinate measurement consistency + 预算内语义条件/风险控制**。本轮只做机制与公式复核，未更改主线、未运行新实验、未联网或下载；详细记录已补入 `LITERATURE.md`。

### 2026-07-15：channel-state-matched latent diffusion 取得机制级阶段成果

- 在任何训练输出前冻结 `EXP-S17-LATDIFF-001`：exact-rate `19,712 real` 中只对 `19,632` 个图像活动坐标训练 masked epsilon predictor，项目 `P/(2×SNR)` 口径使用 `alpha=2*gamma/(2*gamma+1)`；selection/holdout 各 256 图且严格不重叠。
- 原 AMP 运行在 epoch0/batch13 出现 non-finite loss，未产生 selection 输出，失败目录保留；只把 AMP 改为 FP32 的 `EXP-S17-LATDIFF-002` 完整训练，最佳 epoch5 checkpoint SHA `cfc52716...b4e1`，selection matched−B0 PSNR `+0.151483 dB`。
- 一次性 256图×5SNR holdout 上，matched DDIM 相对 B0 为 PSNR `+0.148715 dB`（image-cluster 95% CI `[+0.129607,+0.168857]`）、LPIPS `-0.035305`（`[-0.038907,-0.031814]`）；活动 latent MSE `0.145516→0.060453`，五档均改善。
- Step matching 得到直接支持：matched 相对固定 7 dB step 错配为 `+0.233455 dB`（95% CI `[+0.220234,+0.246661]`）；scalar LMMSE 仅 `+0.001526 dB` 且 LPIPS 恶化，不能解释 learned prior 的收益。
- 收益明显集中在低 SNR：1/4/7/13/19 dB 的 matched−B0 PSNR 为 `+0.61495/+0.19950/+0.01033/-0.05408/-0.02712 dB`。13/19 dB 虽 latent MSE 改善但图像 PSNR 下降，定位出 decoder-unaware latent loss 的瓶颈。
- 当前冻结 B1 仍更强：`27.0162 dB/0.19090 LPIPS`，matched latent diffusion 为 `26.1319/0.27117`；直接串联旧 B1 反而比 B1 `-0.23127 dB`，确认输入分布偏移，不能包装成系统提升。
- pseudo 语义诊断仍为净修复：AlexNet eligible rows 上 matched new/repair `26/83`，三分类器多数票 `7/38`；但 COCO pseudo label 不替代正式监督审计。
- 预注册 8 项检查通过 6 项，正式 verdict 为 `NEGATIVE_OR_PARTIAL`。阶段结论是 **物理匹配 latent diffusion 机制成立、当前系统融合未成立**。中文报告：`reports/channel_matched_latent_diffusion_stage_result_2026-07-15.md`；全程本地离线，official Imagenette validation 未访问。
- 验证：全仓 `108/108` 标准库 `unittest`、3 个新 Python 文件 `py_compile`、10,000-replicate image-cluster bootstrap、输入 CSV/checkpoint SHA 和 `git diff --check` 全部通过。

### 2026-07-15：decoder-aware latent diffusion 获得同预算显著增益

- 在新 selection/holdout 结果前预注册同 parent、同三轮预算的 control 与 decoder-aware 两分支；后者唯一增加 frozen DeepJSCC decoder 图像 MSE。纯训练 loss 尺度规则在 16 batches 上冻结 `lambda_img=20`，未读取 selection/holdout。
- 旧 S17 已暴露 validation 0--511；本轮冻结 512--767 为 selection、768--999 为 232 图 fresh holdout。control/decoder 最佳 checkpoint SHA 分别为 `edbcbdbd...2b1f` 与 `5b708117...5d98f`。
- fresh 232图×5SNR 上，decoder-aware 相对同预算 control 为 `+0.021605 dB` PSNR（image-cluster 95% CI `[+0.018883,+0.024640]`）、`-0.002502` LPIPS（`[-0.002824,-0.002203]`），活动 latent MSE 也下降 `-0.002324`；相对 parent PSNR 为 `+0.021006 dB`。
- decoder-aware 相对 B0 整体为 `+0.174221 dB/-0.038540 LPIPS`，比上一版更强；但分 SNR PSNR 仍为 `+0.66845/+0.24078/+0.03716/-0.04771/-0.02758 dB`，只有 3/5 档为正。
- pseudo 语义没有总体恶化：AlexNet control new/repair=`27/66`、decoder=`29/72`；三分类器多数票为 `5/21→5/31`。COCO pseudo audit 不替代监督安全审计。
- naive decoder-aware DDIM→旧 B1 仍比 B1 `-0.198803 dB`，确认输入分布融合尚未解决。8 项预注册检查通过 7 项，verdict `NEGATIVE_OR_PARTIAL`：**decoder-aware objective 有真实小幅贡献，但还不是全 SNR/最终系统成功。**
- 下一方法变量冻结为带 `g(alpha)→0` 高 SNR identity limit 的 SNR-conditioned correction envelope，不再扫描 image-loss 权重。现有 1000 validation 已全部暴露，下一正式阶段必须建立新的未使用 COCO selection/holdout manifest。
- 中文报告：`reports/decoder_aware_latent_diffusion_stage_result_2026-07-15.md`；输出：`outputs/EXP-S17-LATDIFF-003-CONTROL/`、`outputs/EXP-S17-LATDIFF-004-DECODER/`、`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-003/`、`outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-002/`。全程本地离线，official Imagenette validation 未访问。

### 2026-07-15：SNR identity envelope 首次让 matched diffusion 全五档 PASS

- 在任何新结果前预注册 frozen decoder-aware diffusion 的单调 correction envelope；网络、6-step DDIM、码率、AWGN 和 payload 全部冻结，只改变 `z=y+g(SNR)(z_diff-y)` 的强度控制。
- 从 COCO train2017 剩余 107,287 图按 SHA rank 建立全新 256 selection + 256 holdout；与旧 11,000 图 source path/SHA overlap=`0/0`，manifest SHA `c467d2cc...a8bed`。
- selection 比较 smooth `p={0.25,0.5,1,2}` 与 frozen hard identity。`p=0.5` mean PSNR更高但 13/19 dB 仍负；预注册可靠性优先级选出 `hard_identity_7db`，policy SHA `c31d6853...d05eb`：1/4/7 dB `g=1`，13/19 dB `g=0`。
- 一次性 holdout 上 selected 相对 B0 为 `+0.189717 dB`（95% CI `[+0.170601,+0.210902]`）、LPIPS `-0.036284`；五档 PSNR delta=`+0.677172/+0.240940/+0.030472/0/0 dB`，低中 SNR 保留 full gain `99.999998%`。
- selected 相对 full diffusion PSNR `+0.015642 dB`，95% CI `[+0.014230,+0.016915]`；代价是 LPIPS 相对 full 回吐 `+0.001005`，但仍显著优于 B0。该结果明确记录为 distortion/reliability 与 perception 的取舍，不声明全面支配。
- pseudo semantic 也更保守：AlexNet full→selected new/repair=`17/81→16/71`，majority=`7/32→5/31`；13 dB full 的 majority `2 new/1 repair` 被恒等回退清零。COCO pseudo 指标仍不替代监督审计。
- B1 仍比 selected 高 `+0.830617 dB`（95% CI `[+0.791172,+0.869723]`），所以整体最强 anchor 未变。本阶段成功的是 **diffusion 支路的全 SNR identity-safe strength control**，不是超过 B1。
- 预注册 10/10 checks 全过，最终 verdict=`PASS`。下一步不再扫描 envelope；应训练接收 `B0 + identity-controlled diffusion decode` 的同容量融合器，并与只接收 B0 的同预算 B1 做因果对照。
- 中文报告：`reports/snr_identity_envelope_stage_result_2026-07-15.md`；全程本地离线，未访问 official Imagenette validation。全仓 `112/112` unittest 通过。
