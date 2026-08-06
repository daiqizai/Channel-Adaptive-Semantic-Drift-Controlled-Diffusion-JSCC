# 数据集、管线与比较协议时间线

审计日期：2026-08-03
性质：静态证据审计；未重跑实验。本文追踪会改变结论可比性的总体、信道、码率、模型、选择集与指标口径。阶段号是研究 Stage，不是同一模型的连续版本。

## 1. 协议演变总表

| 日期/阶段 | 数据与总体 | 分辨率 | 信道/SNR | 码率合同 | 训练/选择合同 | 主要指标与语义口径 | 审计解释与可比边界 | 主要证据 |
|---|---|---:|---|---|---|---|---|---|
| S1 | CIFAR-10 test 或固定子集 | 32² | AWGN | 早期 sanity 合同 | 接入 DeepJSCC checkpoint | PSNR、分类一致性 | 只证明模型—信道—指标链可运行，不外推自然图 | `PROJECT.md`；`MILESTONES.md`；`reports/group_meeting_progress_2026-07-31.md` §2.4 |
| S2-HR/S2 | COCO2017 train/val 子集 | 256² crop | AWGN，后统一为 `[1,4,7,13,19]` dB | 旧最小闭环记为 CBR 0.17 | 高分辨率 JSCC + blind diffusion pilot | PSNR/MS-SSIM/LPIPS；冻结语义模型逐步接入 | 建立自然图入口；blind diffusion 的视觉改善不能自动算语义提升 | 同上；`PROJECT.md` |
| S3–S6 | COCO 开发总体 | 256² | AWGN 五档 | 旧 B0/B1 合同逐步固定 | CLIP/分类器/caption-CLIP、SNR-aware strength、fallback、gate | classification consistency、drift、failure；CLIP 仅辅助 | 多轮 controller/gate 结论依赖旧弱 backbone；只能作机制证据 | `reports/group_meeting_progress_2026-07-31.md` §2.4 |
| S7–S12 | COCO 开发/holdout | 256² | canonical paired-real AWGN 五档 | 从总码率审计演化到 `c6 main+c2 structure` 等合同 | structure/semantic sketch、matched-rate short-chain diffusion、同容量 refiner control | 低层+感知+语义 gate | side signal、短链和 semantic control 多数未同时通过质量与风险门槛 | 同上 S7–S12 表 |
| S13–S16 | COCO 10k/1k 与独立 holdout | 256² | AWGN 五档 | 最终形成 exact `19,712 real` 的低码率 B1 | 强 B1 anchor；diffusion 不得用 holdout 选点 | PSNR/LPIPS/semantic new-repair | B1 增强显著改变参照分布；更早 diffusion 结果不能直接接到后续强 backbone | 同上 S13–S16 表 |
| S17–S19 | COCO 新 512 图总体及独立 holdout | 256² | channel-state-matched AWGN 五档 | exact `19,712 real` | FP32 matched latent diffusion、SNR identity envelope、B0-only 等容量 control | source-cluster CI；高 SNR exact fallback；多数分类器审计 | 证明 diffusion observation 含不可被同容量 control 完全替代的信息；不是最终外部性能结论 | `reports/group_meeting_progress_2026-07-31.md` §3.2 |
| S20 | Imagenette policy-dev clean-correct 64 图×3 seeds×5 SNR，共 960 键 | 256² | canonical AWGN | B1 `19,712 real`；SGD main+edge `19,712`，caption 后最低 `≥21,856` | 冻结已有模型，无 checkpoint 选择 | PSNR/MS-SSIM/LPIPS、冻结 `T_cls` failure、延迟 | SGD 是 unequal-rate/perfect-caption paper upper，仅能作 non-ranking 感知上界 | `reports/group_meeting_progress_2026-07-31.md` §3.1 |
| S21–S25 | COCO fresh selection/holdout | 256² | AWGN 五档 | exact `19,712 real` | learned gate、bounded residual、convex mix、feature injection、oracle headroom | cluster CI、exact fallback、pseudo semantic | gate 塌零/质量崩坏；安全非零方向效应仅约 `0.00057 dB`，controller 路线关闭 | 同上 §3.2、§4 |
| S26/S27 | S27 为与既往 path/SHA 零重叠的全新 COCO 512 图 | 256² | AWGN 五档 | exact `19,712 real` | 1/4/7 dB 用 S19 fusion，13/19 dB exact B1；冻结后复制 | PSNR/MS-SSIM/LPIPS、majority pseudo failure、cluster CI | fusion 相对 B1 约 `+0.0927 dB/-0.00792 LPIPS`，属于旧 backbone 上可复现机制证据 | 同上 §3.2 |
| S28–S30 | 同一 Imagenette policy-dev 960 键 | 256² | canonical AWGN 五档 | current/B1 `19,712`；author/DiffJSCC `16,384`；SGD `≥21,856` | external checkpoint 复现；S29 原 batch 精确重放 | PSNR/MS-SSIM/LPIPS/`T_cls` failure | S30 暴露旧 14 万参数 backbone 才是主要瓶颈；跨码率不能作纯算法归因 | 同上 §3.3 |
| S31/S31B | COCO train2017；固定 COCO val512 | 256² | AWGN，离散五档逐图采样 | exact `19,712 real` | 31M 四级双侧 SNR-conditioned，MSE-only；AMP 失败后 FP32 | val aggregate PSNR/MS-SSIM；不用 LPIPS/语义选 checkpoint | 得到强保真基座，但比 author 多码率，只是预算上限内比较 | `README.md` S31；`EXPERIMENTS.md` |
| S33/S33B | COCO train2017 + 固定 val512；外部定位仍是已知 64 图 policy-dev | 256² | canonical AWGN `[1,4,7,13,19]` | 原生 exact `16,384 real`=`1/24`，无 mask/side info | 随机初始化，FP32 4+8 epoch，MSE-only；按 val512 五档质量选择 | PSNR/MS-SSIM/LPIPS/`T_cls` failure、source-cluster CI | 相对 author-JSCC 的 `+0.4799 dB` 只成立于已知 policy-dev；优势由低中 SNR 驱动，19 dB 未过逐档非劣 | `reports/strong_jscc_16384_equal_rate_stage_result_2026-07-21.md` |
| S34A equal-budget | COCO 同训练/val 合同；Imagenette policy-dev 960 键 | 256² | canonical AWGN 五档 | 三臂 exact `16,384 real` | Swin Base/CM 与 S33 equal optimizer-step、12 epoch；两 Swin 尚未完全收敛 | PSNR/MS-SSIM/LPIPS/failure/cluster CI | S33 对 Base 的开发集优势、对 CM 的 Pareto 只回答 equal-budget，不回答各自收敛上限 | `reports/swinjscc_equal_budget_stage_result_2026-07-22.md` |
| A0/A1 | Kodak 24×5×3；CLIC2020 test 428×5×1 | 原生大图，以共同 256 tile/padding 处理 S33/Swin | canonical AWGN | 三判别式臂逐图 actual CBR 相同；Kodak `1/24`，CLIC `0.041667–0.063210` | 训练仍用冻结 COCO-S33/Swin；无大图重训 | PSNR/MS-SSIM/LPIPS/DISTS/CLIP；CLIC FID/KID；无监督 failure | 独立高分辨率主表否定“S33 强于 Swin”；CLIC 仅 1 channel seed，CI 主要反映图像总体 | `paper_idea1b/A1_DISCRIMINATIVE_RESULT.md` |
| S34C-Lite | 冻结 Imagenette policy-dev 960 共同键 | 256² | canonical AWGN 五档 | S33/DiffJSCC exact `16,384`；SGD `≥21,856` | 只读统一账本；不训练/推理 | PSNR/MS-SSIM/LPIPS/failure；无共同 FID/KID | S33–DiffJSCC 是 fidelity–perception Pareto；训练数据/范围不同限制算法归因；SGD 永久 non-ranking | `reports/s34c_lite_rate_transparency_result_2026-07-23.md` |
| S34D | 冻结 policy-dev；质量曲线 64图×1 seed×5 SNR=320 键 | 外部入口统一 256²；DiffJSCC 内部 512² | canonical AWGN | 沿用历史合同 | 同卡 RTX4090D、batch1、persistent-resident、common PyTorch 2.1；不含加载/I/O/指标 | latency、参数、supported-op FLOPs 下界、LPIPS/PSNR/failure | 25-step 是已测候选中最低 LPIPS PASS 点，不是理论最小；`165.1×` 不外推所有未来 diffusion | `reports/s34d_generative_inference_cost_result_2026-07-23.md` |
| 低 SNR 人工审计 | 从 policy-dev 1 dB 异常信号中事前分层选 15 source；另测 −3/−5 dB | 256² | 1 dB 与范围外 −3/−5 dB | 各历史合同；SGD non-ranking | 固定样本后人工 faithful/重建失败/clear-wrong | 多模型分类、CLIP、人工三分 | 样本量小且定向；“未观察到 clear-wrong”不能证明不存在 hallucination | `reports/low_snr_semantic_drift_visual_audit_2026-07-23.md`；`EXPERIMENTS.md` |
| S35R-P0/P1 | P0 复用 S34D；P1 计划 COCO train/val512 + 冻结960键 | 256² | AWGN 五档 | 冻结 S33 `16,384 real`；refiner 额外符号/side-info=0 | P0 只读；P1 2M–6M residual U-Net 仅预注册 | P1 要同时过 LPIPS、PSNR `>-0.10 dB`、failure 三重 gate | P1 尚无 smoke/训练/结果，不得写成已完成方法 | `README.md` S35R；`MILESTONES.md` 2026-07-23 修订 |
| RDD-P0 | 主体为 Imagenette policy-dev 64×3×5；CLIC-428 仅判别式补充 | 256²；CLIC 原生高分辨率 | 复用 AWGN 输出 | 沿用各冻结方法合同 | 只读恢复 4 臂；KID 主/FID 必报；指纹按 source GroupKFold | 分布偏移、轻量频域指纹；不作质量排名 | 主体与原计划 CLIC 三生成臂不同；只支持可识别实现/分布偏移，不支持生成先验因果归因 | `reports/rdd_p0_distribution_shift_result_2026-07-30.md` |
| CVaR P0/P1 | COCO val 中与 S33 val512 不相交的 200 图×64 realizations×5 SNR×4 arms | 256² | Rayleigh block fading+ZF；AWGN control | `1/24` 单 backbone | P0 冻结 AWGN S33；P1 Rayleigh matched mean 6 epoch；两独立 seed | PSNR、outage、CVaR-MSE、方差归因；无语义指标 | P0 尾部主要是 train/test mismatch；P1 后残余不再信道主导，判 `END-CVAR`；Rayleigh 不自动进入主线 | `reports/cvar_p1_rayleigh_matched_result_2026-07-31.md` |

## 2. 会改变结论解释的五次关键协议迁移

1. **弱 B0/B1 → 强 S31/S33 backbone。** 旧 diffusion 增益是在不同表示与码率合同上得到，不能直接宣称可叠加到 S33。
2. **`19,712 real` → 原生 `16,384 real`。** S32 的胜出不是 strict rate-match；只有 S33 才与 author-JSCC/DiffJSCC 在 256² 共同总体上 exact-rate。
3. **Imagenette policy-dev → Kodak/CLIC。** S33 在 64 图开发总体上的强 backbone 结论没有泛化；A1 将其收缩为低代价、较低质量端点。
4. **统一 256²分析 → 原生高分辨率补充。** RDD 的“判别式偏向 blur”在 CLIC-428 消失，说明分辨率和样本量会改变分布结论。
5. **AWGN → Rayleigh matched mean。** P0 大尾部不能归因均值目标；匹配训练吸收大部分尾部后才可判断 CVaR 是否有独立优化对象。

## 3. 当前权威解释顺序

在本项目已冻结的 scope 内发生冲突时，按以下顺序解释：`reports/METHOD_TERMINATION_REPORT_2026-08-03.md` 最终终止层 → `audit/CLAIM_REGISTRY.csv` 逐 claim 边界 → `PROGRESS.md`/`MILESTONES.md`/`README.md` 当前冻结记录 → 正式结果报告及机器可读 summary → `EXPERIMENTS.md` 索引 → 旧 `PROJECT.md` 最小闭环与历史阶段叙述。此前带日期的“授权”“下一步”“待执行”和“若未来重启”只表示其生成时点，均不能覆盖最终终止层。未来用户可以授权具有新问题、预算、预注册和 ID 的独立课题，但不得将其解释为本项目冻结主线的恢复。旧 M0–M3 继续约束研究纪律，不再构成当前待执行任务表。
