# SGD-JSCC 与 SwinJSCC 公平性澄清（2026-07-23）

## 最终定性

本项目现有 SGD-JSCC 结果属于：

> **作者发布权重的 paper-protocol / favorable upper-bound 外部定位，不是 COCO + 16,384 real + 五档训练合同下的公平等码率重训对比。**

因此，现有数据不能支持“SGD-JSCC 不如 SwinJSCC”或“SwinJSCC 战胜 SGD-JSCC”。仓库中也没有做过一个满足统一训练、统一总码率并包含 FID 的 SGD-vs-Swin 确认性实验。

## 逐项核验

| 问题 | 现有 SGD-JSCC 实际做法 | 是否与 S33/S34 Swin 公平对齐 |
|---|---|---|
| 权重来源 | 直接加载作者发布的 `JSCC_model.pth`、`diffusion_backbone.pth`、`diffusion_controlnet.pth`、`muge-epoch-19-checkpoint.pth`；作者源码 commit=`2188acc0...b4da5` | **否**；没有在项目 COCO 数据上重新训练 |
| SGD backbone 训练 SNR | 论文第一阶段 JSCC encoder/decoder 在 ImageNet、固定 AWGN `10 dB` 下训练；本地正式运行没有改变这一训练合同 | **否**；S33/Swin 使用逐图离散 `[1,4,7,13,19] dB` 训练 |
| 测试 SNR | 推理时把作者模型实际放到 `[1,4,7,13,19] dB` canonical AWGN 上测试 | 只对齐了**测试点**，没有对齐训练分布 |
| 训练数据 | 作者权重；论文 JSCC 第一阶段为 ImageNet，生成模型后续阶段使用作者的大规模数据；未在本项目 COCO 合同重训 | **否**；而且 S20 的 Imagenette population 可能与作者 ImageNet 训练源重叠，未做逐文件去重审计 |
| 主图码率 | `16,384 real` | 单看主图等于 S33/Swin，但这不是 SGD 总码率 |
| edge 码率 | active edge 另用 `3,328 real`，已计入 S20 的 image+edge ledger | 加上 edge 已为 `19,712 real`，比 `16,384` 高 `20.31%` |
| caption 码率 | 四个 128×128 patch 各一个 caption；按原论文假设完美、免费传输，正式 S20/S28 没有从预算扣除 | **否**；最低未保护计费为 `4×536=2,144 real`，严格总计至少 `21,856 real` |
| 相对 16,384 的总超额 | main `16,384` + edge `3,328` + 最低 caption `2,144` | 至少多 `5,472 real`，即 `+33.40%`；若给文本纠错保护，成本只会更高 |
| 指标 | PSNR、MS-SSIM、LPIPS、冻结 `T_cls` failure/new-error/repair、运行时间、显存、码率 | **FID 未计算**，不能声称已覆盖 SGD 的完整感知主场 |

作者仓库当前公开的是推理代码与 checkpoint，README 仍把训练/微调指南列为 TODO。因此即使要对齐，也不能把现有权重简单“改个 SNR 参数”称为重训；需要重新建立并冻结完整训练合同。

## 现有观测值：只能并排展示，不能排名

下表三者确实使用了相同的 64 张 S20 policy-dev 图像、3 个 channel seeds 和五个测试 SNR，但 SGD 的训练数据、训练 SNR与总码率均不同。数值只能帮助理解方法取向，不能作为公平胜负结论。

| 方法 | 实际物理口径 | PSNR | MS-SSIM | LPIPS ↓ | FID ↓ | semantic failure |
|---|---|---:|---:|---:|---:|---:|
| SGD-JSCC paper upper | `19,712 real` main+edge；caption 免费；若最低计费则至少 `21,856 real` | `27.74037` | `0.952973` | **`0.072101`** | **未计算** | `25/960 = 2.6042%` |
| SwinJSCC Base-SA-12ep | 严格总计 `16,384 real` | `30.29212` | `0.969685` | `0.117921` | **未计算** | `25/960 = 2.6042%` |
| SwinJSCC CM-SA-12ep | 严格总计 `16,384 real` | `30.53197` | `0.970981` | `0.111465` | **未计算** | `22/960 = 2.2917%` |

这张表恰好说明不能只看 PSNR：SGD 的 PSNR/MS-SSIM 低，但 LPIPS 明显优于两条 SwinJSCC 臂，符合生成式方法用像素失真换感知质量的定位。没有 FID/KID/DISTS，且总码率和训练合同不一致，所以不能进一步宣称 SGD 在“感知主场”已经公平获胜或失败。

SGD-JSCC 的分 SNR 观测值如下；每档有 64 图×3 seeds=`192` 行：

| SNR | PSNR | MS-SSIM | LPIPS ↓ | failure |
|---:|---:|---:|---:|---:|
| 1 dB | `26.11820` | `0.932872` | `0.089137` | `5/192 = 2.6042%` |
| 4 dB | `26.89275` | `0.944893` | `0.084817` | `8/192 = 4.1667%` |
| 7 dB | `27.91116` | `0.956979` | `0.067310` | `5/192 = 2.6042%` |
| 13 dB | `28.56824` | `0.962917` | `0.063549` | `6/192 = 3.1250%` |
| 19 dB | `29.21149` | `0.967206` | `0.055694` | `1/192 = 0.5208%` |

## FID 为什么没有数值

S20/S28 的 per-sample 表没有 FID 字段，正式聚合也没有运行 FID。作者发布代码虽然包含 FID 类，但其公开 inference 路径中的当前实现不能被当作本项目统一 FID 结果；本项目没有把代码中的占位值 `0` 误报为 FID。

此外，当前总体只有 64 个 source clusters。即使重生成全部输出，在这么小的总体上直接报告单个 FID 也会非常不稳定。公平感知评估应在冻结且足够大的同一总体上，同时计算 LPIPS 与 FID/KID，并对所有方法使用同一特征实现、预处理和样本数。

## 现有 SGD 结果允许和禁止的表述

允许：

- “作者发布权重在有利的免费完美文本协议下表现出很强的 LPIPS，并与高保真方法形成 fidelity/perception Pareto。”
- “现有运行证明作者 pipeline 可以在五个测试 SNR 上执行，并揭示 main/edge/text 的码率账本问题。”
- “这是 paper-protocol upper bound / external positioning，不是严格同总码率排名。”

禁止：

- “SGD-JSCC 不如 SwinJSCC”或“SwinJSCC 全面超过 SGD-JSCC”。
- “SGD-JSCC 已在 16,384 real 下公平比较”。
- “SGD-JSCC 已按我们的 COCO/五档 SNR 训练合同重训”。
- 用没有实际计算的 FID=0 或省略 FID后声称完成生成感知全面评价。

## 真正公平的 SGD-vs-Swin 对比需要什么

1. 使用作者结构从头训练，而不是直接使用现成权重；训练数据、增强、优化步数/收敛口径与 Swin/S33 分层对齐。
2. SGD backbone 至少按同一 `[1,4,7,13,19] dB` 逐图离散训练合同重训，不能只在推理时改 SNR。
3. 把 `main + edge + caption` 全部塞进统一的 `16,384 real` 总预算；这必然要求重新分配各支路并重训，不能沿用当前 released weights。
4. 相同图像、相同 channel seeds、相同 canonical AWGN、相同量化与 evaluator；报告分 SNR和聚合 PSNR/MS-SSIM/LPIPS/FID或KID/semantic failure及 CI。
5. 使用足够大的冻结感知测试总体，并对作者 ImageNet 训练源与测试图像做污染/重叠审计。

由于作者仓库尚未发布训练/微调指南，当前无法把“官方 SGD 严格等码率重训”描述为已经可复现完成的 baseline。若后续自行补齐训练流程，必须标明是 **official-architecture reimplementation under our common contract**，不能冒充作者原生训练结果。

## 证据索引

- S20 冻结配置：`configs/s20_sgd_b1_decision.yaml`
- S20 汇总：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/aggregate/summary.json`
- S28 配置与汇总：`configs/s28_external_sgd_positioning.yaml`、`outputs/external_baselines/ANALYSIS-S28-CURRENT-VS-SGD-001/summary.json`
- 作者权重与运行配置：`configs/external_sgdjscc_native_smoke.yaml`
- SGD 阶段报告：`reports/sgd_b1_decision_stage_result_2026-07-17.md`
- Swin equal-budget 汇总：`outputs/external_baselines/ANALYSIS-S34A-SWINJSCC-EQUAL-BUDGET-COMPARISON-001/summary.json`
