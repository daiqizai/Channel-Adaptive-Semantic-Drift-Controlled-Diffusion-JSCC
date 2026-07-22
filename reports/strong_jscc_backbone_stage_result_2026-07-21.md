# 强 JSCC 基座阶段结果（S31/S31b/S32，2026-07-21）

## 结论先说

**“我们的 JSCC 为什么不能做到和 DiffJSCC 的 JSCC 一样强”这个问题，已经得到正面结果：在项目固定总预算内，新 strong-JSCC 的聚合 PSNR、MS-SSIM、LPIPS 都显著优于 DiffJSCC 的 author-JSCC 前端。**

冻结 S20 policy-dev 的同 64 图、3 个 channel seed、5 个 SNR（960 行）上：

| 方法 | 实符号 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | T_cls failure ↓ |
|---|---:|---:|---:|---:|---:|
| **S31b strong-JSCC** | **19,712** | **30.419910** | **0.970266** | **0.122824** | **14** |
| author-JSCC | 16,384 | 29.986135 | 0.963092 | 0.128342 | 22 |
| 完整 DiffJSCC | 16,384 | 27.598398 | 0.940799 | **0.100223** | 23 |
| 旧 current | 19,712 | 28.223678 | 0.949057 | 0.152084 | 29 |
| 旧 B1 | 19,712 | 28.124602 | 0.946698 | 0.159396 | 35 |

strong−author-JSCC 的 source-image cluster 95% CI 为：PSNR `+0.433774 dB [ +0.328020, +0.554007 ]`，MS-SSIM `+0.007174 [ +0.006349, +0.007996 ]`，LPIPS `-0.005518 [ -0.007775, -0.003147 ]`。三项质量轴都显著有利；failure `14 vs 22` 的差值 CI 为 `[-0.01979, 0]`，点估计更好，但不包装成严格显著的语义胜出。

必须保留的边界是：strong 使用完整项目预算 `19,712 real`，author-JSCC 只用 `16,384 real`。因此结论是“**在统一项目预算上限内，我们已经做出比作者前端更强的基座**”，不是“同实符号数 exact-rate matched 胜出”。S32 是已知 S20/S30 结果后的 policy-dev 定位，也不是独立 final test。

## 为什么这次真的变强了

旧 exact-rate JSCC 只有 140,239 个参数、两级下采样、固定 7 dB 训练、无网络内部 CSI modulation，还先生成 24,576 个实 latent 再固定 mask 到 19,712。新基座则为：

- 31,118,032 个参数，约为旧 JSCC 的 222 倍；
- 四级编码器/解码器，encoder 与 decoder 的每个残差块都接受 SNR 条件；
- 原生 `77x16x16=19,712 real`，无裁剪、补零或 side information；
- `[1,4,7,13,19] dB` 按图像采样，完整 COCO train2017 端到端训练；
- 只用 MSE 和固定 COCO validation PSNR 选 checkpoint，没有用 LPIPS、分类器或 external population 调模型。

这说明之前与 author-JSCC 的差距主要确实来自主干容量、原生码率表示和训练合同，而不是“我们的 JSCC 原理上做不到”。

## 训练曲线与失败记录

原 `EXP-S31-STRONG-JSCC-001` 在 COCO 固定 512 图上的五档平均从 epoch0 `25.6984 dB`，经历高学习率震荡后升到 epoch3 `28.044783 dB / 0.958405`；随后 epoch4 batch418 检出 AMP unscale 后非有限梯度范数，按 fail-closed 规则停止。失败目录、STATE 和 checkpoint 全部保留。

systems-only 审计证明相同 checkpoint 在 FP32 batch 8--32 下前向、反向、gradient clip 和 AdamW step 都有限。`EXP-S31B-STRONG-JSCC-FP32-001` 又在任何 validation 输出前发现总 seed 会改变 val512 population，因此主动中止并保留 0-row 合同失败。修正后的 `-002` 恢复原 seed，逐项确认 512 个 validation 索引一致，只加载 S31 epoch3 模型权重，不加载旧 optimizer/scheduler/scaler。

S31b `-002` 八轮全部 finite：

| epoch | train MSE | 五档 PSNR | MS-SSIM |
|---:|---:|---:|---:|
| init（S31 epoch3） | — | 28.044783 | 0.958405 |
| 0 | 0.001896 | 28.568559 | 0.961921 |
| 1 | 0.001814 | 28.751901 | 0.963498 |
| 2 | 0.001740 | 28.931111 | 0.964799 |
| 3 | 0.001681 | 29.043150 | 0.965561 |
| 4 | 0.001638 | 29.201233 | 0.966515 |
| 5 | 0.001606 | 29.271211 | 0.966946 |
| 6 | 0.001588 | 29.325812 | 0.967266 |
| **7** | **0.001577** | **29.360583** | **0.967330** |

最终 COCO 分 SNR PSNR 为 `27.422064/28.623177/29.511816/30.463982/30.781875 dB`；最大归一化功率误差 `2.38e-7`。best checkpoint 为 epoch7，SHA-256 `2f8972a943599bae016f6f64550ca81ea5f861654d9ace6931aebe6cf9057ca8`。

## 外部分 SNR 结果

| SNR | strong PSNR | author PSNR | strong−author | strong LPIPS | author LPIPS | failure（strong/author） |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.286815 | 27.195399 | **+1.091415** | **0.160345** | 0.193366 | 7 / 10 |
| 4 | 29.581747 | 28.750015 | **+0.831732** | **0.133118** | 0.146909 | 5 / 8 |
| 7 | 30.567789 | 30.067189 | **+0.500600** | **0.117057** | 0.119716 | 2 / 3 |
| 13 | 31.649311 | **31.678271** | -0.028961 | 0.103573 | **0.095360** | 0 / 1 |
| 19 | 32.013887 | **32.239801** | -0.225914 | 0.100026 | **0.086360** | 0 / 0 |

所以“聚合三轴胜出”不等于每个工作点全胜。新基座的优势主要集中在 1--7 dB；author-JSCC 在 13/19 dB 仍保留小幅 PSNR/LPIPS 优势，说明我们的高 SNR 表示饱和仍有改进空间。这个观察来自已打开的 external population，后续不能在同一总体上继续调参后再称盲验证。

主结果先做 `floor(255*x)/255`，贴近 S30 author 的 uint8 路径。未量化 float 相对 uint8 的平均变化只有 `+0.042465 dB PSNR / +0.000075 MS-SSIM / +0.000479 LPIPS`；量化不会推翻 strong−author 的结论。

## 对完整 DiffJSCC 和项目主线的含义

strong-JSCC 相对完整 DiffJSCC 为：PSNR `+2.821512 dB`、MS-SSIM `+0.029467`、LPIPS `+0.022600`（更差），failure `14 vs 23`。因此二者仍是清楚的保真/感知 Pareto：

- 现在不再需要用 diffusion 掩盖一个弱 JSCC 前端；
- 但完整 DiffJSCC 仍证明生成先验能进一步改善 LPIPS，所以不应放弃 diffusion；
- 旧 current 被 strong-JSCC 单独在 PSNR `+2.196232 dB`、MS-SSIM `+0.021209`、LPIPS `-0.029260` 和 failure `14 vs 29` 全面压过，不能继续作为最终方法，只保留为“diffusion 确有互补信息”的历史因果证据。

下一阶段的合理目标已经非常具体：冻结 strong-JSCC 作为保真锚点，重新训练与其 latent/重建分布匹配的 channel-state-matched residual diffusion；目标不是追求最大生成幅度，而是从完整 DiffJSCC 的 `0.0226` LPIPS headroom 中取回一部分，同时把 PSNR 损失压到例如 `0.1 dB` 内，并继续以 semantic new-error/repair 和 exact fallback 约束风险。旧 B1、S19 或旧 latent diffusion checkpoint 均不能直接接到新基座后冒充完整方法。

## 可复现产物

- 训练输出：`outputs/train/EXP-S31B-STRONG-JSCC-FP32-002/`
- 外部比较：`outputs/external_baselines/ANALYSIS-S32-STRONG-JSCC-COMPARISON-001/`
- S32 `per_sample.csv` SHA-256：`74997b3c29775848a7cbe6e489828daadc4780f72e0cb7290024c65c10e8714a`
- S32 `summary.json` SHA-256：`a4187ae085af565b2e0f546e064759c2b8380d0d0c9602b2e2e715770ae20999`
- S32 config/script SHA-256：`b9766fd7...655f5b` / `ca931872...09817`

本阶段没有联网或下载；使用已有本地 COCO、冻结 S20/S30 总体、分类器和 LPIPS cache。official Imagenette validation 未访问。
