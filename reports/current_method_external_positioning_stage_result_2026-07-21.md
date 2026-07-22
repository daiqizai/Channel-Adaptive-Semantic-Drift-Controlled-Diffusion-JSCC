# 当前方法外部定位阶段结果（S27--S29，2026-07-21）

## 一句话结论

当前方法已经不只是“在 B1 后面加一个模块”：**它利用功率归一化 AWGN 观测与扩散中间噪声状态的解析对应，在低 SNR 从同一接收码字生成零额外码率的扩散观测并与确定性 JSCC 重建融合，而在高 SNR 结构性精确回退 B1，从而只在生成先验确实有用时使用 diffusion，并主动封住高信道质量下的 semantic drift。**

## 本轮回答了什么

S27 已经在与 S16/S18/S19/S21 全部按 path 和 SHA-256 去重的 512 张新 COCO 图像上完成一次性复现。S28 进一步把冻结方法放到 S20 的 64 张 Imagenette policy-dev 图像、三个信道种子和五个 SNR 上，与冻结 B1 和 SGD-JSCC 逐样本对齐，共 960 行/方法。S29 又按 S20 原始 batch=64 重放 B1，排除了 batch-dependent 浮点差异造成的合同疑点。

本轮没有训练、调参、选择样本或访问 Imagenette 官方验证集，也没有联网下载。

## 主要结果

| 方法 | 总实符号 | 免费文本 | PSNR | MS-SSIM | LPIPS | T_cls failures |
|---|---:|---:|---:|---:|---:|---:|
| 当前方法：S19 low-SNR fusion + exact-B1 fallback | 19,712 | 0 | **28.22368** | 0.94906 | 0.15208 | 29 |
| 等容量 matched control + exact-B1 fallback | 19,712 | 0 | 28.16400 | 0.94759 | 0.15507 | 36 |
| B1 | 19,712 | 0 | 28.12459 | 0.94670 | 0.15940 | 35 |
| SGD-JSCC 论文协议上界 | 19,712 图像/边缘 + 至少 2,144 文本 | **2,144 未计费** | 27.74037 | **0.95297** | **0.07210** | **25** |

### 当前方法相对 B1

- PSNR：`+0.099085 dB`，source-image cluster 95% CI `[+0.088053,+0.111284]`；
- MS-SSIM：`+0.002360`，CI `[+0.001875,+0.002887]`；
- LPIPS：`-0.007314`，CI `[-0.009042,-0.005702]`；
- T_cls failure：`35→29`，`6 new / 12 repair`；failure-rate CI 跨 0，因此只能说计数未恶化，不能说语义改善已显著；
- 分 SNR PSNR 增益：1/4/7 dB 为 `+0.14276/+0.15977/+0.19290 dB`，13/19 dB 只有批处理浮点量级差异；方法结构上的 current/B1 图像张量差严格为 0。

这与 S27 的 512-image pristine COCO 结果 `+0.092662 dB / -0.007922 LPIPS` 高度一致，说明低 SNR 增益跨总体、跨数据域可复现。

### diffusion 信息是否真的有用

当前方法相对参数量和训练方式匹配、但把辅助输入替换成重复 B0 的 control：

- PSNR `+0.059681 dB`，95% CI `[+0.050030,+0.069327]`；
- MS-SSIM `+0.001465`，CI `[+0.001139,+0.001839]`；
- LPIPS `-0.002990`，CI `[-0.004083,-0.001982]`；
- failure `36→29`，`6 new / 13 repair`，但 failure-rate CI 仍跨 0。

因此增益不能只归因于“多了一个 CNN”或“多了训练容量”；冻结 diffusion observation 确实提供了额外可利用信息。这是当前论文叙事中最关键的因果消融。

### 当前方法与 SGD-JSCC

在同图像和同 canonical AWGN 噪声下，当前方法相对 SGD 论文协议上界：

- PSNR `+0.483309 dB`，95% CI `[+0.258829,+0.711956]`；
- MS-SSIM `-0.003916`，CI `[-0.006015,-0.001796]`；
- LPIPS `+0.079983`，CI `[+0.068804,+0.091894]`；
- T_cls failure 为 `29 vs 25`，差异 CI 跨 0。

结论是明确的 Pareto trade-off：**当前方法有更高的像素保真度，SGD 的生成感知质量更强**，两者都不能宣称全面支配另一方。更重要的是，SGD 的 released main+edge 已经占满 19,712 个实符号，四个 caption packet 至少还需 2,144 个未保护实符号，总计至少 21,856，超预算 `10.88%`。所以这里对 SGD 已是有利上界，不是严格同总物理码率排名。

## 给非专业读者的数据流程

1. **发端压缩图像。** DeepJSCC 把 256×256 RGB 图像压成 19,712 个实数；其中 80 个固定位置承载很小的语义载荷，其余 19,632 个位置承载图像信息。
2. **一次通过无线信道。** 全部符号共同经历同一 AWGN。接收端拿到的是一份带噪 latent，不会重发同一图片。
3. **先得到可靠锚点 B0/B1。** 接收端擦除 80 个载荷位置后解码出 B0，再由冻结的确定性接收器得到 B1。它保真、稳定，但低 SNR 下容易残留噪声和结构破坏。
4. **把同一信道观测解释成 diffusion 状态。** 若线性信道为 `y=x+n`，单位功率且每实维噪声方差为 `0.5/γ`，归一化后有 `x_t=√α·y`、`α=1/(1+0.5/γ)`；这正是扩散正向过程的形式。这里不是把通信噪声“类比”成 diffusion 噪声，而是在冻结 AWGN 约定下解析对应到具体累计噪声水平。
5. **只在低 SNR 走 6-step DDIM。** 1/4/7 dB 时，从该 matched state 生成第二份受控观测 D；D 与 B0 来自同一接收码字，不增加任何信道符号。
6. **融合而非让 diffusion 独断。** 冻结 fusion 同时看 B0、D、SNR、B0 的 Sobel/Laplacian 结构，将生成先验当作补充证据，而不是让生成器完全重画图片。
7. **高 SNR 精确停用 diffusion。** 13/19 dB 直接输出 B1；这是代码结构保证的逐像素相等，不依赖学习到一个可能失效的 gate。
8. **同时检查“好看”和“没认错”。** 每个输出都评估 PSNR、MS-SSIM、LPIPS，并用冻结分类器统计 semantic failure/new error/repair；视觉更好但语义错了不算真正提升。

## S28/S29 数值审计

S28 使用 batch=16 以运行 latent diffusion。其噪声 SHA、B1 类别预测和 failure 事件与 S20 全部一致，但重新计算的单样本 PSNR 最大相差 `0.000476837 dB`，超过预注册 `0.0001 dB` 技术阈值，因此 S28 的原始 `verdict` 诚实保留为 `NEGATIVE`，没有事后改阈值。

S29 随后注册为已知结果后的纯诊断，恢复 S20 原 batch=64 重放 960 行 B1。PSNR、MS-SSIM、LPIPS、预测、failure 和 noise SHA 的最大差全部为 **0**，6/6 checks PASS。这证明 S28 的唯一形式失败来自 batch shape 引起的浮点运算次序，不是人口、载荷或信道噪声错位。S29 不是新的盲测，不能增加统计独立性。

## 现在对项目水平的判断

当前已经形成一个可写成论文的最小技术闭环：

1. **有原理，不只是模块堆叠：** AWGN 到 diffusion state 的解析映射；
2. **有新的系统设计：** diffusion 是同码字派生的第二观测，与确定性 anchor 融合；
3. **有可靠性机制：** 高 SNR exact fallback，从结构上限制 semantic drift；
4. **有因果消融：** 等容量 matched control 证明收益来自 diffusion 信息；
5. **有跨总体复现：** 256-image、512-image fresh COCO 与 64-image×3-seed Imagenette 均显示相近低 SNR 增益；
6. **有外部定位和诚实码率审计：** 相对 SGD 是 fidelity/perception Pareto，而不是伪造全面胜出。

它的弱点也很明确：增益约 0.09--0.10 dB，属于稳定但不大的 receiver-side improvement；当前 Imagenette 仍是 policy-dev 而非官方最终测试；严格同总码率的 SGD 文本版本尚不存在；语义 failure 的改善在 S28 上没有达到显著性。因此现阶段更像一篇**方法逻辑完整、实验纪律很强的论文主线**，而不是已经可以宣称 SOTA 的终稿。

## 下一步优先级

1. 冻结当前方法，不再继续扫 gate、alpha 或小模块；
2. 增加一个可执行的严格总码率生成基线：要么重分配 SGD 的 19,712 符号并重训/适配，要么选择无需免费文本的公开 diffusion-JSCC；
3. 把当前方法和 B1/SGD 在更大、最终隔离的监督总体上做一次性统计评估；
4. 补 FID/KID 或 DISTS 等感知分布指标，但仍把 semantic drift 作为硬约束；
5. 将主贡献写成“matched observation + anchor fusion + exact reliability boundary”，而不是泛化成“信道好时不用 diffusion”。

## 可复现产物

- S28 配置：`configs/s28_external_sgd_positioning.yaml`
- S28 脚本：`scripts/s28_external_sgd_positioning.py`
- S28 输出：`outputs/external_baselines/ANALYSIS-S28-CURRENT-VS-SGD-001/`
- S29 配置：`configs/s29_s28_b1_exact_batch_audit.yaml`
- S29 脚本：`scripts/s29_s28_b1_exact_batch_audit.py`
- S29 输出：`outputs/analysis/ANALYSIS-S29-S28-B1-EXACT-BATCH-001/`
