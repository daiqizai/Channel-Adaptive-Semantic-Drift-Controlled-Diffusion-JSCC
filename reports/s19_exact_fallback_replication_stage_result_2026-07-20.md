# S19 强融合 + exact-B1 fallback 跨总体复现结果（S26）

日期：2026-07-20。分析 ID：`ANALYSIS-S26-S19-XF-REPLICATION-001`。

## 一句话结论

**取得阶段性明确正结果：低 SNR 使用冻结 S19 diffusion fusion，高 SNR 强制回到 B1，在另一批 256 图×5 SNR 上同时保留了接近 S19 的质量增益、diffusion 因果优势和 exact fallback，9/9 预注册检查全部通过。**

当前方法可以概括为：

> 信道差时，让同一接收观测产生的 matched diffusion 为联合恢复器提供生成先验；信道好时不再让生成模型参与，逐像素精确返回保真 B1。

## 核心结果

目标总体为 S21/S23 使用的另一批 COCO holdout，共 256 张图、每张 5 个 SNR、1,280 个配对观测。S19 control/fusion checkpoint 在运行前冻结，不访问目标 selection。

| 方法 | PSNR | MS-SSIM | LPIPS | majority failure |
|---|---:|---:|---:|---:|
| B1 | 27.567717 | 0.943591 | 0.183951 | 744/1280 |
| routed matched control | 27.595498 | 0.944364 | 0.179391 | 734/1280 |
| **routed S19 fusion** | **27.660984** | **0.945779** | **0.176290** | **720/1280** |

routed fusion 相对 B1：

- PSNR `+0.093267 dB`，source-image cluster 95% CI `[+0.087945,+0.098806]`；
- MS-SSIM `+0.002188`，CI `[+0.001972,+0.002412]`；
- LPIPS `-0.007661`，CI `[-0.008438,-0.006915]`；
- majority failure `744→720`，failure-rate difference CI `[-0.03203,-0.00547]`；
- 相对 B1 为 `27 new / 51 repair`，净修复显著，但不是零 new error。

## diffusion 不是视觉包装

matched control 与 fusion 都是 `450,115` 参数，来自同一个 B1 expansion，训练时使用相同 batch、crop、flip 和预算；control 的第二 RGB 输入只是 B0，fusion 才读取 matched diffusion。

routed fusion 相对 routed control：

- PSNR `+0.065486 dB`，95% CI `[+0.061088,+0.069994]`；
- MS-SSIM `+0.001414`，CI `[+0.001269,+0.001570]`；
- LPIPS `-0.003100`，CI `[-0.003670,-0.002528]`。

三项质量 CI 都显著有利，因此额外收益不能解释为“多加了一个同容量 CNN”。这在第二个 population 上复现了 S19 的 diffusion-information 因果结论。

## 分 SNR 结果与 exact fallback

| SNR | fusion−B1 PSNR | fusion−B1 LPIPS | fusion−control PSNR | 路由 |
|---:|---:|---:|---:|---|
| 1 dB | +0.141105 | -0.010769 | +0.094364 | S19 fusion |
| 4 dB | +0.154616 | -0.014165 | +0.106808 | S19 fusion |
| 7 dB | +0.170612 | -0.013372 | +0.126256 | S19 fusion |
| 13 dB | 0 | 0 | 0 | exact B1 |
| 19 dB | 0 | 0 | 0 | exact B1 |

13/19 dB routed fusion/control 相对 B1 的最大逐像素差均为 0。这不是网络学出来的近似 gate，而是代码结构直接选择 B1，因此不会出现 S19 原始方案的高 SNR 负迁移。

## 这个结果意味着什么

S25 说明 S23 的弱 feature direction 没有足够逐图上限；S26 则说明问题不在“融合思路本身”，而在表示强度。更强的 S19 joint-fusion representation 已经能提供约 `0.093 dB` 的 aggregate PSNR 增益，而简单、可解释的 SNR 可信域可以切掉高 SNR 风险区。

因此当前项目最好的方法不再是 S19 或 S23 二选一，而是：

`S26 = S19 low-SNR fusion + exact-B1 high-SNR fallback`。

它已经同时满足：严格同码率、diffusion 因果增益、有意义质量效应量、三项质量显著、净语义修复和高 SNR exact fallback。

## 必须保留的边界

- 目标图片总体此前已用于 S21/S23，B1/S23 outcome 已知；只是 S19 checkpoint 在该总体的输出未知，且本轮没有目标 selection。因此这是可信的 frozen cross-population replication，但还不是完全 pristine final population。
- 语义结果来自 AlexNet/ResNet18/MobileNetV3-Small 相对原图预测的一致性，不是 COCO 人工真值；`27 new` 仍说明不能宣称绝对 semantic-safe。
- S26 没有减少低 SNR 6-step diffusion 的端到端计算量；它只在 13/19 dB 完全跳过 diffusion。
- 与 SGD-JSCC 的现有数字不在同一图片总体和 side-information 合同，不能直接宣布 SOTA。

下一步论文级验证只需要收敛地做两件事：一次完全 fresh 256/512-image population 的冻结复现，以及 common contract 下的外部方法对比。不要再回到 S23 alpha/controller 扫描。

## 产物

- summary SHA：`f104ea4a8c870817c2e99b3c0aeb97fb20fa8da4e6e34cf0054004a37f1a376c`
- per-sample CSV SHA：`0f29efc31922da8c3a002a8b3e3dee35da1d509dfe817c93b2a98aa0ab87dd0c`
- 本轮无训练、无联网、无下载、official Imagenette validation 未访问。
