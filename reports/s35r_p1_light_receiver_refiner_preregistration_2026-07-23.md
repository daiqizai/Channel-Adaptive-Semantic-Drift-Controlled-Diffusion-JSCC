# S35R-P1：冻结 S33 上的轻量接收端生成式精修头预注册

日期：2026-07-23
实验 ID：`EXP-S35R-P1-LIGHT-RECEIVER-REFINER-001`
状态：只完成预注册，等待用户确认；smoke 与训练均未获授权。

## 一、方法一句话

冻结严格 `16,384 real` 的 S33 编码器/解码器，在接收端对 S33 RGB 重建做一个几百万参数、SNR-conditioned 的确定性生成式残差精修；它不发送新符号，用感知+对抗目标改善纹理，同时用 MSE 和 S33-anchor 一致性限制语义漂移。

这不是 diffusion sampler，也不使用文本或外部生成大模型。论文主贡献仍锚定“严格等码率下的代价—质量—可靠性公平刻画”，P1 是低代价载体和未来主方法 b 的内部 baseline/ablation。

## 二、数据流与 exact-rate

```text
256×256 RGB
  → 冻结 S33 encoder
  → 64×16×16 = 16,384 real
  → 单位功率归一化 + canonical paired-real AWGN
  → 冻结 S33 decoder
  → x_hat: 3×256×256 RGB
  → lightweight refiner(x_hat, normalized SNR)
  → x_refined: 3×256×256 RGB
```

refiner 完全位于接收端，额外信道符号、side information、mask、prefix、padding均为0。S33 checkpoint 永久冻结，SHA-256：

`2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`

## 三、P1 generator 与 discriminator

Generator 冻结为三尺度 residual U-Net：

- base width=48，宽度为48/96/192；
- encoder每尺度2个 residual block，bottleneck 4个，decoder每尺度2个；
- GroupNorm+SiLU，stride-2 conv 下采样，bilinear+3×3 conv 上采样；
- 归一化 SNR `s=(SNR-1)/18` 经 FiLM 注入每个 residual block；
- 最后3×3卷积零初始化；
- 输出为 `clip(x_hat + 0.10*tanh(residual),0,1)`。

推理参数目标为2M–6M，精确值必须由1-batch smoke 报告；超出区间即停止，不允许临时加宽。

训练期使用小型 conditional 70×70 PatchGAN：输入为 `[x_hat, clean/refined]` 六通道，base width=32、四级32/64/128/256，hinge loss。判别器只用于训练，不计入部署参数和延迟，但必须单独报告。

## 四、训练合同

- 数据：与 S33 相同 COCO train2017、256×256、crop scale `[0.6,1.0]`、随机水平翻转；
- SNR：逐图从 `[1,4,7,13,19] dB` 离散均匀采样，不能写成连续 SNR；
- 信道：逐图单位功率、paired-real half-variance AWGN；
- S33：`eval()`、全参数冻结、无梯度；
- FP32，12 epochs 硬上限，不因结果延长；
- effective batch=32，暂定 microbatch=8、accumulation=4；只允许 smoke 因显存调整 microbatch，effective batch 不变；
- G/D 均 Adam，lr=`1e-4`，betas=`(0.5,0.999)`。

Generator loss 冻结为：

```text
L_G =
  1.0 * LPIPS(x_refined, x)
  + 5.0 * MSE(x_refined, x)
  + 0.01 * hinge_GAN(x_refined | x_hat)
  + 0.5 * L1(x_refined, x_hat)
```

LPIPS AlexNet 权重只用本地缓存。anchor 是冻结 S33 的接收端 RGB，而不是原图；最后一项直接惩罚偏离 S33 的修正。

## 五、checkpoint selection 与泄漏隔离

每 epoch 在冻结 COCO val512 上评估。只在平均 PSNR 相对 S33不低于 `−0.10 dB` 的 epoch 中选 LPIPS最低者，平局取PSNR更高者。若没有 epoch 满足，P1直接记负，不回选“看起来最好”的模型。

64图 policy-dev、三个 channel seeds 和 official validation 均不参与 checkpoint selection。只有 checkpoint 冻结后，才允许一次性跑既定960键的 go/no-go。

## 六、go/no-go

冻结总体：64图×3 seed×5 SNR，共960个 paired keys。报告 PSNR、MS-SSIM、LPIPS、`T_cls` failure、new error、repair，aggregate 和 per-SNR 都要给。

10,000次 source-image cluster bootstrap，把同一源图的3 seed×5 SNR整体重采样。定义差值均为 `refiner−S33`：

- LPIPS 显著改善：95% CI上界 `<0`；
- PSNR在0.10 dB margin下非劣：95% CI下界 `>−0.10 dB`；
- semantic failure不显著上升：failure-rate差的95% CI下界 `≤0`。

P1 的 `GO` 要求三项同时满足。若 LPIPS连显著改善都没有，或任一可靠性/PSNR gate失败，就记录负结果，停止向更复杂主方法 b 升级，回到“代价—质量—可靠性刻画为主”的分析型 letter。

## 七、分阶段放行

当前不运行任何代码。用户确认后仅做一次真实 COCO microbatch 的：

- G/D forward、backward、optimizer step；
- finite loss/gradient；
- exact `16,384 real` 与输出shape/range；
- G/D精确参数量；
- peak allocated/reserved VRAM；
- microbatch step time和12 epochs工期估算；
- checkpoint strict round-trip。

smoke 报告后再次等待用户明确放行正式训练。official Imagenette validation 全程封存。
