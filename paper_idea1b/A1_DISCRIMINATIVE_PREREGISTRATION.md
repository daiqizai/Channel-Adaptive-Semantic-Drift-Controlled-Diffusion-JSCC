# A1 判别式主表预注册

日期：2026-07-23
实验ID：`ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001`

## 验收问题

在冻结COCO训练权重、方法相同原生256 tile、相同逐图actual CBR、相同canonical paired-real AWGN下，S33相对SwinJSCC official Base-SA和capacity-matched CM-SA分别是显著超过、追平非劣还是劣于？

总体措辞继续取两条Swin臂中对S33更不利者。PSNR使用0.10 dB非劣margin；次指标冲突时写Pareto，不包装成全面超过。

## 数据与范围

- Kodak：24张，5 SNR × 3 channel seeds。
- CLIC2020 test：428张，5 SNR × 1 channel seed。
- official Imagenette validation继续封存。
- 本阶段不加载DiffJSCC、SGD或任何refiner。

## 原生大图与码率

三臂使用A0冻结的同一非重叠256 tile坐标。边界采用`numpy reflect`，单轴有效长度为1时退化为edge padding；输出只保留每tile有效区域并拼回原像素网格。

每个tile都真实发送`16,384 real`。每图报告：

```text
real symbols = tile_count × 16,384
complex uses = real symbols / 2
actual CBR = complex uses / (3HW)
```

所以三臂逐图码率严格相同。Kodak恰为`1/24`；CLIC因边界padding可能高于`1/24`，但三臂在同一图上仍完全相等，不把nominal CBR冒充actual CBR。

每张图、每个seed/SNR生成一条长度为该图总real symbols的canonical standard-normal序列，三臂使用同一序列和tile切片。

## 指标

- Kodak/CLIC：PSNR、MS-SSIM、LPIPS、DISTS。
- 无标签语义：原图-重建OpenCLIP ViT-B/32 image cosine。
- CLIC：每个SNR/seed分开计算FID/KID；不把SNR或方法图混在一个分布里。
- 配对差值以source image为bootstrap cluster，10,000次，报告95% CI。

DreamSim当前环境未安装，不属于smoke成功条件；若在正式summary前加入，必须先冻结实现、权重SHA和identity sanity，不能根据结果选择是否加入。

## 分阶段执行

1. preflight：SHA、checkpoint epoch/arm、参数量、16,384-real、manifest与封存边界。
2. smoke：Kodak一张 + CLIC最大2048×2048一张，固定7 dB/seed 20260748；逐臂报告完整图wall time、GPU model time、峰值allocated/reserved、tile数、actual CBR、输出shape与功率误差。
3. smoke通过并先向用户报数后，运行Kodak全量。
4. Kodak完成后运行CLIC 1 seed。
5. 全部完成后统一计算指标、CI和保守双臂verdict。

所有输出使用新ID，禁止覆盖A0或既有S33/S34A资产。
