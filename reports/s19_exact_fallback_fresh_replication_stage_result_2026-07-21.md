# S19 强融合 + exact-B1 fallback 完全新总体复现结果（S27）

日期：2026-07-21。分析 ID：`ANALYSIS-S27-S19-XF-FRESH-001`。

## 结论

**S26 的阶段性正结果已在完全 fresh 的 512-image population 上高精度复现，9/9 预注册检查全部通过。**

当前主方法正式固定为：

> 低 SNR（1/4/7 dB）使用同一信道观测驱动的 S19 matched-diffusion fusion；高 SNR（13/19 dB）跳过生成增强并逐像素精确返回 B1。

## 数据独立性

S27 在任何新图输出产生前冻结 seed、512 张数量、checkpoint、route 和成功门槛。随后从本地 COCO train2017 排除 S16/S18/S19/S21 共 `22,536` 个唯一 source path/SHA，并排除 val2017 同名及新总体内部重复 SHA。

- path overlap：0；
- SHA overlap：0；
- pristine holdout：512 张；
- 5 个 SNR，共 2,560 个观测；
- 2,560 个 B0 和 1,536 个低 SNR diffusion cache 全部通过 PNG/manifest 检查；
- 不存在 selection、重训、换 seed 或结果后改规则。

## 主要指标

| 方法 | PSNR | MS-SSIM | LPIPS | majority failure |
|---|---:|---:|---:|---:|
| B1 | 27.323569 | 0.943408 | 0.188371 | 1561/2560 |
| routed matched control | 27.350433 | 0.944204 | 0.183942 | 1537/2560 |
| **routed S19 fusion** | **27.416232** | **0.945718** | **0.180449** | **1517/2560** |

routed fusion 相对 B1：

- PSNR `+0.092662 dB`，source-image cluster 95% CI `[+0.089147,+0.096313]`；
- MS-SSIM `+0.002310`，CI `[+0.002149,+0.002482]`；
- LPIPS `-0.007922`，CI `[-0.008465,-0.007398]`；
- majority failure `1561→1517`，差值 CI `[-0.02813,-0.00664]`；
- 相对 B1 为 `60 new / 104 repair`，净修复显著，但仍不是绝对零 new error。

## diffusion 的独立贡献再次复现

routed fusion 与 matched control 都为 `450,115` 参数，唯一区别是第二 RGB observation 是否为 matched diffusion：

- PSNR `+0.065799 dB`，95% CI `[+0.062673,+0.068775]`；
- MS-SSIM `+0.001515`，CI `[+0.001410,+0.001624]`；
- LPIPS `-0.003494`，CI `[-0.003884,-0.003110]`。

这是继 S19、S26 后第三份 diffusion-information 证据，也是第一份完全新图总体证据。三项质量 CI 均显著有利。

## 分 SNR

| SNR | fusion−B1 PSNR | fusion−B1 LPIPS | fusion−control PSNR | route |
|---:|---:|---:|---:|---|
| 1 dB | +0.135781 | -0.011313 | +0.088610 | S19 fusion |
| 4 dB | +0.153492 | -0.015094 | +0.108564 | S19 fusion |
| 7 dB | +0.174039 | -0.013204 | +0.131823 | S19 fusion |
| 13 dB | 0 | 0 | 0 | exact B1 |
| 19 dB | 0 | 0 | 0 | exact B1 |

13/19 dB 最大逐像素差为 0；五档 PSNR 均不低于 B1。S26 与 S27 aggregate PSNR 增益分别为 `+0.093267/+0.092662 dB`，差异只有约 `0.00060 dB`，说明结果高度稳定。

## 当前论文判断

现在已经可以把内部主结论写得更强：

1. channel-matched diffusion 在三个 population 上均提供等容量 B0-only CNN 无法替代的信息；
2. 低 SNR 增益稳定在 `+0.136～+0.174 dB`；
3. 高 SNR exact fallback 从结构上消除生成先验负迁移；
4. 质量与辅助语义 failure 都有显著净改善。

仍不能声称绝对 semantic-safe 或外部 SOTA，因为有 `60` 个 majority new error，且尚未在同一 population、同一 side-information 预算下与 SGD-JSCC 直接比较。下一步应只做外部统一定位和论文汇总，不再调整内部方法。

## 产物

- source manifest SHA：`fa1a2ae172fd06f0a00efcd31d482b6848c6775ffbd25d2c20b027474b910caa`
- cache manifest SHA：`2f5b6ec34a681e3725e05378b47fdc54eb80a4fd0515a4437a5f7abab8e556fa`
- summary SHA：`4d0d1426962be2c92fc3b85993b68e42935e6670d7ae4c3b9d5439660e8864f1`
- per-sample SHA：`d20fbf0e90870070b84db444b5a3db2455f2d2722f4fa094ae7ec190136a4dc1`
- 本轮无训练、无联网、无下载，official Imagenette validation 未访问。
