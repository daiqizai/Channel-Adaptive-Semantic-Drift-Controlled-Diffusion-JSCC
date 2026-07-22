# 低码率 M3 最小闭环阶段结果（2026-07-15）

## 结论

本阶段取得了一个可复现但边界很清楚的成果：**低码率 B1 是强正结果；原始短链 diffusion 是明确负结果；经过活动坐标 posterior consistency、预算内语义载荷和 SNR 尾部路由后，diffusion 在独立 holdout 的 19 dB 高信噪比尾部取得了小而稳定的感知收益。**

因此项目没有放弃 diffusion，但也不能把当前 diffusion 写成全 SNR 的主要增益来源。现阶段最可靠的主体是低码率 B1；diffusion 只在高 SNR 下作为受约束的感知尾部启用。

## 1. 严格码率与数据闭环

- 输入：`256×256×3 = 196608` 个实数维度。
- 总预算：`19712` 个实坐标，即 `9856` 个复信道使用，精确 `CBR=0.050130208333333336`。
- DeepJSCC：`c=3` 稠密潜变量 `24576` 个实坐标，固定均匀选择 `19712` 个活动坐标。
- 语义载荷：10 维 UInt2，共 20 bit；BPSK×4，占 `80` 个实坐标。
- 图像活动坐标：`19632`；载荷坐标在图像解码前擦除，posterior consistency 也明确排除这 80 个位置。
- 缓存：COCO train2017 冻结 `10000/1000` 内部分割，五个 SNR `[1,4,7,13,19]`，共 `55000` 条重建；manifest SHA-256 为 `93ae3f3b...2de9`。
- 全过程未访问 Imagenette official validation，未联网、未下载。

低码率 B0 的 11000 图均值为：

| SNR | B0 PSNR |
|---:|---:|
| 1 | 24.1888 |
| 4 | 25.5855 |
| 7 | 26.4744 |
| 13 | 27.2784 |
| 19 | 27.5063 |

## 2. B1：明确正结果

`EXP-S16-B1-001` 完整使用 10000 图训练、1000 图验证和冻结的 S13 结构/gate。best checkpoint：

`outputs/EXP-S16-B1-001/checkpoints/best.pt`

SHA-256：`7a295976105a9c43c25604c9070e676d25512c7a09b5c50655b6671477b7615a`

| SNR | B0 PSNR | B1 PSNR | ΔPSNR | B0 LPIPS | B1 LPIPS | ΔLPIPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.0723 | 25.4540 | +1.3817 | 0.4108 | 0.2382 | -0.1726 |
| 4 | 25.4598 | 26.5650 | +1.1053 | 0.3349 | 0.2040 | -0.1308 |
| 7 | 26.3415 | 27.3278 | +0.9863 | 0.2860 | 0.1784 | -0.1077 |
| 13 | 27.1373 | 28.0237 | +0.8864 | 0.2413 | 0.1573 | -0.0840 |
| 19 | 27.3607 | 28.1917 | +0.8311 | 0.2285 | 0.1529 | -0.0756 |
| 平均 | — | — | **+1.0381** | — | — | **-0.1141** |

五个 SNR 的 PSNR 与 LPIPS 全部同向改善。旧的“B1 必须保持 B0 top-1 才接受”规则在低码率下会误拒大量修复，因为 B0 自身已经不可靠；它不能继续作为最终裁判。

## 3. 原始 diffusion：明确负结果

`EXP-S16-DIFF-001` 在新 B1 anchor 上重新训练，并没有复用高码率输出充数。best checkpoint SHA-256 为 `44915d7e...8a`。

| SNR | B1 PSNR | raw diffusion PSNR | ΔPSNR | B1 LPIPS | raw LPIPS | ΔLPIPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 25.4540 | 24.4420 | -1.0120 | 0.2382 | 0.2603 | +0.0221 |
| 4 | 26.5650 | 25.7011 | -0.8639 | 0.2040 | 0.2236 | +0.0195 |
| 7 | 27.3278 | 26.7239 | -0.6039 | 0.1784 | 0.1941 | +0.0157 |
| 13 | 28.0237 | 27.5578 | -0.4659 | 0.1573 | 0.1703 | +0.0130 |
| 19 | 28.1917 | 27.9418 | -0.2499 | 0.1529 | 0.1624 | +0.0095 |

raw diffusion 的五档 PSNR 和 LPIPS 全部变差；伪语义新增错误/修复为 `318/139`。预注册检查除采样步数外全部失败，正式判为 `NEGATIVE`。该失败表明沿用高码率 gate 会在强 B1 上过修复，当前训练目标也没有学出有效的低码率感知方向。

## 4. 严格 8×5 闭环与 posterior

### 第一组冻结 pilot

`ANALYSIS-S16-LOWRATE-M3-STAGE-001` 使用外部作者工作点相同的 8 张图、五个 SNR 和 base seed `20260730`。

- 40/40 行 payload 零比特错误。
- 无载荷 exact B0：PSNR/LPIPS `25.9260/0.2872`，失败 3。
- 严格载荷 B0：`25.8765/0.2887`，失败 3；80 个载荷坐标的真实代价约 `-0.0495 dB/+0.00156 LPIPS`。
- B1：`26.8461/0.1714`，失败 0。
- raw diffusion：`26.1961/0.1888`，失败 0。
- posterior：`26.3329/0.1764`，失败 0；normalized consistency `0.13253→0.11138`，40/40 行下降。
- 原三重语义门接受 21/40 行，但 final 为 `26.6434/0.1739`，仍比 B1 差 `-0.2027 dB/+0.00250 LPIPS`，所以整体失败。

这一组结果说明 posterior 能稳定把 raw 候选拉回信道可行域，但语义一致性门只能防 hard drift，不能判断微小感知收益；低中 SNR 不应启用当前 diffusion。

### 独立高 SNR 尾部 holdout

根据第一组只在 19 dB 出现非伤害感知信号的事实，在任何 holdout 输出前冻结策略：`1/4/7/13 dB → B1`；`19 dB → 仅允许通过三重语义门的 posterior`。随后使用哈希排名第 9–16 的另一组 8 张 clean policy-dev 图和新 base seed `20260731`，得到 `ANALYSIS-S16-LOWRATE-M3-TAIL-HOLDOUT-001`。

- 预注册 7 项检查全部通过。
- payload BER `0`，40/40 整向量精确恢复。
- B0→B1：PSNR `26.7321→27.7594`（`+1.0273 dB`），LPIPS `0.24781→0.13811`，失败 `3→0`。
- raw diffusion 再次为负：相对 B1 `-0.6099 dB/+0.01510 LPIPS`，并出现 1 个新增错误。
- posterior consistency 40/40 行下降，均值 `0.13114→0.10951`。
- 高 SNR 尾部 final 相对 B1：全五档平均 `-0.0099 dB/-0.000389 LPIPS`，失败保持 0，新错误/修复 `0/0`。
- 19 dB 单档：PSNR `28.9632→28.9137`（`-0.0495 dB`），LPIPS `0.113956→0.112010`（`-0.001945`），失败仍为 0。

这只授权“小范围高 SNR 感知尾部可行”的方向性结论，不授权全 SNR diffusion 成功或最终 semantic-safe 声明。

## 5. 与 SGD-JSCC 的当前关系

在第一组相同 8×5 作者工作点 pilot 上，严格载荷 B1 的 PSNR 为 `26.8461 dB`，SGD-JSCC 免费/无误 caption 论文协议上界为 `26.8389 dB`，两者在 PSNR 上基本持平（B1 高约 `0.0073 dB`）。但 SGD 的 LPIPS 为 `0.07856`，显著优于 B1 的 `0.17140`；SGD 的 MS-SSIM 也更高。

所以当前可以说：**项目已经有希望在严格低码率下追平甚至超过 SGD 的失真指标，但还没有在感知质量上超过它。** 更不能直接声称优于论文，因为 SGD 这组结果的 caption 免费且无误，而本项目的 80 个语义坐标已计入总预算。

## 6. 下一步

1. 冻结当前低码率 B1，作为新的主 baseline，不再围绕旧 B0 一致性路由。
2. diffusion v2 只改低码率增量目标：增加 no-op/anchor regularization，显式降低低中 SNR gate，并把高 SNR 感知尾部作为首要验证点；不改变 AWGN 主线和总码率。
3. 在更大、独立、多 seed 的 Imagenette train holdout 上验证 19 dB 尾部，不能用本次 8 张结果做最终声明。
4. 补严格同预算 SGD 文本传输版本；免费 caption 版本继续只作论文协议上界。
5. 对外主张暂时收敛为：`exact-rate B1 restoration + in-budget semantic payload + active-coordinate posterior-constrained high-SNR diffusion tail`。

## 7. 主要产物

- 预注册：`reports/lowrate_m3_minimal_closure_preregistration_2026-07-15.md`
- 缓存：`outputs/eval/s16_lowrate_m3_exact19712_uint2r4_coco10k_1k/`
- B1：`outputs/EXP-S16-B1-001/`
- diffusion 负结果：`outputs/EXP-S16-DIFF-001/`
- 第一组闭环：`outputs/analysis/ANALYSIS-S16-LOWRATE-M3-STAGE-001/`
- 独立尾部 holdout：`outputs/analysis/ANALYSIS-S16-LOWRATE-M3-TAIL-HOLDOUT-001/`
