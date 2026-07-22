# 16,384-real Strong JSCC 严格等码率阶段结果

日期：2026-07-21
阶段：S33 / S33b / `ANALYSIS-S33-STRONG-JSCC-16384-COMPARISON-001`
结论：**严格等码率下显著超过 author-JSCC；最高优先级论文 gate 通过。**

## 一、最重要结论

在双方都严格使用 `16,384 real`（`8,192 complex channel uses`）的同图、同 canonical AWGN noise prefix、同 64 图×3 channel seeds×5 SNR policy-dev 协议下：

- strong−author-JSCC 聚合 PSNR=`+0.479929 dB`；source-image cluster 95% CI=`[+0.370006,+0.598197]`。
- 该 CI 下界大于 0，按用户预注册规则判定为 **显著超过**，不是“追平/非劣”。
- 聚合 MS-SSIM=`+0.006616`，95% CI=`[+0.005856,+0.007375]`；聚合 LPIPS=`-0.008357`，95% CI=`[-0.010343,-0.006339]`。三项质量指标的聚合 CI 都显著有利于 strong。
- T_cls failure 为 strong `9/960`、author `22/960`；failure-rate delta=`-0.013542`，95% CI=`[-0.027083,-0.003125]`。strong 相对 author 有 `4 new / 17 repair` 行，对应 `1/6` 个 source clusters。

因此 S32 的“项目 19,712-real ceiling 内胜出”已经升级为 **16,384-real 严格等码率胜出**。这通过了 strong backbone 小论文成立所需的最高优先级经验门槛，但 S33 population 仍是已知 policy-dev，不是 independent final test；S34 消融与最终封存测试仍是投稿前必需证据。

## 二、冻结方法与训练合同

新 strong 与 author-JSCC 的关键公平性如下：

| 项目 | S33 strong | author-JSCC |
|---|---:|---:|
| 实传实符号 | `16,384` | `16,384` |
| 复信道使用 | `8,192` | `8,192` |
| 相对 256×256 RGB 源 CBR | `1/24` | `1/24` |
| 可训练参数 | `31,028,163` | 约 `31.289M` |
| mask/padding/裁剪 | 0 | 0（作者 C16 latent） |
| 发送 side information | 0 | 0（仅比较纯 author-JSCC） |

S33 strong 原生 latent=`64x16x16`，不是从更宽 latent 裁剪。训练严格使用用户确认的：

- 随机初始化；
- FP32 12 epochs：前 4 epochs 主阶段，后 8 epochs只加载主阶段 best model、fresh optimizer 的低学习率 continuation；
- 每张训练图从 `[1,4,7,13,19] dB` **离散均匀采样**；
- COCO train2017、MSE-only，checkpoint 只由固定 COCO val2017 512 图的五档 aggregate PSNR/MS-SSIM 选择；
- 未使用连续 `Uniform[1,19]`，未使用 LPIPS、语义标签、S32 outcome 或 official Imagenette validation 选 checkpoint。

连续 SNR 只能写 future work，不能写成当前反超原因。

## 三、训练曲线和运行审计

### 3.1 随机初始化主阶段

| epoch | train MSE | 五档 PSNR | MS-SSIM |
|---:|---:|---:|---:|
| 0 | 0.005584 | 26.062294 | 0.934945 |
| 1 | 0.002624 | 27.459733 | 0.952470 |
| 2 | 0.002099 | 28.280950 | 0.959113 |
| 3 | 0.001881 | 28.587876 | 0.961212 |

epoch 2 完整 checkpoint 落盘后，后台进程曾被外部终止；GPU 空闲且 `STATE.json` 停在 `next_epoch=3`。config、snapshot、checkpoint 三个 SHA 一致，checkpoint epoch=`2`、global step=`11,091`，随后按同配置 `--resume` 只重跑未落盘的 epoch 3。没有覆盖或重复正式 history 行。

主阶段 best epoch 3 SHA-256=`b698797f93f56cd6d1617ee18fdd39493fe08e58e994b21ccb059ffb19ce26c4`。

### 3.2 低学习率续训

| epoch | train MSE | 五档 PSNR | MS-SSIM |
|---:|---:|---:|---:|
| 0 | 0.001924 | 28.510495 | 0.961273 |
| 1 | 0.001816 | 28.631286 | 0.962248 |
| 2 | 0.001746 | 28.938953 | 0.964118 |
| 3 | 0.001687 | 29.028286 | 0.965076 |
| 4 | 0.001639 | 29.235818 | 0.965742 |
| 5 | 0.001605 | 29.319864 | 0.966483 |
| 6 | 0.001584 | 29.389344 | 0.966547 |
| 7 | 0.001572 | **29.415098** | **0.966782** |

续训 epoch 0 相对初始化回落 `-0.077381 dB`，此负点完整保留；epoch 1 开始超过初始化，最终 best 为 epoch 7。最终 checkpoint：

- 路径：`outputs/train/EXP-S33B-STRONG-JSCC-16384-FP32-001/checkpoints/best.pt`
- SHA-256：`2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`
- COCO per-SNR PSNR：`27.311886/28.571358/29.544483/30.634386/31.013377 dB`
- COCO aggregate：`29.415098 dB / 0.966782 MS-SSIM`
- 最大归一化功率误差：`2.3842e-7`
- 12 个 epoch 的纯训练用时合计 `11,110.56 s`，约 `3 h 05 min`；不含 smoke、外部中断等待、外部比较和报告。

## 四、严格等码率聚合对比

primary 继续使用 S30/S32 的 floor-uint8 口径：

| 方法 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | T_cls failure ↓ |
|---|---:|---:|---:|---:|
| S33 strong `16,384 real` | **30.466064** | **0.969708** | **0.119985** | **9** |
| author-JSCC `16,384 real` | 29.986135 | 0.963092 | 0.128342 | 22 |
| strong−author | **+0.479929** | **+0.006616** | **-0.008357** | **-0.013542 rate** |
| 95% CI | `[+0.370006,+0.598197]` | `[+0.005856,+0.007375]` | `[-0.010343,-0.006339]` | `[-0.027083,-0.003125]` |

float-output sensitivity 的平均 float−uint8 为 `+0.044682 dB/+0.0000746 MS-SSIM/+0.000453 LPIPS`；主结论保持统一 uint8 口径，不用 float 数字抬高排名。

## 五、分 SNR 结果和边界

| SNR | strong PSNR / LPIPS / MS-SSIM / fail | author PSNR / LPIPS / MS-SSIM / fail | PSNR delta (95% CI) | 预注册 PSNR 判定 |
|---:|---:|---:|---:|---|
| 1 | 28.1617 / 0.1637 / 0.9464 / 6 | 27.1954 / 0.1934 / 0.9293 / 10 | `+0.9663 [0.8814,1.0571]` | 显著超过 |
| 4 | 29.5217 / 0.1317 / 0.9631 / 2 | 28.7500 / 0.1469 / 0.9534 / 8 | `+0.7717 [0.6754,0.8688]` | 显著超过 |
| 7 | 30.5933 / 0.1137 / 0.9729 / 1 | 30.0672 / 0.1197 / 0.9677 / 3 | `+0.5261 [0.4270,0.6254]` | 显著超过 |
| 13 | 31.8095 / 0.0976 / 0.9817 / 0 | 31.6783 / 0.0954 / 0.9805 / 1 | `+0.1312 [0.0044,0.2688]` | PSNR 显著超过；LPIPS 显著更差 |
| 19 | 32.2441 / 0.0933 / 0.9844 / 0 | 32.2398 / 0.0864 / 0.9846 / 0 | `+0.0043 [-0.1615,0.2027]` | CI 下界 `<-0.10`，按冻结规则判为劣于/未过非劣；LPIPS 显著更差 |

13 dB 的 LPIPS delta=`+0.002205`，95% CI=`[+0.000303,+0.004231]`；19 dB 为 `+0.006905`，CI=`[+0.004852,+0.009055]`。因此虽然五档聚合三项质量指标都显著优于 author，不能写成“每个 SNR、每个指标都全面超过”。更准确的论文表述是：**严格等码率下的聚合显著优势主要由 1/4/7 dB low-to-mid-SNR regime 驱动；13/19 dB 仍有 perceptual boundary，19 dB 未通过逐档 PSNR 非劣 gate。**

## 六、独立 artifact 审计

派生脚本在不重跑模型的情况下验证：

- `960/960` 行、`960/960` 唯一 `(sample,seed,SNR)` key；
- author prediction/failure/PSNR/MS-SSIM/LPIPS 与冻结 S30 CSV `960/960` 逐字段一致；
- 先重建完整 `19,712-real` canonical noise 并逐行匹配 S30 SHA，再取前 `16,384` 坐标；完整/prefix noise `960/960` 均匹配；
- checkpoint/config/snapshot/CSV 全部 SHA 固定；无 non-finite；official Imagenette validation 未访问。

核心哈希：

- `per_sample.csv`：`5e585ca5a513434512f985ebe43a3e97ff77326920256a59fc3233703fe7dbe1`
- immutable runner `summary.json`：`eda17b9027ffc4524d86ac07c5fcf8383891d69e6ccedc704c2d96289d4ac1ce`
- `post_analysis.json`：`d7a89ef2f2f59a8385c346ef72ddfa012cdf07aa8be72aada5c3e51113e8bdfc`

原 runner 是从 S32 复用，immutable `summary.json` 的 `claim_boundary` 说明文字仍残留 “after S31 checkpoint selection”；冻结 config、analysis ID、checkpoint 和全部指标均是 S33。未覆盖原 summary，另由 `post_analysis.json` 显式更正标签；源 runner 已改为后续从 config 读取 claim scope。该问题只影响说明文字，不影响数值或 verdict。

另记录两项输出前诊断失误：标准库 unittest 曾对 pytest 风格 `test_strong_jscc.py` 发现 0 tests，随后直接调用四个测试函数全部 PASS；续训只读 SHA helper 首次调用传入 string 而非 Path，修正后通过。比较 config 的 split SHA 曾误填为 T_cls SHA，静态审计在任何 preflight/比较输出前发现并修正。三者均未污染正式结果。

## 七、阶段判定和停止点

S33 的最高优先级验收问题回答为：

> **是。在严格 `16,384 real` 等码率下，strong backbone 聚合 PSNR 显著超过 author-JSCC；95% CI 下界为 `+0.370 dB`，高于 0，也远高于 `-0.10 dB` 非劣 margin。**

而且不是 PSNR 单轴胜出：聚合 MS-SSIM、LPIPS 和 failure 的 CI 也显著有利。由此，strong backbone 作为独立论文方向的核心经验 gate 已成立。

但当前只能声称 known policy-dev exact-rate positioning。下一步按既定顺序应是 S34 的 SNR conditioning / 四级结构 / 离散五档训练合同消融；只有方法、消融和 claim 全部冻结后才运行 S36 official Imagenette validation。本轮按用户要求在 S33 完成后停止，未启动 S34、S35 或 S36。
