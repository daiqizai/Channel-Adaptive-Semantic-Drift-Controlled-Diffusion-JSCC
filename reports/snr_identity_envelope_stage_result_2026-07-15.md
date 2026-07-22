# SNR-Conditioned Identity Envelope 阶段结果（2026-07-15）

## 结论

本阶段首次把 channel-state-matched latent diffusion 从“机制显著但高 SNR 部分失败”推进为预注册 **PASS**：在一个与旧 11,000 图完全去重的新 COCO population 上，selection 冻结的 `hard_identity_7db` policy 在 1/4/7 dB 使用完整 decoder-aware diffusion，在 13/19 dB 于 codeword 层严格回到同噪声 B0。一次性 256 图×5 SNR holdout 上：

- selected − full diffusion PSNR：`+0.015642 dB`，image-cluster 95% CI `[+0.014230,+0.016915]`；
- selected − B0 PSNR：`+0.189717 dB`，95% CI `[+0.170601,+0.210902]`；
- selected − B0 LPIPS：`-0.036284`，95% CI `[-0.039420,-0.033268]`；
- 1/4/7 dB PSNR delta：`+0.677172/+0.240940/+0.030472 dB`；
- 13/19 dB PSNR/LPIPS delta：严格 `0/0`；
- 低/中 SNR 保留 full diffusion PSNR gain 的 `99.999998%`；
- 10/10 预注册检查全部通过。

这不是“diffusion 在高 SNR 也能提升”的结论，而是更可靠的系统结论：**信道足够好时不强行生成/去噪，本身就是 channel-adaptive diffusion control 的必要组成。** 本 policy 不增加参数、信道符号或 DDIM 计算；它只把无益的高 SNR correction 设为严格恒等。

## 新 population 与泄漏控制

- 本地 COCO train2017 共 118,287 图；旧 exact-rate manifest 的 10,000 train + 1,000 validation 全部视为已使用；
- 用旧 source path 与 source SHA-256 双重排除 11,000 图，并排除 val2017 同名图；剩余候选 107,287 图；
- 按 `SHA256("20260738:"+relative_path)` 排序，前 256 为 selection、后 256 为 holdout；
- selection/holdout channel seed 分别为 `20260739/20260740`；
- 新 manifest path overlap=`0`、SHA overlap=`0`，SHA-256 `c467d2ccadd94242f51ff683f09d7b43d91a07d5edc538d06c06f2b6d93a8bed`；
- selection policy JSON SHA-256 `c31d68533bf4e470d585ff5d279e948a71dca3ead1964902647393b2f37d05eb`，在任何 holdout 输出前冻结；
- 无联网、无下载、未访问 official Imagenette validation。

## 冻结方法

网络完全冻结：

- exact-rate DeepJSCC：`bca5b67a...bb606`；
- decoder-aware matched latent diffusion：`5b708117...5d98f`；
- B1 reference：`7a295976...b7615a`。

令同噪声 received codeword 为 `y`，6-step matched DDIM 为 `z_diff`：

`z_final = y + g(SNR)*(z_diff-y)`。

候选为四个平滑幂 envelope `g=r^p, p∈{0.25,0.5,1,2}` 和一个冻结 hard identity policy。selection 首先最大化五档非负数量，再比较 mean PSNR、LPIPS 和配置顺序。

## Selection 结果

| 候选 | Mean ΔPSNR vs B0 | Mean ΔLPIPS | 非负 SNR 数 |
|---|---:|---:|---:|
| smooth `p=0.25` | `+0.195123` | `-0.036356` | 3/5 |
| smooth `p=0.5` | **`+0.199099`** | `-0.034347` | 3/5 |
| smooth `p=1` | `+0.193932` | `-0.030561` | 3/5 |
| smooth `p=2` | `+0.174850` | `-0.025588` | 3/5 |
| hard identity 7 dB | `+0.194020` | **`-0.036302`** | **5/5** |

平滑 `p=0.5` 的平均 PSNR 略高，但在 13/19 dB 仍为 `-0.007737/-0.003579 dB`，因此按预注册可靠性优先级没有入选。hard policy 的 selection PSNR delta 为 `+0.684812/+0.250225/+0.035064/0/0 dB`，随后冻结进入 holdout。

## Fresh holdout 主表

| 方法 | PSNR | MS-SSIM | LPIPS | ΔPSNR vs B0 |
|---|---:|---:|---:|---:|
| B0 | `26.240533` | `0.919839` | `0.295673` | — |
| full decoder-aware diffusion | `26.414608` | `0.931601` | **`0.258384`** | `+0.174075` |
| smooth `p=0.5` | **`26.435647`** | `0.931347` | `0.261061` | **`+0.195114`** |
| selected hard identity | `26.430250` | **`0.931665`** | `0.259389` | `+0.189717` |
| frozen B1 | `27.260867` | `0.942118` | `0.189690` | `+1.020334` |

selected 相对 full 提升 distortion reliability，但会回吐 full 在高 SNR 的微小 LPIPS收益：selected−full LPIPS=`+0.001005`，95% CI `[+0.000698,+0.001323]`。这不是隐藏的“全面支配”；它是明确的 perception/distortion/tail 取舍。相对 B0，selected 的 LPIPS仍显著改善 `-0.036284`。

平滑 `p=0.5` 相对 selected 的平均 PSNR为 `+0.005397 dB`，95% CI `[+0.003771,+0.006932]`，但它用 13/19 dB 的确定性负尾换取均值。本项目把 semantic/reliability tail 置于微小均值优势之前，因此 selected hard identity 是正式 policy，smooth `p=0.5` 只保留为 tradeoff 消融。

## 五档统计

| SNR | Full ΔPSNR vs B0 | Selected ΔPSNR vs B0（95% CI） | Selected ΔLPIPS vs B0 | `g` |
|---:|---:|---:|---:|---:|
| 1 | `+0.677172` | `+0.677172` `[+0.627707,+0.728323]` | `-0.099622` | 1 |
| 4 | `+0.240940` | `+0.240940` `[+0.208412,+0.275055]` | `-0.054826` | 1 |
| 7 | `+0.030472` | `+0.030472` `[+0.012319,+0.050133]` | `-0.026970` | 1 |
| 13 | `-0.050464` | `0` `[0,0]` | `0` | 0 |
| 19 | `-0.027746` | `0` `[0,0]` | `0` | 0 |

full diffusion 在 13/19 dB 的负 PSNR 95% CI 分别为 `[-0.055486,-0.044679]` 与 `[-0.029325,-0.025983]`，说明回退不是对随机零附近波动的过度反应。selected 相对 full 的改善完全来自删除这两个稳定负尾，低/中 SNR 与 full 只剩浮点舍入量级差异。

## 语义漂移诊断

COCO 仍只有 pseudo semantic endpoint。AlexNet 原图置信度≥0.2 的 850 rows 上：

- B0 failure=`453`；
- full failure=`389`，new/repair=`17/81`；
- selected failure=`398`，new/repair=`16/71`。

三分类器多数票：full new/repair=`7/32`，selected=`5/31`。selected 相比 full 少 1 个 AlexNet new error、少 2 个 majority new error，同时失去部分 high-SNR repair；总 failure 比 full 多 9。具体地，13 dB full 出现 AlexNet new/repair=`1/9`、majority=`2/1`，selected 恒等后全部变为 0；19 dB full 的 `0/1` AlexNet repair 也随恒等回退消失。

因此 identity control 降低了 refinement-induced new-error tail，但不追求最大 repair 数。这与 PSNR/LPIPS 一样是可靠性取舍。COCO pseudo 审计仍不能替代后续 Imagenette/有监督 semantic-tail 复核。

## 与 B1 的边界

B1 仍比 selected 高 `+0.830617 dB`，95% CI `[+0.791172,+0.869723]`，LPIPS也明显更好。故当前最强 deterministic reconstruction anchor 仍是 B1；本阶段成功的是 **diffusion 支路从 B0 出发的全 SNR 安全强度控制**，不是整体系统超过 B1。

下一阶段不应再扫描 envelope exponent/cutoff。更有价值的方向是训练显式接收 `B0 + identity-controlled diffusion decode` 的同容量融合器，并与只接收 `B0` 的 B1 做严格同容量/同训练预算对照；只有这样才能判断 diffusion prior 是否提供 B1 之外的互补信息。该融合训练必须使用新训练 population，并另建未暴露 selection/holdout；本轮 512 图已经全部封存为 S18 selection/holdout，不能再调融合器。

## 最终判定与复现路径

预注册 10 项检查全部通过，最终 verdict=`PASS`。

- 预注册：`reports/snr_identity_envelope_preregistration_2026-07-15.md`
- 配置：`configs/s18_snr_identity_envelope.yaml`
- population runner：`scripts/s18_prepare_fresh_coco_population.py`
- envelope runner：`scripts/s18_snr_identity_envelope.py`
- bootstrap：`scripts/s18_snr_identity_bootstrap.py`
- 核心实现：`src/cadsd_jscc/snr_identity_envelope.py`
- population：`outputs/eval/s18_fresh_coco_identity_envelope_population/`
- selection：`outputs/analysis/ANALYSIS-S18-IDENTITY-SELECTION-001/`
- holdout：`outputs/analysis/ANALYSIS-S18-IDENTITY-HOLDOUT-001/`
- bootstrap：`outputs/analysis/ANALYSIS-S18-IDENTITY-BOOTSTRAP-001/`

frozen holdout CSV SHA-256=`b488fc0205fe3535ea5d128a99f7808ce0c701d379ed242c08e458e1502f95ea`；bootstrap JSON SHA-256=`3c2815a2e7bd247dd2adc03d2e063f5094e515da36408525b389c3cff71e9d26`。
