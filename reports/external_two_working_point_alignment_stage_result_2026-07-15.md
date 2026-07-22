# 外部方法双工作点码率对齐阶段结果（2026-07-15）

## 阶段结论

两套此前混在一起的问题已经拆开并得到首轮可复现答案：

1. **作者约 1/20 工作点：SGD-JSCC 并不差。** 在相同 19,712 个图像分支实符号和相同 canonical AWGN 下，完整 COCO 继续训练后的精确低码率 DeepJSCC 为 `25.9260 dB`，SGD-JSCC 论文协议上界为 `26.8389 dB`，SGD 平均领先 `0.9128 dB`，LPIPS 也显著更低，并把 3 个低码率 DeepJSCC 语义失败全部修复。
2. **项目 1/6 工作点：原 common adapter 的预算确实分得不合理。** 把 SGD 主 latent 从 R1 改为 R2、caption 从 R21 降到 R13 后，PSNR 从 `26.8882` 升至 `27.3933 dB`，40/40 条件提升，且 160/160 caption packets 仍通过 CRC。
3. **当前项目 M3 在 1/6 小 pilot 上仍明显领先发布权重 SGD adapter。** 重分配后差距由 `6.1712` 缩小为 `5.6661 dB`，但仍是 40/40 PSNR 和 MS-SSIM 条件更优，LPIPS 38/40 更优，双方均无 `T_cls` hard failure。
4. 这些结果支持继续保留并发展 diffusion，而不是退回纯 DeepJSCC。低码率结果尤其说明：带生成先验的方法在受限带宽下有更高上限；本项目下一步应把现有 semantic-drift control 迁移到低码率 diffusion 链，而不是停止 diffusion。

全部结论仍只来自 8 张已暴露 policy-dev 图×5 SNR 的方向性 pilot，不授权论文级领先声明。

## 两套冻结协议

| 工作点 | 总图像/物理预算 | SGD 处理 | 能回答的问题 |
|---|---:|---|---|
| 作者工作点 | 19,712 real = 9,856 complex uses，精确 CBR `0.0501302` | main 16,384 + active edge 3,328；caption 完美且不计码率 | 发布方法在论文原假设附近是否仍有生成优势 |
| 项目工作点 | 65,536 real = 32,768 complex uses，CBR `1/6` | main R2 32,768 + edge R1 3,328 + text R13 27,872 + padding 1,568 | 发布权重如何更合理地使用项目总预算 |

作者工作点的 SGD caption 是论文明确采用的免费且无误文本假设，所以该数字只能叫“论文协议上界”，不能叫严格端到端物理码率公平。项目工作点的 R2/R13 只增强抗噪性，不增加发布模型的表示容量，也不能叫“宽 latent SGD-JSCC”。

预注册见 `reports/external_two_working_point_alignment_preregistration_2026-07-15.md`。

## 精确低码率 DeepJSCC

DeepJSCC 的整数 `c` 只能给出 `c/48` 的 CBR，无法直接得到 `1/20`。本轮使用 `c=3` 的 24,576 维稠密 latent，按冻结均匀索引只发送 19,712 个实坐标：

- 活动坐标按样本重新归一化为单位平均功率；
- 每个实坐标使用 `P/(2×SNR)` 的复 AWGN 口径；
- 接收端把未发送坐标置零，训练和测试使用同一掩码；
- 从稳定 c8 checkpoint 按 encoder/decoder 联合 latent importance 裁剪热启动。

首个 AMP、`lr=5e-5` 训练在 epoch 1 batch 213 出现 non-finite loss，失败目录和 `failure.json` 已保留。FP32、`lr=2e-5` 的 20k×12 稳定化复跑达到 COCO-512 `25.8609 dB`。检查到原 c8 基线使用完整 118,287 张 COCO 后，又在不改码率、掩码和信道的前提下，以 `lr=1e-5` 继续训练完整 COCO 12 epoch，最终 COCO-512 为：

- PSNR `26.6981 dB`
- SSIM `0.77855`
- checkpoint SHA-256 `bca5b67a3bca93f17d23688cc5ec2d30ffe8790191e07fbf41926649cf1bb606`

这一继续训练使冻结外部 pilot 的 DeepJSCC PSNR 从 `25.0682→25.9260 dB`，LPIPS 从 `0.34148→0.28716`，failure 从 `5→3`。说明训练预算检查是必要的；首轮弱模型不能作为最终公平结论。

## 作者工作点结果

最终采用 `ANALYSIS-EXT-AUTHOR-RATE-PILOT-002`。两方法使用相同 40 个 sample/SNR keys、相同 19,712 维 canonical noise SHA、相同复 AWGN 口径。

| 方法 | PSNR | MS-SSIM | LPIPS | failure | 延迟/图 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| 精确低码率 DeepJSCC（full-COCO follow-up） | `25.9260` | `0.92189` | `0.28716` | `3/40` | `2.89 ms` | `203.6 MiB` |
| SGD-JSCC 论文协议上界 | **`26.8389`** | **`0.94861`** | **`0.07856`** | **`0/40`** | `2043.7 ms` | `7374.9 MiB` |

SGD−DeepJSCC 的逐行配对结果：

- PSNR `+0.91283 dB`，SGD wins `31/40`；
- MS-SSIM `+0.02672`，wins `33/40`；
- LPIPS `-0.20860`，wins `40/40`；
- SGD-only correct `3`，DeepJSCC-only correct `0`。

分 SNR 看，低码率 DeepJSCC 的 3 个失败全部集中在 1 dB；SGD 五个 SNR 都没有 hard failure。这符合 SGD-JSCC 主打低 SNR 感知/生成质量、而非纯失真最优的论文定位。

限制：低码率 DeepJSCC 是本项目构造的 exact-mask baseline，不是某篇论文的官方 `R=1/20` checkpoint；SGD 又额外获得免费 caption。因此该表证明“SGD 在作者协议下不是弱方法”，但不能单独证明 SGD 架构在严格物理端到端相同信息下必然更强。

## 项目工作点重分配结果

`ANALYSIS-EXT-SGD-REALLOC-PILOT-001` 与旧 common pilot 使用相同 65,536 维 noise、相同 40 个 keys 和相同发布权重。

| 方法 | PSNR | MS-SSIM | LPIPS | failure |
|---|---:|---:|---:|---:|
| SGD R1 + text R21 | `26.8882` | `0.94862` | `0.07763` | `0/40` |
| SGD main R2 + text R13 | **`27.3933`** | **`0.95534`** | **`0.07246`** | `0/40` |
| 当前项目 M3 | **`33.0594`** | **`0.98203`** | **`0.03532`** | `0/40` |

新分配相对旧 SGD：

- PSNR `+0.50510 dB`，wins `40/40`；
- MS-SSIM `+0.006721`，wins `40/40`；
- LPIPS `-0.005167`，wins `33/40`；
- 160/160 caption packets CRC 通过，0 个新语义错误。

当前 M3 相对重分配 SGD：PSNR `+5.66606 dB`、MS-SSIM `+0.026694`、LPIPS `-0.037143`；PSNR/MS-SSIM wins `40/40`，LPIPS wins `38/40`。这比旧的 `+6.171 dB` 更公平，但仍只是小 pilot 的方向性效果量。

## 对项目方向的直接启发

### 1. 不应放弃 diffusion

作者工作点下，生成式 SGD 在感知、失真和 hard semantic failure 上都超过精确低码率 DeepJSCC。纯 JSCC 在带宽收紧后上限确实更低，用户此前“没有 diffusion 上限较低”的判断得到支持。

### 2. 当前 M3 的强项与外部方法强项不同

当前 M3 在 CBR `1/6` 的强项是高 PSNR、低 LPIPS、低延迟和显式 semantic-drift failure handling；SGD 的强项是极低码率下借助 caption/edge/生成先验维持感知与语义。最有价值的下一步不是二选一，而是把二者结合：低码率生成恢复 + 严格计码率 semantic checksum + received-latent consistency + new-error tail gate。

### 3. 下一阶段应做低码率项目 M3，而不是直接移植现有权重

现有 B1、S14 diffusion 和 posterior consistency 都围绕 c8/CBR `1/6` measurement 训练，不能直接接到 c3 exact-mask latent 后声称是低码率 M3。应基于新的 19,712-real DeepJSCC 重新生成 COCO cache，训练低码率 B1/diffusion，并让 posterior data term 只作用于活动坐标。这样才能回答“本项目的 drift control 能否保留 SGD 的低码率生成优势”。

### 4. 外部比较的下一次升级

- 把 8 图 pilot 升为预注册 64 图×3 channel seeds，并按图像聚类给 failure/new-error 上界；
- 继续保留作者协议上界表和严格物理协议表，禁止混写；
- SGD 若要真正使用 CBR `1/6` 的表示容量，需要作者训练流程或自行重训更宽 latent，重复传输不能替代容量；
- SING 对照应升级为逐 reverse-step DDNM projection，final-only 版本不再承担论文级比较；
- 之后再接 DiffJSCC，优先复现与本项目低码率 diffusion restoration 最接近的公开 checkpoint/协议。

## 产物与验证

- 训练：`outputs/train/external_author_rate_deepjscc_c3_mask19712_fullcoco_continue/`
- 作者工作点最终聚合：`outputs/external_baselines/ANALYSIS-EXT-AUTHOR-RATE-PILOT-002/aggregate/summary.json`
- 项目工作点聚合：`outputs/external_baselines/ANALYSIS-EXT-SGD-REALLOC-PILOT-001/aggregate/summary.json`
- 精确码率/重复适配器测试：3/3 pytest 通过；全仓标准库测试 99/99 通过；关键脚本 `py_compile` 与 `git diff --check` 通过。
- 本轮复用本地数据与已下载作者资产，无新增下载，official Imagenette validation 未访问。
