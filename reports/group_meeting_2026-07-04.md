# 2026-07-04 组会汇报材料草稿

主题建议：信道自适应语义漂移控制的 Diffusion-JSCC：从 DeepJSCC baseline 到 blind diffusion 负结果

这份材料用于明天组会快速整理，不是论文定稿。所有结论都基于当前仓库已有记录和输出，未新增实验。

## 一句话主线

DeepJSCC 在低 SNR 下会模糊，diffusion refinement 看起来像一个自然的增强器，但当前小规模实验说明：固定强度、无语义约束的 blind SD img2img 会系统性改写图像语义。这个负结果正好构成后续 SNR-aware strength 和 semantic failure handling 的动机。

## 不能讲过头的边界

- 当前 M1 结论只针对 `strength=0.25`、空 prompt、`guidance_scale=1.0`、Stable Diffusion v1.5 img2img 的 blind refinement。
- 当前语义结果是三条辅助诊断：CLIP image-image、ImageNet AlexNet pseudo-label、COCO caption CLIP image-text。它们还不是 `MILESTONES.md` 要求的最终 clean-correct 冻结分类器主指标。
- 当前 M1 只跑了 1/7/19 dB、每个 SNR 16 张图，是用于发现 failure mode 的小规模正式负结果，不是完整论文实验。
- 不能把 M1 的画质变化包装成提升。已有指标显示 PSNR、SSIM、MS-SSIM、LPIPS 都变差。

## 8-10 分钟汇报结构

### Slide 1. 研究问题

标题：Diffusion-JSCC 的关键风险不是只恢复不清楚，而是恢复错了

要讲：

- DeepJSCC 在低 SNR 下退化，但通常还能保留主体结构。
- Diffusion 后处理可能让图像更像自然图，但也可能 hallucinate。
- 本项目关心的是 channel condition 下的 semantic drift：图像看起来被修复了，但语义偏离原图。

可用句子：

> 这不是单纯追 PSNR 或 LPIPS 的问题。对通信来说，接收端恢复出的内容如果语义错了，即使图像更锐，也不能算可靠传输。

### Slide 2. 项目最小闭环

标题：先收敛在 COCO-256 + AWGN 的最小闭环

设置：

| 项 | 当前设置 |
|---|---|
| 主数据集 | COCO2017 `train2017/val2017` |
| 图像尺寸 | 256x256 |
| 信道 | AWGN |
| CBR | 0.17 |
| SNR sweep | `[1, 4, 7, 13, 19]` dB |
| 方法组 | M0 DeepJSCC, M1 BlindDiffusion, M2 SNRAdaptiveDiffusion, M3 Ours |
| 语义主线 | semantic drift / final failure |

要讲：

- CIFAR-10 已跑通，但只作为 sanity baseline。
- diffusion 主实验必须在高分辨率自然图像上做，所以转向 COCO-256。
- 在 AWGN 最小闭环完成前，不扩展到 Rayleigh、DiT-JSCC、大型 adaptive JSCC baseline。

### Slide 3. 当前 pipeline

标题：当前已经跑通 M0 -> M1 -> 三类语义诊断

流程：

```text
COCO image x
  -> DeepJSCC encoder/channel/decoder under AWGN
  -> reconstruction x_hat                         (M0)
  -> Stable Diffusion img2img blind refinement
  -> refined x_refined                            (M1)
  -> image metrics + semantic diagnostics
```

已完成：

- M0 CIFAR-10 sanity baseline：`EXP-S1-001`
- COCO-256 DeepJSCC 正式训练：`EXP-S2HR-003`
- COCO-256 M0 SNR sweep/export：`EXP-S2HR-004`
- M1 blind diffusion 小规模实验：`EXP-S2-002`
- CLIP / classifier / caption 三类辅助语义诊断：`EXP-S3-001` 到 `EXP-S3-003`

注意点：

- 正式 high-res DeepJSCC checkpoint 必须使用 `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`。
- `latest.pt` 已 NaN，不可用于后续实验。

### Slide 4. M0 DeepJSCC baseline 结果

标题：COCO-256 M0 baseline 随 SNR 稳定提升

实验：`EXP-S2HR-004`，COCO2017 val subset 512 张，AWGN，CBR 0.17，checkpoint 为 `best.pt`。

| SNR(dB) | PSNR(dB) | SSIM | MS-SSIM |
|---:|---:|---:|---:|
| 1 | 28.019 | 0.809 | 0.936 |
| 4 | 30.047 | 0.870 | 0.962 |
| 7 | 31.559 | 0.905 | 0.976 |
| 13 | 33.195 | 0.935 | 0.988 |
| 19 | 33.726 | 0.943 | 0.991 |

要讲：

- M0 的趋势合理，说明高分辨率 DeepJSCC pipeline 是可用的。
- 7 dB 训练得到的模型在 sweep 上表现平滑。
- 高 SNR 下 M0 已经很接近原图，因此 diffusion 后处理如果还强烈改写，责任更多在 refinement，而不是信道噪声。

### Slide 5. M1 BlindDiffusion 设置

标题：先测试最简单的 blind SD img2img 后处理

实验：`EXP-S2-002`

| 项 | 设置 |
|---|---|
| 输入 | M0 export 中的 `x_hat` |
| SNR | `[1, 7, 19]` dB |
| 样本数 | 每个 SNR 16 张 |
| diffusion | `runwayml/stable-diffusion-v1-5` img2img |
| prompt | 空字符串 |
| strength | 0.25 |
| steps | 25 |
| guidance scale | 1.0 |

要讲：

- 这是 blind diffusion baseline，不使用 SNR-adaptive strength，也不使用 semantic guidance。
- 目标是先回答：直接把 diffusion 当后处理会不会稳健改善？

### Slide 6. M1 图像指标是明确负结果

标题：Blind diffusion 没有增强，反而大幅破坏重建

| SNR(dB) | M0 PSNR | M1 PSNR | Delta | M0 LPIPS | M1 LPIPS |
|---:|---:|---:|---:|---:|---:|
| 1 | 28.175 | 16.223 | -11.952 | 0.175 | 0.502 |
| 7 | 31.827 | 16.781 | -15.046 | 0.054 | 0.460 |
| 19 | 34.136 | 16.888 | -17.248 | 0.025 | 0.455 |

聚合：

- 平均 PSNR delta M1-M0：`-14.7485` dB
- 平均 LPIPS delta M1-M0：`+0.3877`，LPIPS 越低越好，所以这是退化
- 这说明当前固定强度 blind diffusion 不是有效 visual refinement

可用图：

- `outputs/EXP-S2-002/samples/snr_01db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_07db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_19db_original_reconstruction_refined.png`

要讲：

- 高 SNR 下 M0 已经非常好，但 M1 仍然把结构改掉。
- 这使得后续控制模块不是锦上添花，而是必要的安全约束。

### Slide 7. 语义诊断也一致指向 drift

标题：退化不只是低层指标，语义一致性也下降

CLIP image-image consistency，`EXP-S3-001`：

| SNR(dB) | CLIP sim original-M0 | CLIP sim original-M1 | Drop | M1 lower rate |
|---:|---:|---:|---:|---:|
| 1 | 0.902 | 0.662 | 0.240 | 1.000 |
| 7 | 0.959 | 0.687 | 0.272 | 1.000 |
| 19 | 0.985 | 0.695 | 0.289 | 1.000 |

Frozen AlexNet pseudo-label consistency，`EXP-S3-002`：

| SNR(dB) | M0 match original top-1 | M1 match original top-1 | M1 pseudo drift |
|---:|---:|---:|---:|
| 1 | 0.500 | 0.125 | 0.875 |
| 7 | 0.688 | 0.063 | 0.938 |
| 19 | 0.938 | 0.125 | 0.875 |

COCO caption CLIP image-text consistency，`EXP-S3-003`：

| SNR(dB) | M0 caption-max | M1 caption-max | Drop | M1 lower rate |
|---:|---:|---:|---:|---:|
| 1 | 0.331 | 0.282 | 0.049 | 1.000 |
| 7 | 0.331 | 0.282 | 0.049 | 0.813 |
| 19 | 0.326 | 0.288 | 0.039 | 0.813 |

要讲：

- 三套诊断口径不同，但方向一致：M0 更接近原图语义，M1 明显偏离。
- 这些指标目前是辅助证据，下一步要固定正式 `T_cls` 和 clean-correct subset。

### Slide 8. Failure case 1：高 SNR 下也会被改写

推荐图：

`outputs/EXP-S3-002/failure_cases/triptychs/snr_19db_sample_000002_origconf_0p984.png`

图中信息：

- SNR = 19 dB
- 原图分类：`Pomeranian`，confidence 0.984
- M0 分类：`Pomeranian`，confidence 0.958
- M1 分类：`gondola`，confidence 0.257

要讲：

- 这个例子很关键，因为它不是低 SNR 导致语义没传过去。
- M0 已经保留了小狗主体，M1 refinement 反而把主体和背景结构改坏。
- 因此语义漂移来自 blind diffusion 后处理，而不是 DeepJSCC baseline 的必然失败。

### Slide 9. Failure case 2：caption 语义被拉开

推荐图：

`outputs/EXP-S3-003/failure_cases/triptychs/snr_07db_sample_000008_caption_drop_0p1198.png`

COCO caption：

> A car parked by a clock and some flowers.

图中信息：

- Original caption-max：0.3608
- M0 caption-max：0.3775
- M1 caption-max：0.2577
- caption drop：0.1198

要讲：

- M0 虽然有模糊，但车、钟、花、街景关系还在。
- M1 局部纹理更强，但场景结构被扰乱，和人工 caption 的语义对应下降。
- 这是“看起来更像生成图”和“通信语义可靠”之间的冲突。

### Slide 10. 下一步

标题：从发现 failure mode 进入 semantic control

短期目标：

1. 固定正式 semantic drift 主指标。
   - COCO 主线可考虑 object detector / caption consistency。
   - 若需要严格 clean-correct 分类统计，可补 Imagenette/ImageNet subset。
2. 实现最小 failure handling。
   - 当 refined 的语义一致性明显低于 M0 时 fallback 到 `x_hat`。
   - 统计 accept rate、reject rate、Final-Failure。
3. 做更保守的 diffusion validation 小网格。
   - 新建实验 ID。
   - 优先尝试 `strength <= 0.10` 或更少 steps。
   - strength 随 SNR 升高不能增加。
4. 进入 M2/M3 对比。
   - M2：SNR-adaptive diffusion strength，无 semantic control。
   - M3：SNR-adaptive strength + semantic consistency control + fallback。

收束句：

> 目前的贡献苗头不是“diffusion 一定提升 JSCC”，而是“diffusion 在信道重建上会引入可量化的语义漂移，因此需要按信道和语义可靠性来控制它”。

## 3 分钟短版

如果时间被压缩，只讲四件事：

1. 我们的课题不是做新的大 JSCC backbone，而是研究 diffusion refinement 在信道退化下的 semantic drift。
2. 已经完成 COCO-256 DeepJSCC baseline，AWGN/CBR 0.17/SNR sweep 下 M0 指标随 SNR 稳定提升。
3. 固定强度 blind SD img2img 是系统性负结果：平均 PSNR 下降 14.75 dB，LPIPS 变差 0.388，CLIP 和分类器一致性也显著下降。
4. 下一步不是继续堆视觉增强，而是先固定正式 drift metric，再做 SNR-aware strength 和 semantic fallback。

## 可能被问的问题

### Q1：为什么 M1 这么差，是不是 strength 太大？

可以回答：

有可能。当前 `strength=0.25` 是一个 blind baseline，用来测试直接后处理是否可靠。结果说明这个设置不可用。下一步会做 validation 小网格，比如 `strength <= 0.10`，但必须新建实验 ID，且不能只看视觉，要同步看 semantic drift。

### Q2：这是不是说明 diffusion 不适合 JSCC？

可以回答：

不能这么下结论。当前只能说明固定强度、空 prompt、无语义约束的 blind SD img2img 不适合当前 COCO-256 DeepJSCC 输出。更合理的方向是 SNR-aware strength、semantic guidance 或 failure handling，让 diffusion 在不破坏语义时才介入。

### Q3：为什么用 COCO，而不是 CIFAR-10？

可以回答：

CIFAR-10 已用于 sanity baseline，但 32x32 图像不适合作为 diffusion 主实验。COCO-256 更接近 generative JSCC 文献中的高分辨率自然图像设置，并且有 captions，可以辅助分析语义一致性和 failure cases。

### Q4：现在的 semantic metric 是否已经满足论文主指标？

可以回答：

还没有。CLIP image-image、AlexNet pseudo-label 和 COCO caption CLIP 都是辅助诊断。按照项目里程碑，正式主指标还需要固定语义模型 `T_cls`、clean-correct subset，以及 Drift-Origin / Refinement-Drift / Final-Failure。下一步就是把这个口径定下来。

### Q5：M1 只跑 48 张样本，结论够吗？

可以回答：

不够作为完整论文结论，但足够作为 failure mode 证据和方法动机。它跨图像指标、CLIP、分类器、caption 三条语义诊断方向一致，而且在高 SNR 下也失败。后续 M2/M3 要扩展到固定 split 和完整 SNR。

### Q6：为什么不直接上 DiT-JSCC 或 DeepJSCC-l++？

可以回答：

这些是相关工作或后续扩展，但现在引入会把主线拉到大模型 generative decoder 或 adaptive JSCC backbone。当前最小闭环先保持 DeepJSCC baseline 简洁，聚焦 diffusion refinement 引入的 semantic drift 及其控制。

### Q7：训练里 `latest.pt` NaN 会影响结果吗？

可以回答：

不影响当前结果，因为后续 M0 export 和 M1 都固定使用 epoch 73 的 `best.pt`。`latest.pt` 已明确禁止使用，并且训练脚本后来加了 NaN 防护。这个风险已经记录在 `PROGRESS.md` 和 `EXPERIMENTS.md`。

## 推荐展示文件

主结果表：

- `outputs/analysis/m1_negative_result_summary/REPORT.md`
- `outputs/analysis/m1_negative_result_summary/summary.csv`

M0 baseline：

- `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/metrics.json`

M1 图像样例：

- `outputs/EXP-S2-002/samples/snr_01db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_07db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_19db_original_reconstruction_refined.png`

语义 failure case：

- `outputs/EXP-S3-001/failure_cases/triptychs/snr_19db_sample_000013_drop_0p4026.png`
- `outputs/EXP-S3-002/failure_cases/triptychs/snr_19db_sample_000002_origconf_0p984.png`
- `outputs/EXP-S3-003/failure_cases/triptychs/snr_07db_sample_000008_caption_drop_0p1198.png`

全局 gallery：

- `outputs/EXP-S3-001/failure_cases/sheets/global_top_clip_drop.png`
- `outputs/EXP-S3-002/failure_cases/sheets/global_top_classifier_drift.png`
- `outputs/EXP-S3-003/failure_cases/sheets/global_top_caption_clip_drop.png`

## 建议 PPT 标题和图表排布

1. 标题页：Channel-Adaptive Semantic-Drift Controlled Diffusion-JSCC
2. 问题定义：DeepJSCC blur vs diffusion semantic drift
3. 最小闭环：COCO-256/AWGN/CBR 0.17/M0-M3
4. 当前 pipeline：M0 export -> M1 refinement -> semantic diagnostics
5. M0 baseline 表格
6. M1 设置和图像指标表格
7. CLIP + classifier + caption 三个语义表格
8. Pomeranian -> gondola failure case
9. car/clock/flowers caption failure case
10. 下一步：formal metric, failure handling, M2/M3

如果 PPT 页数更少，合并 5 和 6，合并 8 和 9。

## 备份页内容

### CIFAR-10 sanity baseline

`EXP-S1-001` 在 CIFAR-10 test subset 1024 张上完成：

| SNR(dB) | PSNR(dB) | SSIM |
|---:|---:|---:|
| 1 | 23.543 | 0.822 |
| 4 | 26.379 | 0.893 |
| 7 | 28.986 | 0.935 |
| 13 | 32.861 | 0.970 |
| 19 | 34.799 | 0.979 |

讲法：

- 这只是验证 channel / checkpoint / metric pipeline。
- CIFAR-10 不作为 diffusion 主统计集。

### 相关工作定位

可以讲：

- DiffJSCC / SGD-JSCC / DiT-JSCC / JSCGC 已经说明 generative 或 diffusion JSCC 很热，所以本项目不声称首次引入 diffusion。
- 本项目的差异点在于把 semantic drift / final failure 当作核心失败模式，并研究 SNR-aware refinement strength 和 semantic control。

### 正式 semantic drift 定义

项目里程碑中的定义：

```text
Drift-Origin = mean[ c(x_refined) != c(x) ]
Drift-GT = mean[ c(x_refined) != y ]
Refinement-Drift = mean[ c(x_refined) != c(x_hat) ]
Final-Failure = mean[ c(x_final) != c(x) ]
```

讲法：

- 当前 pseudo-label 诊断接近这个方向，但还缺正式 clean-correct subset。
- M3 需要报告 fallback 后的 `x_final`，不能只报告 `x_refined`。

## 明天建议强调的结论

- 已经有可复现的 COCO-256 DeepJSCC M0 baseline。
- blind diffusion 后处理不是可靠默认增强器。
- 负结果很有价值，因为它把 semantic drift 这个核心问题从“担心”变成了“可量化 failure mode”。
- 下一步的研究焦点应收紧到 metric formalization 和 failure handling，而不是急着换复杂 backbone。
