# 2026-07-10 阶段性彻底总结与方向判断

> 更新说明（2026-07-10 后续）：本报告早于 `EXP-S4-008/009/010` 的受控消融、单调 schedule、fresh-holdout 和 classifier-ensemble 审计。最新结论见 `reports/edge_conditioning_significant_result_2026-07-10.md`；本文件保留为此前阶段快照。

本报告基于当前仓库已有文档、实验记录和输出报告整理，不新增实验结果。

## 结论先行

这个方向 **值得继续**，但前提是必须把主线从：

```text
DeepJSCC + blind diffusion refinement
```

明确收缩为：

```text
DeepJSCC reconstruction
+ channel-aware residual restoration
+ semantic-risk-aware strength / fallback control
```

如果继续把项目讲成“用 Stable Diffusion 空 prompt 后处理 JSCC 图像”，当前证据不支持继续。
如果把项目讲成“生成式/恢复式后处理在 JSCC 中会造成 semantic drift，因此需要信道感知和语义风险控制”，当前证据支持继续，并且已经有可展示的阶段性成果。

## 当前项目实际已经完成了什么

### 1. 基础闭环已经跑通

已经完成：

- CIFAR-10 sanity baseline。
- COCO2017 256x256 + AWGN high-resolution DeepJSCC baseline。
- 固定 CBR `0.17`。
- SNR sweep `[1, 4, 7, 13, 19]` dB。
- M0/M1/M2/M3 初步命名和同表 closure report。
- PSNR / MS-SSIM / LPIPS / pseudo semantic failure / accepted new error 等指标链路。

这意味着项目不是还停留在想法阶段，已经有可复现实验骨架。

### 2. Blind diffusion 的负结果已经很明确

M1 Stable Diffusion img2img，空 prompt / blind refinement：

- mean PSNR delta vs M0: `-14.7485` dB。
- mean LPIPS delta: `+0.3877`。
- SD VAE roundtrip 本身也显著损伤高保真 M0。

结论：

- 普通 SD img2img 后处理当前不可用。
- 这不是小调 strength 或 prompt 就能自然解决的问题。
- 盲目 diffusion 会把 JSCC 重建拉向生成先验，带来 hallucination / semantic drift。

这个负结果本身有价值：它说明为什么本项目不能只看视觉真实感，必须引入 semantic drift 度量和控制。

### 3. Pixel-domain residual restoration 是第一个稳定正向 anchor

M2 SNR-conditioned pixel residual CNN：

- mean PSNR delta vs M0: `+0.7235` dB。
- mean LPIPS delta: `-0.0274`。
- 5 个 SNR 上均有正向质量提升。

Per-SNR：

| SNR | M2 PSNR delta |
|---:|---:|
| 1 dB | `+1.1323` |
| 4 dB | `+0.7837` |
| 7 dB | `+0.5859` |
| 13 dB | `+0.5504` |
| 19 dB | `+0.5654` |

结论：

- “后处理恢复 JSCC 图像”是有效的。
- 问题不是恢复模块完全无效，而是恢复强度和语义可靠性之间存在冲突。
- M2 应作为后续主 anchor。

### 4. M3 语义控制已经有保守闭环

最小 M3：top-1 semantic fallback。

```text
if candidate_top1 == m0_top1:
    output candidate
else:
    fallback to M0
```

结果：

- mean PSNR delta: `+0.4011` dB。
- mean LPIPS delta: `-0.0104`。
- pseudo semantic failure 不高于 M0。

这说明已经有第一版“安全但保守”的 M3。

### 5. Residual strength / alpha control 是真正有希望的方向

Fixed shrink schedule：

- validation / held-out / test-like PSNR delta:
  - `+0.4584 / +0.4689 / +0.4552` dB。
- accepted new error:
  - `0 / 0 / 0`。

Adaptive alpha：

- validation / held-out / test-like PSNR delta:
  - `+0.5584 / +0.5664 / +0.5691` dB。
- accepted new error:
  - `0 / 0 / 0`。

结论：

- residual amplitude 不只是调参，而是核心控制变量。
- per-sample alpha 明显强于固定 alpha。
- 当前最清晰的贡献点是 semantic-risk-aware residual strength control。

### 6. Learned continuous alpha 是当前训练侧最强候选

Continuous-alpha tail-only residual refiner：

- validation / held-out / test-like PSNR delta:
  - `+0.5010 / +0.5049 / +0.5012` dB。
- LPIPS delta:
  - `-0.0149 / -0.0149 / -0.0162`。
- AlexNet accepted new error:
  - `0 / 0 / 0`。

但 ensemble audit：

- any-classifier new error:
  - `17 / 9 / 14`。
- majority-vote new error:
  - `1 / 0 / 0`。

结论：

- 它是当前最强 learned candidate。
- 但还不能宣布最终 M3，因为跨模型语义安全没有完全过关。

## 现在最大的问题是什么

### 问题 1：真正的 diffusion 还没有被救回来

已经失败的路线：

- SD img2img blind refinement。
- SD VAE roundtrip。
- naive residual DDPM from random noise。

`EXP-S4-007` naive residual diffusion 结果很差：

- refined PSNR 相比 M0 下降约 `-7.16/-7.48/-7.09/-5.42/-4.42` dB。

判断：

- 如果继续 diffusion，只能做短链、条件化、靠近 M0 或 residual CNN 输出的 correction diffusion。
- 不能从高斯噪声随机采 residual。
- 不能回到空 prompt blind generation。

### 问题 2：语义指标还不是最终 supervised clean-correct

当前 COCO 上主要是：

- AlexNet pseudo-label。
- CLIP image-image。
- COCO caption CLIP。
- COCO object zero-shot CLIP。
- ResNet18 / MobileNetV3 ensemble audit。

这些是有用的辅助诊断，但还不是正式分类标签主指标。

风险：

- 如果论文主实验只依赖 COCO pseudo-label，评审可能质疑 semantic drift 定义不够硬。
- 需要补一个有监督标签的 clean-correct subset，例如 Imagenette / ImageNet subset。

### 问题 3：保守安全和语义修复之间还没统一

当前安全方法能做到：

- 不增加 accepted new error。
- 稳定提升 PSNR/LPIPS。

但它的问题是：

- repair 很少或为 0。
- 很多 candidate 可以修复 M0 语义，但被 top-1 fallback 拒绝。

也就是说，当前 M3 更像：

```text
safe quality enhancement
```

还不是：

```text
semantic repair method
```

第一版论文可以接受“安全增强”，但如果要更强贡献，需要进一步提升 repair 或证明 tradeoff 优于 naive 方法。

### 问题 4：浅层 receiver-side gate 已经接近瓶颈

已经试过：

- top-1 equal。
- confidence gain。
- CLIP veto。
- SNR-calibrated CLIP veto。
- selected risk rule。
- ensemble veto。
- receiver-side risk score。

结论基本一致：

- 放宽 gate 可以提高 repair 和 PSNR，但会引入 accepted new error。
- 收紧 gate 可以安全，但收益明显回吐。

所以继续堆浅层规则的收益有限。下一步应该把 risk-aware 目标放进训练或 model selection，而不是继续手写阈值。

## 这个方向还值不值得

### 如果主线是“普通 diffusion 后处理 JSCC”

不值得继续作为主线。

理由：

- 当前 SD img2img 和 residual DDPM 都是明确负结果。
- 相关工作已经有 DiffJSCC、SGD-JSCC、DiT-JSCC、JSCGC 等，单纯说“我也加 diffusion”没有优势。
- 盲目 generative prior 在 JSCC reconstruction 上很容易 hallucinate。
- 继续堆 prompt / strength 大概率会陷入调参，而不是形成稳定贡献。

### 如果主线是“semantic drift controlled restoration”

值得继续。

理由：

- 已经有正向 anchor：M2 residual CNN。
- 已经有安全闭环：M3 fallback / shrink / adaptive alpha。
- 已经有明确核心失败模式：accepted new error / semantic drift。
- 已经有清晰贡献差异：不是追求视觉最好，而是在信道自适应恢复中控制语义漂移。
- 负结果、风险分析和 failure case 可以形成论文论证的一部分。

更准确的论文方向应该是：

> Channel-adaptive semantic-risk-controlled post-restoration for JSCC image transmission.

或者：

> Semantic-drift-aware residual refinement control for DeepJSCC reconstruction.

标题里保留 diffusion 可以，但正文贡献不应押在 blind diffusion 成功上。

## 现在适合怎么对外讲

建议讲法：

> 我们最开始尝试 diffusion refinement，发现 blind diffusion 在 JSCC reconstruction 上会造成明显质量损伤和 semantic drift。随后我们把问题转成：如何在信道自适应后处理恢复中控制语义风险。当前 M2 residual CNN 在 COCO-256 AWGN 上稳定提升 PSNR/LPIPS；M3 fallback/shrink/adaptive-alpha 能在 pseudo semantic constraint 下获得稳定质量收益并控制 accepted new error。最新 continuous-alpha learned refiner 已经接近可部署候选，但跨分类器安全仍需进一步收敛。

这比说“我们 diffusion 效果不好但还在调”要强很多。

## 继续推进的建议路线

### 第一优先级：补 supervised clean-correct 评估

目标：

- 用 Imagenette / ImageNet subset 建立真正带标签的 clean-correct 子集。
- 把现在的 pseudo-label 指标升级为：

```text
A = {i | T_cls(original_i) = y_i and confidence >= threshold}
Final-Failure = mean[ T_cls(final_i) != y_i ], i in A
```

价值：

- 解决语义指标说服力问题。
- 让后续 M3 安全性更容易写成论文主结果。

### 第二优先级：围绕 continuous alpha 做 risk-aware training

不要再回到普通 CE alpha 分类，也不要全量 unfreeze。

建议尝试：

- continuous alpha + semantic-risk penalty。
- listwise utility loss：候选按 PSNR gain 和 new-error risk 排序。
- ensemble-aware validation selection。
- 轻量扩容 tail / alpha head，而不是改大整个 refiner。

目标：

- 保持 PSNR delta `>= +0.50` dB。
- AlexNet new error 继续 `0`。
- ensemble majority new error 清到 `0/0/0`。
- 尽量降低 any-model new error。

### 第三优先级：如果做 diffusion，只做受控短链 residual diffusion

可做，但不能作为下一步最高优先级。

如果做，必须满足：

- 从 M0 或 residual CNN 输出附近初始化。
- 预测小残差或 correction，不生成整图。
- 低噪声 short-chain schedule。
- identity / reconstruction loss 优先。
- semantic gate 或 alpha control 参与选择。

成功标准：

- 至少不能低于 M2 residual CNN。
- 必须报告 accepted new error。

## 停止或转向条件

建议设置明确 kill criteria：

1. 如果补 supervised clean-correct 后，当前 M3 的安全性不成立，必须先修 semantic metric / detector，不能继续包装现有结果。
2. 如果连续 alpha 加 risk-aware 训练后，ensemble majority new error 仍不能清零，M3 只能作为 pseudo-safe candidate，论文贡献要降级。
3. 如果短链 diffusion 仍明显低于 residual CNN，就不要再把 diffusion 作为方法主体，只作为负结果和讨论保留。
4. 如果后续 1-2 周内无法形成 supervised clean-correct + M3 同表结果，应优先收敛为 workshop/阶段报告型成果，而不是扩展新主线。

## 最终判断

项目不是不乐观，而是已经证明了一个关键事实：

```text
普通 generative diffusion 对 JSCC reconstruction 很危险；
但 semantic-risk-aware residual restoration 是可行的。
```

因此，继续做是值得的，但要停止追逐“直接 diffusion 变好看”，转向更稳的贡献：

```text
semantic drift measurement
+ failure handling
+ channel-aware residual strength control
+ learned continuous alpha policy
```

当前最现实的论文闭环是：

- M1 blind diffusion 作为负参考。
- M2 residual CNN 作为质量 anchor。
- M3 semantic-risk-controlled alpha/fallback 作为 ours。
- 再补 supervised clean-correct 评估提高可信度。

这个方向有继续价值，前提是下一阶段必须围绕 M3 语义安全收敛，而不是继续横向扩新模型。
