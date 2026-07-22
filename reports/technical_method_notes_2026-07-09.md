# 2026-07-09 当前技术路线说明

本说明用于解释当前项目实际用到的技术和指标口径。它不新增实验结果。

## 1. 整体 pipeline

当前实验链路是：

```text
original image x
  -> DeepJSCC encoder / AWGN channel / decoder
  -> M0 reconstruction x_hat
  -> refinement module
  -> candidate x_refined
  -> semantic failure handling / alpha selection
  -> final output x_final
```

其中：

- `M0`：DeepJSCC-HR baseline，只输出 `x_hat`。
- `M1`：Stable Diffusion img2img blind refinement，输入 `x_hat`，固定/空 prompt，作为负结果参考。
- `M2`：SNR-conditioned pixel residual CNN refinement，输出 `x_refined`。
- `M3`：在 M2 或其 alpha 候选上加入 semantic fallback / alpha selection，输出 `x_final`。

## 2. DeepJSCC / AWGN / SNR / CBR

DeepJSCC 负责把图像直接映射到信道符号，再经过信道噪声后恢复图像。当前主闭环使用：

- 数据：COCO2017 256x256。
- 信道：AWGN。
- SNR sweep：`[1, 4, 7, 13, 19]` dB。
- CBR：`0.17`。
- baseline checkpoint：COCO-256 AWGN `best.pt`。

SNR 越低，信道越差，M0 重建越模糊或失真。后处理模块的目标是在不改变 JSCC encoder/decoder 主结构的情况下提升恢复质量。

## 3. Blind diffusion 负结果

最初的 M1 是 Stable Diffusion img2img：

```text
x_hat -> SD VAE encode -> UNet denoise -> VAE decode -> x_refined
```

当前实验中它使用空 prompt / blind prompt。问题是：

- SD VAE roundtrip 本身会损伤高保真 JSCC 重建。
- blind denoising 会把图像拉向生成先验，可能改变物体类别或场景语义。
- 所以 M1 被保留为 negative reference，而不是当前主方法。

## 4. Pixel-domain residual refiner

M2 改成 pixel-domain residual restoration，避免 SD VAE：

```text
input:  concat(x_hat, snr_map)
model:  small CNN residual refiner
output: x_refined = clamp(x_hat + gate(SNR) * residual, 0, 1)
```

技术点：

- 输入是 M0 图像和归一化 SNR map。
- CNN 预测一个残差图，不直接重新生成整张图。
- `gate(SNR)` 控制残差强度，且要求低 SNR 不弱于高 SNR。
- 训练损失以 MSE + L1 为主，目标是靠近 original。

这一路线的意义是：它更像 restoration，不像 generative hallucination，因此语义风险比 blind diffusion 小。

## 5. Semantic drift 怎么算

项目定义里的正式形式是：

```text
c(z) = frozen semantic classifier T_cls 对图像 z 的 top-1 类别

Drift-Origin      = mean[ c(x_refined) != c(x) ]
Refinement-Drift  = mean[ c(x_refined) != c(x_hat) ]
Final-Failure     = mean[ c(x_final) != c(x) ]
Prediction-Consistency = mean[ c(x_final) == c(x) ]
```

当前 COCO 实验里没有天然单标签分类真值，所以实际采用的是 pseudo-label 口径：

```text
original_top1 = T_cls(original)
m0_top1       = T_cls(x_hat)
candidate_top1 = T_cls(x_refined or alpha candidate)
final_top1    = T_cls(x_final)
```

然后：

```text
M0 failure rate
  = mean[ m0_top1 != original_top1 ]

Candidate drift-origin / candidate failure
  = mean[ candidate_top1 != original_top1 ]

Refinement drift
  = mean[ candidate_top1 != m0_top1 ]

Final failure rate
  = mean[ final_top1 != original_top1 ]

Prediction consistency
  = mean[ final_top1 == original_top1 ]
```

当前主接收端分类器是冻结 ImageNet AlexNet。后续也用 ResNet18 和 MobileNetV3-Small 做离线 ensemble audit，但 receiver-side 决策默认不是 ensemble。

## 6. Accepted new error / repair / missed repair

这些是 M3 failure handling 的关键统计：

```text
accepted_repair
  = accept candidate
    and M0 was wrong relative to original_top1
    and candidate/final becomes correct relative to original_top1

accepted_new_error
  = accept candidate
    and M0 was correct relative to original_top1
    and candidate/final becomes wrong relative to original_top1

missed_repair
  = M0 was wrong
    and some candidate could match original_top1
    but final output still does not match original_top1
```

研究上最不能接受的是 `accepted_new_error`：它代表后处理把本来语义正确的 M0 改错了。

## 7. Top-1 fallback

最小 M3 的 detector 很简单：

```text
if candidate_top1 == m0_top1:
    x_final = candidate
else:
    x_final = x_hat
```

直观含义：

- 如果增强前后 frozen classifier 的 top-1 没变，就认为语义风险较低，接受增强。
- 如果 top-1 变了，就认为可能发生 semantic drift，回退到 M0。

优点：

- 简单、可复现。
- 在 AlexNet pseudo-label 口径下可以避免 accepted new error。

缺点：

- 很保守。
- 如果 M0 本来错了，而 candidate 修好了，它也可能被拒绝，因为 candidate top-1 不等于 M0 top-1。

## 8. Residual shrink / alpha control

M2 full-strength residual 有质量收益，但也可能过强。于是引入 alpha：

```text
x_alpha = clamp(x_hat + alpha * (x_refined - x_hat), 0, 1)
```

当前用过几类 alpha 策略：

- fixed shrink schedule：每个 SNR 选一个固定 alpha。
- adaptive max top1-consistent alpha：每个样本选最大的、且 `candidate_top1 == m0_top1` 的 alpha。
- two-stage alpha：先试 full strength，不安全再用 fixed schedule。
- receiver alpha predictor：用 receiver-visible features 预测 alpha。
- continuous-alpha tail refiner：训练模型直接输出连续 alpha。

核心结论：

- alpha 不是简单调参，它是 semantic-risk-aware residual amplitude control。
- 固定 alpha 有效，但 per-sample alpha 更强。
- learned continuous alpha 是当前训练侧最强候选，但还没通过最终跨模型安全验证。

## 9. CLIP / caption / COCO-object 辅助诊断

除了 frozen classifier，还做过辅助语义检查：

- CLIP image-image similarity：比较 original/M0/refined 的图像语义 embedding。
- COCO caption CLIP image-text consistency：看图像是否仍匹配原 caption。
- COCO object label zero-shot CLIP：用 dominant object label 做 GT-like 辅助判断。
- classifier ensemble audit：AlexNet / ResNet18 / MobileNetV3-Small 多模型复核。

这些只能作为辅助证据，不能替代正式 classification consistency / final failure 口径。

## 10. 当前技术定位

当前项目不是简单的：

```text
DeepJSCC + diffusion decoder
```

更准确的定位是：

```text
DeepJSCC reconstruction
+ channel-aware residual restoration
+ semantic-risk-aware strength / fallback control
```

如果后续重新引入 diffusion，设计边界应是：

- 从 M0 或 residual CNN 输出附近初始化。
- 做短链 conditional residual correction。
- 不从高斯噪声随机采样完整残差。
- 不用空 prompt blind generation 作为主方法。
