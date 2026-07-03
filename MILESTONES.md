# 里程碑与收敛标准

本文件用于防止课题无限扩张。CIFAR-10 只作为 JSCC sanity baseline；diffusion 主路线必须使用高分辨率自然图像数据。

## 最小论文闭环

第一版论文必须先完成以下闭环：

- Sanity 数据集：CIFAR-10 test set 或固定 test subset，用于验证 JSCC/channel/metric 流程。
- 主数据集：COCO2017 train/val subset，训练和评估统一裁剪到 `256x256`。
- 补充数据集：Kodak，只用于视觉质量补充测试和样例展示，不作为 semantic drift 主统计集。
- 主信道：AWGN。
- SNR：沿用阶段1配置的 `[1, 4, 7, 13, 19]` dB。
- CBR：先固定为 `0.17`。
- JSCC baseline：CIFAR-10 使用已接入 checkpoint；COCO-256 必须训练或接入高分辨率 DeepJSCC checkpoint。
- 方法对比：至少包含 `M0` 到 `M3`。
- 指标：必须同时报告图像质量、感知质量、语义可靠性和推理开销。
- 输出：必须保存指标表、配置副本、样例图、semantic drift failure case。

完成 COCO-256 AWGN 最小闭环前，不引入 Dynamic_JSCC、DeepJSCC-l++、PJSCC、DiT-JSCC 或新的大模型主线。

## 方法分组

正式实验至少包含四组：

- `M0-DeepJSCC`：只使用 DeepJSCC reconstruction，输出 `x_hat`。
- `M1-BlindDiffusion`：对 `x_hat` 做固定强度 diffusion refinement，不使用 SNR-adaptive 或 semantic control。
- `M2-SNRAdaptiveDiffusion`：根据 SNR 调整 diffusion strength，不使用 semantic guidance 或 failure detector。
- `M3-Ours`：SNR-adaptive diffusion strength + semantic consistency control + semantic failure handling。

可选扩展必须放在最小闭环之后：

- Rayleigh 信道。
- ImageNet 子集。
- Dynamic_JSCC / DeepJSCC-l++ 对照。
- CLIP-guided 或 DiT-style diffusion 变体。

## Semantic Drift 定义

第一版必须使用一个冻结语义模型 `T_cls`，例如 CIFAR-10 classifier。设：

- `c(z)`：`T_cls` 对图像 `z` 的 top-1 类别。
- `p(z)`：`T_cls` 对 `c(z)` 的置信度。
- `y`：数据集真实类别。
- `x`：原图。
- `x_hat`：DeepJSCC decoder 输出。
- `x_refined`：diffusion refinement 输出。
- `x_final`：经过 failure handling 后的最终输出。

不同方法中的默认关系：

```text
M0: x_final = x_hat
M1/M2: x_final = x_refined
M3: x_final = accepted x_refined, fallback x_hat, or weaker refinement output
```

正式统计优先在 clean-correct 子集上进行：

```text
A = {i | c(x_i) = y_i and p(x_i) >= tau_clean}
```

第一版主指标：

```text
Drift-Origin = mean_i[ c(x_refined_i) != c(x_i) ], i in A
Drift-GT = mean_i[ c(x_refined_i) != y_i ], i in A
Refinement-Drift = mean_i[ c(x_refined_i) != c(x_hat_i) ], i in A
Final-Failure = mean_i[ c(x_final_i) != c(x_i) ], i in A
Prediction-Consistency = mean_i[ c(x_final_i) = c(x_i) ], i in A
```

若使用 CLIP 或其他语义特征模型，只能作为辅助指标：

```text
CLIP-Drift = mean_i[ sim(T_clip(x_i), T_clip(x_refined_i)) < tau_clip ]
```

不能只用 CLIP similarity 替代分类一致性主指标。

## Semantic Failure Handling

第一版 failure detector 可以简单，但必须可复现：

- 若 `x_refined` 的语义不可信，输出回退到 `x_hat`，或降低 diffusion strength 后重试。
- 必须记录 detector 的接受率、拒绝率和最终 failure rate。
- 若 detector 接受了语义错误结果，记为 false accept。
- 若 detector 拒绝了语义正确且质量更好的结果，记为 false reject，可选统计。

第一版不要求 detector 完美，但必须证明它不是只提高视觉指标、同时放任语义错误。

## Diffusion 第一版边界

第一版 diffusion refinement 只允许作为 DeepJSCC 后处理模块：

- 输入是 `x_hat`，输出是 `x_refined`。
- 不从零训练大型 diffusion 或 DiT-JSCC。
- 不把 diffusion 替换成主 JSCC decoder。
- 不使用需要人工文本 prompt 的流程作为主实验。
- 不在 test set 上调 diffusion strength、guidance weight 或 threshold。

SNR-adaptive strength 必须满足：

- strength 随 SNR 升高而不增加。
- 高 SNR 少修，低 SNR 多修。
- semantic guidance 或 failure handling 在低 SNR 下不能弱于高 SNR。

示例约束：

```text
strength(1 dB) >= strength(4 dB) >= strength(7 dB) >= strength(13 dB) >= strength(19 dB)
semantic_weight(1 dB) >= semantic_weight(4 dB) >= ... >= semantic_weight(19 dB)
```

具体数值必须只在 validation subset 上确定。

## 成功判据

`M3-Ours` 的目标不是在所有指标上绝对最优，而是在感知质量和语义可靠性之间取得更好的 tradeoff。

优先成功判据：

- 相比 `M1-BlindDiffusion`，`M3-Ours` 在低/中 SNR 下有更低 semantic drift 或 final failure。
- 相比 `M0-DeepJSCC`，`M3-Ours` 保留主要感知质量收益，例如 LPIPS 或 FID 改善。
- 相比 `M2-SNRAdaptiveDiffusion`，`M3-Ours` 证明 semantic control 不是多余模块。

以下情况不能算课题成功：

- 只提升 PSNR/MS-SSIM，但 semantic drift 没有统计。
- 只提升 LPIPS/FID，但 drift 或 final failure 明显上升。
- 只在高 SNR 有效，低 SNR 下 diffusion 大量 hallucination。
- 只展示少量好看样例，没有固定 test split 的统计。

## 阶段门槛

### S1 DeepJSCC Baseline

完成标准：

- smoke test 能加载 checkpoint、切换 SNR、输出重建图和 PSNR。
- mini-eval 能在固定 CIFAR-10 subset 上输出 `M0` 指标。
- `EXP-S1-001` 写入 `EXPERIMENTS.md`。

### S2-HR High-Resolution DeepJSCC

完成标准：

- 准备 COCO2017 train/val 或等价自然图像高分辨率数据。
- 训练或接入 `256x256` DeepJSCC checkpoint，至少覆盖 AWGN 和 CBR `0.17`。
- 在固定 COCO val subset 上输出 `M0-HR` 指标和样例图。
- 记录 checkpoint、训练配置、数据 split 和训练日志。

### S3 Blind Diffusion

完成标准：

- `M1` 在相同图像、相同 SNR、相同 CBR 上可复现运行。
- 保存 `x_hat`、`x_refined` 和样例对比。
- 报告视觉指标和初步 semantic drift。

### S4 Semantic Metrics

完成标准：

- 冻结 `T_cls`。
- 固定 clean-correct 子集和阈值。
- Drift-Origin、Drift-GT、Refinement-Drift、Final-Failure 中至少实现前三项；进入 adaptive control 前必须实现 Final-Failure。

### S5 Adaptive Control

完成标准：

- 实现 SNR-adaptive diffusion strength。
- 实现 semantic consistency control 或 failure handling。
- 完成 `M0` 到 `M3` 的同表对比。

### S6 完整实验

完成标准：

- 在所有固定 SNR 上完成正式实验。
- 至少输出一张 tradeoff 图。
- 至少整理一组 semantic drift failure case。
- 明确写出方法成功、部分成功或失败的结论。

## 复现记录

每个正式实验必须保存：

- 实验 ID。
- 日期。
- 项目 commit；如果当前目录不是 git 仓库，写 `N/A (not a project git repo)`。
- 第三方 baseline commit。
- config 路径和 config 内容副本。
- 运行命令。
- 数据 split 或样本 ID。
- 随机种子。
- checkpoint 路径。
- 输出路径。
- 环境信息，至少包含 Python 版本和核心依赖版本。

如果没有项目 git commit，必须额外记录本次使用的脚本路径、配置路径和关键源码路径，避免实验无法追溯。
