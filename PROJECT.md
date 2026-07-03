# Channel-Adaptive Semantic-Drift Controlled Diffusion-JSCC

## 核心问题

DeepJSCC 在低 SNR 信道条件下重建质量下降。扩散模型可以提升感知质量，但也可能引入 hallucination 或 semantic drift：重建图像看起来更真实，但语义上偏离原图。

本项目研究：如何在不同信道条件下，用扩散模型增强 JSCC 图像恢复，同时控制 semantic drift。

完整收敛标准见 `MILESTONES.md`。如果新增想法与 `MILESTONES.md` 的最小闭环冲突，优先完成最小闭环。

## 核心假设

1. Blind diffusion refinement 可以提升感知质量，但可能增加 semantic drift。
2. Semantic consistency guidance 可以降低 semantic drift。
3. SNR-aware / channel-adaptive diffusion strength 可以在视觉质量和语义可靠性之间取得更好平衡。

## 方法边界

本项目不做：

- 不从零训练大型 DiT-JSCC。
- 不把论文写成单纯 DeepJSCC + diffusion decoder。
- 不只报告 PSNR/MS-SSIM，必须报告 semantic drift / semantic failure。
- 不用扩散模型生成与原图语义不一致的“好看图”冒充提升。
- 在 AWGN 最小闭环完成前，不切换到大型 DiT-JSCC、复杂 adaptive JSCC baseline 或多数据集扩展。

## 研究问题

1. Blind diffusion refinement 在不同 SNR 下能带来多少感知质量提升？
2. 哪些信道条件下，扩散增强更容易产生 semantic drift 或 semantic failure？
3. Semantic consistency guidance 是否能降低 drift，同时保留主要感知收益？
4. SNR-aware diffusion strength 是否优于固定扩散强度？

## 实验边界

实验必须区分以下部分：

- diffusion 前的 JSCC 重建质量。
- blind diffusion refinement。
- semantic-guided diffusion refinement。
- channel-adaptive / SNR-aware diffusion refinement。

评估必须同时包含低层图像重建指标和语义可靠性指标。

## 最小完成标准

第一版论文闭环必须至少完成：

- CIFAR-10 + AWGN sanity baseline。
- COCO2017 256x256 + AWGN high-resolution main baseline。
- 固定 CBR：`0.17`。
- 固定 SNR sweep：`[1, 4, 7, 13, 19]` dB。
- 四组方法：DeepJSCC、Blind diffusion、SNR-adaptive diffusion、Ours。
- 主语义指标：classification consistency / semantic drift rate / final failure rate。
- 至少一组 semantic drift failure case 可视化。

Kodak 用于视觉补充测试和展示，不作为唯一主统计集。Rayleigh、ImageNet subset、Dynamic_JSCC、DeepJSCC-l++ 和 DiT-style diffusion 都是 COCO-256 AWGN 最小闭环之后的扩展。

## Semantic Drift 主定义

第一版以冻结分类器 `T_cls` 为主语义模型：

```text
Drift-Origin = mean[ c(x_refined) != c(x) ]
Drift-GT = mean[ c(x_refined) != y ]
Refinement-Drift = mean[ c(x_refined) != c(x_hat) ]
Final-Failure = mean[ c(x_final) != c(x) ]
```

正式统计优先在 clean-correct 子集上进行，即原图能被 `T_cls` 正确识别的样本。CLIP similarity 可以作为辅助指标，但不能替代分类一致性主指标。

## 成功与失败判据

本项目的成功不是单个指标最高，而是证明 semantic control 改善了 diffusion refinement 的可靠性：

- 相比 blind diffusion，Ours 应降低 semantic drift 或 final failure。
- 相比 DeepJSCC，Ours 应保留主要感知质量收益。
- 相比只做 SNR-adaptive diffusion，Ours 应证明 semantic control 有额外价值。

如果 diffusion 让图像更真实但语义错误更多，必须记录为失败或负结果，不能算作有效提升。
