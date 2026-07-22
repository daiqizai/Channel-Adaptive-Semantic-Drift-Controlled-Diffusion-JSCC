# B1 特征注入逐图幅度上限诊断预注册（S25）

日期：2026-07-20。分析 ID：`ANALYSIS-S25-B1FA-HEADROOM-001`。

## 为什么先做这个诊断

S23 已经证明固定 `alpha=0.15` 可以在独立 holdout 上小幅改善三项质量指标并在 13/19 dB 精确回退 B1，但 PSNR 增益只有约 `0.00057 dB`。在训练新的 controller 之前，必须先回答：同一条 S23 feature direction 是否真的存在足够大的逐图幅度选择空间。

如果连使用原图质量和三分类器结果的理想 oracle 都没有明显 headroom，那么任何 receiver-visible controller 都不可能把这条方向提升成强方法，应立即停止继续堆 gate。

## 冻结输入与边界

- 已知 S23 selection 和 holdout 结果；本分析明确是 development diagnostic，不伪装成独立验证。
- 只访问既有 S21/S23 `selection` 256 图×5 SNR，不访问 S23 holdout，不生成新 holdout 结论。
- B1、S23 epoch-1 projection direction、DeepJSCC cache、diffusion cache 和三分类器全部冻结并校验 SHA。
- 不训练新参数；只枚举 S23 已经注册过的 12 个 alpha。
- 13/19 dB 的 architecture envelope 仍为 0，因此所有 alpha 都必须精确等于 B1。

## 三个只用于诊断的策略

1. `fixed_0.15`：当前 S23 固定策略。
2. `psnr_oracle`：逐 sample/SNR 使用原图选择 PSNR 最大的 alpha。这是不可部署的纯质量上限。
3. `semantic_safe_psnr_oracle`：若 B1 的三分类器多数票相对原图正确，则禁止选择多数票失败的 candidate；在剩余候选中选择 PSNR 最大者。它同样使用原图和评估器，只是“无新增 majority failure”的理想上限，不是可部署 controller。

所有 oracle 平局均选择更小 alpha。额外记录 `lpips_oracle`，仅用于描述感知上限。

## 继续/停止门槛

只有以下四项同时成立，下一轮才允许开发 receiver-visible amplitude controller：

- semantic-safe oracle 相对 fixed `0.15` 的 PSNR 至少 `+0.02 dB`；
- 其 source-image cluster bootstrap 95% CI 下界大于 0；
- LPIPS 不差于 fixed `0.15`；
- 在 1/4/7 dB 中，至少 10% 行选择不同于 `0.15` 的 alpha。

若失败，结论是“S23 one-epoch feature direction 的逐图幅度 headroom 不足”，后续返回 S19 的更强 residual representation，而不是继续调 threshold。

## 声明边界

本分析的 oracle 使用原图和评估分类器，绝不作为方法结果、部署方案或 semantic-safety 证据。它只回答可行性上限问题。official Imagenette validation 保持封存；本轮无联网、无下载。
