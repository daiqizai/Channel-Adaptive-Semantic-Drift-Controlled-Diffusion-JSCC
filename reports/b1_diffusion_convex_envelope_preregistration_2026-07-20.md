# S21-004 B1–Diffusion 单调凸融合诊断预注册（2026-07-20）

## 1. 目的与边界

S21-001/002 的 learned sigmoid gate 发生数值塌缩，S21-003 的 fixed-gate residual 发生 tanh 全饱和；三者都未访问 holdout，且均没有训练 checkpoint 通过 selection 的 B1 Pareto 门槛。按照预先停止规则，本轮不再修改同类 adapter 的 LR、scale、gate 或 loss。

S21-004 改做一个无训练、无额外参数的稳定合并诊断：直接在 B1 与 frozen matched-diffusion 图像之间做 SNR 单调凸组合。它不是最终创新方法，而是回答更基础的问题：两条现有输出是否存在无需学习即可达到的非退化 Pareto 合并；若存在，它将成为后续风险控制器必须超过的下界。

## 2. 冻结候选

`x_blend(SNR)=(1-alpha_SNR)×B1+alpha_SNR×D`。

- alpha 候选：`[0,0.025,0.05,0.075,0.10,0.15,0.20,0.30]`。
- 只选择 1/4/7 dB；13/19 dB 冻结 alpha=0，逐像素精确 B1。
- 单调约束：`alpha_1 >= alpha_4 >= alpha_7 >= 0`。
- 使用 S21 frozen selection 选择，holdout 继续封存。
- 每个低 SNR 的 selection PSNR 必须不低于 B1，五 SNR aggregate LPIPS 不得劣于 B1。
- 满足约束的候选按 aggregate mean PSNR 最大选择；并列选 alpha 总和更小者，再按字典序。

所有候选、约束和顺序在计算任何 blend selection 指标前冻结。本轮不扫描图像自适应阈值，不使用原图进行逐样本选择。

## 3. Holdout 开启条件与成功判据

只有 selection 选出至少一个非零 alpha，才冻结 policy SHA 并首次访问 256 张 holdout。若全零 B1 胜出，直接记负结果，不访问 holdout。

holdout 主判据：

1. blend−B1 PSNR 的 source-cluster 95% CI 下界 `>0`；
2. blend−B1 LPIPS 的 95% CI 上界 `<0`；
3. 五 SNR PSNR delta `5/5` 非负，13/19 dB逐像素最大差为 0；
4. 三分类器 majority、相对 B1 的 new 不大于 repair。

COCO 分类仍是 pseudo 语义辅助诊断。即使全部通过，本轮只证明稳定凸融合下界，不声称完成 semantic-risk controller。
