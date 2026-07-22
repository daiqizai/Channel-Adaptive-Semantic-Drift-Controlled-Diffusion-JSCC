# S21-002 门控塌缩修正预注册（2026-07-20）

## 失败事实

`EXP-S21-B1AGF-001` 在 epoch1 后，control/fusion selection spatial gate 分别降到约 `1.32e-44/0`，注入量约 `1.37e-45/0`，两者 PSNR/LPIPS 完全相同并精确退化为 B1。原因是 residual head 零初始化时，gate 暂时没有 reconstruction gradient，而预注册的 `0.0001×mean(gate)` 单独驱动 gate logits 快速降到 sigmoid 数值饱和区。

该训练在 epoch2 selection 完成前人工终止，exit code 130；holdout 未访问，失败目录与 `failure.json` 保留，不覆盖、不包装成方法负结果。

## 唯一修正

S21-002 继承 S21-001 的全部 frozen population、cache、B1、diffusion、模型结构、初始化、loss、训练预算、selection 规则、holdout 和成功判据，只修改：

`spatial_gate_mean_weight: 0.0001 → 0.0`

这样 gate 在第一步保持 0.5，zero residual head 先从 reconstruction/perceptual loss 获得梯度；后续 gate 再通过非零 residual 获得任务梯度。没有增加新模块或新调参。

## 数据使用边界

- S21-001 已访问 train 与 selection，因此 S21-002 仍把它们作为 development 数据。
- S21 holdout 从未被训练/selection/人工诊断读取；S21-002 两个 checkpoint 及 SHA 冻结后才能首次访问。
- official Imagenette validation 继续封存。
- 若 S21-002 再发生 gate 数值塌缩或没有 selection 改善，停止该 spatial-gate 参数化，不继续针对同一 selection 扫描 gate penalty/bias。
