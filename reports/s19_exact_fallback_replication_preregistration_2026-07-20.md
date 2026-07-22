# S19 强融合 + exact-B1 fallback 跨总体复现预注册（S26）

日期：2026-07-20。分析 ID：`ANALYSIS-S26-S19-XF-REPLICATION-001`。

## 唯一问题

S19 已证明 diffusion 相对等容量 B0-only control 提供额外信息，并相对 B1 提升约 `0.102 dB`，但 13/19 dB 相对 control 有负迁移。S25 又证明 S23 的弱 feature direction 即便做 oracle 逐图幅度选择也没有足够上限。

本轮只回答：**把冻结 S19 fusion 限定在真正有 diffusion observation 的 1/4/7 dB，并在 13/19 dB 从结构上强制返回冻结 B1，能否在另一批图片上同时保留有意义增益、diffusion 因果优势和 exact fallback。**

## 冻结方法

- 1/4/7 dB：`routed_fusion = frozen S19 fusion(B0, D)`；matched control 为相同结构、参数量和训练流程的 `frozen S19 control(B0, B0)`。
- 13/19 dB：`routed_fusion = routed_control = frozen B1`，必须逐像素精确一致。
- 路由只读取已知 SNR，不读取原图、分类器、标签或 sample ID；新增参数和 side-information symbols 均为 0。
- S19 control/fusion checkpoint 及其 SHA 已冻结，不训练、不微调、不访问目标 selection。

## 目标总体和事后性

目标是 S21/S23 的 256 图×5 SNR holdout。该总体的 B1/S23 outcome 已知，因此不是全新未暴露图像总体；但 S19 checkpoint 从未在该总体运行，本轮 S19/control 输出在注册时未知。它只能称 frozen cross-population method replication，不能伪装成完全 pristine final test。

在任何目标 S19 输出产生前，冻结上述 route、checkpoint、输入 manifest/cache SHA、10,000 次 source-image cluster bootstrap、语义判据和全部成功门槛。运行中不做 selection 或 threshold 调整。

## 成功门槛

1. 13/19 dB routed fusion/control 相对 B1 最大逐像素差不超过 `1e-7`。
2. routed fusion−control：PSNR CI 下界大于 0，LPIPS CI 上界小于等于 0。
3. routed fusion−B1：平均 PSNR 至少 `+0.05 dB`，CI 下界至少 `+0.03 dB`，LPIPS CI 上界不超过 0。
4. 五个 SNR 的 routed fusion−B1 PSNR 均非负。
5. majority new 不多于 repair，且 routed fusion majority failure 不多于 routed control。

全部通过才称“强表示 + exact fallback 的跨总体阶段性正结果”。这仍不等于最终 SOTA：目标 population 已用于 S23，语义仍是三分类器辅助诊断，之后还需要一次真正 fresh population 或统一外部 common contract。

本轮全部使用本地冻结 cache/checkpoint，无联网、无下载。
