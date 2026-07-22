# S31b 强 JSCC FP32 稳定续训预注册（2026-07-21）

## 触发原因

`EXP-S31-STRONG-JSCC-001` 在 epoch 3 得到五档平均 `28.044783 dB / 0.958405 MS-SSIM`，曲线仍明显上升；随后 epoch 4 batch 418 的 AMP unscale 后梯度范数为非有限值，原实验按 fail-closed 规则停止，失败目录和 `STATE.json` 保留。失败前 best/latest 相同，SHA-256 为 `8e8f3b7b...fb0156`。

独立 systems-only 审计把同一 checkpoint 置于 FP32，在 batch `8/12/16/20/24/28/32` 上逐一执行前向、反向、gradient clip 和 AdamW step，全部有限；batch 32 峰值 allocated VRAM 约 `12,369.7 MiB`。因此本 follow-up 只处理数值稳定性和未收敛问题，不改变架构、码率、数据、信道、损失、SNR 或选择指标。

## 冻结修改

- 新实验 ID：`EXP-S31B-STRONG-JSCC-FP32-001`，绝不覆盖原失败输出。
- 只从原 epoch 3 best 加载模型权重；不加载原 optimizer、scheduler 或 AMP scaler 状态。
- 关闭 AMP，batch 仍为 32；新 AdamW 从 `5e-5` 开始，以 cosine 降到 `1e-6`，无 warmup，训练 8 epoch。
- 固定 COCO val 512 图与验证噪声保持原 S31 的 seed `20260752`，确保内部曲线可直接比较。
- checkpoint 仍只按五档平均 PSNR、再按 MS-SSIM 破平局选择；不读取 LPIPS、语义标签或 S20/S30 外部总体。

## 成功与停止判据

技术上要求全程 finite、功率误差不超过 `1e-5`、严格 `19,712 real` 且 checkpoint 可恢复。阶段进展要求 S31b best 五档平均 PSNR 严格超过初始化的 `28.044783 dB`。若末两轮仍持续显著上涨，只能依据本 COCO 内部曲线另行注册延长；S32 外部总体在最终停止前不得运行。
