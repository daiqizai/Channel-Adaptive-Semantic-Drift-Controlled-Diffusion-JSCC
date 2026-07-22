# S21-003 B1 锚定辅助残差预注册（2026-07-20）

## 1. 两次门控失败后的限定修正

S21-001 的显式 gate sparsity 在 epoch1 把 sigmoid gate 压到 0；S21-002 移除该正则后，fusion gate 在 epoch1–3 保持非零并获得显著 LPIPS 改善，但 epoch4 仍自发塌缩到 0，eligible best 退回 epoch0 B1。两次均未访问 holdout，失败目录和记录保留。

这说明当前“learned sigmoid gate × learned residual”存在尺度/可识别性退化。按照 S21-002 的预先边界，本轮不再扫描 gate penalty、bias 或温度，而是删除可塌缩自由度。

## 2. 唯一结构变化

S21-003 继承相同 fresh population/cache、冻结 B1、matched diffusion、12 通道输入、452,420 参数 adapter、SNR envelope、loss、训练预算、selection 和成功判据。唯一方法变化为：

`A=sigmoid(gate_logits) → A=1`。

因此：

`x_final = clip(B1 + g_max(SNR) × tanh(R(B1,aux,|aux-B1|,SNR,structure)))`。

gate head 参数保留但 forward 不使用，使 control/fusion 参数量仍完全相同；结论不会把这些无效参数包装为贡献。13/19 dB 的 `g_max=0` 继续逐像素精确回 B1。低中 SNR 的不确定性由 `|aux-B1|` 等输入隐式决定 residual 内容，本轮不再声称学到了显式可解释 risk gate。

## 3. 因果对照与数据边界

- control auxiliary=B0；fusion auxiliary=matched diffusion；其余完全相同。
- train/selection 已作为 development 数据使用。
- 256 张 holdout 从未读取；两个 S21-003 checkpoint SHA 冻结后一次性访问。
- selection 仍要求 PSNR 高于 epoch0 B1 且 LPIPS 不劣于 epoch0，否则保留 epoch0。
- 原 S21 八项 holdout 判据不变，其中“spatial gate”只作固定值 1 的实现诊断；核心仍是 fusion−control PSNR/LPIPS cluster CI、分 SNR、B1 anchor 与 majority new/repair。

若 S21-003 selection 仍不能超过 epoch0 B1，则停止当前 pixel auxiliary adapter，不用同一 selection 继续改 residual scale/loss；阶段结论记为 matched diffusion 有感知信息但本参数化未形成保真 Pareto 改善。
