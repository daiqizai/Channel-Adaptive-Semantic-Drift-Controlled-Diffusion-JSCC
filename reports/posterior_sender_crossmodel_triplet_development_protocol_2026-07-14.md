# 固定码率 cross-model triplet controller 开发协议

日期：2026-07-14。编号：`ANALYSIS-PC-SENDER-CROSSMODEL-DEV-001A/B`。

## 来源与开发属性

`G_aux source-JS ∩ G_gate(anchor/posterior top-1)` 在已暴露 seed `20260726` 上没有减少任何 T_cls new-error。事后诊断发现，五个漏检行均满足 recovered `G_aux` source top-1 与 `G_gate(anchor)` top-1 不一致；静态重算三方 top-1 自然一致性规则得到 new-error `5→0`。因此本规则明确由 seed `20260726` 导出，`20260725/20260726` 均只能作为 development，任何正结果都不是独立审计证据。

## 冻结规则

仅在以下条件全部成立时选择 posterior，否则回退 anchor：

1. `JS(q_recovered,G_aux(posterior))-JS(q_recovered,G_aux(anchor)) <= 0`；
2. `argmax(q_recovered) == argmax(G_gate(anchor))`；
3. `argmax(G_gate(anchor)) == argmax(G_gate(posterior))`。

其中 `q_recovered` 是现有 40-bit UInt4+BPSK×4 payload 恢复的 `G_aux(source)` 概率。规则不增加 source bit、不改变 160 payload symbols、不访问标签或 `T_cls`，也不引入 margin/confidence 阈值或 SNR 特例。`G_aux`、`G_gate`、`T_cls` checkpoint 保持互异且冻结。

## 判据与停止条件

两个 development seed 分别沿用 strict-rate 全部 gate；两者都必须 `POSITIVE` 才能冻结规则并启动全新 seed。任一失败则停止，不在已暴露 seed 上增加阈值。即便通过，也必须明确报告其平均质量余量和低/中 SNR 相对 reference raw 的质量回吐，不能只报告 new-error 清零。
