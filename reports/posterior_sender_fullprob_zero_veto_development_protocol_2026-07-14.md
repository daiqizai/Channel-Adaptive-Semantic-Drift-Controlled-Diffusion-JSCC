# Sender 全概率零阈值 Veto 开发协议

日期：2026-07-14。分析编号：`ANALYSIS-PC-SENDER-DEV-001`。

## 目的与证据边界

`ANALYSIS-PC-RISK-FAIL-001` 已事后看到 seed `20260725` 的 2 个漏检新错和 11 个误拒修复，因此本工作包只能是 policy-dev 可达性开发，不能作为独立验证。它只回答：一个自然、无阈值搜索的 source-grounded score 在完整 1894×5 总体上是否仍有价值，从而决定是否值得继续做严格计码率和信道错误实现。

原 receiver-risk verdict 保持 `NEGATIVE`。官方 Imagenette validation 继续封存。

## 冻结方法

- sender 与 receiver 共用已冻结、只在 `cls_train/cls_cal` 训练选点的 scratch `G_aux` EfficientNet-B0；`T_cls` 仍只作 outcome evaluator；
- sender 对原图计算校准 10-way probability，每维四舍五入为 uint8 后归一化，共 80 raw bits；
- 本轮假设 description 无误到达，只是 feasibility，不宣称 matched CBR、可部署或可忽略码率；
- receiver 计算 anchor 和 corrected posterior 的 `G_aux` probability；
- 风险定义为：

```text
r_JS = JS(q_source, p_posterior) - JS(q_source, p_anchor)
```

- 唯一规则：`r_JS <= 0` 接受 posterior，否则回退 B1 anchor；
- threshold 固定为自然零点，不扫描 threshold、不使用 per-SNR 规则、不设例外；
- DeepJSCC、S13 B1、S14 diffusion、三步 posterior correction、seed `20260725`、SNR `[1,4,7,13,19]` 全部保持不变。

## 完整总体检查

完整输出必须覆盖 9470 个唯一 `(sample_id, seed, SNR)` 行，并报告：

1. description 接受率；
2. primary raw/posterior/final failure、new-error、repair，分 SNR 结果；
3. final new-error image-cluster 单侧 95% Clopper-Pearson upper；
4. final-minus-raw PSNR/LPIPS，整体及五个 SNR；
5. 2 个已知漏检是否被保护，11 个已知 posterior repair 是否被保留；
6. source probability score 与 top-1 description 的差异。

可达性只有在以下条件全部满足时才算值得进入 rate-accounted pilot：primary final new-error 不高于 raw（总量与逐 SNR）、primary final failure 不高于 raw、cluster upper 不超过 `0.5%`、mean final-minus-raw PSNR 为正且 LPIPS 不增。即使通过，也只能说明 80-bit noiseless description 值得编码，不代表最终 M3 已通过。

若失败，停止该 sender full-probability veto，不允许继续在 seed `20260725` 扫 threshold。新增结果报告必须使用中文。
