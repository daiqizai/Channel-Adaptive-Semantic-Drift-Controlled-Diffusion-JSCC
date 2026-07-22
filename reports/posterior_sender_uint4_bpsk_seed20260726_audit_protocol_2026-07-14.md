# UInt4-BPSK 固定方法新信道种子审计协议

日期：2026-07-14。审计编号：`ANALYSIS-PC-SENDER-DIGITAL-SEED-AUDIT-001`。

## 冻结点

开发 seed `20260725` 已完成，`UInt4+BPSK×4+zero veto` 的开发输出 verdict 为 `POSITIVE`。从本协议写入起，以下内容全部冻结：

- 新 channel seed 只取此前未使用的 `20260726`；
- Imagenette policy-dev 1894 张、clean threshold `0.50`、SNR `[1,4,7,13,19]`；
- 10 维 `G_aux` 概率、每类 UInt4、40 bit、BPSK、每 bit 重复 4 次、160 个保留实符号；
- `c=8` 总 65536 实符号、CBR `1/6`、共同 AWGN、receiver 擦除和 masked consistency；
- S13 B1、S14 六步 diffusion、三步 posterior correction、JS 自然零阈值 veto；
- `G_aux` 和 `T_cls` checkpoint；官方 Imagenette validation 继续封存。

先用同一 seed 生成 unpunctured `c=8` reference raw 表，只用于公平质量/失败率对照；生成 reference 后仅把其 SHA256 写入审计配置，不据其 outcome 改方法。

## 审计门槛

完全沿用开发阶段门槛：

1. 五个 SNR 的 source/recovered top-1 和 cosine 均不低于 95%；
2. 五个 SNR 的 40-bit 整向量无误率均不低于 95%；
3. primary final failure 不高于同 seed unpunctured reference raw；
4. primary final new-error 总数和逐 SNR 均不高于 in-budget raw；
5. final new-error image-cluster 单侧 95% upper 不超过 0.5%；
6. 五个 SNR 平均 final-minus-reference-raw PSNR 为正、LPIPS 不增加；
7. masked data-consistency 每个 SNR 均下降。

所有门槛同时通过才记为新 seed 审计 `POSITIVE`。任何门槛失败均保持原样报告，不允许在 `20260726` 上重新选择量化位数、重复数或阈值。结果报告使用中文。
