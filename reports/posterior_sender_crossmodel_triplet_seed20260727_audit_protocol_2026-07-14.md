# Cross-model triplet 固定方法新 seed 审计协议

日期：2026-07-14。审计编号：`ANALYSIS-PC-SENDER-CROSSMODEL-SEED-AUDIT-001`。

## 冻结依据与证据隔离

cross-model triplet controller 在已暴露 development seeds `20260725/20260726` 上分别得到 `POSITIVE`。这两份结果仅用于冻结方法，不能作为泛化证据。从本协议写入后，方法、门槛和全新 seed `20260727` 全部冻结。

先生成 seed `20260727` 的 unpunctured `c=8` reference。reference 只提供同 seed 的 S14 raw 公平对照；生成后只允许把 `per_sample.csv` SHA-256 写入 strict-rate audit config，不得根据 reference failure/new-error/质量改变 controller、payload 或成功判据。

## 冻结方法

- DeepJSCC、S13 B1、S14 diffusion、三步 `0.001` posterior correction 不变；
- 10 维 `G_aux(source)` probability 每类 UInt4，共 40 bit；BPSK、每 bit 重复四次；
- 总 `c=8` 65536 实符号，其中 payload 160、图像 65376，总 CBR `1/6`；payload 与图像共同通过一次 AWGN；
- receiver 擦除 payload 位置，posterior consistency 排除同一位置；
- final 取 posterior 当且仅当：

```text
source_fullprob_js_risk <= 0
AND argmax(q_recovered) == argmax(G_gate(anchor))
AND argmax(G_gate(anchor)) == argmax(G_gate(posterior))
```

- 不使用 `T_cls`、标签、source `G_gate` prediction、margin/confidence threshold 或逐 SNR 例外；
- `G_gate` checkpoint SHA-256 固定为 `708aa9e47db27d37080f39564886488beb7795ee9e9f21c4ae0325e6789d4f47`。

## 一次性成功门槛

沿用 strict-rate gate：五个 SNR 载荷 top-1/cosine 和整向量恢复率、每 SNR masked consistency、primary final failure 不高于同 seed unpunctured M2、primary final new-error 总量和逐 SNR 不高于 in-budget raw、new-error image-cluster 单侧 95% upper 不超过 0.5%、五 SNR 平均 final-minus-reference-raw PSNR 为正且 LPIPS 不增加。

所有 gate 同时通过才记为审计 `POSITIVE`。任一失败都必须原样记录，不允许在 seed `20260727` 调规则、位宽、重复数、阈值或 SNR 特例。即使通过，也应报告低/中 SNR 的 PSNR 回吐和约 45% 的 posterior coverage，不能称为全面优于 M2。
