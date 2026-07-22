# SGD-inspired Source Semantic Guidance: Significant Result and Boundary

Date: 2026-07-10

## Outcome first

模仿 SGD-JSCC 是有价值的，但结果把路线清楚地分成了两半：

- coarse semantic description 只用于末端 accept/reject gate：失败；
- fine sender-side structure 直接进入 restoration：有很强的可达收益，值得继续。

## Coarse description negative result

Imagenette policy-dev 内固定拆分 `semantic-select=945`、`semantic-audit=949`。发送端描述为 scratch `G_gate(original)` 的 4-bit top-1 或 80-bit uint8 probability vector，假设无噪声；`T_cls` 只作独立 outcome evaluator。

预注册的 CE/JS/cosine/source-class probability 距离规则没有任何候选同时满足选择集零 accepted-new-error image cluster 与至少 50% M2 PSNR 保留。fallback 选中的最保守规则在 audit 上：

- failure `0.028627`，比 M2 高 `+0.016078`；95% CI `[+0.008627,+0.023922]`；
- accepted-new-error conservative upper `0.005622`；
- PSNR `+0.0258 dB`，只保留 `3.26%` M2 gain。

因此瓶颈不是“缺一句描述”，而是 description-to-image grounding 不够可靠；继续扫 receiver-side threshold 不再是优先路线。

## Fine source-edge positive feasibility result

`EXP-S4-011` 与 receiver-edge `EXP-S4-008` 匹配 split、seed、五个 SNR、模型容量、输入通道数、residual gates、optimizer/loss、crop 和 60 epochs。唯一方法差异是 Sobel/Laplacian 条件来自 sender original，而不是 receiver M0。

| SNR | Receiver-edge ΔPSNR | Source-edge ΔPSNR | Source − receiver |
|---:|---:|---:|---:|
| 1 | +1.3121 | +5.1594 | +3.8473 |
| 4 | +0.9537 | +4.7981 | +3.8443 |
| 7 | +0.7831 | +4.5086 | +3.7255 |
| 13 | +0.7965 | +4.0409 | +3.2444 |
| 19 | +0.8536 | +3.7664 | +2.9128 |

跨 SNR paired sample-cluster bootstrap：`+3.5149 dB`，95% CI `[+3.2602,+3.7652]`；五个 SNR 均为正。辅助 AlexNet pseudo failure 从 `0.3625` 降到 `0.2062`。

## What this proves—and what it does not

它证明当前 residual restoration architecture 能有效利用 sender fine structure；它不证明通信系统在公平预算下提升 3.51 dB。oracle 让 full-resolution structural maps 完美到达，edge rate、调制、FEC 和 channel error 都未计，总 CBR 未定义。

允许表述：source fine semantics 是强可行方向；末端 coarse router 不是。

禁止表述：EXP-S4-011 是 matched-CBR 方法、可部署 M2/M3，或已解决 supervised semantic safety。

## Fixed next direction

1. 训练/接入 main-image CBR≈`1/8` 的 DeepJSCC。
2. 建立 edge CBR≈`1/24` 的独立 lossy JSCC path，使 total≈`1/6≈0.167`。
3. 把 `edge_hat` 注入 residual restoration；加入 edge-channel SNR/error ablation。
4. 用当前 CBR 0.17 no-side-info baseline 做 matched-total-rate 对照。
5. 在 Imagenette policy-dev 用独立 `T_cls` 重新检查 failure、repair、accepted new error；通过前 official val 继续封存。

关键产物：

- `outputs/analysis/imagenette_source_semantic_description_policy_dev/REPORT.md`
- `outputs/analysis/exp_s4_011_source_edge_oracle_vs_receiver_edge/REPORT.md`
- `reports/imagenette_source_semantic_description_preregistration_2026-07-10.md`
- `reports/source_edge_oracle_preregistration_2026-07-10.md`
