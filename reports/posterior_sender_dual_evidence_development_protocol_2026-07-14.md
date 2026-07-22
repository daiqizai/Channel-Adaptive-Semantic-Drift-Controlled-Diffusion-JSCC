# 固定码率双证据 semantic decision 开发协议

日期：2026-07-14。编号：`ANALYSIS-PC-SENDER-DUAL-EVIDENCE-DEV-001A/B`。

## 目的与边界

前一版 `UInt4+BPSK×4 + G_aux source-JS zero veto` 的通信层在 seed `20260725/20260726` 均稳定，但 seed `20260726` 的 final new-error 为 `5>3`。本开发只判断一个无新阈值的自然交集规则能否覆盖单模型盲区，不改变 posterior correction、payload bit 数、重复数、符号位置、总 CBR 或 evaluator。

seed `20260725` 和 `20260726` 均已暴露，只能作为 development；即使结果为正也不能晋级。若两者均通过，才允许冻结同一规则并为全新 channel seed 单独写一次性审计协议。

## 冻结方法

- sender evidence：保持接收的 10 维 `G_aux` UInt4 概率，要求 `JS(q_recovered,G_aux(posterior))-JS(q_recovered,G_aux(anchor)) <= 0`；
- receiver guard：冻结、独立的 scratch `G_gate`，要求 `G_gate(posterior).top1 == G_gate(anchor).top1`；
- final 仅在两个条件同时成立时取 posterior，否则回退 anchor；
- `G_gate` 不看原图、标签或 `T_cls`，不产生额外发送符号；`G_aux`、`G_gate`、`T_cls` checkpoint 必须彼此不同；
- 不扫描 margin、confidence、JS 阈值或 SNR 特例。

固定 `G_gate` SHA-256：`708aa9e47db27d37080f39564886488beb7795ee9e9f21c4ae0325e6789d4f47`。

## 码率与信道合同

- 总 `c=8`、65536 个实符号、CBR `1/6`；
- 40 bit UInt4 payload，BPSK，每 bit 重复 4 次，共 160 个保留符号；
- 图像载荷剩余 65376 个实符号；payload 和图像一次共同通过 AWGN；
- receiver 擦除载荷位置，posterior measurement consistency 排除同一位置。

## Development 判据

每个 seed 分别沿用前一版全部 gate：载荷恢复、masked consistency、相对 unpunctured reference raw 的平均 PSNR/LPIPS、相对 in-budget raw 的 primary new-error 总量和逐 SNR、相对 reference raw 的 failure，以及 0.5% image-cluster 上界。两份 config 必须各自产生 `POSITIVE` 才允许进入新 seed 审计。

双证据规则若失败，不得在这两个 seed 上增加 margin 或选择阈值；应记录为组合仍不足或过度拒绝的负结果。
