# S31b FP32 稳定续训 002 预注册修正（2026-07-21）

`EXP-S31B-STRONG-JSCC-FP32-001` 启动后、完成任何 epoch 或 validation 输出前，发现其总 seed 从原 S31 的 `20260751` 改成 `20260753`。当前 loader 同时用总 seed 选择 COCO val2017 的固定 512 图，因此 `-001` 与原 S31 不是同一 validation population，违反 `reports/s31b_strong_jscc_fp32_continuation_preregistration_2026-07-21.md` 中“固定 512 图保持原 S31”的文字合同。

`-001` 已主动 SIGINT，`STATE.json` 记录 `KeyboardInterrupt`，没有 history row、checkpoint 或任何 validation 指标；输出保留，不复用 ID。新实验 `EXP-S31B-STRONG-JSCC-FP32-002` 只做以下修正：

- 总 seed 恢复原 S31 的 `20260751`，从而同时恢复相同的 512 图子集；
- experiment/output/smoke ID 改为 `-002`，禁止覆盖 `-001`；
- 其余初始化 checkpoint/SHA、FP32、batch、optimizer、学习率、8 epoch、架构、码率、信道、损失和选择规则逐项不变。

该错误在任何目标 validation outcome 产生前发现，修正不使用质量结果。S32 external population 仍未访问。
