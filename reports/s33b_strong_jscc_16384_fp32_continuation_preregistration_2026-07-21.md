# S33b：16,384-real Strong JSCC FP32 续训预注册

日期：2026-07-21
状态：在任何 S33b 输出产生前冻结。
父预注册：`reports/s33_strong_jscc_16384_preregistration_2026-07-21.md`。

S33 主阶段按冻结配置完成 4 个 FP32 epoch。期间在 epoch 2 完整落盘后发生一次外部进程终止；config/snapshot/checkpoint SHA 一致，按同配置 `--resume` 只重跑未落盘的 epoch 3。四个正式点的 aggregate PSNR 为 `26.062294/27.459733/28.280950/28.587876 dB`，全部 finite 且单调提升。best 为 epoch 3：

- checkpoint：`outputs/train/EXP-S33-STRONG-JSCC-16384-FP32-001/checkpoints/best.pt`
- SHA-256：`b698797f93f56cd6d1617ee18fdd39493fe08e58e994b21ccb059ffb19ce26c4`
- aggregate：`28.587876 dB / 0.961212 MS-SSIM`
- per-SNR PSNR：`26.751881/27.895735/28.738001/29.632153/29.921610 dB`
- max normalized-power error：`2.3842e-7`

本阶段不引入新方法选择。严格执行父预注册已冻结的后 8 epochs：

- 只加载上述 best model state，不加载 optimizer、scheduler 或 scaler；
- fresh AdamW，FP32，LR `5e-5→1e-6` cosine，无 warmup；
- seed、COCO 数据、离散五档逐图 SNR、固定 validation noise、MSE-only selection 全部不变；
- 8 epochs 全部运行，不依据 S32/author-JSCC 结果早停或延长；
- non-finite/功率/码率失败则 fail-closed；official Imagenette validation 继续封存。

配置：`configs/s33b_strong_jscc_16384_fp32_continuation.yaml`。本阶段完成并冻结最终 best SHA 后，才创建 S33 外部等码率比较配置。
