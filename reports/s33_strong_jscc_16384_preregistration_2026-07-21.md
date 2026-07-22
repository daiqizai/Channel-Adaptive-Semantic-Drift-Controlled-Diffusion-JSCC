# S33：16,384-real 等码率 Strong JSCC 预注册

日期：2026-07-21
状态：在任何 S33 模型输出产生前冻结；用户已确认配置与判定 margin。
范围：本轮只执行 S33；S34 消融、S35 matched diffusion、S36 official Imagenette validation 均不执行。

## 已知信息与未知问题

已知 S31b `19,712-real` strong 在 S32 policy-dev 上相对 `16,384-real` author-JSCC 聚合 PSNR 为 `+0.433774 dB`，95% CI `[+0.328020,+0.554007]`。该结果不是严格等码率结论；13/19 dB PSNR 分别低 `0.028961/0.225914 dB`，LPIPS 也较差。

S33 唯一主问题是：在双方都严格使用 `16,384 real` 时，clean-room strong backbone 是否仍显著超过、非劣追平或劣于 author-JSCC？S33 outcome 当前未知。S32 population 已知，故本轮仍是 policy-development 等码率 gate，不冒充 independent final test。

## 冻结模型与物理合同

- 新 strong 原生 latent=`64x16x16=16,384 real`，即 `8,192 complex channel uses`，source-relative CBR=`1/24`。
- trainable parameters=`31,028,163`；author-JSCC 约 `31.289M`，参数差不到 1%。
- mask、padding、固定裁剪、重发、side information 均为 0。
- AWGN 沿用 paired-real half-variance：每实坐标噪声方差 `P/(2*10^(SNR/10))`。
- 外部比较逐 key 使用与 author-JSCC 相同 canonical noise 的前 `16,384` 个实坐标。
- S31b `19,712-real` checkpoint SHA-256=`2f8972a943599bae016f6f64550ca81ea5f861654d9ace6931aebe6cf9057ca8` 永久冻结，仅作历史参照，不用于 S33 初始化。

## 训练合同

用户确认使用：**随机初始化 + FP32 12 epochs + 离散五档 SNR 训练**。

- seed=`20260751`；COCO train2017 全量，val2017 固定 512 图；沿用 S31 的 crop/flip 与 validation noise seed=`20260752`。
- SNR 对每张训练图从 `[1,4,7,13,19] dB` 离散均匀采样。禁止描述为连续随机 SNR。
- 连续 `Uniform[1,19] dB` 只允许写 future work/后续扩展，不得倒写为当前 strong 的原因。
- 只优化 MSE；checkpoint 按五档 aggregate PSNR、再按 aggregate MS-SSIM 选择。S32 population、LPIPS、语义标签和 official Imagenette validation 均不得参与 checkpoint selection。
- 主阶段：随机初始化、FP32、4 epochs、AdamW、LR `2e-4→2e-6` cosine、warmup 1,000 steps。
- 续训阶段：主阶段 best checkpoint 冻结 SHA 后，只加载 model state；fresh AdamW/scheduler/scaler，FP32 8 epochs、LR `5e-5→1e-6` cosine、无 warmup。
- 任一 non-finite loss/gradient/metric、功率或码率合同失败均 fail-closed，保留失败目录，不覆盖重跑。

主阶段配置：`configs/s33_strong_jscc_16384_fp32_main.yaml`。续训配置必须在主阶段 best SHA 已知后生成并冻结，不得预填未知 SHA。

## 外部比较与指标

最终 checkpoint 冻结后，复用 S32 的 64 张 Imagenette policy-dev 图、channel seeds `[20260748,20260749,20260750]`、SNR `[1,4,7,13,19]`，形成每方法 960 个观测。

必须报告：

- per-SNR 和五档聚合 PSNR、LPIPS、MS-SSIM；
- `T_cls` failure、strong 相对 author 的 new/repair；
- strong−author 配对差和按 source image 聚类、10,000 次 bootstrap 的 95% CI；
- config/script/checkpoint/CSV/noise SHA、symbol ledger、功率误差；
- 13/19 dB 独立边界，不用聚合值隐藏。

## 用户确认的判定规则

主判据为 aggregate `PSNR(strong)-PSNR(author)`：

- 双侧 95% CI 下界 `>0 dB`：**显著超过**。
- CI 下界位于 `(-0.10,0] dB`：**在 0.10 dB margin 下追平/非劣**。
- CI 下界 `<-0.10 dB`：**劣于**。
- 恰等于 `-0.10 dB` 时按边界不确定处理，不写追平或劣于，需原样报告。

LPIPS、MS-SSIM 或 semantic failure 若与 PSNR 方向冲突，只能写 Pareto，不称全面超过。该 `0.10 dB` margin 也适用于后续所有 vs author-JSCC 表述，禁止结果后修改。

## 封存和停止规则

- official Imagenette validation 全程封存。
- S33 明确结论完成后停止；本轮不自动进入 S34--S36。
- 不因 S33 结果临时延长 epoch、切换连续 SNR、warm-start、增加感知损失或回选 S31b。
