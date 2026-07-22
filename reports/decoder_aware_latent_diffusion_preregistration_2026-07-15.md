# Decoder-Aware Channel-Matched Latent Diffusion 预注册（2026-07-15）

## 研究问题

`EXP-S17-LATDIFF-002` 已证明 channel-state matching 能在五个 SNR 上降低 DeepJSCC 活动 latent MSE，并在一次性 holdout 上相对 B0 得到平均 `+0.148715 dB/-0.035305 LPIPS`。但 13/19 dB 的 latent MSE 改善没有转化为 PSNR，且旧 B1 直接接在 diffusion 输出后出现输入分布偏移。本后继只检验一个解释：**等权 latent loss 没有刻画 frozen DeepJSCC decoder 对不同 codeword 方向的图像敏感性；加入通过 frozen decoder 回传的图像重建损失后，能否改善这种错配。**

这不是放弃 diffusion，也不更换表示空间、step matching、采样器、码率或信道。本轮不加入 measurement consistency、感知损失、语义 loss、文本/结构条件或新恢复器。

## 单变量与严格对照

两个分支都从同一 `EXP-S17-LATDIFF-002` 最佳 checkpoint warm-start，使用相同的三轮训练、学习率、batch、训练样本、随机种子、alpha 采样、6-step DDIM、selection split 和 checkpoint 指标：

1. `EXP-S17-LATDIFF-003-CONTROL`：继续使用 `epsilon MSE + 0.1*x0 latent MSE`；
2. `EXP-S17-LATDIFF-004-DECODER`：在完全相同的 base loss 上增加 `lambda_img * MSE(D(x0_pred), x_original)`。

`D` 是冻结 exact-rate DeepJSCC decoder。其参数不更新，但梯度允许从图像 MSE 经过 decoder 回到 epsilon predictor。reserved 80 坐标和未发送坐标仍固定为零；图像 loss 不增加任何信道符号。

因此正式因果比较是 `DECODER − CONTROL`，不是只把 decoder-aware 模型与少训练三轮的 parent 比较。parent 仍作为稳定性参照同表报告。

## loss 尺度诊断与冻结规则

在任何新 selection 指标和任何第 768--999 validation 图 holdout 指标之前，允许在训练 population 上用 warm-start checkpoint 做一次**无参数更新**的 loss 尺度诊断：

- 固定 seed `20260734`，只读取训练 loader 前 16 batches；
- 计算 `L_base=L_epsilon+0.1*L_x0` 与未加权 `L_img`；
- 目标是让初始化时 `lambda_img*L_img/L_base` 接近 `0.075`；
- 候选权重只允许 `[5,10,20,40]`；选择与 `0.075*mean(L_base)/mean(L_img)` 对数距离最小者，平局取较小值；
- 诊断只决定量纲，不查看 PSNR、LPIPS、分类器或 selection；输出写入独立目录并冻结到正式配置。

不得根据后续 selection/holdout 重新扫描 `lambda_img`。control 与 decoder-aware 各自只按冻结 selection mean matched-DDIM PSNR 选一个 checkpoint。

## 冻结合同

- frozen exact-rate DeepJSCC：SHA-256 `bca5b67a...bb606`；
- warm-start latent diffusion：SHA-256 `cfc52716...b4e1`；
- frozen B1：SHA-256 `7a295976...b7615a`；
- AWGN：`sigma^2=P/(2*gamma)`，SNR `[1,4,7,13,19]`；
- channel step：`alpha_channel=2*gamma/(2*gamma+1)`；
- 总预算：19,712 real symbols，其中 19,632 image coordinates、80 reserved payload coordinates；
- 模型：同一个 323,574 参数 masked epsilon predictor；
- sampler：matched 6-step deterministic DDIM，`measurement_blend=0`；
- 训练：同一 10,000 train population，3 epochs，batch 24，FP32，AdamW，LR `1e-4`，weight decay `1e-4`。

## 新 population 隔离

S17-002 已暴露 validation role 的第 0--511 张。本轮只使用剩余 488 张，并在读取任何新指标前冻结为：

- checkpoint selection：validation 第 512--767 张，共 256 张，channel seed `20260735`；
- 一次性 holdout：validation 第 768--999 张，共 232 张，channel seed `20260736`。

两个训练分支共享 selection，但 holdout 在两者 checkpoint 和哈希都冻结后才允许打开一次。official Imagenette validation 继续封存；COCO classifier 结果仍只是 pseudo semantic audit。

## 比较项与成功判据

holdout 在完全相同 sample/SNR/noise 上报告：B0、scalar LMMSE、fixed-7dB DDIM、parent matched DDIM、control matched DDIM、decoder-aware matched one-step/DDIM、B1、decoder-aware DDIM→旧 B1。

主要阶段成功要求全部满足：

1. decoder-aware matched DDIM 平均 PSNR 高于同预算、同训练步数 control；
2. decoder-aware 相对 control 至少 3/5 SNR 的 PSNR 为正；
3. decoder-aware 相对 B0 平均 PSNR 为正，且至少 4/5 SNR 为正；
4. decoder-aware 平均 LPIPS 不差于 B0；
5. decoder-aware 活动 latent MSE 在五个 SNR 都低于 raw；
6. AlexNet 和三分类器多数票 pseudo new error 均不多于 repair。

`decoder-aware DDIM→旧 B1` 是否超过 B1 继续报告，但不作为 decoder-aware loss 的主要因果成功条件；若仍失败，只能说明旧 B1 的输入分布问题尚未解决。所有配对差异按 image cluster bootstrap 95% CI 报告。若主要条件失败，结论必须记为部分成功或负结果。

## 预定输出

- 尺度诊断：`outputs/analysis/ANALYSIS-S17-LATDIFF-LOSS-DIAGNOSTIC-001/`
- control 训练：`outputs/EXP-S17-LATDIFF-003-CONTROL/`
- decoder-aware 训练：`outputs/EXP-S17-LATDIFF-004-DECODER/`
- fresh holdout：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-003/`
- 中文阶段报告：`reports/decoder_aware_latent_diffusion_stage_result_2026-07-15.md`

本实验最多证明 frozen-decoder-aware objective 是否改善 channel-matched latent diffusion，不授权声称已经超过 B1、SGD-JSCC 论文或完成 semantic-tail safety。

## loss 尺度冻结记录（正式训练前）

预注册诊断已按训练 loader 前 16 batches 完成，未访问 selection 或 holdout：`mean L_base=0.67200185`、`mean L_img=0.002449218`，连续目标权重为 `20.5781`。按冻结候选和对数最近规则选定 `lambda_img=20.0`，对应初始化 weighted-image/base ratio `0.0728932`。诊断摘要 SHA-256 为 `b43b1192753606ce44e0396d475725c4ae306edd3ec90fdbcc99a450c242f0d8`；正式配置在任何新 selection 输出前据此把唯一待定尺度冻结为 `20.0`，不得后续扫描。

control 已按冻结合同完成三轮，selection 仅用于在 epoch 1 冻结最佳 checkpoint；其 SHA-256 为 `edbcbdbd7f78384decab40572728fab384e06bef854445af9452ee555aab2b1f`。该哈希在 decoder-aware 正式训练和 holdout 前写入干预配置；fresh holdout 此时仍未访问。

decoder-aware 三轮训练完成后，selection 按原指标冻结 epoch 2 checkpoint，SHA-256 为 `5b708117a5d25cad0a5909a24f85bb32d1b5dc11c83146ba8c98fad5ee35d98f`。selection mean matched-DDIM PSNR 为 `26.377630 dB`，control 最佳为 `26.356077 dB`；该差值只作 checkpoint 冻结记录，不替代一次性 holdout。两个哈希均已写入 holdout 合同，随后才允许访问 validation 第 768--999 张。
