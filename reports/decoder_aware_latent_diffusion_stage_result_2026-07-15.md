# Decoder-Aware Channel-Matched Latent Diffusion 阶段结果（2026-07-15）

## 阶段结论

本轮得到一个**统计稳定但幅度有限的正结果**：给 channel-matched latent diffusion 增加 frozen DeepJSCC decoder-aware 图像重建目标，确实优于同一 parent、同三轮训练预算、同 seed 的无图像损失 control。在一次性 232 图×5 SNR fresh holdout 上：

- decoder-aware − control：PSNR `+0.021605 dB`，image-cluster 95% CI `[+0.018883,+0.024640]`；
- decoder-aware − control：LPIPS `-0.002502`，95% CI `[-0.002824,-0.002203]`；
- decoder-aware − control：活动 latent MSE `-0.002324`，95% CI `[-0.002370,-0.002278]`；
- decoder-aware − parent：PSNR `+0.021006 dB`，95% CI `[+0.019132,+0.022939]`；
- decoder-aware − B0：PSNR `+0.174221 dB`，95% CI `[+0.150695,+0.199403]`，LPIPS `-0.038540`。

这说明上一轮提出的 decoder-loss mismatch 解释是成立的：图像域梯度能让 latent denoiser 学到更有利于 frozen decoder 的修正方向，而且收益不能用“多训练三轮”解释。

但预注册总判定仍为 `NEGATIVE_OR_PARTIAL`。decoder-aware 相对 control 只在 1/4/7 dB 明确改善，13 dB 的 PSNR 差异跨零，19 dB 小幅显著下降；相对 B0 也仍只有 3/5 SNR 为正，没有达到 4/5。旧 B1 直接接在 diffusion 后依旧受输入分布偏移影响。因此本轮升级的是**低/中 SNR 的 diffusion prior**，尚未完成全 SNR identity-safe 控制或最终 B1 融合。

## 实验合同与隔离

- parent：`EXP-S17-LATDIFF-002`，SHA-256 `cfc52716...b4e1`；
- control：`EXP-S17-LATDIFF-003-CONTROL`，最佳 epoch 1，SHA-256 `edbcbdbd...b2b1f`；
- decoder-aware：`EXP-S17-LATDIFF-004-DECODER`，最佳 epoch 2，SHA-256 `5b708117...5d98f`；
- 两个后继都从同一 parent warm-start，训练 3 epochs、FP32、AdamW、LR `1e-4`，其余 batch/alpha/step/mask/码率合同相同；
- 唯一方法差异是 decoder-aware 分支加入 `20*MSE(D(x0_pred),x)`；`D` 冻结但允许对输入回传；
- `lambda_img=20` 来自训练 population 前 16 batches 的预注册纯尺度规则，对应初始 weighted-image/base loss ratio `7.289%`；诊断未读取 selection/holdout；
- selection：此前未暴露的 validation 第 512--767 张，seed `20260735`；
- fresh holdout：validation 第 768--999 张，共 232 张，seed `20260736`；
- AWGN `[1,4,7,13,19]`，`sigma²=P/(2*gamma)`，19,712 real symbols，其中 19,632 image coordinates、80 reserved coordinates；
- 没有联网、没有下载、没有访问 official Imagenette validation。

selection 上 decoder-aware/control 相对 B0 分别为 `+0.177963/+0.156410 dB`，差 `+0.021553 dB`；与 fresh holdout 的 `+0.021605 dB` 几乎一致，说明小幅收益没有明显 selection 偶然性。

## Fresh holdout 主表

| 方法 | PSNR | MS-SSIM | LPIPS | 相对 B0 PSNR |
|---|---:|---:|---:|---:|
| B0 | `25.949091` | `0.919500` | `0.301593` | — |
| parent matched DDIM | `26.102306` | `0.931252` | `0.265138` | `+0.153215` |
| same-budget control | `26.101707` | `0.931412` | `0.265555` | `+0.152616` |
| decoder-aware one-step | `26.063233` | `0.929988` | `0.266952` | `+0.114141` |
| decoder-aware 6-step DDIM | **`26.123313`** | **`0.931980`** | **`0.263053`** | **`+0.174221`** |
| frozen B1 | `26.986561` | `0.942907` | `0.187668` | `+1.037469` |
| decoder-aware DDIM→旧 B1 | `26.787758` | `0.936546` | `0.221261` | `+0.838667` |

decoder-aware DDIM 仍明显低于 B1；它的意义是让 diffusion 支路比 parent/control 更强，而不是取代当前 deterministic anchor。

## 五档配对结果

| SNR | decoder−control PSNR（95% CI） | decoder−B0 PSNR（95% CI） | decoder−control LPIPS |
|---:|---:|---:|---:|
| 1 | `+0.066430` `[+0.059277,+0.073915]` | `+0.668454` `[+0.612611,+0.727155]` | `-0.006573` |
| 4 | `+0.031984` `[+0.027502,+0.036850]` | `+0.240784` `[+0.203729,+0.281774]` | `-0.003317` |
| 7 | `+0.012261` `[+0.009443,+0.015409]` | `+0.037157` `[+0.015024,+0.060621]` | `-0.001813` |
| 13 | `-0.000190` `[-0.002043,+0.001477]` | `-0.047710` `[-0.053113,-0.041847]` | `-0.000585` |
| 19 | `-0.002457` `[-0.003915,-0.001244]` | `-0.027578` `[-0.029278,-0.025616]` | `-0.000221` |

图像 loss 的收益随信道变好而单调缩小：低 SNR 改善最明显，高 SNR 则已进入“任何 prior correction 都可能超过真实噪声”的区域。13/19 dB 虽然 LPIPS仍改善，但 PSNR 负值不能忽略；这正是下一版需要显式 identity limit 的证据。

## Latent 与图像目标的关系

decoder-aware 相对 control 不仅图像质量更好，latent MSE 也更低，五档分别为 `-0.007021/-0.003027/-0.001302/-0.000231/-0.000039`。因此图像 loss 没有通过牺牲 latent fidelity 换取视觉指标，而是帮助优化器找到更好的局部 codeword correction。相对 parent 的整体 latent MSE 也下降 `-0.000713`，95% CI `[-0.000781,-0.000646]`。

不过“latent MSE 改善但高 SNR PSNR 仍低于 B0”的现象没有完全消失。原因是高 SNR 下可改进空间已经很小，decoder 局部 Jacobian 的方向权重只能减少错误，不能保证 correction 幅值自动收敛到严格零。

## 语义漂移诊断

COCO 没有本实验需要的监督类别真值，以下仅是冻结 ImageNet 分类器相对原图预测的 pseudo audit：

- AlexNet clean-confidence eligible 805 rows：B0 failure `469`；control `430`，new/repair=`27/66`；decoder-aware `426`，new/repair=`29/72`；
- 三分类器多数票：control new/repair=`5/21`；decoder-aware=`5/31`；
- decoder-aware 比 control 多 2 个 AlexNet new error，但多 6 个 repair，总 failure 再降 4；多数票 new error 不增加、repair 增加 10。

所以本轮没有看到 pseudo semantic drift 总体恶化，但也不能据此声明 semantic-safe。后续若把该支路接入 M3，仍需在监督 population 上重新做 hard new-error/tail-risk 审计。

## 未解决的 B1 融合

decoder-aware DDIM→旧 B1 相对 B1 仍为：

- PSNR `-0.198803 dB`，95% CI `[-0.208762,-0.188763]`；
- LPIPS `+0.033593`，95% CI `[+0.031729,+0.035402]`。

低 SNR 分布偏移最重，1 dB PSNR 为 `-0.339171 dB`；19 dB 已缩小到 `-0.035129 dB`。这进一步说明旧 B1 是按 B0 输入训练的，不能直接当作任意 latent denoiser 后端。下一轮不能再用 naive 串联宣称系统融合。

## 判定与下一步

预注册 8 项检查通过 7 项：decoder-aware 显著优于 control、3/5 SNR 优于 control、总体优于 B0、LPIPS 改善、五档 latent MSE 改善、两种 pseudo new≤repair均通过；唯一未过的是相对 B0 只有 3/5 SNR PSNR 为正。

下一方法变量应从“继续调图像 loss 权重”转向**带高 SNR identity limit 的 SNR-conditioned correction envelope**：把 denoiser correction 写成 `x_final = measurement + g(alpha)*Delta`，结构上约束 `g(alpha)→0` 当 `alpha→1`，并保持低 SNR 的 decoder-aware prior。这样针对的是已重复出现两轮的 13/19 dB 过修复，而不是用 holdout 后处理硬编码结论。由于现有 1,000 张 validation population 已全部暴露，下一正式检验必须从 COCO train2017 未使用图像建立新的 selection/holdout manifest；本 holdout 只能作为下一阶段 development evidence。

## 可复现路径

- 预注册：`reports/decoder_aware_latent_diffusion_preregistration_2026-07-15.md`
- control 配置：`configs/s17_decoder_aware_latent_diffusion_control.yaml`
- decoder-aware 配置：`configs/s17_decoder_aware_latent_diffusion.yaml`
- runner：`scripts/s17_channel_matched_latent_diffusion.py`
- bootstrap：`scripts/s17_decoder_aware_latent_bootstrap.py`
- loss 诊断：`outputs/analysis/ANALYSIS-S17-LATDIFF-LOSS-DIAGNOSTIC-001/`
- control：`outputs/EXP-S17-LATDIFF-003-CONTROL/`
- decoder-aware：`outputs/EXP-S17-LATDIFF-004-DECODER/`
- holdout：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-003/`
- bootstrap：`outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-002/`

frozen per-sample CSV SHA-256 为 `9a42ce71c05036f6401a0509b4aa6cde200b660a5acad603b0ce0293926baf92`；bootstrap JSON SHA-256 为 `88aefd7b613543cf99b60fcb75799b14b437d0cb8880a1ffde11165d61fadcbd`。
