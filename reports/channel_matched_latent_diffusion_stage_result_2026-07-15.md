# Channel-State-Matched Latent Diffusion 阶段结果（2026-07-15）

## 结论

本阶段得到一个边界清楚的正结果：**把真实 AWGN 输出匹配到 DeepJSCC latent diffusion 的对应前向时刻是有效的；固定错配 step 和标量 LMMSE 都不能解释该收益。** 在预冻结的 256 图×5 SNR holdout 上，6-step matched DDIM 相对同信道噪声的 B0 为：

- PSNR `+0.148715 dB`，image-cluster 95% CI `[+0.129607,+0.168857]`；
- LPIPS `-0.035305`，95% CI `[-0.038907,-0.031814]`；
- 图像活动 latent MSE `0.145516→0.060453`，下降 `58.46%`；
- 相对固定 7 dB step 错配，PSNR 再高 `+0.233455 dB`，95% CI `[+0.220234,+0.246661]`。

但预注册总判定为 `NEGATIVE_OR_PARTIAL`：PSNR 只在 1/4/7 dB 三档为正，未达到 4/5；冻结 B1 仍明显更强，且把旧 B1 直接接在 latent diffusion 后会产生输入分布偏移，平均比 B1 低 `-0.231266 dB`。因此本阶段证明的是**物理匹配 latent diffusion 的机制价值**，不是“当前系统已超过 B1”或“最终 M3 已完成”。

## 实验合同

- 训练：现有 COCO train2017 内部 10,000 图；selection 为 validation role 前 256 图。
- 一次性 holdout：validation role 第 256--511 图；独立 channel seed `20260733`。
- 信道：AWGN，SNR `[1,4,7,13,19]`；每实坐标方差 `P/(2×SNR)`。
- 码率：`19,712` 个活动实坐标=`9,856` 次复使用，CBR `0.050130208333333336`。
- 坐标：`19,632` 个图像活动坐标参与 diffusion；`80` 个语义载荷坐标和未发送坐标固定为零。
- 模型：323,574 参数 masked convolutional epsilon predictor；6-step deterministic DDIM。
- 无联网、无下载、未访问 official Imagenette validation。

原 `EXP-S17-LATDIFF-001` 使用 AMP，在 epoch 0 / batch 13 出现 non-finite loss，且尚未产生 selection 输出。失败目录与配置保留。`EXP-S17-LATDIFF-002` 仅把 AMP 改为 FP32，其他协议不变；最佳 checkpoint 为 epoch 5，SHA-256 `cfc52716...b4e1`。selection 上 matched DDIM 相对 B0 为 `+0.151483 dB`，与 holdout 的 `+0.148715 dB` 一致。

## 五档结果

| SNR | raw→matched latent MSE | matched−B0 PSNR | 95% CI | matched−B0 LPIPS | 95% CI |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.397367→0.138182` | `+0.614949` | `[+0.568775,+0.661880]` | `-0.092860` | `[-0.099993,-0.086032]` |
| 4 | `0.199117→0.087429` | `+0.199503` | `[+0.169609,+0.231052]` | `-0.052260` | `[-0.057752,-0.046884]` |
| 7 | `0.099757→0.053399` | `+0.010328` | `[-0.006439,+0.028435]` | `-0.026238` | `[-0.030233,-0.022545]` |
| 13 | `0.025049→0.017912` | `-0.054083` | `[-0.058803,-0.049033]` | `-0.004864` | `[-0.006301,-0.003489]` |
| 19 | `0.006292→0.005344` | `-0.027123` | `[-0.028591,-0.025536]` | `-0.000304` | `[-0.000715,+0.000078]` |

这组结果揭示了两个事实：

1. latent MSE 五档都改善，说明 denoiser 确实学到了 DeepJSCC codeword prior；
2. 13/19 dB 的 latent MSE 改善没有转化为 PSNR，说明当前等权 latent loss 没有建模 decoder 对不同 codeword 方向的敏感性。

因此下一版不应继续盲目增加 DDIM step，而应加入 frozen decoder-aware reconstruction objective，或在 latent-diffusion 输出分布上重新训练融合/恢复模块。

## Step matching 与对照

| 方法 | PSNR | LPIPS | 说明 |
|---|---:|---:|---|
| B0 | `25.983234` | `0.306472` | 原始 received latent 解码 |
| scalar LMMSE | `25.984761` | `0.311932` | PSNR 几乎不变、LPIPS 恶化 |
| fixed 7 dB DDIM | `25.898494` | `0.297611` | SNR 错配；高 SNR 明显过修复 |
| matched one-step | `26.059838` | `0.275861` | 已有正收益 |
| matched 6-step DDIM | `26.131949` | `0.271167` | 比 one-step 再高约 `0.0721 dB` |
| frozen B1 | `27.016222` | `0.190905` | 当前系统强基线 |
| matched DDIM→旧 B1 | `26.784956` | `0.228130` | 输入分布偏移，低于 B1 |

固定 7 dB 对照在真实 7 dB 与 matched 完全相同；在 1/4 dB 去噪不足，在 13/19 dB 过修复。matched 相对 fixed 的整体 `+0.233455 dB` 直接支持 step matching，而不是只支持“又训练了一个 CNN”。

## 语义漂移诊断

COCO 没有本实验所需的监督类别真值，因此这里只使用原图 ImageNet classifier prediction 作为 pseudo 参照，不替代后续严格 Imagenette 审计。

- AlexNet 原图置信度≥0.20 的 865 个 image/SNR rows 中：B0 failure `494`；matched DDIM failure `437`，new/repair=`26/83`。
- 三分类器多数票（AlexNet/ResNet18/MobileNetV3-Small）：matched DDIM new/repair=`7/38`。
- 冻结 B1 的 AlexNet new/repair=`84/251`；matched DDIM→旧 B1 为 `115/196`。这再次说明 naive 串联损伤了 B1 的语义行为，不能以“两个模块都各自有效”为由直接组合。

样例表中没有观察到 SGD-JSCC 式明显文本驱动内容替换，但仍存在低码率纹理噪声和过平滑；该观察只作定性记录。

## 判定与下一步

预注册 8 个检查通过 6 个：

- 通过：五档 latent MSE 全改善、平均 PSNR 正、平均 LPIPS 改善、matched 优于 fixed、AlexNet new≤repair、多数票 new≤repair；
- 未通过：PSNR 正收益只有 3/5；matched DDIM→旧 B1 不及 B1。

阶段定位因此冻结为：

> **channel-state-matched latent diffusion 是成立的低 SNR codeword prior，但当前等权 latent objective 与旧 B1 串联还不是最终系统。**

下一方法实验应在新的 selection/holdout population 上预注册以下一个变量：给 epsilon/x0 loss 加 frozen decoder-aware image reconstruction loss；若仍不能接近 B1，再训练显式接收 `B0 + matched latent decode` 的融合恢复器，并以同容量的无-diffusion B1 对照隔离 diffusion 贡献。逐步 measurement consistency 暂不与 decoder-aware loss同时引入。

## 可复现路径

- 预注册：`reports/channel_matched_latent_diffusion_preregistration_2026-07-15.md`
- 配置：`configs/s17_channel_matched_latent_diffusion.yaml`
- 实现：`src/cadsd_jscc/channel_matched_latent_diffusion.py`
- runner：`scripts/s17_channel_matched_latent_diffusion.py`
- 训练：`outputs/EXP-S17-LATDIFF-002/`
- holdout：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-002/`
- bootstrap：`outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-001/`

验证：全仓 `108/108` 项标准库 `unittest` 通过；新模块/runner/bootstrap 脚本 `py_compile` 通过；`git diff --check` 通过。
