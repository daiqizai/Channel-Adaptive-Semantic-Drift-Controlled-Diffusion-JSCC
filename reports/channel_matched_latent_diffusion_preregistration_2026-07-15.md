# Channel-State-Matched Latent Diffusion 预注册（2026-07-15）

## 研究动机

`EXP-S16-DIFF-001` 的 B1 后图像残差短链在五个 SNR 上全部恶化，但该实现没有让真实信道输出对应 diffusion scheduler 的前向状态。SGD-JSCC 的代码级复核显示，其关键机制是发送潜变量与 diffusion latent 共用表示空间，并把归一化 AWGN 输出匹配到 `alphas_cumprod` 的对应时刻。因此，本实验只回答一个更窄的问题：**在本项目精确 19,712-real DeepJSCC 链路上，物理匹配的 latent diffusion 能否稳定降低活动潜变量误差，并把收益传递到解码图像。**

本实验不改变 AWGN 主线，不访问 official Imagenette validation，不引入新下载，不使用文本或真实标签。它是机制验证，不是新的正式外部排名。

## 冻结表示与码率

- 冻结 exact-rate DeepJSCC checkpoint：`bca5b67a...bb606`。
- 稠密 encoder latent：`6×64×64=24,576` 个实坐标。
- 实际活动坐标：`19,712`；其中 `80` 个为 UInt2×BPSK-r4 语义载荷预留，图像活动坐标为 `19,632`。
- 总复信道使用：`9,856`，CBR=`0.050130208333333336`。
- diffusion 输入、训练损失和反向更新只作用于 `19,632` 个图像活动坐标；载荷坐标和未发送坐标始终置零。不得把稠密 tensor 冒充额外传输预算。

## 信道到 diffusion 的匹配

项目当前冻结的实坐标信道为：

`y = x0 + sqrt(P/(2*gamma)) * epsilon`，其中 `gamma=10^(SNR_dB/10)`。

发送活动向量归一化为单位平均功率。在 `P=1` 口径下：

`alpha_channel = 1 / (1 + 1/(2*gamma)) = 2*gamma/(2*gamma+1)`，

`x_t = sqrt(alpha_channel) * y`。

于是 `x_t` 与 `sqrt(alpha_channel)x0 + sqrt(1-alpha_channel)epsilon` 同形。训练在 1 dB 对应 log-SNR 到 `alpha=0.999` 之间采样前向状态；推理从每个真实 SNR 的 `alpha_channel` 以 6-step deterministic DDIM 反演。

## 模型与对照

模型是作用在散射后的 `6×64×64` DeepJSCC latent 上的小型 masked convolutional epsilon predictor，显式输入完整坐标 mask 和 log-SNR。它不是 Stable Diffusion，也不声称复现 SGD-JSCC。

冻结比较项：

1. `B0`：同一 received latent 直接解码；
2. `scalar_lmmse`：只做 `y/(1+sigma²)` 标量收缩；
3. `fixed_step_ddim`：所有 SNR 错配为 7 dB step，用于检验 step matching；
4. `matched_one_step`：匹配时刻单步 `x0` 预测；
5. `matched_ddim`：匹配时刻 6-step DDIM；
6. `B1`：冻结 `EXP-S16-B1-001`；
7. `matched_ddim_b1`：在 matched latent 解码结果上使用同一个冻结 B1，不重新训练。

第一轮 `measurement_blend=0`，避免把 step matching 和 posterior/data-consistency 强度混在一起。只有本实验取得正信号后，才另行预注册逐步 measurement consistency。

## 数据隔离

- 训练：现有 manifest 的 10,000 个 `train` 样本。
- checkpoint selection：`validation` 的前 256 个样本，channel seed `20260732`。
- 一次性 holdout：`validation` 的第 256--511 个样本，channel seed `20260733`。
- selection 与 holdout 在读到任何 holdout 指标前已经冻结，禁止根据 holdout 调模型或采样参数。

## 指标与成功判据

必须同时报告：图像活动 latent MSE、PSNR、MS-SSIM、LPIPS，以及以原图预测为参照的 AlexNet pseudo failure/new-error/repair；另用 AlexNet、ResNet18、MobileNetV3-Small 报告多数票 pseudo new/repair。COCO 无监督类别真值，因此该语义结果只作机制期漂移诊断，不能替代后续 Imagenette 严格监督审计。

阶段成功要求：

- matched latent MSE 在五个 SNR 都低于 raw received latent；
- matched DDIM 相对 B0 平均 PSNR 为正，至少 4/5 SNR 为正，平均 LPIPS 不恶化；
- matched DDIM 平均 PSNR 不低于固定 7 dB 错配对照；
- `matched_ddim_b1` 平均 PSNR 不低于冻结 B1；
- AlexNet 和三模型多数票的 pseudo new error 均不多于 repair。

任一关键条件失败都按部分成功或负结果记录；不得只凭 latent MSE 或少量样例宣布 diffusion 成功。

## 输出与边界

- 训练：`outputs/EXP-S17-LATDIFF-001/`
- holdout：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-001/`
- 配置：`configs/s17_channel_matched_latent_diffusion.yaml`

本实验允许证明“channel-state matching 对当前 DeepJSCC latent 有/无机制收益”，不允许声称已超过 SGD-JSCC 论文、已完成 semantic-tail safety，或已得到最终 M3。

## 数值失败后的稳定版登记

`EXP-S17-LATDIFF-001` 按上述原始配置启动后，在 epoch 0 / batch 13 出现 non-finite training loss，且发生在任何 epoch selection 输出之前。失败目录与原始 `amp: true` 配置保留，不作为结果。其直接后继登记为 `EXP-S17-LATDIFF-002`：唯一方法性改动是 `amp: false`，其余数据、模型宽度、训练轮数、alpha 分布、采样步数、selection/holdout 划分、对照和成功判据全部不变。稳定版正式输出改为：

- 训练：`outputs/EXP-S17-LATDIFF-002/`
- holdout：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-002/`

在正式稳定版前只允许独立 debug 目录做有限 batch 的 finite 检查；debug 数字不得用于 checkpoint selection 或结论。
