# S21 B1 锚定门控融合预注册（2026-07-20）

## 1. 研究问题

S19 已证明 channel-matched diffusion 观测含有等容量 B0-only CNN 无法替代的信息，但其共享融合网络在 13/19 dB 相对 control 有轻微负迁移。S20 又证明 SGD 论文协议上界相对 B1 呈现明确的 fidelity/perception Pareto tradeoff，而不是全面支配。

S21 在任何新 population、cache、训练或评估输出产生前预注册以下问题：

> 冻结 B1 作为保真锚点，只让一个参数解耦、零初始化、低中 SNR 有效的辅助 adapter 注入 matched-diffusion 信息，能否相对同参数量且看不到 diffusion 的 control 同时改善 PSNR 与 LPIPS，并在 13/19 dB 由架构保证精确回到 B1？

本轮是 AWGN、exact `19,712 real` 主线内的下一最小闭环，不引入新信道、大模型或免费文本。

## 2. 冻结 population 与泄漏边界

- 本地 COCO train2017 候选，统一 resize/center-crop 到 256×256 后保存 uint8 PNG。
- 排除 S16 的 11,000 张、S18 的 512 张、S19 的 5,512 张，path 与 source SHA-256 双重排除。
- 排除与 COCO val2017 同文件名的图片。
- 用 `SHA256(20260751:relative_path)` 排序后固定 5,000 train、256 selection、256 holdout。
- 三段互不重叠；holdout 在 control/fusion checkpoint 与 SHA 冻结前不得访问。
- official Imagenette validation 保持封存。

population materialization、cache、training、holdout 使用不同的状态门；每阶段输出存在即拒绝覆盖。

## 3. 共同物理合同

- AWGN，SNR `[1,4,7,13,19] dB`。
- 每实坐标噪声方差 `P/(2γ)`。
- total `19,712 real = 9,856 complex channel uses`，CBR `0.0501302083`。
- 其中 80 real 为既有 payload reservation，图像有效坐标 19,632；接收端在图像 decode 前擦除 reservation。
- B0、B1、matched diffusion、control、fusion 使用同图同 canonical channel realization。
- diffusion 与 B1 都从同一接收 codeword 派生；融合不增加信道侧信息，额外信道符号为 0。

## 4. 冻结分支

### 4.1 B1 保真锚点

冻结 `EXP-S16-B1-001`，不更新任何 B1 参数。B1 输入为 B0 RGB、SNR map、B0 Sobel 与 Laplacian，输出为严格低码率 residual restoration。

### 4.2 matched diffusion 辅助观测

冻结 `EXP-S17-LATDIFF-004-DECODER` 与 S18 `hard_identity_7db`：

- 1/4/7 dB：用解析 `alpha(SNR)=2γ/(2γ+1)` 进入 6-step deterministic DDIM；
- 13/19 dB：auxiliary 严格等于 B0，不运行 diffusion。

### 4.3 B1-anchored spatial-gated adapter

control 与 fusion 都冻结 B1，只训练完全相同的 auxiliary adapter。adapter 输入 12 通道：

`[B1_RGB, auxiliary_RGB, |auxiliary-B1|_RGB, SNR, Sobel(B1), Laplacian(B1)]`。

adapter 输出 RGB proposal `R` 与单通道 spatial gate `A=sigmoid(logits)`：

`x_final = clip(B1 + g_max(SNR) × A × tanh(R))`。

`g_max={1:0.12,4:0.10,7:0.08,13:0,19:0}`。因此 13/19 dB 无论网络参数如何都逐像素精确等于 B1。RGB residual head 零初始化，gate logits 零初始化；epoch0 的 control/fusion 必须都等于 B1。

唯一因果差异：

- control auxiliary：B0；
- fusion auxiliary：S18 identity-controlled matched diffusion。

两者 adapter 参数量、初始 state、batch、crop/flip、优化器、训练预算完全相同。

## 5. 训练与选择

- 10 epochs；每 epoch 5,000×5=25,000 行。
- batch 16、128×128 paired random crop、paired horizontal flip。
- Adam，LR `1e-3`，无 weight decay，gradient clip 1.0。
- loss：`MSE + 0.10 L1 + 0.01 LPIPS + 0.0001 mean(A)`；高 SNR 的 `g_max=0` 使 adapter 不影响输出。
- epoch0 纳入候选。
- control/fusion 分别选择 mean PSNR 最大且 selection LPIPS 不劣于各自 epoch0 的最早 epoch；若没有训练 epoch 满足 LPIPS 约束，保留 epoch0。

## 6. 一次性 holdout 指标

- PSNR、MS-SSIM、LPIPS。
- spatial gate mean、`|x_final-B1|` 注入幅度。
- AlexNet pseudo failure，以及相对 B1 anchor 的 new/repair。
- AlexNet、ResNet18、MobileNetV3-Small 三分类器 majority pseudo failure，以及相对 B1 anchor 的 new/repair。
- source-image cluster bootstrap 10,000 次；每个源图聚合五个 SNR，seed `20260756`。
- 保存配置、checkpoint SHA、逐样本 CSV、summary、各 SNR 样例图和失败结果。

COCO pseudo semantic 只作辅助诊断，不替代最终监督 `T_cls` 审计。

## 7. 预注册成功判据

主判据：

1. fusion−control PSNR 的 source-cluster 95% CI 下界 `>0`；
2. fusion−control LPIPS 的 95% CI 上界 `<0`；
3. 1/4/7 dB 的 fusion−control PSNR `3/3` 非负。

架构/锚点判据：

4. epoch0 control、fusion 与 B1 最大绝对差 `≤1e-6`；
5. holdout 13/19 dB control/fusion 与 B1 最大绝对差 `≤1e-7`；
6. fusion−B1 平均 PSNR `>0`，五 SNR `5/5` 非负。

语义辅助判据：

7. fusion majority new `≤` repair；
8. fusion majority new `≤` control majority new。

若主判据失败，则只能说明 B1 锚定结构可运行，不能声明 diffusion 融合有效。若质量判据通过但语义辅助判据失败，结论必须标为质量正向、语义风险未闭合。

## 8. 预先限定的解释

- 这不是作者 SGD-JSCC 的复现或改造：不使用其 VAE channel latent、BLIP2 caption 或免费文本。
- 这不是固定图像平均：B1 是不动点，adapter 学习空间 gate 与受限增量。
- 高 SNR 精确回 B1 是架构保证，不作为学出来的贡献。
- S19 已暴露的数据、checkpoint 和 holdout 不用于 S21 训练、选择或最终统计。
- 本轮若成功，只证明 frozen matched diffusion 对 B1-anchored adapter 有可利用的互补信息；semantic-risk controller 仍需在独立监督 population 上闭合。
