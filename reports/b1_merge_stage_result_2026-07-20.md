# B1 与 matched diffusion 合并阶段结果（2026-07-20）

## 结论先行

这一阶段第一次在 frozen B1 上完成了 matched diffusion 的非零安全合并，但效应量很小。证据链为：

1. S19 已经证明 matched diffusion 含有 B1/B0-only 网络不能完全替代的互补信息；
2. S21 证明简单输出层 gate、残差叠加和凸平均都不能稳定提取这部分信息；
3. S22 证明仅用 `1,728` 个参数把 `D-B0` 注入冻结 B1 特征，可以稳定、大幅改善感知指标，但单位训练端点与 B1 的 PSNR 形成约 `0.018--0.020 dB` 的小冲突；
4. S23 在事先冻结的全局 shrink 网格中选出 `alpha=0.15`，随后在一次性独立 holdout 上同时取得 PSNR `+0.000568 dB` 与 LPIPS `-0.001731`，两项 source-cluster 95% CI 均不跨零，5/5 预注册检查通过。

按预注册规则，S22 选择了 epoch 0，未访问其 holdout。S23 是在已知 S22 结果后明确注册的 development follow-up；alpha 只在 selection 上选择，S23 的 256×5 holdout 在 checkpoint/policy SHA 冻结后才首次访问。S21/S22 数字仍只属于 development，S23 holdout 可以用于本轮阶段结论，但不消除 S23 设计受 S22 启发的事后性。

## 1. 冻结实验合同

- 数据：全新 COCO train2017 `5,000 train / 256 selection / 256 sealed holdout`，与 S16/S18/S19 的 path 和 SHA 重叠均为 0；
- cache：`27,560` 行，SHA256 `dd79fe2f...84b87`；
- 信道与码率：AWGN，`1/4/7/13/19 dB`，严格 `19,712 real`；
- B1：`EXP-S16-B1-001` frozen receiver residual anchor；
- matched diffusion：与 S18/S19 相同的 6-step identity-controlled channel-state-matched 分支，同一 received codeword，不增加 side information；
- official Imagenette validation 未访问；S21/S22 holdout 均未访问。

## 2. S21：输出层合并的四个负结论

### 2.1 带 gate penalty 的 learned gate

`EXP-S21-B1AGF-001` 第 1 轮 spatial gate 直接塌到约 `0`，control/fusion 都精确退化为 B1。根因是 residual 零初始化时 gate 暂时收不到重建梯度，而 gate penalty 先把 logits 推入数值饱和区。

### 2.2 移除 gate penalty

`EXP-S21-B1AGF-002` 的 gate 在前三轮保持非零，并把 LPIPS 从 B1 的 `0.187629` 降到约 `0.1537--0.1553`；但 PSNR 仍低于 B1。第 4 轮 gate 又塌到 0，说明 jointly learned gate×residual 参数化本身不可辨识，不只是 penalty 设错。

### 2.3 固定 gate 的 bounded residual

`EXP-S21-B1AR-003` 去掉 learned gate 后，第 3 轮输出达到预设 mean-absolute envelope 上限 `0.06`，PSNR 从约 `27.05 dB` 崩到 `22.73 dB`。这排除了继续扫描 gate bias/penalty 的必要性。

### 2.4 无训练凸融合

`ANALYSIS-S21-CONVEX-SELECTION-004` 穷举 `120` 个单调低 SNR alpha 组合。唯一同时满足每个低 SNR PSNR 不低于 B1、aggregate LPIPS 不劣于 B1 的候选是全零 B1。最小非零 alpha 已能取得微小 PSNR 增益，但 LPIPS 有微小退化，说明直接像素平均的方向也不对。

## 3. S22：冻结 B1 的最小特征注入

S22 冻结 B1 的 head/body/tail，只新增一个零初始化 `Conv3x3(3→64,bias=False)`：

`h = h_B1 + e(SNR) P(D-B0)`。

其中 `e(1/4/7)=1`、`e(13/19)=0`。新增参数仅 `1,728`；control 使用 `B0-B0=0`，训练后仍严格等于 B1；13/19 dB 也由结构保证严格等于 B1。

真实 cache smoke 结果：初始最大 B1 差为 `0`，projection gradient L1 为 `0.03054`，证明新增参数不是死支路。

### 3.1 Selection 轨迹

| epoch | fusion−B1 PSNR (dB) | fusion−B1 LPIPS | mean abs feature injection |
|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 |
| 1 | -0.01887 | -0.01096 | 0.002411 |
| 2 | -0.01875 | -0.01267 | 0.003511 |
| 3 | -0.01917 | -0.01363 | 0.004288 |
| 6 | **-0.01789** | -0.01487 | 0.005825 |
| 10 | -0.02031 | **-0.01580** | 0.007025 |

10 个非零训练 epoch 的 LPIPS 全部优于 B1，且改善随注入幅度总体增加；但 aggregate PSNR 全部低于 B1。按“最大化 PSNR且 LPIPS 不劣于 epoch0”的冻结规则，最终 checkpoint 是 epoch0：

- checkpoint SHA256：`b7eac7ec...a0d79`；
- `selected_nonzero_training_epoch=false`；
- holdout accessed：`false`。

### 3.2 这项负结果说明什么

与 S21 不同，S22 没有 gate collapse、没有输出残差饱和，且感知增益非常稳定。因此它给出了更强的机理证据：`D-B0` 中确有可以穿过 frozen B1 body 改善感知质量的方向，只是单位幅度的训练端点越过了 PSNR 最优区间。

不能据此声称 S22 已优于 B1，也不能用 LPIPS 单指标挑 checkpoint 后看 holdout。S23 因此事先冻结零点附近的全局幅度协议，只检验是否存在“PSNR 与 LPIPS 同时不差”的非零点。

## 4. S23：零点附近的全局 Shrink

### 4.1 预注册边界

S23 注册时已经知道 S22 的 selection 轨迹，因此明确标为 follow-up。它固定复现 S22 第 1 轮 projection endpoint，不扫描 epoch；在运行前固定全局 alpha 网格 `[0,.01,.025,.05,.075,.1,.15,.2,.35,.5,.75,1]`，不做 per-SNR alpha 搜索。

候选必须同时满足：aggregate LPIPS 不劣于 B1、1/4/7 dB PSNR 全不低于 B1、aggregate PSNR 不低于 B1；随后最大化 aggregate PSNR。只有非零候选胜出才允许访问 holdout。

### 4.2 Selection

`alpha=0.01--0.20` 均可行，说明零点附近确有非零 Pareto 区间；`alpha>=0.35` 开始出现低 SNR PSNR 退化。冻结选择为 `alpha=0.15`：

- aggregate fusion−B1：PSNR `+0.000536 dB`，LPIPS `-0.001681`；
- 1/4/7 dB PSNR：`+0.000566/+0.001114/+0.001000 dB`；
- 13/19 dB：精确 B1；
- selected checkpoint SHA：`53692278...1abbf`；
- selected policy SHA：`54c2639f...8c68f`。

### 4.3 一次性 Holdout 与 Bootstrap

全新 256 图×5 SNR，共 1,280 行；per-sample CSV SHA `9f4dd87d...71fa3`。

| 指标 | fusion−B1 mean | source-cluster 95% CI |
|---|---:|---:|
| PSNR | `+0.000568 dB` | `[+0.000378,+0.000771]` |
| LPIPS | `-0.001731` | `[-0.001849,-0.001622]` |

分 SNR PSNR 为 `+0.000701/+0.001158/+0.000979/0/0 dB`，LPIPS 为 `-0.003789/-0.003000/-0.001864/0/0`。13/19 dB fusion/control 与 B1 的最大逐像素差均为 0。

语义辅助诊断中，AlexNet clean-confident 子集相对 B1 为 `1 new / 2 repair`；三分类器 majority 为 `3 new / 7 repair`。这满足预注册的 `new<=repair`，但 new error 并非 0，不能宣称语义绝对安全。

Bootstrap 的五项检查全部通过：PSNR CI 下界大于 0、LPIPS CI 上界小于 0、三个低 SNR PSNR 全非负、高 SNR 精确 B1、majority new 不大于 repair。

### 4.4 正确解读

S23 是关键的机制闭环：它证实 S22 的 diffusion feature 方向不是错误，只是未经幅度控制时过冲；把幅度压回零点附近后，可以在独立 holdout 上同时改善 distortion 与 perception。

但 PSNR 增益只有约 `5.7e-4 dB`，远小于 S19 joint fusion 相对 B1 的 `+0.10168 dB`，没有实际显著的保真价值。当前成果适合写成“严格回退、非零 Pareto 可行性和幅度过冲诊断”，不够支撑强主方法。下一步应学习/解析出更有效的 SNR/sample-adaptive amplitude，同时保留当前 exact-B1 fallback 与语义审计；不能只继续细扫全局 alpha。

## 5. 当前主线如何表述

现阶段最稳妥的一句话是：

> 在严格同码率的 JSCC 接收端，先由 B1 提供确定性的保真锚点，再把与信道状态严格匹配的 diffusion 重建视作同一观测的感知先验；系统只提取其相对 B0/B1 的互补特征，并在不能同时改善失真与感知质量时严格回退 B1。

这比“信道好时不用 diffusion”更准确：高 SNR identity 是当前已冻结的安全边界；低 SNR 是否使用 diffusion 由可验证的 Pareto 条件决定，而不是先验假定 diffusion 永远更好。
