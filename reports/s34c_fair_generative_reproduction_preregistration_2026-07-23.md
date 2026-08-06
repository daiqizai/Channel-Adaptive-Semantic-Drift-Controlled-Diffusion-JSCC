# S34C：严格总码率公平的 SGD-JSCC / DiffJSCC 复现预注册（2026-07-23）

## 0. 状态与授权边界

状态：**用户已于 2026-07-23 在任何执行前暂停。** 本文保留为将来可能重启时的预注册，不构成 smoke、训练或输出授权。暂停原因是：DiffJSCC 本来已等码率；SGD 无官方 trainer、只能近似适配；算力/容量不等削弱可解释性；14–29 天工期与尽快投稿冲突。当前先做只读的 S34C-Lite 码率透明度分析，再决定是否重启；official Imagenette validation 继续封存。

这轮采用“对冲路线”：先回答生成式 JSCC 在真正相同总码率下还剩多少感知优势，再决定论文重心。若公平生成方法仍在 LPIPS/KID/FID 上显著领先且 semantic failure 不恶化，形成适合 ICASSP 的 fidelity–perception–reliability 结果；若优势大幅缩水或消失，则 S33 的简单、强、严格等码率故事更适合 WCL 快速闭环。

## 1. 技术可行性结论

### 1.1 DiffJSCC：可行，而且能做到官方代码基础上的公平重训

本地官方源码 `mingyuyng/DiffJSCC@13aeb62451b872ce41ceba132c9c30a9ca172c53` 提供完整两阶段入口：

1. `DeepJSCC.training_step/configure_optimizers` 与 `train_jscc_cnn.yaml`，可训练 JSCC encoder/decoder；
2. `ControlLDM.configure_optimizers` 与 `train_cldm.yaml`，可从 Stable Diffusion 2.1 + JSCC 初始化后训练 ControlNet。

需要适配但不缺算法主体：把作者绝对数据路径换成项目 COCO manifest；把连续 `Uniform[0,14] dB` 改成与 S33 相同的逐图离散 `[1,4,7,13,19] dB`；把信道替换为项目 paired-real half-variance AWGN；增加 exact-symbol、功率、canonical evaluation noise、恢复和输出防覆盖审计。

DiffJSCC 必须澄清一个常见误解：其文本由接收端 BLIP2 从带噪 JSCC 初始重建本地生成，不是发送端传来的 side information，因此计 `0` 个信道符号。作者 512 网格上的 C16 latent 原生是 `16×32×32=16,384 real`，与 S33 正好等码率。它的公平问题不是“免费文本超码率”，而是此前权重训练于 OpenImage、训练 SNR 与项目不同；本轮重训正好消除这两项混杂。

### 1.2 SGD-JSCC：端到端运行可行，但严格意义的官方公平重训不可行

本地官方源码 `SGDJSCC@2188acc0dd2805355d3d0d2e478cbc27b46b4da5` 的 README 明确只发布 inference 与 checkpoint，`Training guideline to fine-tune the diffusion model or controlnet` 仍是未完成 TODO。虽然 VAE、SNR predictor、edge-JSCC、diffusion backbone 和 ControlNet 的模块定义及权重都在，但缺少：

- 作者训练阶段次序与每阶段冻结规则；
- 主要损失、权重、optimizer/scheduler 与停止规则；
- caption/edge 训练数据预处理；
- step-matching/SNR predictor 的精确训练目标；
- 主 latent 缩小时如何联合重训 diffusion 的官方合同。

因此不能把项目自行补出的 trainer 称作“官方 SGD-JSCC 重训”，也不能用它单独否定原论文。最强可行方案是 **SGD-inspired released-component rate-constrained adaptation**：保留作者模块和初始化，增加可审计的主 latent 降维适配器、真实 caption codec 与低码率 edge 分支，再用标准 latent diffusion/ControlNet 噪声预测目标在 COCO 上微调。它能回答“已发布组件被迫承担真实总码率后还能保留多少优势”，但结论必须带 approximate reproduction 标签。

## 2. 冻结公平合同

所有可排名方法共享：

- 原始输入：`256×256 RGB`；源实维度 `196,608`；
- 总预算：每图严格 `16,384 real = 8,192 complex channel uses`，CBR=`1/24`；
- 训练任务数据：项目 COCO train2017 manifest；
- 训练 SNR：逐图离散均匀 `[1,4,7,13,19] dB`；不得写成连续 SNR；
- 信道：paired-real half-variance canonical AWGN；
- 测试：同图、同 SNR、同 channel seed，并从同一 16,384-D canonical 标准正态向量取对应坐标；
- 禁止发送端 side information 不计码率；padding 必须登记且不得承载信息；
- S33 checkpoint 永久冻结为 SHA `2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`。

### 2.1 码率账本

| 方法 | main image | edge | caption | padding | 总计 | 排名资格 |
|---|---:|---:|---:|---:|---:|---|
| S33 pure JSCC | 16,384 | 0 | 0 | 0 | 16,384 | 有 |
| DiffJSCC-COCO-fair | 16,384 | 0 | 接收端本地生成，0 | 0 | 16,384 | 有 |
| SGD-RC-fair | 12,288 | 2,048 | 1,744 | 304 | 16,384 | 有，但标 approximate |
| SGD paper protocol | 16,384 | 3,328 | 免费完美文本 | 0 | 至少 19,712 + text | 无，仅 upper bound |

SGD-RC 的预注册主分配为：四个 `128×128` patch，每块把原 `16×16×16` 主 latent 通过可学习投影变成 `12×16×16`，四块共 `12,288 real`；edge 使用作者 variable-rate 分支 `cr=8`，每块 512、共 2,048 real；caption 使用冻结 CLIP BPE，最多 12 token/patch，`length + 12×16-bit token id + CRC16` 后用 rate-1/2 convolutional code，含 6 tail bits，每块 436、共 1,744 real；余下 304 个坐标固定为无信息 padding。CRC 失败时接收端使用空 caption。

每个有效实坐标平均功率为 1，不允许用未申报的支路功率提升。SGD 各支路使用同一 canonical 16,384-D 向量的不重叠区间，避免重复利用同一噪声样本。

## 3. 训练计划

### 3.1 DiffJSCC 主裁决臂

阶段 D1：从随机初始化训练 official ResNet JSCC，COCO 与五档离散 SNR，最多 100k optimizer steps；先按固定 COCO selection 曲线判断收敛，不因外部测试结果续训。

阶段 D2：冻结 D1 JSCC，按官方方式从 SD2.1 初始化并训练 ControlNet，最多 25k steps。BLIP2 caption 继续由接收端初始重建生成，不使用 COCO ground-truth caption 作为免费条件。checkpoint 只根据冻结 COCO selection 的 diffusion loss、LPIPS/KID proxy 与 semantic auxiliary 预先固定规则选择，不读取 Imagenette policy-dev 或 official validation 做选择。

为回答 diffusion 到底贡献了什么，必须保存同一 checkpoint 的 `author-JSCC initial reconstruction` 与最终 `DiffJSCC`，报告最终相对初始重建的 PSNR/LPIPS/KID/semantic new-error 与 repair。

### 3.2 SGD 近似公平臂

阶段 G1：加载作者 released VAE/JSCC/edge 权重，先冻结大 diffusion backbone，训练 `16→12→16` 主 latent 投影、`cr=8` edge branch 与 SNR predictor，使五档 received latent 与 edge 可稳定使用。

阶段 G2：caption codec 是确定性通信协议，不从测试集学习；在训练期真实经过相同 AWGN，CRC 失败即空文本。

阶段 G3：在 G1 的 matched received distribution 上，用 clean-room 标准 ε-prediction 目标微调 released ControlNet；若 24 GB 无法 full fine-tune，则只允许预注册的 ControlNet/LoRA 路径，并报告 trainable parameters。不得用 S33 或测试原图作为 diffusion condition。

强制诊断臂：

- `fair/no-text`：仍占用同样 1,744 text coordinates，但发送零信息，衡量文本贡献；
- `fair/oracle-caption`：把同一 fair 主/edge 接收结果配上干净 caption，只作非排名上界，隔离 caption channel error；
- released paper protocol：沿用作者 main+edge+perfect-caption，只作非排名上界。

## 4. 评估与统计

### 4.1 配对质量与语义主表

复用冻结 64 张 Imagenette policy-dev、3 个 channel seeds `20260748/49/50`、5 个 SNR，共每方法 960 行。报告 aggregate 与每档：

- PSNR、MS-SSIM、LPIPS；
- 冻结 T_cls 的 failure、相对 S33/各自 pure-JSCC 端点的 new-error 与 repair；
- source-image cluster bootstrap 10,000 次的双侧 95% CI；
- 参数量、trainable parameters、optimizer steps、GPU-hours、推理时间、显存和 sampler steps。

这仍是已知 policy-development population，不是独立最终测试。

### 4.2 FID/KID 感知主场

64 个 source 不足以支撑可靠 FID。因此在任何方法训练前，以 SHA 排序冻结 COCO val2017 中 2,048 张未参与 S33 selection、旧 policy development 或新训练选择的图；方法全部冻结后才一次性解封该 perception holdout。每个 SNR 使用共同 seed `20260748`，报告：

- InceptionV3 同一预处理下的 FID 与 KID；
- source bootstrap 10,000 次的 FID/KID 差值 95% CI；
- KID 作为较少样本下的主要分布指标，FID 为必报支持指标；
- 同一 2,048 图上的 PSNR/MS-SSIM/LPIPS 与 COCO dominant-object/CLIP clean-correct 辅助语义统计。

COCO-object/CLIP 仍是辅助语义指标；正式 supervised semantic failure 以冻结 Imagenette T_cls clean-correct population 为主。official Imagenette validation 继续封存。

### 4.3 “感知优势缩水”定义

对 LPIPS/KID/FID 分别在同一 population 计算：

```text
paper gap = metric(S33) - metric(paper-protocol generative)
fair gap  = metric(S33) - metric(fair generative)
shrinkage = 1 - fair gap / paper gap
```

因三者都是越低越好，`gap>0` 表示生成方法有感知优势。必须同时报告 raw gaps、pairwise CI 和 shrinkage CI；若 paper gap 符号不对或 CI 跨零，百分比记为 undefined，禁止制造夸张比例。SGD 还必须报告 fair/oracle-caption 与 fair/transmitted-caption 的差，隔离 text channel penalty。DiffJSCC 没有发送端文本，所以不计算“caption 码率缩水”，只比较官方 OpenImage checkpoint 与 COCO/five-SNR 公平重训的合同变化。

## 5. 验收与 venue gate

只有同时满足以下条件，才称“公平条件下仍保留感知优势”：

1. 相对 S33，LPIPS 差和 KID 差的 95% CI 上界均 `<0`；FID 同向且必报；
2. 冻结 T_cls semantic failure 不显著更高；
3. 逐档至少低 SNR 结论成立，不能由单一 SNR 聚合掩盖；
4. 码率账本、功率、canonical noise、输出数和所有 side information 全部审计通过。

若 LPIPS/KID 不再显著优于 S33，或感知收益伴随 semantic failure 显著上升，则判“公平感知优势被抹平/不可无条件成立”。PSNR 继续沿用 `0.10 dB` 非劣 margin；质量轴冲突一律写 Pareto。

Venue 只在结果后决定：

- **ICASSP 倾向**：至少一个公平生成臂保留显著 LPIPS+KID 优势且语义不劣，或形成足够完整、可复现的公平性 benchmark/negative result；
- **WCL 倾向**：公平感知优势消失，S33 以更简单、低延迟、exact-rate 方法维持最干净的质量—可靠性结论。

## 6. 预计时间（单张 RTX 4090D，需 smoke 后校准）

| 工作 | 预计时间 | 主要不确定性 |
|---|---:|---|
| 合同实现、数据/码率审计、两套 1-batch smoke | 1–2 天 | Lightning 兼容、SGD trainer 补全、显存 |
| DiffJSCC D1 JSCC 训练 | 2–4 天 | 100k step 实测吞吐、512 推理网格 |
| DiffJSCC D2 ControlNet 训练 | 4–8 天 | BLIP2 在线 caption、batch=1、activation checkpoint |
| SGD G1 rate/edge adaptation | 1–3 天 | 16→12 bottleneck 收敛、edge cr=8 |
| SGD G3 diffusion adaptation | 4–8 天 | 非官方 trainer、可训练范围与显存 |
| 64×3×5 + COCO-2048 全指标评估 | 2–4 天 | 多个 50/100-step diffusion 臂、FID/KID feature cache |
| **顺序总计** | **约 14–29 天** | smoke 若发现 FP32 OOM 或未收敛会靠近上限 |

现有 S30 实测 DiffJSCC 100-step 推理约 `5.238 s/图`、peak allocated `14.93 GiB`；它说明推理可行，但不能推出训练能在 FP32/batch8 下装入 24 GB。因此获准后第一步必须只跑 1-batch forward/backward smoke，并先汇报显存与 step time；不能直接放任到 100k/25k steps。

## 7. 不得不接受的局限

1. **SGD 不能称官方复现。** 缺训练代码使任何补全都带实现者判断；最严谨定位只能是 released-component 的 rate-constrained adaptation。
2. **“同 COCO”只能对齐 task adaptation 数据，不能抹掉生成先验的外部预训练。** DiffJSCC 使用 SD2.1/BLIP2/OpenCLIP，SGD 使用其 released generative backbone；若要求所有参数只见 COCO、全部从零训练，技术与算力上都不现实，也不再是论文方法的复现。
3. **训练算力不等。** S33 约 31M 且 12 epochs；生成系统为十亿级冻结/部分可训练模块和更长训练。可以公平比较码率、数据任务、信道与结果，但不能声称 equal-compute；必须报告 GPU-hours 和 trainable/total parameters。
4. **24 GB 显存可能迫使 diffusion 阶段使用 BF16/FP16、gradient checkpointing、batch1+accumulation。** 这是系统可行性让步，需在 smoke 后冻结并报告。
5. **SGD 主码率被压到 12,288 real 后，架构已经不是作者原生 16-channel latent。** 这正是总预算约束的必然代价，但也意味着“缩水”是公平系统设计后的结果，不是对作者 checkpoint 做无损等价变换。

## 8. 等待确认的三项原则

在任何 smoke/训练前需要用户确认：

1. 接受 DiffJSCC 为主要可裁决公平基线、SGD 为 approximate secondary baseline；
2. 接受“同 COCO task adaptation、但保留论文所需外部生成预训练”，并如实报告；
3. 接受 diffusion 阶段在 FP32 smoke OOM 时改用 BF16/gradient-checkpoint/batch1+accumulation，同时不主张 equal-compute。

预注册机器可读配置：`configs/s34c_fair_generative_reproduction_preregistration.yaml`。
