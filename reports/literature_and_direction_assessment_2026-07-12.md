# 文献、现状与未来方向评估

日期：2026-07-12

## 0. 结论先行

项目已经不是“只有想法”的早期课题，而是一个证据链较完整的强 pilot：有正式 DeepJSCC 基线、blind diffusion 负结果及 VAE 归因、pixel residual 正结果、edge 受控消融、严格等总码率结构侧信道、独立监督 clean-correct 审计、预注册安全门槛和失败记录。

但它还不是可直接投稿的最终方法。当前最重要的缺口不是再加一个模块，而是先解决归因和公平性：`c=6 main + c=2 structure + refiner` 主要与裸 `c=8 DeepJSCC` 比较，尚缺同协议下的 `c=8 + 同等 refiner`、双分支但无结构语义的容量匹配对照，以及模型参数、训练预算、推理开销和多信道随机种子审计。因此现有结果证明“等信道使用量的完整系统有效”，还不能把全部收益归因于结构/语义表示。

推荐主线收紧为：

> **Rate-accounted semantic-risk control for generative/restorative JSCC：在严格等总码率下，显式测量 refinement 相对基线新制造的语义错误，并用接收端可见、信道条件化的 semantic anchor 与 selective failure handling 约束该风险。**

未来顺序应为：

1. 先补因果和公平基线；
2. 再把当前随机投影 sketch 换成真正样本特异、可解释、可传输的 semantic anchor；
3. 再做风险受约束的选择性控制；
4. 方法冻结后扩大训练/信道随机种子/监督审计；
5. 最后决定是否加入短链 conditional diffusion 和 Rayleigh。

若最终稳定正方法仍是 residual CNN，应从标题中移除 `Diffusion-JSCC`；只有短链、近 M0 初始化、强条件化的 diffusion correction 真正优于 CNN anchor 后，才保留 diffusion 作为主方法。

## 1. 当前项目实际完成了什么

### 1.1 已成立的资产

| 资产 | 当前证据 | 可以支持的结论 |
|---|---|---|
| 高分辨率 DeepJSCC | COCO-256、AWGN、CBR≈0.17、五个 SNR；稳定 `best.pt` | 主实验骨架已经具备；`latest.pt` NaN，正式流程只能用 `best.pt` |
| Blind SD img2img | mean ΔPSNR `-14.7485 dB`、ΔLPIPS `+0.3877` | 盲生成式后处理是明确负参考；不能包装为感知提升 |
| SD VAE roundtrip | 不运行 UNet 已损失约 `3.49–7.33 dB` | blind SD 失败有具体归因，不只是 strength 没调好 |
| Pixel residual restoration | `EXP-S4-006` mean ΔPSNR `+0.7235 dB`、ΔLPIPS `-0.0274` | JSCC 后恢复本身有效；稳定正 anchor 是 pixel-domain restoration，不是 diffusion |
| Edge conditioning | matched 2×2 后 large edge raw effect `+0.1389 dB`，跨四 split CI 均为正 | receiver-visible structure 对质量有独立贡献；不等于语义安全 |
| Supervised audit | scratch `T_cls`、clean-correct、cluster bootstrap、new-error upper、official val sealed | 语义风险评价纪律强，是项目最重要的可复用资产之一 |
| Matched-total-rate structure | `c8` 对 `c6+c2`；COCO frozen downstream `+0.3772 dB`，Imagenette `+1.8341 dB` | 等信道使用量下，main+decoded structure 完整系统有质量和净语义收益 |
| Hybrid semantic sketch | payload 占 `c2` real latent 的 `0.78125%`；AWGN 后 cosine 高 | 小带宽 semantic payload 可稳定传输 |
| Mainline M3 controller | new-error clusters `23→18`，保留 `74.8%` raw PSNR gain | semantic control 有部分作用，但未显著改善 raw failure，未过安全门槛 |

### 1.2 不能越界的结论

- Blind diffusion 的负结果不能证明所有 conditional diffusion 都失败。
- Edge 的质量增益不能写成 semantic reliability 增益。
- AlexNet gate 定义下的 0 new error 是规则内生结果，不是跨模型安全证据。
- Matched-total-rate 结果不是“语义无损”：其 supervised new-error upper 为 `2.4764%`，高于预注册 `0.5%`。
- Hybrid sketch 的 received-vs-shuffled PSNR CI 跨 0，不能称为有效 semantic grounding。
- M3 将 new-error clusters 降约 22%，但 raw-minus-M3 failure CI 跨 0，不能称为控制器显著有效。
- 所有 `c6/c2` 组件仍是 20k warm-start pilot，refiner 只用 160 张 COCO development images；不能当作 full-scale final model。

## 2. 本轮技术审计发现的关键缺口

### 2.1 最大缺口：matched rate 不等于 matched system complexity

当前 reference 是单个 `c=8` DeepJSCC，而 proposed 使用 `c=6` main DeepJSCC、`c=2` structure DeepJSCC 和 residual refiner。信道使用量严格相等，但 proposed 的网络数量、参数、编码/解码计算和训练过程更多。

正式归因至少需要：

| Arm | 目的 |
|---|---|
| B0 `c8 DeepJSCC` | 裸通信基线 |
| B1 `c8 + same-capacity receiver-only refiner` | 判断收益是否只是来自后处理 |
| B2 `c6 main + c2 generic RGB/detail path + same refiner` | 控制双网络参数和额外编码分支 |
| B3 `c6 main + c2 decoded structure + same refiner` | 隔离 structure representation 的增量价值 |
| B4 `c6 main + c2 structure/semantic + same refiner/controller` | 隔离 semantic payload 和 risk control 的增量价值 |

还需报告 encoder/decoder/refiner 参数量、MACs/FLOPs、延迟、显存和 semantic teacher 的发送端计算。

### 2.2 训练预算与规模没有完全匹配

- Reference `c8` 来自正式 COCO 训练；`c6/c2` 是从 `c8` warm-start 后在 20k 子集上短训。
- `c6` 与 `c2` 的 epoch 数不同，虽有合理任务差异，但不是训练预算匹配实验。
- S7/S8 refiner 用 160 张图训练、64 张图选优，适合 pilot，不足以支撑最终泛化结论。

正式实验应使用 COCO train2017 生成独立 refiner train/validation，不再用 val export 的 160 张作为最终训练集；同时给所有关键 arm 相同的数据规模、选择规则和可比训练预算。

### 2.3 当前统计没有覆盖信道随机性的全部不确定性

现有 bootstrap 以 image ID 聚类，能处理同一图像跨 SNR 的相关性，但每个正式 audit 主要使用一个 channel seed。它估计的是“当前信道随机实现下跨图像的差异”，不是对重复传输噪声的完整期望。

正式版本需要多个固定 channel seeds，并区分：

- `R_new(snr)`：一次传输在给定 SNR 下的新错误概率；
- image susceptibility：同一图像跨若干信道实现是否反复易错；
- worst-SNR / worst-class / CVaR-like tail；
- model-training seed variance。

不能简单把更多 seed 下的“任一事件”全部合并成一个 image-any-error 指标，否则指标会随试验次数机械上升。

### 2.4 当前 semantic sketch 不是充分的样本身份表示

实现上，32-D sketch 是 1000-D AlexNet probability 的固定 Rademacher 投影，经 analog repetition 覆盖 `c2` latent 的 128 个位置，再对整个 latent 重新归一化。该设计有四个边界：

1. received 相比 zero 有收益，但 received 相比 shuffled 不显著，说明网络主要利用非零条件能量；
2. payload 是 post-hoc overwrite，structure encoder/decoder 没有联合学习如何分配符号和功率；
3. 当前因果 gate 主要看 received-vs-shuffled PSNR，未直接要求 semantic failure/new-error 改善；
4. 发送端和控制器都使用 ImageNet AlexNet，虽与 scratch ResNet18 `T_cls` 独立，但仍是单一 teacher-specific semantic loop。

下一版必须同时记录 symbol fraction、实际 power fraction，并把正确/打乱/置零 payload 的比较扩展到 new-error、repair 和最终 failure。

### 2.5 监督协议严谨，但 policy-dev 已经完成其职责

Imagenette `policy_dev` 已用于多轮方法诊断，适合作为开发集，不可再承担最终证据。Official val 尚未访问，这是好事；在方法和协议冻结前继续保持封存。

最终 promotion 还需要：

- 新的、独立的方法开发/校准 population；或在不改变现有 sealed protocol 的前提下使用另一带标签数据集开发；
- official val 一次性 final audit；
- 至少一个不同 evaluator family 的二级审计；
- 如要做 benchmark，增加 object detection/segmentation 或更广分类集，避免 10-class Imagenette 成为唯一 hard semantics。

### 2.6 正向 diffusion 方法仍缺席

目前正结果属于 residual CNN，真正 diffusion 只有 blind SD 和 naive residual DDPM 负结果。SING、DiffJSCC、SGD-JSCC、LRISC、DiT-JSCC、ADDPS 等都表明 conditional/inverse/posterior diffusion 是活跃方向；因此仅凭当前实验不能把项目写成“semantic-drift controlled diffusion 方法”。

## 3. 文献地图与真实 novelty 边界

### 3.1 最相关生成式 JSCC / SemCom

| 工作 | 已覆盖内容 | 对本项目的直接约束 |
|---|---|---|
| DiffJSCC, 2404.17736 | Stable Diffusion、多模态条件、SNR 条件、下游任务 | 不能声称首次 diffusion-aided / SNR-conditioned JSCC |
| RDP-JSCC/DPCT, 2408.14127 | rate-distortion-perception controllable JSCC | 一般多轴曲线和感知控制不是 novelty |
| Rate-Adaptive CDM-JSCC, 2409.02597 | entropy rate control + conditional diffusion decoder | rate-adaptive generative image JSCC 已存在 |
| SGD-JSCC, 2501.01138 | text/edge guidance、AWGN/slow/fast fading、channel-adaptive diffusion | “semantic guidance + channel adaptation”组件组合已高度接近 |
| SING, 2503.12484 | DeepJSCC reconstruction 的 inverse-problem diffusion restoration | 两阶段 diffusion refinement 已被直接覆盖 |
| LRISC, 2504.21577 | latent features、adaptive coding length、conditional diffusion、SNR adaptation | “latent semantic condition + SNR adaptive generation”已存在 |
| RD-JSCC, 2505.21681 | MIMO CSI residual diffusion、2-step inference、channel-conditioned switch | 短链 residual diffusion + adaptive switch 不能单独作为创新；但任务不是自然图像 |
| TOAST, 2506.21900 | real-time channel state 下平衡 reconstruction/classification，并含 latent EDM | 一般 channel-adaptive semantic task balancing 已存在 |
| MTGC, 2512.06344 | caption/HCI/pseudo-word 多模态 guidance，针对 generative hallucination | 多模态语义 anchor 设计竞争激烈，必须强调 rate/risk protocol |
| DiT-JSCC, 2601.03112 | semantics-detail encoder + DiT，CLIP/DreamSim/DINOv2 semantic metrics | semantic consistency 叙事和 feature metric 已被占用 |
| JSCGC, 2601.12808 | generation coding、mutual information、semantic inconsistency lower bound | “generative communication 的 semantic inconsistency”已有理论 framing |
| SBGSC, 2604.17802 | Schrödinger bridge，声称降低 hallucination 和采样成本 | 仅说更短轨迹减少 hallucination 不足；需 hard risk endpoint |
| ADDPS, 2604.16796 | dual-domain posterior sampling、感知最优理论 | 若救 diffusion，应考虑 posterior/data consistency，不要从随机 residual 盲采样 |
| STCC, 2606.11819 | 离散 semantic token codebook、拓扑对齐 constellation、明确 “Semantic Drift” | semantic token 与术语均有近邻；本项目需落在图像 refinement-induced risk |

### 3.2 度量、风险与理论工具

| 工作 | 可借鉴点 | 不能误用的地方 |
|---|---|---|
| Blau & Michaeli PD tradeoff, 1711.06077 | 感知质量和失真存在基本冲突 | 不直接证明分类 new-error |
| CDP tradeoff, 1904.08816 | classification error 是独立评价轴 | 不等于本项目首次提出语义轴 |
| RDPC/JSCM, 2312.14792 | rate/distortion/perception/classification 四元权衡 | 四轴曲线只能是分析，不是核心 novelty |
| Cohen et al., 2405.16475 | uncertainty-perception tradeoff 支撑“越真实越需谨慎” | inherent uncertainty 不能直接替代实际 new-error 统计 |
| Hallucination Index, 2407.12780 | 重复生成分布和 SNR 对 hallucination 的影响；较弱 denoising 可降风险 | 医学/重复采样度量，不是自然图像分类安全标准 |
| HalluGen/SHAFE, 2512.03345 | 可控 hallucination 数据、reference-free detector | 当前主要是 MRI，迁移需重新验证 |
| Selective Classification, 1705.08500 | risk-coverage、reject option | 保证依赖校准分布，不能直接声称逐样本安全 |
| Conformal Risk Control, 2208.02814 | 对单调 loss family 控制期望风险 | 当前 alpha 的 new-error 未必单调，需要先验证或使用更一般校准方法 |

### 3.3 可以守住的窄 gap

当前检索支持、但正式论文仍应避免“首个”绝对表述的差异是：

> 在严格 matched total rate 下，把 **baseline-correct 样本被 generative/refinement 模块新破坏** 定义为独立事件，报告其按 SNR、类别和信道随机性的 tail/UCB，并让 receiver-visible controller/failure handler 直接优化或校准该事件风险。

区别不在于用了分类器、语义 token、SNR 或 diffusion，而在于：

- 风险有明确的因果参照（pre-refinement / matched baseline）；
- repair 不能抵消 new error；
- side information 的 rate/power/error 都计入；
- 控制器需证明相对 raw refiner 的额外价值；
- 失败时允许 abstain/fallback，而不是只追平均分类准确率。

## 4. 候选方向比较

| 方向 | 复用现有资产 | 新颖性 | 技术风险 | 工作量 | 建议 |
|---|---:|---:|---:|---:|---|
| A. 等码率 semantic anchor + risk-constrained selective restoration | 高 | 中高 | 中 | 中 | **主推荐** |
| B. Semantic reliability benchmark / measurement protocol | 中高 | 中 | 中高 | 高 | 方法失败后的备选，或与 A 合并为副贡献 |
| C. 近 M0 的短链 conditional residual diffusion | 中 | 中 | 高 | 中高 | A 的可选生成式后端，不应先做 |
| D. Rayleigh/fading 扩展 | 高 | 低 | 低中 | 中 | 方法冻结后必做的外部验证，不是主创新 |
| E. Reference-free drift detector | 中 | 中高 | 高 | 高 | 后续课题，不作为当前收敛主线 |
| F. 从零训练大型 DiT-JSCC | 低 | 低/撞车 | 极高 | 极高 | 不做 |

## 5. 推荐方法方向

### 5.1 Semantic anchor：从随机投影改为显式身份

优先只筛三类、固定总预算的候选，避免再次横向扩张：

1. **Top-k discrete semantic checksum**：发送端冻结 teacher 的 top-k class IDs、量化概率和置信度；优点是实现快、身份明确，缺点是类别空间依赖强。
2. **Quantized global self-supervised token**：冻结 DINO/CLIP-like encoder 的 global feature 经小 codebook/PQ 压缩；比随机投影更可能保持样本距离。
3. **Global + coarse spatial tokens**：少量全局 token 加 2×2/4×4 spatial semantic tokens；工作量更大，但最可能约束 object identity 和 layout。

所有候选必须：

- 固定占用 `c2` 内相同 real/complex symbols；
- 记录实际 transmit power fraction；
- 经 AWGN，而非无错 side information；
- 与 structure encoder 联合训练或显式分配子信道，不再 post-hoc overwrite 后擦零；
- final `T_cls` 不参与编码、训练、选择或校准；
- 通过 received / shuffled / zero 的质量和 hard semantic 双重因果测试。

### 5.2 Restoration 内部融合，而不是末端只做 router

当前 c6 main 与 raw candidate 存在共同错误，二选一无法通过严格门槛。Semantic anchor 必须进入 refiner 内部，产生新的受约束 candidate。可采用：

- multi-layer FiLM，而非只在 head 后一次调制；
- cross-attention 到全局/空间 semantic tokens；
- semantic consistency residual loss；
- correct-vs-shuffled contrastive/ranking loss；
- reconstruction-dominant 训练，防止分类目标破坏 restoration anchor。

### 5.3 Risk-constrained selective controller

控制器只允许使用 receiver-visible 特征：SNR、decoded main/structure、payload recovery confidence、candidate disagreement、semantic-anchor consistency 和内部 uncertainty。

建议输出 `alpha` 或 `{accept, weaken, abstain}`，并报告完整 risk-coverage 曲线。开发目标应写成约束形式，而不是任意加权和：

```text
maximize    quality_gain(policy)
subject to  UCB[R_new(policy | SNR=s)] <= epsilon_s
            for each primary SNR s
            and retain_raw_gain >= rho
```

Selective classification 和 conformal risk control 可用于校准思想，但有限样本保证只在明确的数据交换性、loss family 和校准协议下成立；本项目不应提前声称“conformal safety guarantee”。

## 6. 推荐实验路线与停止条件

### P0：先补因果基线（最高优先，暂不设计新方法）

任务：在同一图像、SNR、channel seeds、训练预算和 refiner capacity 下完成 B0–B4。

通过条件：

- B3 相对 B1/B2 的 structure 增量在质量或 hard semantics 上有 CI 支持；
- B4 相对 B3 有样本特异 semantic 增量；
- 完整报告 rate、power、params、latency。

停止条件：若 B3 不优于 `c8 + same refiner` 或 capacity-matched dual-path control，则不能继续把结构侧信道写成主要方法，应退回 measurement/risk-control 方向。

### P1：semantic anchor 小型筛选

只在新的 COCO development split 上筛最多三种表示，固定 payload budget。正式 causal gate 应同时要求：

- received-vs-shuffled 的样本聚类 CI 在预注册主端点上优于 0；
- zero/shuffled 不得通过同样 gate；
- structure 质量损失不超过预设比例；
- 至少一个独立 semantic evaluator 同向。

未通过则停止继续加复杂 controller，因为没有可靠 source identity 可供控制。

### P2：risk controller 与校准

- 新建独立 labeled development/calibration population；不再使用现有 Imagenette policy_dev 调参。
- 比较 SNR-only、top-1 fallback、当前 sketch cosine、risk predictor、selective/conformal-style calibration。
- 报告 new error、repair、failure、coverage、risk-coverage/AURC-like summary、worst class、per-SNR UCB。

Promotion gate 至少保留现有标准：controller 相对 raw failure 的 CI 上界严格小于 0、new-error upper `<=0.5%`、保留 `>=50%` raw quality gain。

### P3：formal scale-up

- 用 COCO train2017 训练正式 c8/c6/c2/refiner，而不是 20k/160-image pilot。
- 关键模型至少多个训练 seed；正式评估至少多个 channel seeds。
- 扩大图像数，补 DISTS/CLIP/DINO 等可比指标，但 hard supervised semantics 仍为主。
- 补参数、FLOPs、latency、显存和发送端 semantic extractor 成本。
- 方法和 protocol hash 冻结后，一次性访问 Imagenette official val。

### P4：是否救 diffusion 的 go/no-go

仅在 P0–P2 主线成立后尝试：

- 从 main/refiner output 附近初始化；
- 2–8 step deterministic/low-noise residual diffusion；
- main + decoded structure + semantic anchor 强条件；
- data/posterior consistency；
- 与同容量 residual CNN 正面对比。

只有在 LPIPS/感知质量显著优于 CNN anchor、PSNR 回吐可控、new-error 不恶化且开销可接受时才保留。否则正式改名为 `semantic-risk-controlled restorative JSCC`。

### P5：Rayleigh 与 benchmark 扩展

Rayleigh 用于证明 channel-adaptive 迁移性，应在方法冻结后做。若转 benchmark 路线，则必须覆盖多种 generative/refinement 方法、至少 AWGN+Rayleigh、多个有监督数据集/任务和统一公开协议；仅分析当前自研系统不足以构成强 benchmark。

## 7. 论文叙事建议

### 成功的方法论文版本

1. Blind/generative refinement 会产生无法被平均指标反映的新语义错误；
2. 定义 refinement-induced new error、repair、tail/UCB 和 risk-coverage；
3. 提出严格计 rate/power 的 semantic anchor；
4. 提出 channel-conditioned selective risk controller；
5. 在 matched system baselines、AWGN/Rayleigh 和独立监督审计上证明控制器的额外价值。

候选标题：

- `Rate-Accounted Semantic Risk Control for Generative JSCC Restoration`
- `Do No New Semantic Harm: Selective Risk Control for Generative Image JSCC`
- 若 diffusion 成功：`Semantic-Risk-Constrained Conditional Diffusion for Rate-Accounted Image JSCC`

### 若方法 gate 最终失败

可以转为 measurement/benchmark 论文，但必须扩大方法、信道和数据覆盖。可保留的核心发现是：平均 failure 改善不代表没有 individual new harm；matched-rate structure guidance 能显著提高质量和净语义表现，但简单 gate、随机 sketch 和浅层 router 无法满足严格安全端点。

## 8. 当前最优先的五个动作

1. 不再在现有 Imagenette policy_dev 上调任何 alpha/threshold/model。
2. 先实现/整理 B0–B4 公平对照，而不是立刻训练新 semantic token。
3. 为正式比较加入多 channel seeds、参数/延迟/power accounting。
4. 只有 B3/B4 通过增量归因后，做最多三种 semantic anchor 的预注册筛选。
5. 在 positive diffusion backend 出现前，内部将主线称为 `semantic-risk-controlled restoration`，避免题目和实际方法错位。

## 9. 主要文献链接

- DiffJSCC: https://arxiv.org/abs/2404.17736
- RDP-JSCC: https://arxiv.org/abs/2408.14127
- Rate-Adaptive CDM-JSCC: https://arxiv.org/abs/2409.02597
- SGD-JSCC: https://arxiv.org/abs/2501.01138
- SING: https://arxiv.org/abs/2503.12484
- LRISC: https://arxiv.org/abs/2504.21577
- RD-JSCC: https://arxiv.org/abs/2505.21681
- TOAST: https://arxiv.org/abs/2506.21900
- MTGC: https://arxiv.org/abs/2512.06344
- DiT-JSCC: https://arxiv.org/abs/2601.03112
- JSCGC: https://arxiv.org/abs/2601.12808
- ADDPS: https://arxiv.org/abs/2604.16796
- SBGSC: https://arxiv.org/abs/2604.17802
- STCC: https://arxiv.org/abs/2606.11819
- Perception-Distortion: https://arxiv.org/abs/1711.06077
- CDP: https://arxiv.org/abs/1904.08816
- RDPC/JSCM: https://arxiv.org/abs/2312.14792
- Cohen uncertainty-perception: https://arxiv.org/abs/2405.16475
- Hallucination Index: https://arxiv.org/abs/2407.12780
- HalluGen/SHAFE: https://arxiv.org/abs/2512.03345
- Selective Classification: https://arxiv.org/abs/1705.08500
- Conformal Risk Control: https://arxiv.org/abs/2208.02814
