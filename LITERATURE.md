# 文献记录

## 相关工作

### Diffusion JSCC / Generative JSCC

记录 DiffJSCC、SGD-JSCC、DiT-JSCC、JSCGC 等工作。

关注点：

- 是否把 diffusion 用作 JSCC decoder、post-refinement 或 generative reconstruction。
- 是否报告低 SNR 下的感知质量提升。
- 是否讨论 hallucination、semantic drift 或语义错误。
- 是否有公开代码、checkpoint、数据集和 SNR 设置。

### Channel-adaptive JSCC

记录 SNR-adaptive、bandwidth-adaptive、prompt-based JSCC 等工作。

关注点：

- 自适应变量是 SNR、CBR、带宽、信道类型，还是 prompt / condition。
- 自适应发生在 encoder、decoder、channel embedding、prompt module，还是后处理阶段。
- 是否能作为本项目 SNR-aware diffusion strength 的方法参照。

### Semantic reliability / hallucination

记录语义一致性、CLIP guidance、semantic drift 相关工作。

关注点：

- 如何定义 semantic consistency、semantic drift、semantic failure。
- 使用哪些度量：classification accuracy、prediction consistency、CLIP similarity、检测/分割一致性等。
- 是否区分“视觉更真实”和“语义更可靠”。
- 是否提供 failure case 分析。

## 本项目创新边界

本项目不声称首次使用 diffusion 做 JSCC。

本项目重点是：在信道自适应条件下控制 diffusion refinement 的 semantic drift。

需要明确区分：

- 已有工作：使用 diffusion / generative model 提升 JSCC 重建视觉质量。
- 本项目：研究不同信道条件下 diffusion strength / semantic guidance 如何影响 semantic drift，并给出可测量的语义可靠性指标。

## 论文笔记

### 数据集使用扫描：2026-06-30

结论：

- 传统 DeepJSCC / adaptive JSCC 工作常用 CIFAR-10，适合验证信道和编码器/解码器机制，但不适合作为 diffusion 主实验数据集。
- 最近 generative / diffusion JSCC 工作明显转向高分辨率自然图像：OpenImages、ImageNet、COCO2017 和 Kodak。
- Kodak 常作为小规模高分辨率测试集或视觉展示集；它只有 24 张图，不适合作为 semantic drift 主统计集。
- COCO2017 的优势是有 caption，可用于 CLIP score、text-image semantic consistency 和 failure case 描述；这和本项目 semantic drift/failure 主线一致。
- ImageNet 的优势是类别标签清楚，适合 classification consistency；但完整 ImageNet 获取成本较高，可用 Imagenette/ImageNet subset 做 pilot。

代表性论文设置：

| 工作 | 年份 | 训练数据 | 测试/评估数据 | 与本项目启发 |
|---|---:|---|---|---|
| Dynamic_JSCC | 2021/2022 | CIFAR-10 | CIFAR-10 | 自适应 rate/control 的早期路线，分辨率太低，只适合 sanity 参考 |
| DeepJSCC-l++ | 2023 | CIFAR-10 | CIFAR-10 | Swin/adaptive JSCC 重要对照，但不是 diffusion 主数据集参考 |
| DiffJSCC | 2024/2025 | OpenImages subset，154k 高质量图像 | Kodak，原始高分辨率 | diffusion-JSCC 使用高分辨率自然图像，Kodak 用于视觉和感知指标展示 |
| SGD-JSCC | 2025 | JSCC 用 ImageNet；DM 用 SA-1B、JourneyDB、CC3M、Datacomp、CelebA-HQ 等约 14M text-image pairs | COCO2017 validation | COCO caption 被用于 CLIP/text consistency，和 semantic reliability 评估强相关 |
| DiT-JSCC | 2026 | ImageNet train，256/512 crop | ImageNet validation held-out subset | 语义一致性评估转向 CLIP、DreamSim、DINOv2，ImageNet 适合类别/语义一致性 |
| JSCGC | 2026 | OpenImages subset，500k 图像，256 crop | Kodak 原始分辨率 | 明确讨论 semantic inconsistency；OpenImages + Kodak 是 generative communication 常见组合 |

对本项目的决定：

- 保留 CIFAR-10 作为 sanity baseline。
- 主路线优先使用 COCO2017 256x256，因为它兼顾自然图像、高分辨率和 caption-based semantic evaluation。
- COCO 下载未完成前，COCO `val2017` 固定切分 pilot 可用于训练 high-res JSCC checkpoint 和调试 diffusion 接口，但不能替代正式主实验。
- 后续可补充 Kodak 作为视觉展示集。
- 若需要 classification consistency，建议补 Imagenette/ImageNet subset，而不是把 CIFAR-10 作为 diffusion 主实验。

### 第一轮扫描结论：2026-06-29

初步结论：

- `Diffusion JSCC / Generative JSCC` 已经有较强相关工作，本项目不能声称首次使用 diffusion 做 JSCC。
- `SGD-JSCC` 已经覆盖“semantic guidance + diffusion + 信道条件适应”的一部分，本项目必须进一步强调 semantic drift 的显式定义、度量和 failure case。
- `DiT-JSCC` 和 `JSCGC` 已经把 generative JSCC 推到更系统的位置，尤其关注 semantic consistency / semantic inconsistency。本项目要避免走“大模型 generative decoder”路线，重点放在 plug-in diffusion refinement strength control。
- 第一版 baseline 建议优先从普通 DeepJSCC inference 路线开始，先得到可控的 pre-diffusion reconstruction，再接 diffusion refinement 和 semantic drift 评估。

### Diffusion JSCC / Generative JSCC

| 工作 | 年份 | 重点 | 与本项目关系 | 来源 |
|---|---:|---|---|---|
| DiffJSCC | 2024/2025 | 使用预训练 Stable Diffusion，通过多模态空间/文本特征和 SNR 条件引导 denoising，提升高真实感重建 | 强相关；已经覆盖 diffusion-aided JSCC 和 SNR 条件，需要避开“只是 diffusion decoder”的贡献表述 | https://arxiv.org/abs/2404.17736 |
| SGD-JSCC | 2025 | 使用文本描述、edge map 等语义信息指导 diffusion，并在慢衰落/快衰落信道下适应信道变化 | 强相关；本项目必须突出 semantic drift rate / semantic failure rate 的可测评估，而不是只说语义指导 | https://arxiv.org/abs/2501.01138 |
| DiT-JSCC | 2026 | 语义优先 encoder + DiT generative decoder，强调 extreme channel 下的 semantic consistency | 强相关；本项目不从零训练大型 DiT-JSCC，应定位为轻量 refinement/control 研究 | https://arxiv.org/abs/2601.03112 |
| JSCGC | 2026 | 将接收信号作为 generative sampling 条件，提出 Joint Source-Channel-Generation Coding，并讨论 semantic inconsistency 错误行为 | 强相关；提醒本项目必须认真定义 semantic inconsistency / drift，而不是只看 LPIPS/FID | https://arxiv.org/abs/2601.12808 |

#### SGD-JSCC 借鉴边界补充：2026-07-10

SGD-JSCC 的核心启发不是“直接把 Stable Diffusion 接到 JSCC 后面”，而是给 diffusion denoising 提供明确语义条件和信道自适应机制：语义条件包括 text description、edge map 等，信道侧包括慢衰落下由 channel output 估计瞬时 SNR、快衰落下 training-free denoising adjustment。官方代码仓库目前说明提供 AWGN inference 路径和 checkpoint 下载，preprocessing / training guideline 仍在 TODO，这意味着它可作为设计参照，但不适合直接替换当前最小闭环。

对本项目的可借鉴点：

- 用 receiver-visible semantic side information 约束 diffusion / restoration，而不是空 prompt blind generation。
- 如果重新做 diffusion，应优先考虑 edge/structure-conditioned residual correction，而不是从高斯噪声随机采样完整图像或完整残差。
- SNR/channel condition 应进入 denoising 或 residual strength controller，而不是只作为后处理日志。
- 本项目仍需保留自己的 semantic drift / accepted new error 指标；不能只复刻 SGD-JSCC 的感知质量叙事。

### Channel-adaptive JSCC

| 工作 | 年份 | 重点 | 与本项目关系 | 来源 |
|---|---:|---|---|---|
| DeepJSCC 原始工作 | 2018/2019 | CNN encoder/decoder + AWGN/Rayleigh 信道层，低 SNR 下相对数字方案有 graceful degradation | baseline 理论起点；第一阶段需要复现或接入类似 DeepJSCC | https://arxiv.org/abs/1809.01733 |
| Dynamic_JSCC | 2021/2022 | 单模型自适应 rate control，按信道条件和图像内容动态控制带宽；公开 PyTorch 代码 | 可作为 channel-adaptive 策略参考，不建议作为第一版主 baseline，因为它引入 rate policy 复杂度 | https://arxiv.org/abs/2110.04456 ; https://github.com/mingyuyng/Dynamic_JSCC |
| DeepJSCC-l++ | 2023 | Swin Transformer backbone，单模型适应多个 bandwidth ratio 和 SNR，代码公开 | channel-adaptive JSCC 重要撞车线；可作为后续对照或方法参照 | https://arxiv.org/abs/2305.13161 ; https://github.com/aprilbian/deepjscc-lplusplus |
| SwinJSCC | 2023/2024 | 四级 Swin Transformer JSCC；Channel ModNet / Rate ModNet 分别适应 SNR 与码率，官方代码和权重入口公开 | S33 strong backbone 的关键外部审稿基线；必须在同 COCO、exact rate、SNR 和训练预算下重训，不能直接引用其 DIV2K/CLIC checkpoint 排名 | https://arxiv.org/abs/2308.09361 ; https://github.com/semcomm/SwinJSCC |
| PJSCC | 2024 | Channel State Prompt 生成 learnable prompt，融合 SNR 和信道分布信息，实现跨信道自适应 | prompt-based/channel-adaptive JSCC 撞车线；本项目可借鉴 condition 设计，但要落在 diffusion strength / semantic guidance | https://arxiv.org/abs/2411.10178 |

### SwinJSCC 官方资产与 S33 公平对比边界：2026-07-22

本轮核验 official paper 与 `semcomm/SwinJSCC@a6d0e6da53548976acbe9317839a077ef31f190f`。官方 HR 架构为四级 Swin encoder/decoder，Base depths 为 `[2,2,6,2]`，Channel ModNet 用 SNR 调制 latent feature，Rate ModNet 则对通道维做可学习 mask。对本项目固定码率场景，`SwinJSCC_w/_SA` 比 `w/_SAandRA` 更合适：它保留 SNR adaptation，同时避免动态 mask 及其论文中提到的辅助 mask 信息。取 `C=64` 时，`256x256` 输入原生输出 `64x16x16=16,384 real`，严格对应本项目 `8,192 complex uses / CBR=1/24`。

官方代码的共同对比不能原样照搬：发布训练数据是 DIV2K/CLIC 混合，README 示例 SNR 为 `[1,4,7,10,13]`，源码每个 batch 只随机一个 SNR，且默认功率按整个 batch 统计；本项目 S33 则使用 COCO、逐图离散 `[1,4,7,13,19]` 和逐图单位功率。因此正式比较必须保留官方 Swin block/Channel ModNet 拓扑，但替换数据 adapter、逐图 SNR、逐图 normalization 和 canonical AWGN 注入，并在同 12-epoch/equal-step 合同下从随机初始化重训。这应称为 official-source architecture under a common contract，不是作者论文数值的直接复现。

训练预算还需分层。论文没有给出明确 epoch 数，但报告 DIV2K 上每个训练 step model 使用单张 RTX 3090 约四天，并说明 SNR-adaptive 模型先训练基础/非 ModNet 参数，再训练带 Channel ModNet 的全模型；官方源码则把 `tot_epoch` 设为 `10,000,000` 的开放上限、每 100 epochs 保存。这些事实不能换算成精确的“作者 epoch 数”，但足以说明 12-epoch equal-budget 不能未经曲线检查就称为 SwinJSCC 已收敛。S34A 因此把同预算结果与按冻结 val-curve gate 触发的充分训练结果分开报告。

静态参数审计显示：SA-only official Base、`C=64` 为 `28,182,512` 参数，比 S33 少 `9.17%`；只把第三 stage depth 从 6 增至 8 可得 `31,348,752` 参数，与 S33 相差 `+1.03%`。因此最稳妥的论文对照是同时报告未改 Base 和 capacity-matched official-code control，并以两者中更强者给出保守 verdict。官方 README 提供 Google Drive 权重入口，但这些权重合同不匹配且不参与正式训练/排名；仓库根目录未发现 LICENSE，需保留源码许可边界。完整预注册：`reports/swinjscc_equal_rate_comparison_preregistration_2026-07-22.md`。

### DeepJSCC-l++ 细读结论：2026-06-30

论文：`DeepJSCC-l++: Robust and Bandwidth-Adaptive Wireless Image Transmission`。

可取之处：

- 把 SNR 和 bandwidth ratio 作为 side information 同时送入 encoder 和 decoder，用一个模型覆盖多个信道质量和多个 CBR。
- 采用 Swin Transformer backbone，证明 Transformer 对 JSCC feature prioritization 有帮助。
- 通过 mask/zero-padding 支持不同 bandwidth ratio，不需要为每个 CBR 存一套模型。
- 提出 Dynamic Weight Assignment（DWA），避免多 CBR 联合训练时 loss 被低 CBR 样本主导，导致高 CBR 性能退化。
- 代码公开，可作为后续 channel-adaptive JSCC 对照或复现参考。

局限：

- 实验主要是 CIFAR-10，分辨率偏低，不适合作为 diffusion 主实验数据集依据。
- 指标主要围绕 PSNR/MSE，不处理 diffusion hallucination 或 semantic drift。
- 主贡献在 encoder/decoder 自适应，不是 post-diffusion refinement；如果现在引入，会把主线从 semantic drift controlled diffusion 拉向 adaptive JSCC backbone。

对本项目的使用方式：

- 第一版不改成 DeepJSCC-l++ 主 baseline，继续使用 CNN DeepJSCC + high-res pilot，先完成 diffusion/refinement 和 semantic drift 评价闭环。
- 可借鉴 side information 设计：把 SNR/CBR embedding 用于后续 diffusion strength predictor 或 semantic failure predictor。
- 可借鉴 DWA 思想：若后续同时训练多个 SNR/CBR 的 diffusion controller，可按各条件的 semantic drift/failure gap 动态调整 loss weight。
- 可作为论文 related work 中的 channel-adaptive JSCC 代表，强调本项目不同点是控制 diffusion 引入的 semantic drift，而不是提出新的 bandwidth-adaptive JSCC backbone。

### Baseline 代码候选

| 候选 | 优点 | 风险 | 初步判断 |
|---|---|---|---|
| `chunbaobao/Deep-JSCC-PyTorch` | 结构直接，包含 `train.py`、`eval.py`、`channel.py`、`dataset.py`；支持 CIFAR-10 自动下载，记录 SNR/ratio/channel，输出 checkpoint/config/log/eval 目录 | 非官方复现，README 自述可能有错误；需要先做 smoke test 和代码审计 | 第一优先候选，用于阶段1快速跑通普通 DeepJSCC baseline |
| `mingyuyng/Dynamic_JSCC` | PyTorch；论文代码；包含 train/test 和 SNR 参数；与 adaptive rate control 直接相关 | 方法本身已经是 adaptive rate，不适合作为最干净的 pre-diffusion baseline | 第二候选，用于学习 adaptive control 和后续对照 |
| `aprilbian/deepjscc-lplusplus` | 论文官方代码；支持 SNR/bandwidth adaptive，Swin backbone | 复杂度比第一阶段需要的高；可能把问题带向 adaptive encoder/decoder 而非 diffusion refinement | 第三候选，作为 channel-adaptive baseline / related work 参照 |

### SGD-JSCC 语义侧信息精读与公平比较边界：2026-07-10

arXiv `2501.01138v2` 的 coarse semantics 是由现成 image captioner 提取的 text description，并在实验叙述中假设文本可完美传输、成本可忽略；文本由 CLIP 编码后作为 cross-attention 的 key/value。fine semantics 是 edge map，经独立 DeepJSCC 路径传输并通过 ControlNet 注入 diffusion。论文示例口径中主图 JSCC feature CBR 约 `1/8`、edge CBR 约 `1/24`，合计约 `1/6`。

对本项目的约束：

- 可以借鉴“coarse text + fine edge + channel-adaptive denoising”的分层语义条件，但不能继承“文本成本为零”的假设而不记录 raw bits、FEC 和错误率。
- 当前 4/80-bit Imagenette source-description 诊断显示，coarse semantics 只放在末端 router 无法解决 semantic grounding 瓶颈。
- perfect source-edge oracle 显示 fine structure 注入 restoration 的可达收益很大，但这恰好说明 rate accounting 不能省略：未编码的 full-resolution edge maps 不是通信结果。
- 公平 follow-up 应把当前主图从 CBR `0.17≈1/6` 降到约 `1/8`，为 edge path 保留约 `1/24`，固定 total≈`1/6`；还需加入 edge channel error、独立监督 evaluator 和 accepted-new-error endpoint。
- 本项目的可区分贡献不是再次提出 semantic-guided diffusion，而是“在 matched total rate 下，显式测量并约束 source-guided restoration 引入的 semantic drift/new error”。

2026-07-11 的 matched-rate follow-up 已完成上述公平边界：`c=6` main + `c=2` lossy structure 与 reference `c=8` 都是 total CBR `1/6`。decoded structure 在 COCO frozen downstream splits 上带来 `+0.3772 dB`（95% CI `[+0.3274,+0.4253]`），并在独立 Imagenette policy-dev 上带来 `+1.8341 dB` 与 supervised failure `3.3785%→1.2375%`；但 new-error image-cluster 上界仍为 `2.4764%`，未过 `0.5%` 门槛。这说明 SGD-JSCC 的分层描述思想确实能转化为公平等码率收益，但“fine edge 足够保护 semantics”不成立。

因此下一步对 SGD-JSCC 最合理的模仿不是继续增加 receiver-side threshold，而是保留明确码率账本，把 `c=2` 拆成或改造成可学习的 coarse-semantic checksum + fine-structure representation，并在 restoration 网络内部融合。语义描述编码器不得使用最终 `T_cls`，候选必须先在独立开发数据上冻结，再做新的预注册监督审计。

S8 的 rate-accounted 实现进一步收紧了这个判断：仅占 `c=2` latent 0.78125% 的 32-D 连续 sketch 可以稳定通过 AWGN，并相对 zero condition 带来显著质量收益；但正确 sketch 相对 shuffled sketch 的 downstream CI 仍跨 0。随机投影保留了可用条件能量，却没有充分保留样本特异语义身份。后续更值得尝试显式 top-k class/caption token、空间语义 token 或可解释 checksum，而不是继续扩大随机投影 FiLM 的损失权重。

把该 sketch 进一步接回主线 M3 alpha controller 后，独立 `T_cls` 审计显示它能把 hybrid-raw new-error image clusters 从 23 降到 18，并保留约 75% raw PSNR gain；但总 failure 改善不显著且绝对 new-error 上界仍超标。这支持 SGD-JSCC 式 semantic description 应作为 restoration 内部条件与风险控制共同使用，而不是只做路由；同时也说明随机投影 checksum 仍弱于显式 class/caption/spatial token grounding。

来源：arXiv `https://arxiv.org/abs/2501.01138v2`。

### OpenCode 调研残留材料复核：2026-07-12

本轮从 `/tmp/opencode/session_export.json` 恢复了未落盘报告的摘要、两个调研子任务完整输出和原始 arXiv 查询结果。材料可用于重建报告，但不能原样采用。关键核验与修正如下。

| 工作 | 核验后的直接相关点 | 对本项目定位的影响 | 来源 |
|---|---|---|---|
| Blau & Michaeli, Perception-Distortion Tradeoff | 证明感知质量与失真不可同时任意优化 | 可作为 blind/generative refinement 风险的基础理论，但不包含语义 drift | https://arxiv.org/abs/1711.06077 |
| Liu et al., Classification-Distortion-Perception Tradeoff | 把分类错误率加入 distortion/perception，证明三者存在基本权衡 | 支撑分类一致性作为独立评价轴；不能据此声称本项目首次提出多轴评价 | https://arxiv.org/abs/1904.08816 |
| Fang et al., RDPC/JSCM | 已在 JSCM 中明确研究 channel rate、distortion、perception、classification accuracy 四元权衡 | 直接否定“Rate-Distortion-Perception-Drift 四轴本身无人做”的宽泛 novelty 表述；本项目必须改为 refinement-induced drift/new error、tail risk 和闭环控制 | https://arxiv.org/abs/2312.14792 |
| RDP-JSCC / DPCT | 已提出 rate-distortion-perception controllable JSCC，并用 realism map 控制生成式恢复 | “画多轴曲线”只能作为分析工具，不能单独作为主要方法贡献 | https://arxiv.org/abs/2408.14127 |
| Cohen et al., Looks Too Good To Be True | 给出 uncertainty-perception tradeoff；完美感知对应至少两倍 inherent uncertainty 的下界 | 可解释“更真实但更不可靠”是基本风险，但其 uncertainty 不能直接等同本项目的分类 new-error rate | https://arxiv.org/abs/2405.16475 |
| TOAST | 根据实时信道条件，用 RL 动态平衡 reconstruction fidelity 与 classification accuracy，并包含 latent EDM denoiser | 与“channel-adaptive + semantic task + diffusion”高度接近；本项目不能再把这三个组件的并列组合作为 novelty | https://arxiv.org/abs/2506.21900 |
| DiT-JSCC | 明确讨论 diffusion GenJSCC 的 semantic inconsistency；用 CLIP、DreamSim、DINOv2 评估 semantic consistency | 本项目的区别必须落在离散 failure/new-error、refinement causality 和风险处理，不能只说 semantic consistency | https://arxiv.org/abs/2601.03112 |
| STCC | 在离散 foundation-model token 通信中明确使用并刻画 “Semantic Drift” | “semantic drift”术语本身已被占用；本项目应限定为 generative/refinement-induced image semantic drift | https://arxiv.org/abs/2606.11819 |
| HalluGen / SHAFE | 构造可控的“感知真实但语义错误”恢复结果，并训练 reference-free hallucination detector | 可作为未来 reference-free drift detector 的方法参照，但其当前验证域是低场 MRI，不可直接当作自然图像无线传输指标 | https://arxiv.org/abs/2512.03345 |
| Skocaj et al. | 在生成式信道估计中说明 score-based 方法更适合高 predictive uncertainty，低 uncertainty 下判别式方法更合适 | 为“按信道/不确定性决定是否启用生成先验”提供通信域类比，不是图像 JSCC 的直接结论 | https://arxiv.org/abs/2606.16815 |
| SING | 把受损 DeepJSCC reconstruction 的恢复表述为 inverse problem，并用 null-space / INN-guided diffusion 做两阶段图像恢复 | 直接覆盖“DeepJSCC 后接 diffusion refinement 提升感知质量”；但未把 refinement-induced hard new error 作为风险约束 | https://arxiv.org/abs/2503.12484 |
| RD-JSCC | 轻量 autoencoder + residual diffusion；按信道条件在低复杂度解码与 2-step diffusion refinement 间切换 | 与短链 residual diffusion 和 channel-conditioned switching 组件高度相似，但任务是 MIMO CSI reconstruction，不是自然图像语义可靠性 | https://arxiv.org/abs/2505.21681 |
| Rate-Adaptive Generative SemCom | entropy-based rate control + conditional diffusion decoder，用接收 symbols 条件化图像生成 | 覆盖 rate-adaptive generative image JSCC 的质量主线，不报告 refinement-induced hard semantic failure | https://arxiv.org/abs/2409.02597 |
| Latency-Aware Generative SemCom | 显式传输 prompt 与 edge 等多模态语义流，按信道自适应 modulation/coding、重传、功率与时延 | 说明“语义侧信息通信成本完全无人考虑”不成立；本项目的区别应是 matched-total-rate 公平对照和 new-error endpoint | https://arxiv.org/abs/2403.17256 |
| MTGC / Beyond Hallucinations | caption、highly compressed image、semantic pseudo-words 联合指导 diffusion compression | 生成式压缩已直接以 hallucination/semantic deviation 为动机；但主摘要仍以 DISTS 等为 semantic consistency 结果，本项目应保留 hard new-error endpoint | https://arxiv.org/abs/2512.06344 |
| ADDPS | 把 generative semantic decoding 表述为 Bayesian inverse problem，以 latent/image dual-domain posterior consistency 指导 diffusion | 若重启 diffusion，应优先考虑 measurement/posterior consistency，而不是从随机 residual 盲采样 | https://arxiv.org/abs/2604.16796 |
| SBGSC | 用 Schrödinger Bridge 缩短 semantics-to-image transport，声称降低 hallucination 与采样开销 | 说明短轨迹/直接 transport 已有近邻；本项目仍需 hard semantic risk，而不能只用 FID/SSIM 证明 hallucination 下降 | https://arxiv.org/abs/2604.17802 |
| Hallucination Index | 用重复 measurement/reconstruction 分布相对 zero-hallucination reference 的 Hellinger distance 量化生成式重建 hallucination | 提醒正式实验应覆盖 channel/generation randomness；该医学影像指标不能直接替代自然图像分类 new-error | https://arxiv.org/abs/2407.12780 |
| Selective Classification / Conformal Risk Control | 通过 coverage-rejection 权衡和校准控制预期风险 | 可为 M3 abstention/risk-coverage 提供方法工具；保证依赖校准分布与 loss 条件，不能直接宣称逐样本安全 | https://arxiv.org/abs/1705.08500 ; https://arxiv.org/abs/2208.02814 |

对 SGD-JSCC 的码率表述也需精确区分：论文明确忽略 text description 的传输成本并假设完美传输；edge map 则经独立 DeepJSCC 路径传输，实验中有对应 BCR。因而正确批评是“文本成本和可靠性未计入统一账本，edge 路径虽有 BCR 但仍需核对 matched-total-rate 公平性”，而不是笼统声称所有 side information 都未计码率。

本轮可保留的最窄研究 gap 是：在严格 matched total rate 下，针对 generative/refinement 模块相对 pre-refinement reconstruction 新引入的离散语义错误，报告 average 与 tail risk，并让 receiver-visible、channel-conditioned controller 或 failure handler 直接约束该风险。当前检索支持这是一个更有区分度的方向，但在完成更系统的逐篇全文核对前，不应使用“首个”或“领域空白”的绝对表述。

### 外部 baseline 可执行性与公平协议核验：2026-07-14

本轮不再只按论文摘要排序，而是核验 primary paper、作者仓库、checkpoint inventory 和 released inference contract，并据此冻结外部对比顺序。

| 方法 | primary-source 复现状态 | 公平性关键点 | 本项目排期 |
|---|---|---|---|
| SGD-JSCC | 作者仓库可运行源码已发布，本地固定 `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`；4 个作者权重约 2.931 GB，但另需 BLIP2/CLIP；batch preprocessing 和 training guideline 未发布；仓库未发现 license | 论文假设 text 完美传输且成本可忽略；edge 另走 JSCC；发布代码的 main/edge 实际 symbol count 与论文 complex-BCR 口径需 instrumentation | 第一项：作者原生表 + common-contract 表分开；总码率未知前禁止直接排名 |
| SING | 论文明确让 DeepJSCC/SING-Zero/SING-INN 共用同一 pretrained DeepJSCC；本轮未从论文/primary project page 定位到作者实现 | 与本项目 received-latent posterior restoration 最接近；若自行实现只能标 `SING-Zero-style`，不能冒充精确复现 | 第二项：同 frozen DeepJSCC、同 AWGN、同码率的机制级对照 |
| DiffJSCC | `mingyuyng/DiffJSCC` 已公开 inference/training/model/config，并在 Hugging Face 发布 OpenImage/CelebA 四个完整 checkpoint；源码仓库未发现 license，权重仓库标 Apache-2.0 | Stable Diffusion、多模态 feature、不同训练数据/分辨率；作者会把短边放大到至少 512，论文/模型名 CBR 不能直接照搬到 256 源 | 第三项：作者 C16 checkpoint 已进入 S30 source/checkpoint/rate audit 与共同总体对比 |
| DiT-JSCC | 论文 v2 指向 `semcomm/DiTJSCC`；截至核验日仓库根目录只有 README、metadata 约 5 KiB | 论文强调 semantic/detail 双支路和 semantic consistency，但当前无法精确运行 | 第四项 watch-only；等待真实代码，不做替代实现冒充作者方法 |

统一 common contract 冻结为 COCO-256、AWGN、SNR `[1,4,7,13,19]`、总 CBR `1/6`、同图同 channel realization；必须同时报告 PSNR/MS-SSIM/LPIPS、`T_cls` clean-correct final failure/new-error/repair、image-cluster tail upper bound、总实符号数、side-information 符号数、运行时间与显存。作者原生复现只能证明 paper/code fidelity，只有 common contract 可用于直接优劣结论。

该核验加强了而不是削弱了本项目的 diffusion 路线：外部生成式方法主要证明感知先验的上限，本项目需要用 matched-total-rate 与 refinement-induced new error/tail risk 判断收益是否真实。执行契约与详细报告见 `configs/external_baseline_comparison_contract.yaml`、`reports/external_method_comparison_schedule_2026-07-14.md`。

Primary sources：

- SGD-JSCC：https://arxiv.org/abs/2501.01138；https://github.com/MauroZMJ/SGDJSCC；https://huggingface.co/murjun/SGDJSCC/tree/main
- SING：https://arxiv.org/abs/2503.12484
- DiffJSCC：https://arxiv.org/abs/2404.17736；https://github.com/mingyuyng/DiffJSCC
- DiT-JSCC：https://arxiv.org/abs/2601.03112；https://github.com/semcomm/DiTJSCC

## 撞车风险

- 可能已有论文把 diffusion 作为 JSCC decoder；本项目必须用 semantic drift control 和 SNR-aware refinement 区分贡献。
- 可能已有论文报告 diffusion-enhanced JSCC 的感知指标；本项目必须额外评估语义可靠性和失败模式。
- 可能已有论文使用 semantic guidance；本项目需要明确其 guidance 是否 channel-adaptive，以及是否显式测量 drift。
- 可能已有论文做 channel-adaptive JSCC；本项目需要强调 adaptive diffusion refinement，而不是只做 adaptive encoder/decoder。
- `SGD-JSCC` 已经接近“semantic-guided diffusion + channel adaptation”，本项目需要把 novelty 收紧为：对 diffusion refinement 强度进行 SNR-aware 控制，并以 semantic drift/failure 为显式优化或选择准则。
- `DiT-JSCC` 明确指出 generative decoder 可能存在 semantic consistency 问题；本项目不能只重复这个观察，必须给出可执行的 drift metric 和控制策略。
- `JSCGC` 已经把 semantic inconsistency 当作 generative communication 的核心错误形态；本项目应把它作为理论背景，而不是回避。
- `RDPC/JSCM` 已覆盖 rate-distortion-perception-classification 四元权衡，不能把一般四轴 tradeoff 曲线作为首次贡献。
- `TOAST` 已覆盖信道状态驱动的 reconstruction/classification 权衡并引入 diffusion denoiser；本项目必须进一步限定到 refinement-induced new error、tail risk、matched-rate side information 和显式 failure handling。
- `SING` 已覆盖图像 DeepJSCC 的两阶段 diffusion inverse restoration；`RD-JSCC` 已覆盖 MIMO CSI 的短链 residual diffusion 与 channel-conditioned switching；因此“短链 residual diffusion + adaptive switch”不能单独作为本项目贡献。
- 已有 latency-aware generative SemCom 显式传输 prompt/edge 并计入重传、调制、功率和时延；本项目只能强调统一 matched-total-rate 对照和风险端点，不能宽泛声称首次考虑语义侧信息通信成本。

## 待检索关键词

- "DiffJSCC"
- "SGD-JSCC"
- "DiT-JSCC"
- "JSCGC"
- "DeepJSCC diffusion decoder semantic"
- "diffusion model joint source channel coding image"
- "generative JSCC hallucination"
- "semantic drift diffusion image restoration"
- "hallucination generative image compression"
- "SNR adaptive DeepJSCC"
- "bandwidth adaptive DeepJSCC"
- "prompt based JSCC"
- "semantic consistency guidance diffusion restoration"

### SING 全文机制复核与本项目实现边界：2026-07-15

本轮进一步读取 SING arXiv `2503.12484` 全文，而不只使用摘要。论文把 frozen DeepJSCC 全链写成随机非线性 degradation `y=A(x)`；SING-Zero 用一个可处理的近似线性 `A` 替代该链，并在 DDNM reverse diffusion 的每一步把 denoised estimate 投影为：

`x_hat_0,t = x_0,t - A†(A x_0,t - x_hat_JSCC)`，

等价地可解释为：

`x_hat = A† x_hat_JSCC + (I-A†A)x_0,t`。

前一项保留与 observed DeepJSCC reconstruction 一致的 range-space 成分，后一项让 diffusion prior 补全 null-space。论文示例近似 operator 包含 decolorization 或 mean-pooling downsampling，对应 pseudoinverse 可用上采样；SING-INN 则学习更复杂的非线性 degradation。论文实验使用同一个 pretrained DeepJSCC 做公平底座，CelebA-HQ 512、SNR `[-5,-3,-1,1,3,5]`、BCR `[0.0013,0.0052]` 和 `T=1000`，未报告本项目定义的 refinement-induced hard new-error endpoint。

`ANALYSIS-EXT-COMMON-PILOT-001` 中的 `SING-Zero-style` 明确不是上述算法的精确复现：它用同一个 frozen project DeepJSCC 和本地 SD1.5，但只在 25-step img2img 完成后执行一次 `A†y+(I-A†A)x`，`A` 固定为 2×2 mean pool、`A†` 为 nearest repeat。该实现可检查 range/null 机制，却缺少每个 reverse step 的 DDNM projection，也没有 SING-INN degradation estimator。因此其负结果只能说明 **final-only projection 不足**，不能用来否定或排名 SING 论文。

本轮对项目最直接的启发是：下一版 SING 对照应优先实现逐 reverse-step measurement projection，而不是继续扫描 final-only img2img strength；同时仍需保留 `T_cls` new-error，因为 measurement consistency 接近零并不自动保证语义不漂移。Primary source：https://arxiv.org/abs/2503.12484 。

### DiffJSCC 公开资产与共同对比口径复核：2026-07-21

作者仓库 `mingyuyng/DiffJSCC` 目前已经不再是 README-only：固定提交 `13aeb62451b872ce41ceba132c9c30a9ca172c53` 含训练、推理、ControlLDM、DeepJSCC 和配置代码。作者另在 Hugging Face 发布四个 DiffJSCC checkpoint；S30 选择 OpenImage `C_channel=16` / 模型名 CBR `1/96` 的权重，因为它是公开 OpenImage 两个工作点中最接近项目 19,712-real 预算的一项。该 checkpoint 为 `9,859,655,693` bytes，Hub LFS SHA-256 为 `ae1e6df0...2d399f94cc171d8d0ce44f851d96cb032bd7dec579`，权重 card 标 Apache-2.0；源码仓库根目录仍未发现 license，二者不能混写。源码的 `on_save_checkpoint` 会主动删除 `blip_model.*`，所以精确推理还必须加载代码硬编码的 `Salesforce/blip2-opt-2.7b` base 权重，不能把 9.86 GB 文件误称为完全自包含模型。

代码级审计暴露了一个必须分层报告的 CBR 口径。作者 `inference_cldm.py` 会把任何短边小于 512 的输入先双三次放大到 512，并 pad 到 64 倍数。C16、四次 2× 下采样在 512×512 网格上产生 `16×32×32=16,384` 个实坐标，即 8,192 次复信道使用：相对作者 512 源是 CBR `1/96`，但相对本项目原始 256×256 源是 `8192/(3×256×256)=1/24`。项目当前方法为 9,856 次复使用，所以 DiffJSCC 只用预算的 `83.1169%`、少 3,328 个实坐标。它可以进入“同一预算上限内”表，但不能称 exact-rate matched。

DiffJSCC 与 SGD-JSCC 的 side-information 边界不同：DiffJSCC 的 BLIP2 caption 由接收端从带噪 DeepJSCC 初始重建生成，不需要发送端文本信道，因而 caption 记 0 个传输符号；空间条件同样来自该初始重建的 condition encoder。公开配置的训练 SNR 是 `[0,14] dB`，项目 19 dB 点属于外推，必须单列。S30 冻结 100-step、wavelet color fix、无 intermediate MSE guidance，并使用 S20/S28 的同图、同 `external-common-v1` 噪声向量前缀；先比较 DiffJSCC 相对其作者 JSCC 初始重建，再比较项目 current/B1，避免只按 LPIPS 或 PSNR 单轴选赢家。

S30 完整 960 行复现随后给出两个方向性结论。第一，作者 DeepJSCC 前端自身是项目必须正视的强 baseline：它只用 16,384 real，却相对 current 高 `1.76246 dB`、LPIPS 低 `0.02374`，说明继续在弱 backbone 上堆轻量 controller 不是最优优先级。第二，固定 DiffJSCC 相对该前端平均以 `-2.38774 dB` 换取 `-0.02812 LPIPS`，总 failure `22→23`、new/repair=`10/9`；但事件随 SNR 有清晰转折：1/4 dB 为 `1/3、2/4` 净修复，7 dB 为 `3/2`，13 dB 为 `3/0` 净风险。这个结果支持把论文问题表述为“强保真端点与生成感知端点之间的可校准风险控制”，而不是“首次把 diffusion 用进 JSCC”或“好信道机械关闭 diffusion”。完整实验边界见 `reports/diffjscc_external_comparison_stage_result_2026-07-21.md`。

截至同日复核，`semcomm/DiTJSCC` 官方仓库仍只有 README，说明代码将在最终接收后发布，不能做精确复现或用自制近似冒充。Primary sources：https://arxiv.org/abs/2404.17736 ；https://github.com/mingyuyng/DiffJSCC ；https://huggingface.co/Mingyuyang/DiffJSCC-OpenImage-CBR-1-96 ；https://github.com/semcomm/DiTJSCC 。

### SGD-JSCC 全文码率与文本假设复核：2026-07-15

进一步读取 SGD-JSCC arXiv `2501.01138` 全文后确认：论文所有对比使用标称 CBR `R=1/20`；正文给出的 edge map 与 JSCC feature 开销分别为 `1/24` 和 `1/120`。发布代码在 256×256 四 patch 路径下实测 main/active-edge 共 `16,384+3,328=19,712` 个实坐标，即 `9,856/(3×256×256)=0.0501302`，与论文标称 `1/20` 一致。

更关键的是，论文明确把 text transmission cost 视为可忽略并假设文本无误传输，极低 SNR 下的文本传输留作未来工作。因此，发布权重按论文协议与本项目做作者工作点对照时，必须标成“免费且无误文本的论文协议上界”；若把 caption 经真实信道并计入总预算，则已不再是论文原协议。项目 CBR `1/6` 的 common adapter 也不能把剩余预算用 padding 或过强文本重复后称为“宽码率 SGD-JSCC”，除非重新训练更宽的图像 latent。

论文自身也报告 SGD-JSCC 的 PSNR 仍与 state-of-the-art conventional JSCC 有约 `2.5 dB` 差距，其主优势集中在低 SNR 的感知、FID/CLIP 和生成语义质量，而不是纯失真最优。该事实与本仓库作者工作点首轮结果的方向一致，但本仓库 8 图 pilot 不能替代论文大样本结论。Primary source：https://arxiv.org/abs/2501.01138 。

### SGD-JSCC step matching 的代码级机制与迁移边界：2026-07-15

对固定作者源码 `2188acc0dd2805355d3d0d2e478cbc27b46b4da5` 的进一步核验确认，SGD-JSCC 的核心不只是“按 SNR 调整 diffusion strength”，而是先把归一化 AWGN 观测改写成 diffusion forward marginal 的形式。若清洁潜变量每实维功率为 1，作者口径的线性 SNR 为 `gamma`，则经确定性总方差归一化后：

`y_bar = sqrt(gamma/(gamma+1)) * z0 + sqrt(1/(gamma+1)) * epsilon`。

它与 `q(z_t|z0)=sqrt(alpha_bar_t)z0+sqrt(1-alpha_bar_t)epsilon` 在 `alpha_bar_t=gamma/(gamma+1)` 时同形。发布代码于 `inference_one.py` 明确计算 `signal_scale=gamma/(gamma+1)`，离散 step matching 选择 `argmin_t |alphas_cumprod[t]-signal_scale|`，再从该时刻构造反向轨迹。无真实 CSI 时，则用 `snr_prediction_net` 从归一化接收潜变量估计 signal scale。文本 prompt 提供全局语义条件，edge-JSCC 重建的 soft edge 通过 ControlNet 约束空间结构。

“严格对应”需要保留两个限定。第一，这是归一化 AWGN、独立高斯噪声和一致 SNR/方差定义下的边缘分布等价，不意味着某次信道噪声是某条已实现 diffusion Markov 路径上的唯一噪声。第二，作者代码实际使用逐样本 L2 球面归一化，再用最近的离散 scheduler step；因而对有限维样本更准确的说法是“在理想模型下同形，代码中作球面/离散近似匹配”。

另一个关键是，发送端用的就是同一 VAE/diffusion 系统的潜变量，接收潜变量可以直接作为 diffusion 初态。这与本项目当前“DeepJSCC 专用信道 latent → B1 图像 → image-space residual bridge”不同；仅把 SNR 输入现有 refiner 不会自动建立 channel state 与 diffusion state 的对齐。这为 `EXP-S16-DIFF-001` 在五个 SNR 全部过修复提供了一个更直接的机制解释：该负结果否定当前 residual bridge，不否定物理匹配的 latent diffusion。

迁移到本项目时不能直接照搬 `gamma/(gamma+1)`。本项目定义 `gamma=Es/N0` 每复信道使用，每个实坐标噪声方差为 `P/(2*gamma)`。若对实值 diffusion latent 逐坐标匹配，应为 `alpha_bar_channel=2*gamma/(2*gamma+1)`；只有把 SNR 重定义为每实维 SNR，或构造成对的复值 diffusion 过程时，才会回到作者公式。后续值得预注册的方法不是扫描更多 image-space diffusion strength，而是在 exact-rate JSCC latent 上训练 channel-state-matched score/denoiser，从匹配 `t*` 反演，在活动坐标加入逐步 measurement consistency，并继续使用预算内语义载荷和 hard new-error/tail-risk 门槛。文本和结构条件只能缩小反演解空间，不保证条件本身真实；本仓库已观察到 caption CRC 全通过但 sender BLIP2 仍可将 dog 写成 cat，因此 semantic reliability 不能被 packet reliability 替代。
