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
| JSCGC | 2026 | 将接收信号作为 generative sampling 条件，提出 Joint Source-Channel-Generation Coding，并讨论 semantic inconsistency 错误行为 | 强相关；提醒本项目必须认真定义 semantic inconsistency / drift，而不是只看 LPIPS/FID | https://arxiv.org/abs/2606.12858 |

### Channel-adaptive JSCC

| 工作 | 年份 | 重点 | 与本项目关系 | 来源 |
|---|---:|---|---|---|
| DeepJSCC 原始工作 | 2018/2019 | CNN encoder/decoder + AWGN/Rayleigh 信道层，低 SNR 下相对数字方案有 graceful degradation | baseline 理论起点；第一阶段需要复现或接入类似 DeepJSCC | https://arxiv.org/abs/1809.01733 |
| Dynamic_JSCC | 2021/2022 | 单模型自适应 rate control，按信道条件和图像内容动态控制带宽；公开 PyTorch 代码 | 可作为 channel-adaptive 策略参考，不建议作为第一版主 baseline，因为它引入 rate policy 复杂度 | https://arxiv.org/abs/2110.04456 ; https://github.com/mingyuyng/Dynamic_JSCC |
| DeepJSCC-l++ | 2023 | Swin Transformer backbone，单模型适应多个 bandwidth ratio 和 SNR，代码公开 | channel-adaptive JSCC 重要撞车线；可作为后续对照或方法参照 | https://arxiv.org/abs/2305.13161 ; https://github.com/aprilbian/deepjscc-lplusplus |
| PJSCC | 2024 | Channel State Prompt 生成 learnable prompt，融合 SNR 和信道分布信息，实现跨信道自适应 | prompt-based/channel-adaptive JSCC 撞车线；本项目可借鉴 condition 设计，但要落在 diffusion strength / semantic guidance | https://arxiv.org/abs/2411.10178 |

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

## 撞车风险

- 可能已有论文把 diffusion 作为 JSCC decoder；本项目必须用 semantic drift control 和 SNR-aware refinement 区分贡献。
- 可能已有论文报告 diffusion-enhanced JSCC 的感知指标；本项目必须额外评估语义可靠性和失败模式。
- 可能已有论文使用 semantic guidance；本项目需要明确其 guidance 是否 channel-adaptive，以及是否显式测量 drift。
- 可能已有论文做 channel-adaptive JSCC；本项目需要强调 adaptive diffusion refinement，而不是只做 adaptive encoder/decoder。
- `SGD-JSCC` 已经接近“semantic-guided diffusion + channel adaptation”，本项目需要把 novelty 收紧为：对 diffusion refinement 强度进行 SNR-aware 控制，并以 semantic drift/failure 为显式优化或选择准则。
- `DiT-JSCC` 明确指出 generative decoder 可能存在 semantic consistency 问题；本项目不能只重复这个观察，必须给出可执行的 drift metric 和控制策略。
- `JSCGC` 已经把 semantic inconsistency 当作 generative communication 的核心错误形态；本项目应把它作为理论背景，而不是回避。

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
