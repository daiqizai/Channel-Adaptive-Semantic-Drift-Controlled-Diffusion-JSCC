# Kodak / CLIC / DIV2K 数据与 benchmark 对齐计划

日期：2026-07-23
状态：**Gate A0 已完成；A1方法推理未授权，未训练、未作方法排名。**

## 0. Gate A0 完成记录

专属工作区已按用户要求建立在 `paper_idea1b/`，共享 `src/` 只通过import复用；S33 checkpoint、canonical noise、S34D结果和其他旧SHA资产均保持原位。正式输出为 `paper_idea1b/outputs/GATE-A0-BENCHMARK-SETUP-001/`，状态是 `complete` 且 `a1_authorized=false`。

- 数据：Kodak 24张 + CLIC2020 test 428张，共452张，全部RGB、无内容SHA重复。CLIC Mobile/Professional官方包SHA分别为 `2025f07a...aa732`、`857df244...52884`。Kodak官方逐图CDN对本服务器返回403，故使用公开mirror，但24个成员逐个通过官方页面公布的字节数核验，archive SHA=`44e2569b...00223`。全部下载均清空proxy并走服务器直连。
- manifests：452条source、2,260条method-rate、20,018条S33/Swin tile、882,675条SGD released `split_image_v2` patch。Kodak上S33/Swin实际CBR=`1/24`；DiffJSCC原生整图公式CBR=`1/96`，因此A1中只能标under-budget而非exact-rate，并必须用runtime instrumentation复核；SGD sender caption未定价，仍non-ranking。
- 指标：DISTS、clean-fid FID/KID已接入。Kodak恒等输入得到PSNR=`120 dB`、MS-SSIM=`1`、LPIPS=`0`、DISTS=`5.96e-8–1.19e-7`；完整CLIC自比FID=`−4.5057e-5`、KID=`−0.00205318`，均通过冻结数值容差。
- 失败记录：首次把共享PSNR的恒等上限误要求为∞，且把clean-fid self-FID浮点残差阈值设为`1e-5`，该失败已单独保留；按实现事实改为PSNR≥119.999 dB、`|FID|≤1e-4`后重跑通过。没有修改数据或指标计算。

## 1. 结论先行

建议采用两阶段方案：

1. **先做 A：保留冻结的 COCO-S33，只把主评测迁移到 Kodak + CLIC2020 test。**
   这一步最快、不会抹掉现有训练资产，也能直接回答 S33 是否能跨数据集泛化。Kodak 作为 24 张原生高分辨率重建 benchmark；CLIC2020 test 作为 428 张大图的分布感知 benchmark。Imagenette 不再承担主画质 benchmark，只保留 semantic reliability 的独立 policy-dev / 最终封存职责。
2. **A 的结果成立后，再决定是否做 B：在 DIV2K train 上从随机初始化重训 S33。**
   B 是“完全贴近压缩/JSCC领域训练习惯”的更强口径，但它不是简单换数据：新的 S33 会连带要求 SwinJSCC 公平臂重训、P1 refiner 重新预注册并重训。它应当是一条新实验分支，不能覆盖当前 COCO-S33。

大型测试集优先选 **CLIC2020 test，而不是 DIV2K val**。这样未来若选择 B，DIV2K train 用于训练、CLIC test 用于最终 benchmark，训练和主评测不存在同源划分或 checkpoint selection 污染。DIV2K val 可只作训练期选择集或补充测试，但不能同时承担这两个角色。

## 2. 为什么这样对齐

- Kodak 官方集合包含 24 张无损 PNG，每张为 `768×512` 或 `512×768`；它适合报告逐图 PSNR、MS-SSIM、LPIPS、DISTS 和展示重建图，但样本太少，不适合给 FID/KID 排名。[Kodak 官方页面](https://r0k.us/graphics/kodak/)
- DIV2K 官方集合含 800 张 train、100 张 validation、100 张 test 的 2K 图像。官方可下载的 train HR 压缩包约 `3.29 GiB`，validation HR 约 `428 MiB`。[DIV2K 官方页面](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- TensorFlow Datasets 的 CLIC 页面记录 CLIC2020 lossy split 为 1,633 train、102 validation、428 test，总体约 `7.48 GiB`；本计划只取官方 test 的 mobile + professional 两个压缩包，HEAD 审计合计约 `1.50 GiB`。该数据还包含灰度图，适配时必须统一转成三通道 RGB 并记录转换。[CLIC2020 数据说明](https://www.tensorflow.org/datasets/catalog/clic)
- SwinJSCC 本身就是在 DIV2K 上用随机 `256×256` patch 训练、在 Kodak 和 CLIC 高分辨率集合上评测。因此“DIV2K patch train + Kodak/CLIC test”是最贴近强 Transformer JSCC 对照的口径；“COCO train + Kodak/CLIC test”则是可接受但必须明确标为跨数据集泛化的口径。[SwinJSCC 论文](https://arxiv.org/abs/2308.09361)

## 3. 数据获取与适配工作量

| 数据 | 用途 | 下载规模 | 获取与完整性 | harness 适配 | 预计人工工作 |
|---|---|---:|---|---|---:|
| Kodak | 主重建 benchmark | `14.68 MiB`，24 PNG | 官方逐图下载；固定文件名、尺寸、SHA-256 manifest | 原图不 resize；Kodak 尺寸能被 256 整除，可直接切成无重叠 tile | `0.25–0.5 天` |
| CLIC2020 test | FID/KID 与大图 benchmark | 约 `1.50 GiB`，428 图 | 官方 mobile/professional 两包；解压后做去重、RGB/灰度审计、SHA-256 manifest | 尺寸不统一；各方法使用冻结原生整图/tile/patch路径并记录每图实际码率 | `0.5–1 天` |
| DIV2K valid HR | CLIC 不可用时的备选大集；或 B 的训练选择集 | 约 `428 MiB`，100 图 | 官方单包、manifest | 同 CLIC 的原生尺寸适配 | `0.25–0.5 天` |
| DIV2K train HR | 仅方案 B 训练 | 约 `3.29 GiB`，800 图 | 官方单包、manifest；训练期在线随机 crop | 新 dataset/sampler；随机 `256×256` crop，不预生成海量 patch | `0.5–1 天` |

本轮所有大包均按仓库规则显式清空代理环境变量并使用服务器直连；具体来源、大小与SHA已写入A0 manifest。

## 4. 原生高分辨率与通信码率公平合同（2026-07-23 用户修订）

### 4.1 不强制所有方法统一 tile

共同 tile 会切断生成模型的全局上下文，并可能人为降低 DiffJSCC/SGD 的感知质量。正式主表因此允许每种方法使用**冻结的原生最佳可运行处理路径**，但处理规则必须在看到结果前固定，不能逐图选择：

- S33 / 当前 Swin adapter：允许按固定训练尺寸做 tile；重叠、边界 padding 和重复发送全部计入通信账本。
- DiffJSCC自带的author-JSCC臂跟随同一官方整图auto-resize/pad入口，不人为改成S33的256 tile。
- DiffJSCC：优先使用作者整图入口，即短边小于512时放大到512、再补到64的倍数，并在整幅内部网格上生成；只有整图实测 OOM 时才允许另立、明确标成 fallback 的作者 tiled sampling 臂，不能静默切换。
- SGD-JSCC：作者发布入口原生使用 `split_image_v2` 的128 patch和重叠融合。为保持 checkpoint/代码真实性，主 paper-upper 保留该路径；不能为了满足“整图”字面要求而擅自改成未验证的全图网络。所有重叠 patch 的 main/edge/caption 开销逐次计费。
- 后续轻量 refiner：其输入处理独立记录；receiver-only refinement 不增加通信符号，但计算量必须单列。

画质指标一律在 benchmark 原始像素网格上计算。方法内部 resize/pad 可以不同，但必须保存处理 manifest，并把输出按该方法冻结的官方逆变换恢复到原尺寸。

### 4.2 只统一通信账本，不把计算张量误算成通信

每张图必须记录：

- 原始 `H×W` 和 `3HW` 个 source real dimensions；
- main image、edge、caption、其他 sender-side side information 分别实际发送多少 real symbols/bits；
- padding、重叠 tile、重复发送产生的额外符号；
- `total complex channel uses` 和 `CBR_actual = total_complex_uses / (3HW)`；
- 接收端本地生成的 BLIP/CLIP/VAE/diffusion tensor 记0通信符号，但另计参数、FLOPs、显存和延迟。

S33 的预算锚仍为 `CBR=1/24`。只有 actual CBR 相同且 side information 全部入账的方法才能称 **exact-rate**；actual CBR 更低者只能称“在不超过 `1/24` 预算下”，不能用 padding或重发凑数；actual CBR 更高或 sender caption 未计费者只能分层/non-ranking 报告。

对 Kodak，S33 的无重叠256 tile恰好覆盖整图，每张发送：

`6 × 16,384 = 98,304 real = 49,152 complex uses`，

仍严格为 `CBR=1/24`。CLIC边界 tile的padding也按完整发送tile计费。DiffJSCC/SGD不套用该tile公式，而由它们各自实际channel latent和side-information路径 instrumentation 记账。

## 5. 指标合同

### 5.1 Kodak

只计算全参考、逐图指标：

- PSNR；
- MS-SSIM；
- LPIPS；
- DISTS。

Kodak 不计算 FID/KID，也不把 24 张重复多 seed 后冒充更大的独立图像集合。

### 5.2 CLIC2020 test

计算：

- PSNR、MS-SSIM、LPIPS、DISTS；
- 每个 SNR、每个 channel seed 独立计算 FID 和 KID。

FID/KID 的真实样本数始终是 428，不能把同一原图的 3 个噪声 seed 当成 1,284 个独立 source。聚合时报告各 seed 结果及均值/范围。由于 428 对 FID 仍偏小，**KID 作为主要分布感知指标，FID 作为必须报告但带小样本提示的次要指标**。

### 5.3 实现冻结

- DISTS 使用作者推荐的 PyTorch 实现，输入统一为 RGB `[0,1]`；DISTS 是全参考图像质量指标，强调结构/纹理相似性。[DISTS 官方实现](https://github.com/dingkeyan93/DISTS)
- FID/KID 固定一个实现，不混用不同预处理。推荐使用当前环境已有的 `clean-fid`，冻结版本、Inception 权重 hash、RGB/uint8 PNG 落盘方式和 `clean` resize 模式；其文档明确支持 folder-to-folder FID/KID，且预处理会显著改变数值。[clean-fid 官方实现](https://github.com/GaParmar/clean-fid)
- A0已接入TorchMetrics DISTS与`clean-fid==0.1.35`。VGG16权重SHA=`397923af...a5bf0`，clean-fid Inception权重SHA=`f58cb9b6...8f4`；两者存放在`paper_idea1b/data/metric_weights/`，没有复制旧模型资产。
- 指标接入必须先通过 identity sanity：原图对原图时 PSNR 为无穷或上限、LPIPS/DISTS 接近 0、FID/KID 接近 0；还要用一组冻结小图交叉检查新旧 PSNR/MS-SSIM/LPIPS 实现一致。

### 5.4 semantic reliability 不因换 benchmark 消失

Kodak/CLIC 没有适用于当前 `T_cls` 的 Imagenette 类别标签，因此不能把它们强行当成有监督 semantic-failure 测试集。分层口径为：

- Kodak/CLIC：主画质与感知 benchmark；可补冻结的 source-vs-reconstruction CLIP / 通用分类器一致性，只称辅助无标签语义审计；
- 当前 64 图 Imagenette policy-dev：继续承担方法开发期的有监督 reliability 检查；
- official Imagenette validation：继续封存，方法和所有阈值冻结后才一次性解锁。

## 6. 训练方案 A 与 B

### A：冻结 COCO-S33，只换主评测集

**做什么**

- 不改、不续训 S33 SHA `2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`。
- 接入 Kodak + CLIC2020 test 和新指标。
- 所有外部方法使用各自冻结的原生处理路径，在同 source/SNR/channel-seed 上重新推理；只统一实际通信码率账本。

**成本**

- 数据、method-native processing/rate ledger、指标基础设施已由A0完成。
- S33/Swin/author-JSCC 判别式推理通常可在数小时内完成。
- DiffJSCC 在428张大图、5 SNR上优先执行官方整图diffusion；总成本不能再用统一tile数外推。A1获授权后必须先用少量原生大图测wall time/峰值显存，再估算全量工期；这不是A0已测结果。

**对现有数据的影响**

- 当前 COCO 训练结果、S33 checkpoint、Swin equal-budget checkpoint 全部保留有效。
- 当前 960 行 Imagenette policy-dev 仍是内部 reliability 证据，但不再冒充领域主画质 benchmark。
- A 能形成“COCO 训练、Kodak/CLIC 跨数据集泛化”的主表；论文必须如实写训练数据不同于 SwinJSCC 官方的 DIV2K。

**优点/风险**

- 优点：最快，不重新消耗训练预算，不污染已冻结 S33。
- 风险：审稿人仍可能问“若在 DIV2K 同域训练是否更强”；因此 A 是先验最低成本 gate，不自动排除 B。

### B：DIV2K 重训一条新 S33 分支

**做什么**

- 保留 COCO-S33，不覆盖目录；从随机初始化训练 `S33-DIV2K`。
- 训练使用 DIV2K train 的随机 `256×256` crops，仍为 FP32、离散五档 SNR、paired-real AWGN、exact `16,384 real`。
- 不机械照抄“12 epochs”：800 张 DIV2K 下 12 个字面 epoch 只有约 300 optimizer steps，远少于现有 S33 的 `44,364` steps。公平训练预算应冻结为 `44,364 optimizer steps / 约 1.42M crops`，并另报充分收敛曲线。
- 若 CLIC test 是主 benchmark，可从 DIV2K train 内部固定少量 image IDs 作 checkpoint selection；若 DIV2K val 被用作主 benchmark，就不得再用它选 checkpoint。

**成本**

- S33-DIV2K 本身：预计 `4–6 GPU 小时`，外加约半天做 sampler、日志和曲线核验。
- 若要保留“与 Swin 同数据同预算”的严格结论，Base-SA 和 CM-SA 两臂也必须在 DIV2K 重训：按现有速度预计合计 `17–22 GPU 小时`。
- P1 refiner 必须改为基于新 S33-DIV2K 的 matched distribution 重新预注册并重训：初估另需 `10–24 GPU 小时`，仍须先 smoke。
- 含工程、训练、完整大图评测，单张 4090D 的严格 B 分支约 `2–4 天`；DiffJSCC 大图推理可能再占 `1–2 天`。

**对现有数据的影响**

- 现有所有 COCO-S33 结果仍保留为历史/跨数据集分支，但不能用来替代新 S33-DIV2K 的结论。
- S33-vs-Swin 的严格同训练分布结论必须重做；只把旧 Swin checkpoint 拿来测新 S33-DIV2K，最多叫 cross-training comparison。
- 当前 P1 预注册绑定 COCO-S33，B 下必须明确 supersede，不能沿用旧 refiner 训练或 threshold。

## 7. 哪些既有对比需要重跑

| 既有资产 | 方案 A：COCO-S33 + 新 benchmark | 方案 B：DIV2K-S33 + 新 benchmark |
|---|---|---|
| S33 本身 | **不重训；必须在 Kodak/CLIC 重推理、补 DISTS/FID/KID** | **必须新训练并在新 benchmark 全量推理** |
| DiffJSCC 现有 960 行 | 旧结果保留为 Imagenette policy-dev；**Kodak/CLIC 必须重新生成**，不能由旧 960 行外推 | 同左；若还保留旧 Imagenette reliability 表，只需在相同旧 keys 上补新 S33，不必为“新训练分布”伪造 DiffJSCC 重训 |
| Swin Base-SA / CM-SA | **checkpoint 不重训；Kodak/CLIC 必须重推理**，使用冻结的Swin原生处理与共同实际码率账本 | 若主张同数据公平胜负，**两臂必须在 DIV2K 重训并重新评测** |
| author-JSCC | 若继续放主表，checkpoint 不重训但要在新 benchmark 重推理 | 至少重推理；训练分布不匹配时只能分层报告 |
| S34D 256×256 代价刻画 | **核心延迟、参数量、FLOPs 倍率不需重跑**；应补一组 native benchmark 的每百万像素延迟、峰值显存、tile 数 | 架构不变，核心参数/FLOPs不重跑；新权重只做延迟 spot-check，并补同一高分辨率系统吞吐 |
| SGD paper-upper | 仍非 ranking；P0/S34D 成本不需重跑。若进新 benchmark，只作有合同缺口的视觉/成本参考 | 同左；没有官方训练代码，不能因 B 而把它包装成 DIV2K 公平重训 |
| P1 refiner | 尚未训练；在训练前修订评测章节，训练仍匹配 COCO-S33；冻结后增加 Kodak/CLIC 外部泛化 gate | 当前 P1 预注册失效，必须基于 S33-DIV2K 重新预注册、smoke、训练 |

换句话说：**A 主要重跑“推理与指标”，B 还要重跑“模型训练与公平基线”。**

## 8. 建议的分阶段放行

### Gate A0：只做基础设施

- 下载/校验 Kodak、CLIC2020 test；
- 固定 source manifest、各方法原生处理 manifest 与统一实际码率 schema；
- 接入 DISTS、FID、KID，完成 identity sanity；
- 不运行任何方法推理。

状态：**已完成**。这一步没有做正式排名，也没有自动解锁A1。

### Gate A1：先跑判别式主表

- 冻结 S33、Swin Base/CM、必要时 author-JSCC；
- Kodak：5 SNR × 3 seed；
- CLIC：先 5 SNR × 1 seed，确认成本和指标稳定，再由用户决定是否扩到 3 seed。

预计 `0.5–1 天`。

### Gate A2：生成式对照

- DiffJSCC 先用已经被 S34D 证明的最低保感知步数作主成本点，同时保留论文/既有 100-step 口径作上界；
- 先 CLIC 1 seed，结果和时间报用户后再决定 3 seed；
- SGD 继续是 non-ranking paper upper，不把 side-information 缺口写成公平胜负。

预计 `1–2 GPU 天`，取决于 CLIC 实际 tile 数。

### Gate B：是否切训练分布

只有看完 A 的跨数据集结果后再决定。建议触发 B 的条件是：

- S33 在 Kodak/CLIC 相对当前主要基线明显失去优势，且失效看起来是训练域问题；或
- 目标 venue/审稿意见明确要求 DIV2K 同训练分布；或
- P1 在 COCO 上通过，但在 Kodak/CLIC 完全不泛化，需要用领域训练数据验证。

## 9. 推荐决策

现在不应直接丢弃 COCO-S33 去重训。最稳妥的顺序是：

> **先做 A，把 Kodak + CLIC2020 test 升为领域主 benchmark；保留 Imagenette 只做 reliability；A 通过后，论文可先按跨数据集泛化口径推进。只有 A 暴露明显域差距，或投稿目标强制同域训练，再开 B 的 DIV2K 全套重训。**

这同时保住了已有 S33/Swin 资产，也避免未来 DIV2K 训练和 DIV2K val 测试互相污染。
