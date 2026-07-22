# 外部方法对比排期与 SGD-JSCC 源码审计（2026-07-14）

> 2026-07-15 状态更新：本报告中的“未下载/未运行/码率未闭合”描述是 7 月 14 日最初审计时点的事实，现已被后续阶段推进取代。作者原生 smoke、复信道口径 common smoke，以及当前 M3/SGD common adapter/SING-Zero-style 的 8 图×5 SNR 对比均已完成；120 行共同协议校验通过。最新结果见 `reports/external_common_comparison_pilot_stage_result_2026-07-15.md`。

## 结论先行

外部方法对比已经正式进入项目日程，顺序冻结为：

1. **SGD-JSCC 作者代码**：先做原生复现，再解决总码率后进入统一协议；
2. **SING-Zero-style**：使用本项目同一个 frozen DeepJSCC 做机制级公平对照，明确标注不是论文精确复现；
3. **DiffJSCC 作者代码**：前两项闭环后再接，避免同时引入第二套大型生成系统；
4. **DiT-JSCC**：当前官方仓库只有 README，暂列 watch-only，不虚构可复现性。

本轮完成了文献/代码状态核验、SGD-JSCC 源码固定、统一公平协议和 fail-closed dry-run；**没有下载模型权重，没有运行外部模型，也没有产生外部方法实验结果**。因此当前仍不能回答“我们数值上是否强于 SGD-JSCC / DiffJSCC”。

## 为什么要分两张表

外部生成式 JSCC 的作者协议与本项目不完全相同。如果只把论文图上的 PSNR/LPIPS 数字抄在同一表里，会混入分辨率、数据集、码率、语义侧信息和信道随机性差异。

后续固定输出两类结果：

| 轨道 | 目的 | 是否允许直接说强于本项目 |
|---|---|---|
| 作者原生复现 | 检查作者代码和权重能否复现其自身设置 | 否 |
| common contract | 同一 source images、AWGN realization、SNR、总码率和指标 | 只有总码率完整可计且 semantic tail gate 通过后才允许 |

common contract 固定为 COCO-256、AWGN、SNR `[1,4,7,13,19]` dB、总 CBR `1/6`（65,536 个实坐标 = 32,768 个 complex channel uses），并同时报告：

- PSNR、MS-SSIM、LPIPS；
- `T_cls` clean-correct final failure、new error、repair；
- image-cluster 单侧 Clopper-Pearson upper bound；
- main/side-information/pilot 总符号数、运行时间和峰值显存。

视觉更好但新增语义错误或 tail risk 不过门槛，仍不得算主要提升。

## SGD-JSCC：第一复现对象的源码审计

### 已固定资产

- 作者仓库：`https://github.com/MauroZMJ/SGDJSCC`
- 本地只读路径：`third_party/SGDJSCC`
- 固定 commit：`2188acc0dd2805355d3d0d2e478cbc27b46b4da5`
- GitHub metadata 仓库大小约 1.8 MiB；落盘含 `.git` 约 4.1 MiB。
- 仓库未发现 LICENSE/COPYING，GitHub metadata 也没有可识别 license；因此不直接修改或复制其源码，只在本项目侧写 adapter。

作者代码确实提供 AWGN inference、128×128 VAE latent、文本 caption、edge JSCC、ControlNet/DiT diffusion 和 SNR prediction 路径；不是只有论文描述。

### 当前可复现性阻塞

1. 作者 Hugging Face checkpoint bundle 共约 **2.931 GB**：
   - `JSCC_model.pth` 571 MB；
   - `diffusion_backbone.pth` 938 MB；
   - `diffusion_controlnet.pth` 1.13 GB；
   - `muge-epoch-19-checkpoint.pth` 292 MB。
2. 推理脚本还会加载 `Salesforce/blip2-opt-2.7b-coco` 和 OpenAI CLIP ViT-L/14；额外下载量尚未冻结，不能把 2.931 GB 当成完整资产规模。
3. README 明确说 batch preprocessing script 尚未发布，training guideline 仍在 TODO；现有 `inference_config.py` 依赖作者预处理目录。
4. `configs/inference.yaml` 含作者机器绝对路径，必须由项目侧 adapter 重定向，不能原样运行。
5. `inference_one.py` 的 SNR loop 和逐 SNR control 参数是硬编码的；common SNR grid 需要 adapter，但不能偷偷把适配版标成“原样作者复现”。

### 公平性阻塞

论文把 BLIP2 text description 假设为完美传输，并忽略其通信成本；edge map 则通过独立 JSCC 支路传输。发布代码中主图 latent shape 是 `16×16×16=4096` 个实数，edge 分支还会依据 `canny_cr=0.2` 产生额外 active symbols。

这里同时存在“论文 complex-symbol BCR”和“本项目 real-symbol budget”的记账口径差异。所以下一步必须实际 hook transmitter 输出，分别记录：

- main latent 的实数/复数 channel uses；
- edge branch 的 active mask 和实际 channel uses；
- text payload 的 UTF-8/raw bits、选定调制/FEC 后符号数；
- CSI/pilot 是否另计。

在这些数没有闭合前，SGD-JSCC 只能给作者原生表，不能直接与我们的 CBR `1/6` 表排高低。

## 其他三项的当前状态

### SING

SING 把受损 DeepJSCC reconstruction 恢复写成 inverse problem，SING-Zero 使用近似线性退化，SING-INN 用条件 INN 描述更复杂退化；论文公平性上值得借鉴的一点是各方法共用同一个 pretrained DeepJSCC。

截至本轮 primary-source 核验，没有定位到作者可运行仓库。因此排入第二项的是 **SING-Zero-style mechanism baseline**：同一 frozen DeepJSCC、同一 received reconstruction、同一 AWGN 和码率，只实现论文启发的 inverse-restoration 机制；所有表格必须带 `style / not exact reproduction` 标签。

### DiffJSCC

论文官方链接指向 `mingyuyng/DiffJSCC`，仓库约 27 MiB，包含 inference、training、model 和 config 目录，但没有可识别 license。论文使用 Stable Diffusion、多模态条件和 SNR，并报告 Kodak 768×512、3072 symbols 的设置。

它是重要外部对照，但接入成本与分辨率/数据合同差异都高于 SGD-JSCC，所以放在第三顺位。前两项未闭环前不并行下载另一套大型权重。

### DiT-JSCC

论文 v2 明确把 semantic inconsistency 作为 diffusion GJSCC 问题，并提出 semantic/detail 双分支 encoder 与 coarse-to-fine DiT decoder。官方链接仓库 `semcomm/DiTJSCC` 当前 metadata 仅约 5 KiB，根目录只有 README；因此本轮判为不可运行，只监控代码发布，不做“近似复现冒充作者方法”。

## 冻结排期

| 顺序 | 里程碑 | 当前状态 | 阶段产物 |
|---:|---|---|---|
| 1 | EXT0 source/contract audit | **已完成** | 本报告、固定源码、统一契约、dry-run checker |
| 2 | EXT1 SGD-JSCC no-download adapter | **已完成** | import 隔离、输入 manifest、symbol counter、执行预检 |
| 3 | EXT2 SGD-JSCC native smoke | **已完成** | 作者原生输出；不作直接优劣结论 |
| 4 | EXT2B SGD-JSCC common one-image smoke | **已完成** | 四 patch、显式文本信道、active edge、精确总码率 |
| 5 | EXT3 SING-Zero-style common contract | **下一项** | 同 DeepJSCC / 同码率 / 同 AWGN 的机制对照 |
| 6 | EXT4 common-contract stage | 待执行 | 64 图×3 channel seeds 的质量、语义、系统表 |
| 7 | EXT5 full external audit | 方法冻结后 | 新 frozen population 上的一次性完整对比 |

EXT2 前的大下载将另行明确说明来源、完整预计体积、落盘目录和服务器直连命令；默认清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY` 及小写变量。

## 已落盘的机器约束

- 配置：`configs/external_baseline_comparison_contract.yaml`
- checker：`scripts/check_external_baseline_contract.py`
- 单测：`tests/test_external_baseline_contract.py`

执行：

```bash
python3 scripts/check_external_baseline_contract.py
python3 -m unittest discover -s tests -p 'test_external_baseline_contract.py' -v
```

当前 dry-run 状态为 `PASS`，并确认：官方 Imagenette validation 未访问、SGD-JSCC commit 匹配、资产下载授权/完成状态有记录、共同适配器 rate gate 已通过，外部优劣结果声明仍被禁止。

## 对当前项目方向的实际影响

这个对比轨道不会让项目放弃 diffusion，也不会把主线改成复刻别人。它会回答三个不同问题：

1. 我们的 posterior/data-consistency diffusion 相对作者完整生成系统处于什么位置；
2. 提升究竟来自 diffusion inverse-restoration 机制，还是来自额外 text/edge 语义预算；
3. 我们强调的 refinement-induced new error 与 tail risk，是否揭示了论文平均 PSNR/LPIPS/CLIP/FID 表没有覆盖的失败。

第一项阶段结论的合理目标不是立即“赢 SGD-JSCC”，而是先得到一份不会把免费文本、额外 edge channel 或不同数据集混成同一排名的可信对比表。

## Primary sources

- SGD-JSCC paper: https://arxiv.org/abs/2501.01138
- SGD-JSCC code: https://github.com/MauroZMJ/SGDJSCC
- SGD-JSCC checkpoints: https://huggingface.co/murjun/SGDJSCC/tree/main
- SING: https://arxiv.org/abs/2503.12484
- DiffJSCC: https://arxiv.org/abs/2404.17736
- DiffJSCC code: https://github.com/mingyuyng/DiffJSCC
- DiT-JSCC: https://arxiv.org/abs/2601.03112
- DiT-JSCC repository: https://github.com/semcomm/DiTJSCC
