# S34C-Lite：现有结果的码率透明度分析计划（2026-07-23）

## 状态

**可在 1–2 天内完成；当前只报计划，尚未执行聚合。** 长版 S34C 已在任何 smoke/训练前由用户暂停。本轻量版不训练、不做新模型推理、不下载、不访问 official Imagenette validation，只读取已经冻结的 S33、S30 DiffJSCC 与 S20/S28 SGD-JSCC 逐样本结果。

## 为什么可行

三种方法都已有相同冻结 population：64 张 Imagenette policy-dev 图、channel seeds `20260748/49/50`、SNR `[1,4,7,13,19] dB`，每方法 960 行。S33 与 DiffJSCC 已在 S33 summary 中联合出现；SGD 三个 seed 的原始 `per_sample.csv` 也仍在，可按 `sample_id + seed + SNR` 与 S33 做只读连接。现有数据足以重新检查键集合、噪声来源、指标定义、码率账本，并对所有逐样本指标计算 source-image cluster 95% CI。

预计用时约 6–10 小时：

| 工作 | 时间 |
|---|---:|
| 输入 SHA、960-key、seed/SNR、canonical noise 与指标路径审计 | 1–2 h |
| 统一聚合器、aggregate/per-SNR 表、source-cluster bootstrap CI | 2–3 h |
| main/edge/caption/receiver-prior/训练数据/算力透明账本 | 1–2 h |
| 中文报告、论文可用表格与 claim 边界检查 | 2–3 h |

不需要 GPU；因此 1 个工作日通常足够，2 天包含复核和排版余量。

## 最终表会把什么分开

| 方法 | 信道内真实成本 | 发送端 side-info | 接收端/外部先验 | 现有合同定位 |
|---|---:|---|---|---|
| S33 strong | `16,384 real` | 无 | 无生成式外部先验；COCO 从零训练 | exact-rate pure JSCC |
| DiffJSCC | `16,384 real` | 无 | SD2.1 + BLIP2 + OpenCLIP；caption 在接收端从带噪初始重建生成 | exact-rate generative JSCC，但训练域/算力不同 |
| SGD paper upper | 最低 `21,856 real` | active edge + paper 中免费完美 caption | released diffusion/ControlNet + BLIP2/CLIP | 超 S33 `5,472 real = 33.40%`，只作 non-ranking upper bound |

这里要严格区分：**外部先验不是信道码率。** DiffJSCC 依赖大模型预训练会影响模型容量、数据公平和算力公平，但不会使它在通信码率上作弊；SGD 的问题则是真正发送了 main、edge 和 caption，却没有把完美 caption 计入论文通信预算。

## 能得到的结论

轻量版可以得到以下五类结论：

1. **DiffJSCC 的感知表现不能用“码率白嫖”解释。** 它与 S33 都是 `16,384 real` 且无发送端 side information，因此两者当前结果是合法的 exact-rate positioning。
2. **S33 与 DiffJSCC 的现有证据是 fidelity–perception Pareto。** 已知 aggregate 中，S33=`30.4661 dB / 0.969708 MS-SSIM / 0.119985 LPIPS / 9 failures`，DiffJSCC=`27.5984 / 0.940799 / 0.100223 / 23`。S33 的 PSNR 高约 `2.868 dB`，DiffJSCC 的 LPIPS 低约 `0.01976`；不能宣布单一总冠军。
3. **SGD 的 LPIPS 数值很强，但不能进入公平排名。** 现有 SGD=`27.7404 dB / 0.952973 / 0.072101 LPIPS / 25 failures`，但最低真实成本为 `21,856 real`，比 S33/DiffJSCC 多 `33.40%`，且 paper protocol 给接收端完美 caption。轻量版可以量化“观测优势”，不能量化压回 16,384 后会缩水多少。
4. **S33 的论文价值不会因 DiffJSCC 感知更好而消失。** exact-rate 下它提供明显更高 fidelity、较低观测 semantic failure、约 31M 参数和无 diffusion 的低复杂度端点；DiffJSCC 提供另一端的生成式感知真实性。这个对照支持“不同系统目标/复杂度的 Pareto”，而不是 S33 全面超过。
5. **能明确指出下一项证据缺口。** 三套现有共同结果没有统一 FID/KID；因此只能把 LPIPS 称为现有感知证据，不能声称已完整覆盖生成分布质量。这个缺口会写在主表脚注和结论中。

## 不能得到的结论

- 不能回答 SGD 在总预算压到 16,384 后 LPIPS/FID 会缩水多少；这仍需要不存在的官方重训或近似适配。
- 不能说 S33、DiffJSCC 与 SGD 训练数据、模型容量或算力公平；只能说 S33 与 DiffJSCC 的通信总码率相同。
- 不能把 SD2.1/BLIP2 参数量换算成 channel symbols，也不能把“外部先验更大”写成“码率更高”。
- 不能凭 LPIPS 单轴宣布生成质量全面领先；没有统一 FID/KID。
- 不能把冻结 policy-dev 结果写成 independent final test；official validation 仍封存。

## 计划产物

- 一张论文可直接引用的统一方法表：真实码率、side information、外部先验、训练数据、参数/计算、PSNR/MS-SSIM/LPIPS/failure。
- aggregate 与 per-SNR 表，以及逐样本 pairwise descriptive delta + source-cluster 95% CI。
- `rate_and_prior_ledger.json`：把通信成本和模型先验分开记账。
- 中文结果报告；标题与正文不使用“全面排名”“SOTA”或“公平击败”。

机器可读计划为 `configs/s34c_lite_rate_transparency_preregistration.yaml`。长版 S34C 保持暂停，轻量版完成后再决定是否值得投入重训。
