# SGD-JSCC 全程替代 B1 的扩展判定（2026-07-17）

## 一句话结论

**不能因为 SGD-JSCC 明显优于普通 JSCC，就把当前系统无条件改成“全程 SGD”。** 在对 SGD 最有利的免费、完美文本条件下，它相对 B1 显著改善 LPIPS 和 MS-SSIM，却显著损失 PSNR；语义失败数虽从 35 降到 25，但配对置信区间跨零且产生了 11 次 B1 原本正确、SGD 变错的新错误。更关键的是，公开 SGD 图像与边缘支路已经用满 19,712 个实符号，文本条件没有进入同码率账本。

因此，本轮支持的方向不是放弃 diffusion，也不是照搬公开 SGD 全程替代，而是：**保留严格同码率的保真路径，把 channel-state-matched diffusion 作为有明确收益判据、受 measurement/semantic risk 约束的生成先验。**

## 1. 本轮要回答的问题

此前 8 图 pilot 已确认：

- SGD-JSCC 论文协议上界明显优于精确低码率普通 DeepJSCC；
- 当前 B1 在 PSNR 上很强，但 LPIPS 不如 SGD；
- 小样本不足以判断“既然 SGD 比 JSCC 好，是否应该全部使用 SGD”。

本轮只判定下面的可证伪命题：

> 在相同图像、相同 AWGN 实现和相同 19,712-real 图像支路工作点下，SGD 论文协议上界是否在质量和语义上全面支配 B1，并且公开方法是否能在同一总预算内传输其文本条件？

预注册在任何 S20 结果产生前完成：`reports/sgd_b1_decision_preregistration_2026-07-17.md`。

## 2. 冻结协议

- population：Imagenette `policy_dev` 中由独立监督分类器 `T_cls` 判为 clean-correct 的 64 张图；排除旧 8 图 pilot，按 10 类分层取样；official validation 未访问。
- SNR：`[1, 4, 7, 13, 19] dB`。
- channel seeds：`20260748/20260749/20260750`。
- 每个方法：`64×5×3=960` 行。
- 信道：项目复 AWGN 口径，每实坐标噪声方差 `P/(2γ)`。
- 配对：同一 `(seed, sample_id, SNR)` 必须有相同 canonical noise SHA-256。
- 统计：以源图为 cluster，对五 SNR 和三个信道种子先在图内平均，再做 10,000 次 cluster bootstrap。
- 语义：只在原图 `T_cls` clean-correct population 上统计 final failure；同时报告方法间 new error/repair，不能用平均 failure 掩盖个体漂移。

冻结 population SHA-256：

`a08b0d3f3dead68919bea42a0a28c7854e998aea6173fe62d4669bd537ab393f`

## 3. 方法口径

| 方法 | 物理预算 | 文本/语义条件 | 角色 |
|---|---:|---|---|
| B0-full | 19,712 real | 无 | 普通 exact-rate JSCC 参考 |
| B0-strict | 19,632 image + 80 payload | `G_aux` UInt2、BPSK×4，接收端擦除 payload 坐标后重建 | B1 的严格同码率输入 |
| B1 | 同 B0-strict | 预算内 80-real sender payload；B1 本身使用 SNR/结构输入 | 当前强保真 anchor |
| SGD-paper-upper | main 16,384 + active edge 3,328 = 19,712 real | 四个 sender caption 免费、完美、无误 | 论文发布协议的有利上界，不是严格总物理码率 |

SGD 使用作者发布源码 commit `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`、发布权重和 49 步反向扩散；第三方源码未修改，项目侧 adapter 只负责统一输入、信道、指标和元数据。

## 4. 总体结果

| 方法 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | `T_cls` failure ↓ | 推理时间/图 | 峰值显存 |
|---|---:|---:|---:|---:|---:|---:|
| B0-full | 27.10576 | 0.927514 | 0.255417 | 111/960 | 0.151 ms | 5551.5 MiB |
| B0-strict | 27.05489 | 0.926733 | 0.257027 | 115/960 | 2.642 ms* | 5551.5 MiB |
| B1 | **28.12459** | 0.946697 | 0.159398 | 35/960 | 2.642 ms | 5551.5 MiB |
| SGD-paper-upper | 27.74037 | **0.952973** | **0.072101** | **25/960** | 2064.738 ms | 7458.9 MiB |

`*` B0-strict 与 B1 在同一计时段产生，因此该行不是独立 B0 解码耗时；只用于说明 B1 路径的总量级。

直接观察：

- SGD 论文上界相对普通 B0-full 的确是强提升，不是“论文方法本身很差”。
- B1 相对严格 B0 同时提升 PSNR `+1.06970 dB`、LPIPS `-0.09763`，95% CI 均不跨零；B1 是实质性方法，不是弱 baseline。
- SGD 与 B1 落在不同的 Pareto 位置：B1 更保真，SGD 更感知化。
- SGD 平均推理时间约为 B1 的 `781.4×`，峰值显存多约 `1907.3 MiB`。

## 5. 关键配对判定：SGD-paper-upper − B1

| 指标 | 配对均值差 | 源图 cluster 95% CI | 判定 |
|---|---:|---:|---|
| PSNR | **−0.38422 dB** | `[−0.61529, −0.16026]` | SGD 显著更差 |
| MS-SSIM | **+0.006276** | `[+0.004162, +0.008386]` | SGD 显著更好 |
| LPIPS | **−0.087297** | `[−0.100439, −0.075641]` | SGD 显著更好 |
| failure rate | `−1.0417` 个百分点 | `[−4.2708, +1.4583]` 个百分点 | 方向有利，但未显著 |

语义事件不能只看净数：

- SGD failure：25；B1 failure：35；
- SGD 相对 B1：`21` 次 repair，同时有 `11` 次 new error；
- 因此不能写成“SGD 语义上无条件更安全”。

预注册的“全面支配”要求 PSNR CI 下界大于 0、LPIPS CI 上界小于 0、failure 不更多。第一项明确失败，所以：

`paper_free_text_upper_bound_quality_and_semantic_dominance = false`

## 6. 分 SNR 结果

| SNR | B1 PSNR | SGD PSNR | ΔPSNR | B1 LPIPS | SGD LPIPS | ΔLPIPS | failure（B1→SGD） |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.22567 | 26.11820 | −0.10747 | 0.20772 | 0.08914 | −0.11858 | 11→5 |
| 4 | 27.44660 | 26.89275 | −0.55384 | 0.17681 | 0.08482 | −0.09200 | 8→8 |
| 7 | 28.33776 | 27.91116 | −0.42660 | 0.15330 | 0.06731 | −0.08599 | 8→5 |
| 13 | 29.19198 | 28.56824 | −0.62374 | 0.13189 | 0.06355 | −0.06834 | 5→6 |
| 19 | 29.42096 | 29.21149 | −0.20946 | 0.12727 | 0.05569 | −0.07157 | 3→1 |

SGD 在五个 SNR 上 LPIPS 都更好，但 PSNR 也在五个 SNR 上都更低；13/19 dB 的 MS-SSIM 也略低于 B1。由此不能把结论简化成“低 SNR SGD、高 SNR B1”的单阈值规则；更合理的控制变量还应包括任务偏好、接收不确定性、measurement consistency 和 semantic risk。

## 7. 严格码率审计

公开 SGD 权重的图像相关支路：

- main latent：16,384 real；
- active edge：3,328 real；
- 合计：19,712 real，已经等于本轮总预算。

四个 caption 采用固定 67-byte packet 时，每块为 536 bit。即便不做任何纠错、只做最弱 BPSK，也至少需要：

`4 × 536 = 2,144 real symbols`

因此最低总量为：

`19,712 + 2,144 = 21,856 real`

超预算 `2,144 real`，即 `10.8766%`。若要可靠传输还会更高。公开方法若要严格落在 19,712-real 总预算，必须重新分配/压缩主、边缘、文本支路并重新训练，不能把免费 caption 当作零成本条件。

所以：

`strict_full_sgd_route_supported = false`

## 8. 对“为什么还要融合”的回答

现在融合/受控使用 diffusion 有了实验证据，而不是为了堆模块：

1. 普通 JSCC 的确弱于 SGD；如果项目只有普通 B0，直接换 SGD 在论文协议上很合理。
2. 当前 B1 已经改变了比较对象：它在严格预算内取得更高 PSNR，且远快于 49 步 SGD。
3. SGD 提供 B1 缺少的感知先验，LPIPS 优势巨大；S19 的等容量消融也已证明 diffusion 输入包含 B0-only control 无法完全替代的信息。
4. 但生成先验会牺牲像素保真并产生个体 new error；因此必须受 channel measurement 和 semantic risk 约束，而不是无条件接管输出。

简言之：**融合的理由不是“SGD 不如 JSCC”，而是“B1 与 SGD 各自占据不同 Pareto 前沿，项目要在严格码率内利用 diffusion 的感知先验，同时守住 B1 的保真和语义底线”。**

## 9. 下一阶段方向

不改 AWGN 主线，不直接复制公开 SGD 全链，下一方法阶段收紧为：

> 严格总码率下的 channel-state-matched diffusion auxiliary path：以接收 latent/SNR 做 step matching，用 active-coordinate measurement consistency 保留信道证据，用预算内语义信息与独立 semantic-risk 判据限制生成漂移；B1/融合输出作为可回退的保真路径。

最小下一步：

1. 先把当前 S19 fusion checkpoint 纳入同一套 64×3 外部协议，回答“当前最好系统”而不只是 B1 是否超过 SGD；必须另行预注册，不能事后并入本轮确认性统计。
2. 在相同 19,712-real 总预算内设计不依赖免费 caption 的条件：优先复用已有 80-real payload，或先做无文本的 matched latent/structure ablation。
3. 训练目标同时保留 distortion、LPIPS 和 semantic new-error penalty；不能只追求生成观感。
4. 对 SGD 的 11 个相对 B1 new-error 与 21 个 repair 做 failure taxonomy，寻找 receiver-visible uncertainty，而不是手工按单一 SNR 切换。

## 10. 可复现资产

- 主配置：`configs/s20_sgd_b1_decision.yaml`
- population：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/population/population_reference.yaml`
- 基线：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/baseline/`
- SGD 三种子：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/sgd_jscc_paper_protocol/`
- 聚合：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/aggregate/summary.json`
- 聚合 SHA-256：`3023ac917f4705fd6a705ea08b7ebe99b5dec4e529c0b77dcc1d414a7dd364d5`

本轮全程使用本地数据和模型缓存，没有联网或下载；大任务执行时清空了全部代理变量。`py_compile`、`git diff --check` 和全仓 115 项标准库单测均通过。
