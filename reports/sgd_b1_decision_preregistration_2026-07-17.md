# SGD-JSCC 是否应全面替代 B1：扩大决策实验预注册

日期：2026-07-17
分析 ID：`ANALYSIS-S20-SGD-B1-DECISION-001`

## 要回答的问题

本实验不预设 fusion 必要。它直接检验：在作者发布权重与论文“文本免费且无误”协议下，SGD-JSCC 是否在失真、感知质量和监督语义上全面支配普通精确低码率 DeepJSCC 与强 B1。若全面支配成立，后续方法优先以 SGD-JSCC 为主体；若不成立，才保留锚点、路由或融合的研究必要性。

## 冻结 population

- 数据：Imagenette2-320 `train` 内部 `policy_dev`，official validation 保持封存；
- 从冻结 `T_cls` clean-correct membership 中按类别分别作 SHA-256 排序；
- 排除先前已经暴露的 8 图 external pilot；
- 共 64 图，类别计数为 `7/7/7/7/6/6/6/6/6/6`；
- population reference SHA-256：`a08b0d3f3dead68919bea42a0a28c7854e998aea6173fe62d4669bd537ab393f`。

## 冻结信道与码率

- AWGN，SNR `[1,4,7,13,19]` dB；
- channel seeds：`20260748/20260749/20260750`；
- 每个 `(source, seed, SNR)` 使用 SHA 派生的同一个 19,712 维 CPU float32 标准高斯向量；
- 每实坐标噪声方差为 `P/(2*gamma)`；
- 总图像分支预算为 19,712 real symbols，即 9,856 complex uses，CBR `0.0501302083`。

## 方法

1. `B0-full`：完整 19,712 图像坐标的精确低码率 DeepJSCC。
2. `B0-strict`：为现有 B1 输入合同预留并擦除 80 个 UInt2-R4 payload 坐标，只保留 19,632 图像坐标。
3. `B1`：冻结 `EXP-S16-B1-001`，输入 `B0-strict`。
4. `SGD-paper-upper`：作者发布权重，main 16,384 + active edge 3,328，文本沿用论文免费、无误假设。

`SGD-paper-upper` 不是严格端到端同总码率方法；它是判断“即便给予 SGD 最有利的论文假设，是否仍全面支配 B1”的上界实验。

## 指标与统计

- PSNR、MS-SSIM、LPIPS；
- 冻结独立 `T_cls` 的 final failure；
- SGD 相对 B1 的 new error / repair；
- 推理耗时与峰值显存；
- 以 source image 为 cluster，在三个信道 seed 和五个 SNR 上做 10,000 次 bootstrap。

## 决策规则

只有同时满足以下条件，才判定“全用 SGD”得到支持：

1. SGD−B1 PSNR cluster-bootstrap 95% CI 下界大于 0；
2. SGD−B1 LPIPS 95% CI 上界小于 0；
3. SGD final failure 不高于 B1；
4. 明确解决严格 19,712 总预算中的文本传输，或结论始终限定为论文免费文本上界。

任一条件不满足，只能判定 SGD 与 B1 存在 tradeoff，不能为了简化叙事直接删除 B1。

## 严格码率审计

作者发布图像分支已经用满 `16,384+3,328=19,712` 实符号。每图四个固定 caption packet 在无重复 BPSK 下至少还需要 `4×536=2,144` 实符号。因此不修改/重训 main 或 edge 的前提下，发布方法不存在严格 19,712 总符号且同时传文本的可执行版本。该事实作为物理可执行性判据单独记录，不用不公平的临时 puncturing 冒充作者方法。

## 产物边界

- 主配置：`configs/s20_sgd_b1_decision.yaml`；
- population：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/population/`；
- 所有结果目录禁止覆盖；
