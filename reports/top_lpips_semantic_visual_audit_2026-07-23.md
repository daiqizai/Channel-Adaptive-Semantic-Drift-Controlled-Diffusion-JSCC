# SGD-JSCC 与 S33 top-LPIPS 重建语义人工核查

日期：2026-07-23
分析 ID：`ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004`

## 这次具体看了什么

本次没有训练模型，也没有调参或重新选择 checkpoint。对 SGD-JSCC 和 S33 strong backbone 各自已有的 `960` 条正式 policy-dev 逐样本记录，按 LPIPS 从低到高排序；同一原图即使出现在不同 seed/SNR，也只保留 LPIPS 最低的一次，最终每种方法取 `15` 张不同原图。

这里“LPIPS 最好”准确地说是“重建在 LPIPS 特征空间里最接近原图”，并不等同于无参考的“绝对最真实”，也不等同于 FID 最优。

- SGD-JSCC：直接从 S20 已保存的正式 `64 原图 + 64 重建` montage 中裁图，没有重新推理。
- S33 strong：历史 montage 每档只保存前 4 张，无法覆盖 top-15，因此用冻结 S33 checkpoint、相同 sample、相同 seed/SNR、相同 canonical noise 和历史 batch 合同做纯推理重放。15 张的重放 PSNR 与历史 CSV 最大绝对误差为 `0.0 dB`。
- official Imagenette validation 没有访问。

筛选后，SGD-JSCC 有 `14/15` 张来自 19 dB、`1/15` 张来自 7 dB；S33 有 `14/15` 张来自 19 dB、`1/15` 张来自 13 dB。这说明 top-LPIPS 集合明显偏向容易的高 SNR 样本。

## 并排图

颜色含义：绿色为人工判断语义忠实；橙色为局部数字、文字或微小轮廓发生变化，但主体/物体/场景意义仍一致；红色才表示“看着真，但内容语义与原图对不上”。本次没有红色样本。

### SGD-JSCC top-15

![SGD-JSCC top-15](../outputs/analysis/ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004/sgd_jscc_top15_reviewed.png)

### S33 strong backbone top-15

![S33 strong top-15](../outputs/analysis/ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004/s33_strong_top15_reviewed.png)

单张可放大的 pair 位于：

- `outputs/analysis/ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004/pairs/sgd_jscc/`
- `outputs/analysis/ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004/pairs/s33_strong/`

## 人工判断

| 方法 | 语义忠实 | 局部结构/文字变化 | 看着真但语义错位 | 不确定 |
|---|---:|---:|---:|---:|
| SGD-JSCC | 14 | 1 | 0 | 0 |
| S33 strong | 12 | 3 | 0 | 0 |

橙色案例：

- SGD-JSCC #15：加油机仪表数字和标牌字形发生变化，但仍是同一加油机、同一场景。
- S33 #03：远处降落伞伞面和人体轮廓被平滑，但“有人正在跳伞”的事件没有改变。
- S33 #10：远处微小伞体/物体与烟迹连接处的轮廓变化，但场景意义没有改变。
- S33 #15：加油机仪表数字和文字变模糊，但物体和场景一致。

这 30 张的冻结 `T_cls` 也全部没有报 classification failure；人工核查没有发现它漏掉的主体类别或场景级错位。

## 应该怎样解释

在“各自 LPIPS 最好的 15 张”这个特定切片上，SGD-JSCC 和 S33 都表现出很好的内容忠实度，没有观察到典型的“画面很真但主体换了/物体多了少了/场景含义变了”。S33 的橙色样本略多，但变化集中在很小的目标、仪表数字和文字，不能算 semantic drift。

这个结果不能证明任一方法在全分布上没有 hallucination。top-LPIPS 选择天然排除了较差样本，而且样本几乎都来自高 SNR；它回答的是“最佳感知样本是否仍可能语义错位”，不是“低 SNR 或 semantic-failure 尾部是否安全”。如需专门寻找风险，下一张图应独立选择“低 SNR 中 LPIPS 尚好、但 T_cls/new-error 或跨模型语义检测异常”的样本，不能把它混进本次 top-LPIPS 结果。

另外，这次可视化不改变 SGD-JSCC 与 S33 的公平性边界：SGD 仍是作者权重、免费完美文本且总码率未对齐的 paper upper bound，不能据这些图与 S33 作公平胜负判断。

## 可复核产物

- 冻结配置：`configs/top_lpips_semantic_visual_audit.yaml`，SHA-256=`3286cbcf...e643`
- 脚本：`scripts/top_lpips_semantic_visual_audit.py`，SHA-256=`83c72f99...7dbf`
- 排名表：`selection.csv`，SHA-256=`9bf874e4...25bb`
- 自动审计：`audit.json`，SHA-256=`f0b6fbc1...4499`
- 人工标签：`manual_review.json`，SHA-256=`5ffcddcf...f956`

前 3 次准备目录均因审计而 fail-closed 并保留：前两次是重放阈值过严，第三次进一步定位到误用了 SGD 的 PyTorch 2.1 虚拟环境；正式 S33 历史运行使用系统 PyTorch 2.11。切回正确环境后，`-004` 达到 `0.0 dB` 重放误差，因此只有 `-004` 是有效结果。
