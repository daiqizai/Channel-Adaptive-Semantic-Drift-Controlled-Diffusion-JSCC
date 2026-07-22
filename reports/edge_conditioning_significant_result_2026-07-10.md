# Edge-Conditioned Residual Restoration：受控显著结果

日期：2026-07-10

## 一句话结论

Receiver-visible Sobel/Laplacian 条件对 residual restoration 的质量增益已经通过 matched 2×2、四段 paired bootstrap、fresh holdout 和 LPIPS 复核；但 raw/跨分类器语义风险仍存在，因此当前贡献应写成“稳定质量与感知收益 + 明确安全边界”，不能写成跨模型完全安全的语义改进。

## 1. 归因问题已解决

早先 `EXP-S4-008` 相比 `EXP-S4-006` 同时增加 edge 输入、模型容量和训练轮数，不能单独归因。新增两个 matched arms 后形成：

| Arm | Edge | Capacity / Epochs | Params | Raw ΔPSNR | M3 ΔPSNR |
|---|---|---|---:|---:|---:|
| `EXP-S4-006` | no | `48×5 / 40` | 210,915 | +0.7235 | +0.4011 |
| `EXP-S4-010` | yes | `48×5 / 40` | 211,779 | +0.7736 | +0.4466 |
| `EXP-S4-009` | no | `64×6 / 60` | 447,235 | +0.8009 | +0.4738 |
| `EXP-S4-008` | yes | `64×6 / 60` | 448,387 | +0.9398 | +0.5356 |

Sample-cluster 10,000 次 paired bootstrap：

- small edge raw effect：`+0.0501 dB`，95% CI `[+0.0249,+0.0696]`；
- large edge raw effect：`+0.1389 dB`，95% CI `[+0.1031,+0.1805]`；
- small/large M3 effect：`+0.0455/+0.0617 dB`，CI 均排除 0；
- raw edge×capacity interaction：`+0.0888 dB`，CI `[+0.0461,+0.1369]`。

因此 edge 的独立质量贡献成立，并且在较大模型上更明显。

## 2. 跨 split 与 fresh holdout 复现

Large matched pair 的 raw edge − no-edge：

| Split | Paired ΔPSNR | 95% CI | Positive SNR |
|---|---:|---|---:|
| validation | +0.1389 | `[+0.1031,+0.1805]` | 5/5 |
| held-out | +0.1565 | `[+0.1221,+0.1975]` | 5/5 |
| test-like | +0.1585 | `[+0.1337,+0.1854]` | 5/5 |
| fresh-holdout | +0.1411 | `[+0.1201,+0.1634]` | 5/5 |

Fresh holdout 固定为此前未用于 downstream residual 分析的 `sample_000320`-`sample_000383`。该 split 运行后没有重新调 alpha 或 threshold。

## 3. Schedule 约束与感知质量

旧 per-SNR schedule 的有效强度在 4→7 dB 上升，违反项目单调约束。新的 validation-only 全局选择为：

```text
alpha          = [0.75, 0.75, 0.75, 1.00, 0.75]
residual gate  = [0.12, 0.10, 0.08, 0.05, 0.04]
gate × alpha   = [0.09, 0.075, 0.06, 0.05, 0.03]
```

冻结结果：

| Split | ΔPSNR | ΔLPIPS | Failure Δ |
|---|---:|---:|---:|
| validation | +0.5734 | -0.0145 | +0.0000 |
| held-out | +0.6128 | -0.0148 | +0.0000 |
| test-like | +0.5700 | -0.0163 | +0.0000 |
| fresh-holdout | +0.5668 | -0.0162 | +0.0000 |

四段所有单独 SNR 的 LPIPS 都优于 M0，PSNR 也全部为正。

## 4. Semantic risk 边界

Edge 不是稳定语义改进：

- validation large raw pseudo failure 相比 matched no-edge 增加 `+0.0438`；
- raw new error `26→34`，repair `44→38`；
- 其他 split 的 pseudo failure 变化会改变方向，说明该指标不稳定。

Source-AlexNet top-1 fallback 的 new error 为 0 是决策定义保证，不是独立泛化证据。三分类器离线审计：

| Split | Any-Model New Error | Majority New Error | Any Repair |
|---|---:|---:|---:|
| validation | 20 | 1 | 47 |
| held-out | 7 | 1 | 8 |
| test-like | 17 | 0 | 50 |
| fresh-holdout | 17 | 3 | 55 |

因此当前 edge monotonic policy 不能直接升级为“跨模型安全最终 M3”。

## 5. 可用于论文/组会的表述

可以写：

> Receiver-visible structure conditioning provides a statistically significant and cross-split consistent restoration gain under capacity- and budget-matched comparisons. A monotonic residual-strength controller preserves over 0.56 dB PSNR gain while improving LPIPS on all evaluated splits. Independent classifier audits, however, reveal residual semantic risk, motivating supervised clean-correct evaluation and semantic-risk-aware model selection.

不能写：

- edge 本身提升 semantic reliability；
- AlexNet gate 的零 new-error 证明跨模型安全；
- 当前 residual result 已完成原始 diffusion M2/M3 定义；
- fresh holdout 等价于外部数据集泛化。

## 6. 关键复现产物

- 2×2 report：`outputs/analysis/exp_s4_006_008_009_010_edge_capacity_ablation/REPORT.md`
- cross-split report：`outputs/analysis/exp_s4_008_009_matched_edge_holdout_audit/REPORT.md`
- monotonic validation：`outputs/analysis/exp_s4_008_edge_monotonic_residual_shrink_selection/REPORT.md`
- frozen held-out/test-like/fresh reports：`outputs/analysis/exp_s4_008_edge_monotonic_*_residual_shrink_schedule_check/REPORT.md`
- ensemble report：`outputs/analysis/exp_s4_008_edge_monotonic_policy_ensemble_audit/REPORT.md`

## 7. 下一步优先级

1. 补带标签的 supervised clean-correct subset；这是当前最重要的论文闭环缺口。
2. 在训练或 validation model selection 中加入 independent semantic-risk 约束，而不是继续只调单一 AlexNet 阈值。
3. 若仍保留 diffusion 论文主线，只尝试从 M0/residual-CNN 附近初始化的短链 conditional correction，并把本轮 edge residual result 作为正向 restoration anchor。
4. 在用户确认论文主线口径前，不修改 `PROJECT.md` / `MILESTONES.md` 中原始 diffusion 方法定义。
