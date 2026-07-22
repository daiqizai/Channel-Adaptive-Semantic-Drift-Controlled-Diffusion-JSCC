# 2026-07-09 组会可展示结果

本文件整理当前可以直接放进组会或阶段汇报的结果。所有数字来自已有实验和派生分析，没有新增实验。

## 汇报主线

一句话版本：

> Blind diffusion 在当前 JSCC 输出上会引入明显质量损伤和 semantic drift；改用 pixel-domain residual restoration 后得到稳定正向质量收益，再通过 residual alpha / fallback 控制语义风险。

建议展示为 6 页：

1. 问题动机：blind SD img2img 是负结果。
2. 正向 anchor：SNR-conditioned residual CNN 明显优于 M0。
3. 保守 M3：top-1 fallback / shrink schedule 可以做到质量提升且 accepted new error 为 0。
4. Alpha 控制：per-sample adaptive alpha 明显优于固定 schedule。
5. Learned 候选：continuous-alpha tail refiner 是当前最强训练侧候选。
6. 风险边界：ensemble audit 说明还不能声明最终 M3 完成。

## 可展示数字

### 总表

| 方法 | 角色 | 核心结果 | 语义风险结论 | 展示口径 |
|---|---|---:|---|---|
| M1 Blind SD img2img | 负结果 | mean PSNR delta `-14.7485` dB, LPIPS delta `+0.3877` | 视觉/语义都不可靠 | 用作动机，不作为正结果 |
| M2 SNR-conditioned residual CNN | 正向 anchor | mean PSNR delta `+0.7235` dB, LPIPS delta `-0.0274` | 需要 failure handling | 证明 pixel residual 路线有效 |
| M3 top-1 fallback | 保守闭环 | mean PSNR delta `+0.4011` dB, LPIPS delta `-0.0104` | pseudo semantic failure 不高于 M0 | 第一版安全闭环 |
| M3 fixed shrink schedule | 保守候选 | validation/held-out/test-like PSNR delta `+0.4584/+0.4689/+0.4552` dB | accepted new error `0/0/0` | 固定 schedule 消融/备选 |
| M3 adaptive alpha | 最强保守候选 | validation/held-out/test-like PSNR delta `+0.5584/+0.5664/+0.5691` dB | accepted new error `0/0/0`, repair 仍为 0 | 当前最强 pseudo-safe candidate |
| Continuous-alpha tail refiner | 最强 learned 候选 | validation/held-out/test-like PSNR delta `+0.5010/+0.5049/+0.5012` dB | AlexNet new error `0/0/0`; ensemble majority `1/0/0` | 强候选，但不能称最终 M3 |

### Residual CNN 每个 SNR 的正向结果

| SNR | M0 PSNR | M2 PSNR | M2 delta | M3 top-1 delta |
|---:|---:|---:|---:|---:|
| 1 dB | 28.2390 | 29.3713 | `+1.1323` | `+0.3313` |
| 4 dB | 30.3021 | 31.0858 | `+0.7837` | `+0.3812` |
| 7 dB | 31.8137 | 32.3996 | `+0.5859` | `+0.3815` |
| 13 dB | 33.4944 | 34.0448 | `+0.5504` | `+0.4557` |
| 19 dB | 34.0518 | 34.6172 | `+0.5654` | `+0.4561` |

展示讲法：

> M2 在所有 SNR 上都有 PSNR 增益，低 SNR 增益最大；M3 牺牲一部分增益换取 semantic fallback 安全边界。

### Shrink M3 的展示数字

| Split | PSNR delta | LPIPS delta | Safe accept | Protective reject | Rejected good | M3 new error |
|---|---:|---:|---:|---:|---:|---:|
| validation | `+0.4584` | `-0.0153` | 183 | 17 | 34 | 0 |
| held-out | `+0.4689` | `-0.0150` | 102 | 6 | 19 | 0 |
| test-like | `+0.4552` | `-0.0152` | 156 | 13 | 44 | 0 |

对照：

| Policy | validation new error | held-out new error | test-like new error |
|---|---:|---:|---:|
| M3 shrink fallback | 0 | 0 | 0 |
| Always accept full strength | 28 | 10 | 25 |
| Always accept validation-constrained | 19 | 3 | 12 |

展示讲法：

> 这张表最适合说明为什么不能只展示视觉增强。Always-accept 的 PSNR 更高，但会接受新语义错误；M3 shrink 的价值是把质量收益放在 semantic safety 约束内。

### Adaptive alpha 的展示数字

| Split | top-1 full | fixed shrink | adaptive alpha | adaptive new error |
|---|---:|---:|---:|---:|
| validation | `+0.4011` | `+0.4584` | `+0.5584` | 0 |
| held-out | `+0.4454` | `+0.4689` | `+0.5664` | 0 |
| test-like | `+0.4113` | `+0.4552` | `+0.5691` | 0 |

展示讲法：

> 固定 strength 已经有效，但 per-sample alpha selection 更强。这支持后续把 residual amplitude 控制前移到模型训练或选择策略中。

### Continuous-alpha learned 候选

| Split | PSNR delta | LPIPS delta | AlexNet new error | Ensemble any new error | Ensemble majority new error |
|---|---:|---:|---:|---:|---:|
| validation | `+0.5010` | `-0.0149` | 0 | 17 | 1 |
| held-out | `+0.5049` | `-0.0149` | 0 | 9 | 0 |
| test-like | `+0.5012` | `-0.0162` | 0 | 14 | 0 |

展示讲法：

> Continuous alpha 是当前最强 learned training-side amplitude-control candidate。它已经比离散 alpha-head 稳，但 ensemble audit 还有 1 个 validation majority new error，所以只能说强候选，不能说最终 M3 完成。

## 可直接放进 PPT 的图

### 1. Blind diffusion 负结果样例

- 1 dB: `outputs/EXP-S2-002/samples/snr_01db_original_reconstruction_refined.png`
- 7 dB: `outputs/EXP-S2-002/samples/snr_07db_original_reconstruction_refined.png`
- 19 dB: `outputs/EXP-S2-002/samples/snr_19db_original_reconstruction_refined.png`

用途：说明空 prompt / blind SD img2img 不是可用主线。

### 2. Residual CNN 质量随 SNR 变化

- `outputs/analysis/minimal_closure_report/figures/residual_quality_vs_snr.png`
- `outputs/analysis/minimal_closure_report/figures/residual_semantics_vs_snr.png`

用途：展示 M2 和 M3 在不同信道质量下的 tradeoff。

### 3. Shrink policy tradeoff

- `outputs/analysis/minimal_closure_report/figures/residual_shrink_policy_tradeoff.png`

用途：展示 fixed residual strength control 的收益。

### 4. Adaptive alpha tradeoff

- `outputs/analysis/minimal_closure_report/figures/adaptive_residual_alpha_policy_tradeoff.png`

用途：展示 adaptive alpha 在三段 split 上都向右移动，同时 new error 保持 0。

### 5. M3 安全接受样例

- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/samples/m3_safe_accept_sheet.png`

用途：展示 final 输出相对 M0 有恢复收益，且语义标签保持一致。

### 6. 保护性拒绝样例

- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/samples/m3_protective_reject_sheet.png`

用途：展示 detector/fallback 的必要性。

### 7. Always-accept 负对照

- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/samples/unsafe_accepted_new_error_sheet.png`

用途：展示为什么不能只看 PSNR/LPIPS，必须报告 semantic drift。

### 8. Continuous-alpha ensemble 风险图

- `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/galleries/continuous_majority_classifier_new_errors.png`
- `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/galleries/continuous_any_classifier_new_errors.png`
- `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/galleries/continuous_any_classifier_repairs.png`

用途：展示 learned candidate 的风险边界和下一步目标。

## 不能过度宣传的点

- COCO 上当前主语义判断仍是 pseudo-label / CLIP / ensemble 辅助诊断，不是最终人工或监督标签证明。
- Adaptive alpha 是 post-hoc policy over alpha candidates，不是已经端到端学会的最终模块。
- Continuous-alpha 是 learned 候选，但 ensemble audit 仍有 validation majority new error。
- Selected risk rule 有 repair，但留下 new error，不能作为最终安全方法。
- M1 blind diffusion 只能作为负结果和动机，不能包装成可用 diffusion 方法。

## 建议结论页

可以直接写：

> 这一周的阶段性成果是：我们把 blind diffusion 的失败转化成一个可控 residual restoration 闭环。M2 residual CNN 在 COCO-256 AWGN 上稳定提升质量；M3 fallback/shrink/adaptive-alpha 在 pseudo semantic constraint 下实现 `0` accepted new error 的保守增强。当前最强 learned 候选是 continuous-alpha tail refiner，但跨分类器 audit 还未完全安全，所以下一步应围绕 semantic-risk-aware continuous alpha loss 或 ensemble-aware model selection 收敛最终 M3。
