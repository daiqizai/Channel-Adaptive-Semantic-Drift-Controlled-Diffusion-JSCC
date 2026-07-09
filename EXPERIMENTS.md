# 实验记录

## ID 规则

每个实验必须有唯一 ID。

格式如下：

- `EXP-S1-001`：阶段1，DeepJSCC baseline
- `EXP-S2-001`：阶段3，Blind diffusion refinement
- `EXP-S3-001`：阶段4，Semantic drift metric
- `EXP-S4-001`：阶段5，Channel-adaptive semantic guidance
- `EXP-S5-001`：阶段6，完整实验

即使实验失败，也不能复用 ID。

## 实验索引

| ID | 日期 | 项目版本 | 方法 | 数据集 | 信道 | SNR | CBR | 指标 | 状态 | 输出路径 |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP-S1-001 | 2026-06-29 | N/A (not a project git repo) | M0-DeepJSCC | CIFAR-10 test subset, 1024 images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM | 完成 | `outputs/EXP-S1-001/` |
| EXP-S2HR-001 | 2026-06-30 | N/A (not a project git repo) | M0-DeepJSCC-HR-pilot | COCO2017 val split pilot, 4500 train / 500 val | AWGN | 7 dB | 0.17 | MSE, PSNR, SSIM | 完成（非正式 pilot） | `outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/` |
| EXP-S2HR-002 | 2026-06-30 | N/A (not a project git repo) | M0-DeepJSCC-HR-pilot export | COCO2017 val split pilot, 500 val | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成（非正式 pilot） | `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/` |
| EXP-S2HR-003 | 2026-07-01 | N/A (not a project git repo) | M0-DeepJSCC-HR formal train | COCO2017 train2017 / val2017 | AWGN | 7 dB | 0.17 | MSE, PSNR, SSIM | 完成（best 可用，latest NaN） | `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/` |
| EXP-S2HR-004 | 2026-07-01 | N/A (not a project git repo) | M0-DeepJSCC-HR formal export | COCO2017 val2017 subset, 512 images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成 | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/` |
| EXP-S2HR-005 | 2026-07-03 | 8678e4f | M0-DeepJSCC-HR formal export 256 saved images | COCO2017 val2017 subset, 512 eval / 256 exported images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成（供 residual validation 使用） | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/` |
| EXP-S2HR-006 | 2026-07-06 | 3bcf825 | M0-DeepJSCC-HR formal export 384 saved images | COCO2017 val2017 subset, 512 eval / 384 exported images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成（供 test-like split 复核使用） | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/` |
| EXP-S2-001 | 2026-07-01 | N/A (not a project git repo) | M1-BlindDiffusion preflight/run attempt | COCO2017 val2017 export subset, 16 images/SNR planned | AWGN | [1, 7, 19] dB | 0.17 | 未生成 | 阻塞（模型权重缺失；提权下载/GPU 运行被拒绝） | 未创建 |
| EXP-S2-002 | 2026-07-01 | N/A (not a project git repo) | M1-BlindDiffusion | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, diffusion time | 完成（负结果） | `outputs/EXP-S2-002/` |
| EXP-S3-001 | 2026-07-02 | N/A (not a project git repo) | CLIP image-image consistency diagnostic | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | CLIP cosine similarity, CLIP drop rate | 完成（辅助语义诊断；负结果） | `outputs/EXP-S3-001/` |
| EXP-S3-002 | 2026-07-02 | N/A (not a project git repo) | Frozen classifier pseudo-label consistency diagnostic | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | Pseudo-label prediction consistency, pseudo drift-origin, refinement drift | 完成（辅助分类器诊断；负结果） | `outputs/EXP-S3-002/` |
| EXP-S3-003 | 2026-07-02 | N/A (not a project git repo) | COCO caption CLIP text consistency diagnostic | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | COCO caption CLIP image-text similarity, caption drop rate | 完成（辅助 caption 语义诊断；负结果） | `outputs/EXP-S3-003/` |
| EXP-S4-001 | 2026-07-03 | N/A (not a project git repo) | M3-PseudoClassifierFallbackPilot | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo Final-Failure, accept/reject, false accept/reject | 完成（S5 fallback pilot；非完整 M3） | `outputs/EXP-S4-001/` |
| EXP-S4-002 | 2026-07-03 | N/A (local directory is not yet a git repo) | SNRAdaptiveDiffusionStrengthValidation | COCO2017 val2017 export subset, 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject | 完成（S5 strength validation；负/部分结果） | `outputs/EXP-S4-002/` |
| EXP-S4-003 | 2026-07-03 | N/A (local directory is not yet a git repo) | SD VAE roundtrip diagnostic | COCO2017 val2017 export subset, 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure | 完成（S5 VAE 诊断；负/瓶颈确认） | `outputs/EXP-S4-003/` |
| EXP-S4-004 | 2026-07-03 | 401d4bd + uncommitted local changes at run time | SNR-conditioned pixel residual refiner pilot attempt | COCO2017 val2017 export subset, train 24 images/SNR, eval planned 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | training loss only; final metrics not written | 失败（CSV 写入字段 bug；保留输出，不复用） | `outputs/EXP-S4-004/` |
| EXP-S4-005 | 2026-07-03 | 401d4bd + uncommitted local changes at run time | SNR-conditioned pixel residual refiner pilot | COCO2017 val2017 export subset, train 24 images/SNR, eval 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject | 完成（S5 latent-free restoration pilot；正向小样本结果） | `outputs/EXP-S4-005/` |
| EXP-S4-006 | 2026-07-03 | 709f1c6 | SNR-conditioned pixel residual refiner validation | COCO2017 val2017 export subset, train 160 images/SNR, eval 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject | 完成（S5 residual validation；正向但需 detector error analysis） | `outputs/EXP-S4-006/` |
| EXP-S4-007 | 2026-07-06 | 4f4eefb | SNR-conditioned pixel residual diffusion pilot | COCO2017 val2017 export subset, train 80 images/SNR, eval 16 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject, sampling time | 完成（S5 residual diffusion pilot；负结果） | `outputs/EXP-S4-007/` |
| ANALYSIS-S6-002 | 2026-07-07 | 20f9cc3 + local script | ResidualShrinkSelection | COCO2017 val2017 `EXP-S4-006` eval outputs, 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, accept/new-error | 完成（派生分析；validation-only；不训练不下载） | `outputs/analysis/exp_s4_006_residual_shrink_selection/` |
| ANALYSIS-S6-003 | 2026-07-07 | 7ef1753 + local script | FrozenResidualShrinkScheduleCheck | COCO2017 val2017 test-like `sample_000256`-`sample_000319`, 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, accept/new-error | 完成（frozen schedule 复核；不调参不训练不下载） | `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/` |
| ANALYSIS-S6-004 | 2026-07-07 | 371833e + local script | MinimalClosureReportWithHeldoutShrinkM3 | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 held-out/test-like shrink M3；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-005 | 2026-07-07 | 371833e + local script | FrozenHeldoutResidualShrinkScheduleCheck | COCO2017 val2017 held-out `sample_000000`-`sample_000031`, 32 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, accept/new-error | 完成（frozen schedule held-out 复核；不调参不训练不下载） | `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/` |
| ANALYSIS-S6-006 | 2026-07-07 | c19cc0f + local script | ResidualShrinkM3ArtifactGallery | COCO2017 val2017 validation/held-out/test-like residual shrink outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | policy summary, case counts, safe accept/protective reject/new-error galleries | 完成（派生 artifact；不训练不下载不调参） | `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/` |
| ANALYSIS-S6-007 | 2026-07-07 | fbcfe72 + local script | AdaptiveResidualAlphaPolicy | COCO2017 val2017 validation/held-out/test-like residual alpha candidates | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, selected alpha, accept/new-error | 完成（派生 policy；不训练不下载不调参） | `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/` |
| ANALYSIS-S6-008 | 2026-07-07 | bcfc1f1 + local script/config | MinimalClosureReportWithAdaptiveAlphaM3 | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink/adaptive-alpha tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 adaptive alpha M3；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-009 | 2026-07-07 | 9cacff5 + local script/config | TwoStageResidualAlphaPolicy | COCO2017 val2017 validation/held-out/test-like adaptive alpha decisions | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, two-stage accept/fallback, accepted new error | 完成（派生 policy；不重分类；不训练不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/` |
| ANALYSIS-S6-010 | 2026-07-07 | 9cacff5 + local script/config | MinimalClosureReportWithTwoStageAlphaAblation | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink/adaptive/two-stage alpha tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 two-stage alpha 消融；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-011 | 2026-07-09 | 4a466e8 + local script/config | ReceiverAlphaPredictor | COCO2017 val2017 validation/held-out/test-like adaptive alpha decisions and candidate PNGs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, target alpha accuracy, accept/new-error | 完成（validation-only tabular predictor；不训练图像模型不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_receiver_alpha_predictor/` |
| ANALYSIS-S6-012 | 2026-07-09 | 4a466e8 + local script/config | MinimalClosureReportWithReceiverAlphaPredictor | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink/adaptive/two-stage/predictor alpha tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 receiver predictor；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-013 | 2026-07-09 | a7076eb + local script/config | AlphaHeadResidualRefinerPilot | COCO2017 val2017 validation/held-out/test-like adaptive-alpha pseudo targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, target alpha accuracy, accept/new-error | 完成（冻结 residual CNN，仅训练 alpha head；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/` |
| ANALYSIS-S6-014 | 2026-07-09 | 594db31 + local script/config | WeightedAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like adaptive-alpha pseudo targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, target alpha accuracy, accept/new-error | 完成（冻结 residual CNN，仅训练 class-weighted alpha head；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/` |
| ANALYSIS-S6-015 | 2026-07-09 | 050b0c2 + local script/config | BenefitAwareAlphaPredictor | COCO2017 val2017 validation/held-out/test-like alpha candidates | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（validation-derived safe-PSNR utility soft labels；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_benefit_alpha_predictor/` |
| ANALYSIS-S6-016 | 2026-07-09 | 53b71b3 + local script/config | BenefitAwareAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（冻结 residual CNN，仅训练 benefit-aware alpha head；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/` |
| ANALYSIS-S6-017 | 2026-07-09 | 901420f + local script/config | BenefitAwareJointAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（解冻 residual CNN joint fine-tune；不运行 diffusion 不下载；LPIPS 省略；负/诊断结果） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/` |
| ANALYSIS-S6-018 | 2026-07-09 | c69743a + local script/config | BenefitAwareTailAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（只微调 residual tail + alpha head；不运行 diffusion 不下载；LPIPS 省略；训练侧正向阶段结果） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/` |
| ANALYSIS-S6-019 | 2026-07-09 | 9b6f74a + local script/config | BenefitAwareTailContinuousAlphaResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, continuous alpha, accept/new-error | 完成（只微调 residual tail + continuous alpha head；不运行 diffusion 不下载；LPIPS 省略；训练侧正向突破） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/` |
| ANALYSIS-S6-020 | 2026-07-09 | 3c8a0bd + local script/config | ContinuousAlphaTailRefinerPerceptualEnsembleAudit | COCO2017 val2017 validation/held-out/test-like continuous-alpha outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, AlexNet/ResNet18/MobileNetV3-Small pseudo final failure, ensemble new-error votes | 完成（派生审计；不训练不运行 diffusion；本地 LPIPS 与分类器权重；强候选但非跨模型完全安全） | `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/` |

`项目版本` 优先填写 git commit。若当前项目目录不是 git 仓库，填写 `N/A (not a project git repo)`，并在单实验记录中写明 config、脚本和关键源码路径。

## 指标要求

### 图像质量

- PSNR
- MS-SSIM
- LPIPS
- FID，可选

### 语义可靠性

- Classification accuracy
- Prediction consistency
- Semantic drift rate
- Semantic failure rate
- Detector accept / reject rate，若使用 failure detector
- CLIP similarity，可选

### 系统开销

- Diffusion steps
- Inference time
- 参数量
- FLOPs

## 单实验模板

### EXP-Sx-000：标题

- 日期：
- 项目版本：
- 第三方 commit：
- 阶段：
- 方法：
- 数据集：
- 数据 split / 样本 ID：
- 信道：
- SNR：
- CBR：
- 随机种子：
- checkpoint：
- config：
- 运行命令：
- 关键源码：
- 输出路径：
- 状态：

#### 指标

- PSNR：
- MS-SSIM：
- LPIPS：
- FID：
- Classification accuracy：
- Prediction consistency：
- Semantic drift rate：
- Semantic failure rate：
- Detector accept rate：
- Detector reject rate：
- CLIP similarity：
- Diffusion steps：
- Inference time：
- 参数量：
- FLOPs：

#### 结果总结

-

#### Semantic drift 观察

-

#### 失败案例

-

#### 复现备注

-

#### 下一步

-

## 正式实验要求

正式实验必须满足：

- 使用唯一 `EXP-*` ID 和唯一输出目录。
- 保存 config 副本、metrics 文件和样例图。
- 记录项目版本；如果没有项目 git commit，必须记录脚本、配置和关键源码路径。
- 记录第三方 baseline commit。
- 记录数据 split、随机种子、checkpoint、SNR、CBR 和信道模型。
- smoke test 不写入正式实验索引，但可以写入 `PROGRESS.md`。

### EXP-S1-001：DeepJSCC CIFAR-10 AWGN baseline

- 日期：2026-06-29
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S1 DeepJSCC baseline
- 方法：M0-DeepJSCC
- 数据集：CIFAR-10 test subset, 1024 images
- 数据 split / 样本 ID：`outputs/EXP-S1-001/subset_indices.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；subset seed 42
- checkpoint：`third_party/Deep-JSCC-PyTorch/out/checkpoints/CIFAR10_8_13.0_0.17_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl`
- config：`outputs/EXP-S1-001/config.yaml`
- 运行命令：`python3 scripts/s1_deepjscc_mini_eval.py --device cpu --num-samples 1024 --batch-size 64 --output-dir outputs/EXP-S1-001 --formal`
- 关键源码：`scripts/s1_deepjscc_mini_eval.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S1-001/`
- 状态：完成

#### 指标

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM |
|---:|---:|---:|---:|---|
| 1 | 0.004698 | 23.5428 | 0.8216 | N/A |
| 4 | 0.002464 | 26.3794 | 0.8927 | N/A |
| 7 | 0.001371 | 28.9857 | 0.9350 | N/A |
| 13 | 0.000584 | 32.8612 | 0.9696 | N/A |
| 19 | 0.000390 | 34.7994 | 0.9785 | N/A |

- PSNR：见上表
- MS-SSIM：未计算；CIFAR-10 为 32x32，`pytorch-msssim` 默认 4 次下采样要求图像边长大于 160
- LPIPS：未计算，后续接入 perceptual metric 时补
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用
- Inference time：未单独统计
- 参数量：未单独统计
- FLOPs：未单独统计

#### 结果总结

M0-DeepJSCC baseline 在固定 CIFAR-10 test subset 上跑通。PSNR 和 SSIM 随 SNR 升高单调提升，可作为后续 M1/M2/M3 的 pre-diffusion 对照。

#### Semantic drift 观察

本实验不包含 diffusion refinement，尚未统计 semantic drift。下一步需要冻结 `T_cls` 并实现 classifier consistency 指标。

#### 失败案例

本实验仅保存每个 SNR 的样例对比图，尚未整理 semantic failure case。

#### 复现备注

当前项目目录不是 git 仓库，因此项目版本记为 `N/A (not a project git repo)`。正式复现依赖第三方 commit、配置副本、脚本路径和固定 subset indices。

#### 下一步

实现 semantic drift metric 的最小版本，或开始接入 M1-BlindDiffusion 的后处理接口。

### EXP-S2HR-001：DeepJSCC COCO-val 256x256 AWGN pilot

- 日期：2026-06-30
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC pilot
- 方法：M0-DeepJSCC-HR-pilot
- 数据集：COCO2017 `val2017` 固定切分，4500 train / 500 val
- 数据 split / 样本 ID：`data/coco_val_split/split_manifest.json`
- 信道：AWGN
- SNR：7 dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/checkpoints/best.pt`
- config：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/config.yaml`
- 运行命令：`python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0`
- 关键源码：`scripts/train_deepjscc_highres.py`, `scripts/prepare_image_symlink_split.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/`
- 状态：完成；非正式 pilot，不替代 COCO2017 train/val 主实验

#### 指标

- PSNR：26.6647 dB
- MS-SSIM：未计算；当前训练脚本记录 SSIM
- SSIM：0.7837
- MSE：0.002548
- LPIPS：未计算
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用
- Inference time：未单独统计
- 参数量：未单独统计
- FLOPs：未单独统计

#### 结果总结

使用已下载的 COCO2017 `val2017` 生成不重叠 4500/500 pilot split，并训练 50 epoch 得到可用的 256x256 DeepJSCC checkpoint。该 checkpoint 适合后续 high-res inference、diffusion refinement 接口和样例流程调试。

#### Semantic drift 观察

本实验不包含 diffusion refinement，尚未统计 semantic drift。样例图显示重建能保留主要物体和场景结构，但细节明显模糊，适合作为后续 diffusion semantic drift 控制的调试输入。

#### 失败案例

尚未整理。

#### 复现备注

这是非正式 pilot 实验，训练和验证都来自 COCO2017 `val2017` 的固定不重叠切分。正式论文主实验仍必须等待 COCO2017 `train2017` 下载完成后重新训练。

#### 下一步

用该 checkpoint 调试 high-res inference/export 和 M1-BlindDiffusion 接口；COCO2017 `train2017` 完成后重新训练正式 `M0-HR`。

### EXP-S2HR-002：DeepJSCC COCO-val 256x256 pilot SNR sweep export

- 日期：2026-06-30
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC pilot export
- 方法：M0-DeepJSCC-HR-pilot export
- 数据集：COCO2017 `val2017` pilot validation split, 500 images
- 数据 split / 样本 ID：`data/coco_val_split/split_manifest.json`; evaluated paths copied to `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/config.yaml`
- 运行命令：`python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 32 --output-dir outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/`
- 状态：完成；非正式 pilot，不替代 COCO2017 train/val 主实验

#### 指标

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003426 | 25.1263 | 0.7096 | 0.8991 | 0.7205 |
| 4 | 0.002837 | 26.0905 | 0.7563 | 0.9280 | 0.1874 |
| 7 | 0.002547 | 26.6680 | 0.7836 | 0.9441 | 0.1840 |
| 13 | 0.002333 | 27.1678 | 0.8064 | 0.9572 | 0.1849 |
| 19 | 0.002279 | 27.3030 | 0.8125 | 0.9607 | 0.1830 |

- LPIPS：未计算
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用

#### 结果总结

pilot checkpoint 在高分辨率 COCO-val validation split 上完成 SNR sweep。PSNR、SSIM 和 MS-SSIM 随 SNR 升高整体提升。脚本同时导出 32 张原图和各 SNR 的 DeepJSCC 重建图，用于后续 `M1-BlindDiffusion` 输入。

#### Semantic drift 观察

本实验只导出 pre-diffusion `x_hat`，尚未统计 semantic drift。低 SNR 样例显示纹理和边缘更模糊，但主要物体/场景仍可辨认，适合作为 diffusion hallucination 风险测试输入。

#### 失败案例

尚未整理。

#### 复现备注

这是非正式 pilot export。第一组 SNR 的 inference time 包含 CUDA warmup，计时仅供粗略参考。正式论文主实验仍需等待 COCO2017 `train2017` 完成后重新训练和评估。

#### 下一步

读取 `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/exports/snr_XXdb/reconstruction/` 接入 `M1-BlindDiffusion`，并开始记录 refinement 后的视觉指标和初步 semantic drift。

### EXP-S2HR-003：DeepJSCC COCO2017 256x256 AWGN formal train

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal train
- 方法：M0-DeepJSCC-HR formal train
- 数据集：COCO2017 `train2017` / `val2017`
- 数据 split / 样本 ID：`configs/s2_deepjscc_coco256_awgn.yaml`，验证集使用 config 中固定 val subset
- 信道：AWGN
- SNR：7 dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/config.yaml`
- 运行命令：`python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco256_awgn.yaml --device cuda:0`
- 关键源码：`scripts/train_deepjscc_highres.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/`
- 状态：完成；`best.pt` 可用，`latest.pt` 不可用

#### 指标

- best epoch：73
- best val MSE：0.0008254946042143274
- best val PSNR：31.56180403754115 dB
- best val SSIM：0.9054122059606016
- latest epoch：99
- latest metrics：NaN
- latest 参数：NaN，不可用于后续实验
- LPIPS：未计算
- FID：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算

#### 结果总结

COCO2017 `train2017` 和 `val2017` 已完整就位，正式 COCO-256 DeepJSCC 训练产出了可用 `best.pt`。训练在 epoch 0-88 指标有限，epoch 89-99 出现 NaN，因此本实验的正式 baseline 必须使用 epoch 73 的 `best.pt`，不能使用 `latest.pt` 或最终 `metrics.json` 中的 NaN final。

#### Semantic drift 观察

本实验只训练 pre-diffusion DeepJSCC，不包含 refinement，因此尚未统计 semantic drift。

#### 失败案例

epoch 89 后训练发散为 NaN。已在训练脚本中增加非有限 loss/metrics 防护，后续重训会提前停止并用 best checkpoint 评估 final metrics。

#### 复现备注

当前项目目录不是 git 仓库，因此项目版本记为 `N/A (not a project git repo)`。训练日志见 `outputs/logs/s2_coco256_awgn_train.direct.screen.log`。后续论文主实验和 diffusion 输入应固定使用 `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`。

#### 下一步

基于 `best.pt` 跑正式 SNR sweep/export，并将导出的 `x_hat` 输入 `M1-BlindDiffusion`。

### EXP-S2HR-004：DeepJSCC COCO2017 256x256 formal SNR sweep export

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal export
- 方法：M0-DeepJSCC-HR formal export
- 数据集：COCO2017 `val2017` subset, 512 images
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/config.yaml`
- 运行命令：`python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 32 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/`
- 状态：完成

#### 指标

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0018173862 | 28.0189655945 | 0.8090499612 | 0.9363910668 | 0.6834 |
| 4 | 0.0011532078 | 30.0470464826 | 0.8700513527 | 0.9622469177 | 0.1861 |
| 7 | 0.0008255254 | 31.5589745864 | 0.9054089159 | 0.9763763894 | 0.1969 |
| 13 | 0.0005807463 | 33.1954004802 | 0.9348068793 | 0.9876335945 | 0.1824 |
| 19 | 0.0005199769 | 33.7264324129 | 0.9425818466 | 0.9905498993 | 0.1869 |

- LPIPS：未计算
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用

#### 结果总结

正式 `best.pt` 在 COCO2017 val subset 上完成 SNR sweep。PSNR、SSIM 和 MS-SSIM 随 SNR 增加稳定提升，7 dB 结果与 best checkpoint 训练记录一致。导出目录包含 32 张 `exports/original/` 原图，以及每个 SNR 下 32 张 `exports/snr_XXdb/reconstruction/` 重建图，可直接作为 `M1-BlindDiffusion` 的输入。

#### Semantic drift 观察

本实验只导出 pre-diffusion `x_hat`，尚未统计 semantic drift。下一阶段需要比较 DeepJSCC 原始重建、blind diffusion refinement 和 semantic-controlled refinement 的分类一致性或 CLIP consistency。

#### 失败案例

尚未整理。当前低 SNR 样例应优先用于观察 diffusion 是否把主体语义修偏。

#### 复现备注

第一组 SNR 的 inference time 包含 CUDA warmup，计时仅供粗略参考。该实验是后续正式 high-resolution diffusion 实验的 M0 输入来源，优先级高于 COCO-val pilot export。

#### 下一步

读取 `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/exports/snr_XXdb/reconstruction/`，实现 `M1-BlindDiffusion` 的最小可复现后处理与 LPIPS/semantic drift 指标。

### EXP-S2HR-005：DeepJSCC COCO2017 256x256 formal SNR sweep export 256 saved images

- 日期：2026-07-03
- 项目版本：`8678e4f`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal export
- 方法：M0-DeepJSCC-HR formal export, 256 saved images per SNR
- 数据集：COCO2017 `val2017` subset, 512 images evaluated, first 256 images exported
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 256 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
- 状态：完成；support export for `EXP-S4-006`

#### 指标

该实验仍在同一 512 张 COCO val subset 上评估 M0，因此 MSE/PSNR/SSIM/MS-SSIM 与 `EXP-S2HR-004` 的主指标一致；区别是保存的 PNG 从 32 张/SNR 扩大到 256 张/SNR。

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0018173862 | 28.0189655945 | 0.8090499612 | 0.9363910668 | 0.6802 |
| 4 | 0.0011532078 | 30.0470464826 | 0.8700513527 | 0.9622469177 | 0.1960 |
| 7 | 0.0008255254 | 31.5589745864 | 0.9054089159 | 0.9763763894 | 0.1961 |
| 13 | 0.0005807463 | 33.1954004802 | 0.9348068793 | 0.9876335945 | 0.1942 |
| 19 | 0.0005199769 | 33.7264324129 | 0.9425818466 | 0.9905498993 | 0.1953 |

#### 结果总结

本实验用于给 residual restoration validation 提供更大的固定 M0 PNG 输入，不代表新的 M0 模型。输出包含 `exports/original/sample_000000.png` 到 `sample_000255.png`，以及每个 SNR 对应的 `exports/snr_XXdb/reconstruction/`。后续 `EXP-S4-006` 使用其中 `sample_000032` 到 `sample_000191` 训练，`sample_000192` 到 `sample_000255` 验证。

#### Semantic drift 观察

本实验只导出 pre-refinement `x_hat`，未额外统计 semantic drift。语义可靠性在后续 `EXP-S4-006` 中统计。

#### 复现备注

本实验不下载数据或模型，只读取本地 COCO、已有 `best.pt` checkpoint 和第三方 DeepJSCC 代码。运行命令显式清空代理变量；输出目录是新目录，不覆盖旧 32 张 formal export。

### EXP-S2HR-006：DeepJSCC COCO2017 256x256 formal SNR sweep export 384 saved images

- 日期：2026-07-06
- 项目版本：`3bcf82525ca6760a66d3b9dfa4d846ec275451e7`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal export
- 方法：M0-DeepJSCC-HR formal export, 384 saved images per SNR
- 数据集：COCO2017 `val2017` subset, 512 images evaluated, first 384 images exported
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 384 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/`
- 状态：完成；support export for `EXP-S4-006` test-like split check

#### 指标

该实验仍在同一 512 张 COCO val subset 上评估 M0，因此 MSE/PSNR/SSIM/MS-SSIM 与 `EXP-S2HR-004` 和 `EXP-S2HR-005` 的主指标一致；区别是保存的 PNG 从 256 张/SNR 扩大到 384 张/SNR。

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0018173862 | 28.0189655945 | 0.8090499612 | 0.9363910668 | 0.6610 |
| 4 | 0.0011532078 | 30.0470464826 | 0.8700513527 | 0.9622469177 | 0.1956 |
| 7 | 0.0008255254 | 31.5589745864 | 0.9054089159 | 0.9763763894 | 0.1937 |
| 13 | 0.0005807463 | 33.1954004802 | 0.9348068793 | 0.9876335945 | 0.1929 |
| 19 | 0.0005199769 | 33.7264324129 | 0.9425818466 | 0.9905498993 | 0.1961 |

#### 结果总结

本实验用于给 `EXP-S4-006` 的更正式 test-like gate 复核提供额外 M0 PNG 输入，不代表新的 M0 模型。输出包含 `exports/original/sample_000000.png` 到 `sample_000383.png`，以及每个 SNR 对应的 `exports/snr_XXdb/reconstruction/`。后续 test-like 复核使用 `sample_000256` 到 `sample_000319`，该样本段不与 `EXP-S4-006` 的 train `sample_000032`-`sample_000191` 或 eval `sample_000192`-`sample_000255` 重叠。

#### Semantic drift 观察

本实验只导出 pre-refinement `x_hat`，未额外统计 semantic drift。语义可靠性在后续 `EXP-S4-006` test-like gate 复核中统计。

#### 复现备注

本实验不下载数据或模型，只读取本地 COCO、已有 `best.pt` checkpoint 和第三方 DeepJSCC 代码。运行命令显式清空代理变量；输出目录是新目录，不覆盖旧 256 张 formal export。

### EXP-S2-001：M1-BlindDiffusion preflight/run attempt

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S3 Blind Diffusion
- 方法：M1-BlindDiffusion
- 数据集：COCO2017 `val2017` subset export，计划每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`configs/s3_m1_blind_diffusion_coco256_awgn.yaml`
- 运行命令：
  - `python3 scripts/s3_blind_diffusion_refine.py --dry-run`
  - `python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download`
  - `python3 scripts/s3_blind_diffusion_refine.py --device cpu`
- 关键源码：`scripts/s3_blind_diffusion_refine.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：未创建；`outputs/EXP-S2-001/` 不存在
- 状态：阻塞 / 未生成正式结果

#### 指标

- PSNR：未生成
- MS-SSIM：未生成
- LPIPS：未生成
- Diffusion steps：计划值 25，未执行
- Inference time：未生成

#### 结果总结

已完成 M1 脚本、配置和输入样本对齐 preflight。dry-run 确认正式 M0 export 中 1/7/19 dB 各有 16 张匹配样本可用，且 checkpoint 指向 `best.pt` 而非 `latest.pt`。

正式 diffusion 运行未完成：提权命令因审批层拒绝，无法使用 GPU 和网络下载 Stable Diffusion 权重；local-only CPU 命令在 `local_files_only=true` 下报错，原因是 `runwayml/stable-diffusion-v1-5` 不在本地 Hugging Face cache。

#### Semantic drift 观察

未生成 refinement 图像，不能报告 semantic drift 或视觉提升。

#### 失败案例

本次失败属于环境/模型权重阻塞，不是方法结果失败。不能把该尝试写成 M1 的有效实验。

#### 复现备注

后续若用户显式允许下载并使用 GPU，使用当前配置默认输出 `outputs/EXP-S2-002/`，避免复用本次失败 ID。也可以预先把 diffusion 权重放入 `outputs/cache/huggingface/` 后去掉 `--allow-download` 运行。

#### 下一步

获得 Stable Diffusion img2img 权重和 GPU 运行许可后，运行 `python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download`，生成 refined 图、`metrics.json` 和样例图。

### EXP-S2-002：M1-BlindDiffusion COCO-256 small-scale refinement

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S3 Blind Diffusion
- 方法：M1-BlindDiffusion
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S2-002/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S2-002/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy HF_ENDPOINT=https://hf-mirror.com python3 scripts/s3_blind_diffusion_refine.py --device cuda:0`
- 关键源码：`scripts/s3_blind_diffusion_refine.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S2-002/`
- 状态：完成；负结果

#### 指标

| SNR(dB) | M0 PSNR(dB) | M1 PSNR(dB) | M0 SSIM | M1 SSIM | M0 MS-SSIM | M1 MS-SSIM | M0 LPIPS | M1 LPIPS | Diffusion ms/image |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.1746 | 16.2229 | 0.8107 | 0.3204 | 0.9398 | 0.5421 | 0.1747 | 0.5025 | 107.27 |
| 7 | 31.8274 | 16.7812 | 0.9088 | 0.3795 | 0.9779 | 0.5843 | 0.0542 | 0.4600 | 82.42 |
| 19 | 34.1357 | 16.8880 | 0.9470 | 0.4065 | 0.9915 | 0.5959 | 0.0254 | 0.4549 | 81.57 |

- Diffusion steps：25
- Strength：0.25
- Guidance scale：1.0
- Prompt：空字符串
- LPIPS：成功计算，AlexNet 权重缓存到 `outputs/cache/torch/`
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- CLIP similarity：已由 `EXP-S3-001` 作为辅助语义诊断计算

#### 结果总结

当前固定强度 blind SD img2img 不是有效 refinement。相对 M0，M1 在所有 SNR 下 PSNR、SSIM、MS-SSIM 大幅下降，LPIPS 也显著变差。高 SNR 下 M0 已很接近原图，但 blind diffusion 仍强行改写结构，说明该设置不适合作为正向视觉增强。

#### Semantic drift 观察

尚未用冻结分类器或 CLIP 计算正式 semantic drift 指标，但样例图已经显示明显 hallucination / semantic drift 风险：甜甜圈纹理被改成不稳定的杂乱结构，花瓶和花朵被重新生成，狗和车内猫/座椅场景出现主体和背景结构错乱。该结果应作为后续 semantic control / failure handling 的负例动机，不能包装成提升。

#### 失败案例

样例拼图：

- `outputs/EXP-S2-002/samples/snr_01db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_07db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_19db_original_reconstruction_refined.png`

这些图的第三行均为 M1 refined 输出，显示 diffusion 对主体结构的强烈改写。

#### 复现备注

大模型下载按用户要求走服务器直连，不走 `127.0.0.1:17890` 代理。官方 `huggingface.co` 服务器直连在本机超时，改用 `HF_ENDPOINT=https://hf-mirror.com`。由于 diffusers 多线程下载在 UNet 大文件上不稳定，本次用临时 range downloader 补齐 `unet/diffusion_pytorch_model.safetensors`，并把完整 blob 链接回 `outputs/cache/huggingface/`。该下载过程不改变实验方法，实际运行时脚本使用 local cache。

#### 下一步

实现 semantic drift / CLIP consistency 的初步评估，并把当前样例整理为 failure case。若继续探索 M1，应新建实验 ID，先做更低 `strength` 的 validation 小网格，不能覆盖本实验输出。

### EXP-S3-001：M1-BlindDiffusion CLIP image consistency diagnostic

- 日期：2026-07-02
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S4 Semantic drift metric
- 方法：CLIP image-image consistency diagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S3-001/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S3-001/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_clip_consistency_eval.py --device cuda:0`
- 关键源码：`scripts/s4_clip_consistency_eval.py`, `scripts/s4_make_clip_failure_gallery.py`
- 输出路径：`outputs/EXP-S3-001/`
- 状态：完成；辅助语义诊断，负结果

#### 指标

| SNR(dB) | CLIP sim(original, M0) | CLIP sim(original, M1) | CLIP drop M0-M1 | M1 lower than M0 | Drop >= 0.10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9022 | 0.6619 | 0.2402 | 1.0000 | 0.9375 |
| 7 | 0.9587 | 0.6867 | 0.2720 | 1.0000 | 1.0000 |
| 19 | 0.9848 | 0.6954 | 0.2895 | 1.0000 | 1.0000 |

- CLIP backbone：OpenAI CLIP `ViT-B/32` via `open_clip`
- CLIP checkpoint：`outputs/cache/open_clip/ViT-B-32.pt`
- CLIP checkpoint SHA256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- PSNR / MS-SSIM / LPIPS：本实验不重复计算；见 `EXP-S2-002`
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算正式阈值版；当前以 CLIP drop rate 作辅助诊断
- Semantic failure rate：未计算正式阈值版
- Detector accept rate：不适用
- Detector reject rate：不适用

#### 结果总结

CLIP image-image consistency 进一步确认 `EXP-S2-002` 的 blind diffusion refinement 明显不可靠。所有 48 个样本中，M1 refined 相对原图的 CLIP cosine similarity 都低于 M0 reconstruction；7 dB 和 19 dB 下所有样本的 drop 都大于 0.10。高 SNR 下 M0 已非常接近原图，但 M1 仍把图像改写到 CLIP 空间显著远离原图的位置。

#### Semantic drift 观察

该实验不是最终的分类一致性 semantic drift 指标，但它把视觉样例中的 hallucination 风险量化出来：M0 的 CLIP 相似度随 SNR 升高从 0.9022 增至 0.9848，而 M1 基本停留在 0.66 到 0.70，说明 blind diffusion 并没有利用高 SNR 下更可靠的 JSCC 重建，反而引入额外语义漂移。

#### 失败案例

每个 SNR 的 top failure case 已写入 `outputs/EXP-S3-001/metrics.json`，逐样本指标见 `outputs/EXP-S3-001/per_sample.csv`。按 CLIP drop 排名前列的样本包括：

- 1 dB：`sample_000004.png`, `sample_000013.png`, `sample_000000.png`
- 7 dB：`sample_000005.png`, `sample_000009.png`, `sample_000013.png`
- 19 dB：`sample_000013.png`, `sample_000004.png`, `sample_000008.png`

已用 `scripts/s4_make_clip_failure_gallery.py` 从 `per_sample.csv` 生成 failure case gallery：

- 全局 top sheet：`outputs/EXP-S3-001/failure_cases/sheets/global_top_clip_drop.png`
- 分 SNR sheets：`outputs/EXP-S3-001/failure_cases/sheets/snr_01db_top_clip_drop.png`, `outputs/EXP-S3-001/failure_cases/sheets/snr_07db_top_clip_drop.png`, `outputs/EXP-S3-001/failure_cases/sheets/snr_19db_top_clip_drop.png`
- triptych 目录：`outputs/EXP-S3-001/failure_cases/triptychs/`
- 索引：`outputs/EXP-S3-001/failure_cases/index.json`, `outputs/EXP-S3-001/failure_cases/global_top_clip_drop.csv`

gallery 共包含 18 个不重复 triptych：全局 top 12 和每个 SNR top 6。抽查全局最大失败样本 `snr_19db/sample_000013.png` 时，M0 与原图接近，但 M1 refined 出现明显主体纹理和背景结构改写，符合 CLIP drop 0.4026 的诊断。

#### 复现备注

open_clip 3.3.0 对 `ViT-B-32/openai` 默认优先尝试 Hugging Face Hub；本机服务器直连 `huggingface.co` HEAD 请求超时，因此本实验改为从 OpenAI 官方 URL 直连下载 `ViT-B-32.pt` 到项目缓存，并在配置中显式使用本地权重路径。该 `.pt` 是 TorchScript archive，PyTorch 2.6+ 默认 `weights_only=True` 会拒绝加载；由于文件来源为 OpenAI 官方 URL 且 SHA256 校验完全匹配，本实验配置中对该权重设置 `weights_only: false`。

#### 下一步

补充更正式的 semantic drift metric，例如冻结分类器 prediction consistency 或 object-level / CLIP text consistency。后续如果继续试更保守的 blind diffusion strength，应先用本诊断脚本和 failure gallery 筛查是否仍然破坏语义。

### EXP-S3-002：M1-BlindDiffusion frozen classifier pseudo-label consistency diagnostic

- 日期：2026-07-02
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S4 Semantic drift metric
- 方法：Frozen classifier pseudo-label consistency diagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S3-002/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S3-002/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_classifier_consistency_eval.py --device cuda:0`
- 关键源码：`scripts/s4_classifier_consistency_eval.py`, `scripts/s4_make_classifier_failure_gallery.py`
- 输出路径：`outputs/EXP-S3-002/`
- 状态：完成；辅助分类器诊断，负结果

#### 指标

All-subset，使用原图 ImageNet top-1 作为 pseudo-label：

| SNR(dB) | M0 matches original top-1 | M1 matches original top-1 | M0 pseudo drift-origin | M1 pseudo drift-origin | M1 refinement drift |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5000 | 0.1250 | 0.5000 | 0.8750 | 0.8750 |
| 7 | 0.6875 | 0.0625 | 0.3125 | 0.9375 | 0.9375 |
| 19 | 0.9375 | 0.1250 | 0.0625 | 0.8750 | 0.8750 |

原图 top-1 confidence >= 0.30 的 pseudo-clean subset：

| SNR(dB) | subset n | M0 matches original top-1 | M1 matches original top-1 | M0 pseudo drift-origin | M1 pseudo drift-origin |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 0.8889 | 0.2222 | 0.1111 | 0.7778 |
| 7 | 9 | 1.0000 | 0.1111 | 0.0000 | 0.8889 |
| 19 | 9 | 1.0000 | 0.2222 | 0.0000 | 0.7778 |

- Frozen classifier：torchvision AlexNet `IMAGENET1K_V1`
- Classifier checkpoint：`outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- Classification accuracy：未计算；COCO GT 标签未用于本实验
- Prediction consistency：见上表的 matches original top-1
- Semantic drift rate：未计算正式 clean-correct 版本；当前为 pseudo-label drift diagnostic
- Semantic failure rate：未计算正式版本
- Detector accept rate：不适用
- Detector reject rate：不适用

#### 结果总结

冻结分类器诊断与 CLIP 诊断一致：M0 随 SNR 升高越来越保持原图分类器 top-1，而 M1 refined 在所有 SNR 下都明显偏离原图 pseudo-label。尤其在原图置信度 >= 0.30 的 subset 上，7 dB 和 19 dB 的 M0 一致率为 1.0，但 M1 只有 0.1111 和 0.2222，说明 blind diffusion 会在高质量 DeepJSCC 输入上仍然强行改写语义线索。

#### Semantic drift 观察

该实验比 CLIP 更接近 `MILESTONES.md` 中要求的冻结分类器路线，但仍不是最终 clean-correct 指标，因为 COCO 图像没有使用分类 GT，ImageNet AlexNet top-1 只能作为 pseudo-label。它适合作为当前 M1 负结果的第二条证据，以及后续 failure detector / fallback 规则的调试信号。

#### 失败案例

`outputs/EXP-S3-002/metrics.json` 中保存了每个 SNR 的 top failure cases，筛选条件为 M0 与原图 top-1 一致但 M1 不一致。典型样本包括：

- 1 dB：`sample_000002.png`，原图/M0 为 `Pomeranian`，M1 为 `shoe shop`
- 7 dB：`sample_000002.png`，原图/M0 为 `Pomeranian`，M1 为 `dogsled`
- 19 dB：`sample_000002.png`，原图/M0 为 `Pomeranian`，M1 为 `gondola`
- 19 dB：`sample_000015.png`，原图/M0 为 `broccoli`，M1 为 `indigo bunting`

逐样本预测见 `outputs/EXP-S3-002/per_sample.csv`。

已用 `scripts/s4_make_classifier_failure_gallery.py` 从 `per_sample.csv` 生成 classifier failure case gallery：

- 全局 top sheet：`outputs/EXP-S3-002/failure_cases/sheets/global_top_classifier_drift.png`
- 分 SNR sheets：`outputs/EXP-S3-002/failure_cases/sheets/snr_01db_top_classifier_drift.png`, `outputs/EXP-S3-002/failure_cases/sheets/snr_07db_top_classifier_drift.png`, `outputs/EXP-S3-002/failure_cases/sheets/snr_19db_top_classifier_drift.png`
- triptych 目录：`outputs/EXP-S3-002/failure_cases/triptychs/`
- 索引：`outputs/EXP-S3-002/failure_cases/index.json`, `outputs/EXP-S3-002/failure_cases/global_top_classifier_drift.csv`

gallery 共包含 18 个不重复 triptych：全局 top 12 和每个 SNR top 6。抽查全局最大失败样本 `snr_19db/sample_000002.png` 时，原图和 M0 均被分类为 `Pomeranian`，M1 refined 被分类为 `gondola`，图像中主体结构也明显被破坏。

#### 复现备注

本实验没有联网下载。AlexNet ImageNet 权重已由 LPIPS/torch cache 路径提供，脚本在 `--allow-download` 未开启时会要求 `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` 已存在。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。

#### 下一步

确定正式 semantic drift 主指标的语义模型选择：若继续 COCO 主线，优先考虑 object detector / CLIP-text / caption-based consistency；若需要严格分类 clean-correct 统计，可引入带 ImageNet 标签的 Imagenette/ImageNet subset 作为补充，而不是把当前 pseudo-label 诊断包装成最终指标。

### EXP-S3-003：M1-BlindDiffusion COCO caption CLIP text consistency diagnostic

- 日期：2026-07-02
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S4 Semantic drift metric
- 方法：COCO caption CLIP image-text consistency diagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S3-003/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- COCO 标注：`data/coco/annotations/captions_val2017.json`，来自官方 `annotations_trainval2017.zip`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S3-003/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_coco_caption_clip_eval.py --device cuda:0`
- 关键源码：`scripts/s4_coco_caption_clip_eval.py`, `scripts/s4_make_coco_caption_failure_gallery.py`
- 输出路径：`outputs/EXP-S3-003/`
- 状态：完成；辅助 caption 语义诊断，负结果

#### 指标

使用每张 COCO val 图的 5 条人工 caption，计算每张图像与其 caption 集合的 CLIP image-text cosine similarity；表中 `caption-max` 表示取 5 条 caption 中最高相似度。

| SNR(dB) | Original caption-max | M0 caption-max | M1 caption-max | Drop M0-M1 | M1 max lower than M0 | Drop >= 0.05 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3292 | 0.3306 | 0.2816 | 0.0490 | 1.0000 | 0.6250 |
| 7 | 0.3292 | 0.3305 | 0.2815 | 0.0490 | 0.8125 | 0.5000 |
| 19 | 0.3292 | 0.3263 | 0.2877 | 0.0386 | 0.8125 | 0.3125 |

caption-mean 辅助结果：

| SNR(dB) | M0 caption-mean | M1 caption-mean | Drop M0-M1 | M1 mean lower than M0 | Drop >= 0.05 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3054 | 0.2568 | 0.0486 | 0.9375 | 0.6250 |
| 7 | 0.3063 | 0.2559 | 0.0504 | 0.8125 | 0.5000 |
| 19 | 0.3022 | 0.2605 | 0.0417 | 0.8125 | 0.3125 |

- CLIP backbone：OpenAI CLIP `ViT-B/32` via `open_clip`
- CLIP checkpoint：`outputs/cache/open_clip/ViT-B-32.pt`
- CLIP checkpoint SHA256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- Classification accuracy：未计算；COCO caption 不是分类 GT
- Prediction consistency：未计算冻结分类器版本；见 `EXP-S3-002`
- Semantic drift rate：未计算正式 clean-correct 版本；当前为 caption CLIP drop diagnostic
- Semantic failure rate：未计算正式版本
- Detector accept rate：不适用
- Detector reject rate：不适用

#### 结果总结

COCO caption image-text consistency 与 CLIP image-image 和冻结分类器 pseudo-label 诊断方向一致：M0 reconstruction 与原图 captions 的对齐基本保持在 original 附近，而 M1 refined 在所有 SNR 下都明显下降。尤其 1 dB 下 16/16 个样本的 M1 caption-max 低于 M0；7 dB 和 19 dB 下也有 13/16 个样本低于 M0。

#### Semantic drift 观察

该实验把 COCO 主数据集的人工 captions 接入了 S4 诊断，解决了此前只有 image-image CLIP 或 ImageNet pseudo-label 的局限。它仍是辅助指标，不能替代 `MILESTONES.md` 要求的正式 clean-correct 冻结分类器统计，但能更直接说明 blind diffusion 会把图像从 COCO caption 描述的语义内容上拉开。

#### 失败案例

`outputs/EXP-S3-003/metrics.json` 中保存了每个 SNR 的 top caption drop cases。典型样本包括：

- 1 dB：`sample_000002.png`，caption 为小狗，caption-max drop 0.0957
- 7 dB：`sample_000008.png`，caption 为 car / clock / flowers，caption-max drop 0.1198
- 19 dB：`sample_000003.png`，caption 为 car 中的黑猫，caption-max drop 0.0951

已用 `scripts/s4_make_coco_caption_failure_gallery.py` 从 `per_sample.csv` 生成 caption failure case gallery：

- 全局 top sheet：`outputs/EXP-S3-003/failure_cases/sheets/global_top_caption_clip_drop.png`
- 分 SNR sheets：`outputs/EXP-S3-003/failure_cases/sheets/snr_01db_top_caption_clip_drop.png`, `outputs/EXP-S3-003/failure_cases/sheets/snr_07db_top_caption_clip_drop.png`, `outputs/EXP-S3-003/failure_cases/sheets/snr_19db_top_caption_clip_drop.png`
- triptych 目录：`outputs/EXP-S3-003/failure_cases/triptychs/`
- 索引：`outputs/EXP-S3-003/failure_cases/index.json`, `outputs/EXP-S3-003/failure_cases/global_top_caption_clip_drop.csv`, `outputs/EXP-S3-003/failure_cases/README.md`

gallery 共包含全局 top 12 和每个 SNR top 6 的 triptych。抽查全局最大失败样本 `snr_07db/sample_000008.png` 时，原图/M0 都保留 car、clock 和 flowers 场景，M1 refined 出现明显纹理和结构改写。

#### 复现备注

本实验没有联网下载模型权重，使用已缓存的 OpenAI CLIP 权重。COCO annotations 下载发生在实验前，来源为 `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`，大小 252907541 bytes；下载时清空代理变量并使用服务器直连，`unzip -t` 验证无错误。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。

#### 下一步

结合 `EXP-S3-001`、`EXP-S3-002` 和 `EXP-S3-003` 设计最小 semantic failure handling：优先实现一个可复现的 fallback 规则，统计 detector accept/reject 和 Final-Failure，再进入 M3/Ours。

### EXP-S4-001：M3 pseudo-classifier semantic fallback pilot

- 日期：2026-07-03
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / Semantic Failure Handling pilot
- 方法：M3-PseudoClassifierFallbackPilot
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S4-001/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-001/config.yaml`
- 运行命令：`python3 scripts/s5_semantic_fallback_eval.py --device cuda:0`
- 关键源码：`scripts/s5_semantic_fallback_eval.py`, `src/cadsd_jscc/metrics.py`, `scripts/s4_classifier_consistency_eval.py`
- 输入实验：M0 export `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/`；M1 output `outputs/EXP-S2-002/`；classifier CSV `outputs/EXP-S3-002/per_sample.csv`
- 输出路径：`outputs/EXP-S4-001/`
- 状态：完成；S5 fallback pilot，不是完整 M3/Ours

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价；detector 决策只看 `c(M1) == c(M0)`，不使用原图。

| SNR(dB) | Accept | Reject | M0 PSNR | M1 PSNR | M3 PSNR | M0 LPIPS | M1 LPIPS | M3 LPIPS | M0 Final-Failure | M1 Final-Failure | M3 Final-Failure | False Accept | False Reject |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1250 | 0.8750 | 28.1746 | 16.2229 | 26.8313 | 0.1747 | 0.5025 | 0.2123 | 0.5000 | 0.8750 | 0.5000 | 0.0000 | 0.0000 |
| 7 | 0.0625 | 0.9375 | 31.8274 | 16.7812 | 30.9141 | 0.0542 | 0.4600 | 0.0782 | 0.3125 | 0.9375 | 0.3125 | 0.0000 | 0.0000 |
| 19 | 0.1250 | 0.8750 | 34.1357 | 16.8880 | 32.0135 | 0.0254 | 0.4549 | 0.0733 | 0.0625 | 0.8750 | 0.0625 | 0.0000 | 0.0000 |

Pseudo-clean subset，原图 top-1 confidence >= 0.30：

| SNR(dB) | subset n | Accept | M0 Final-Failure | M1 Final-Failure | M3 Final-Failure | M3 Prediction-Consistency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 0.2222 | 0.1111 | 0.7778 | 0.1111 | 0.8889 |
| 7 | 9 | 0.1111 | 0.0000 | 0.8889 | 0.0000 | 1.0000 |
| 19 | 9 | 0.2222 | 0.0000 | 0.7778 | 0.0000 | 1.0000 |

- Detector：frozen AlexNet top-1 agreement between M0 and M1
- Final output：若 `c(M1) == c(M0)`，输出 M1；否则 fallback 到 M0
- Diffusion steps：沿用 `EXP-S2-002` 的 25 steps
- Strength：沿用 `EXP-S2-002` 的 0.25
- Guidance scale：沿用 `EXP-S2-002` 的 1.0
- Prompt：空字符串
- CLIP similarity：本实验不重新计算；见 `EXP-S3-001` 和 `EXP-S3-003`

#### 结果总结

该 pilot 验证了最小 semantic failure handling 的可复现流程：在不看原图的接收端规则下，detector 拒绝大多数会改变冻结分类器 top-1 的 M1 refined 输出，使 M3 pseudo Final-Failure 回到 M0 水平。相对 M1，M3 在 all-subset 上将 Final-Failure 分别降低 `0.3750/0.6250/0.8125`。

但这不是完整 M3/Ours：底层 diffusion 仍是 `EXP-S2-002` 的固定强度负结果。少量 accepted M1 虽然没有造成 pseudo-label failure，却仍降低 PSNR、MS-SSIM 和 LPIPS，因此 fallback 只能控制语义风险，不能把一个过强的 blind diffusion 设置变成视觉增强。

#### Semantic drift 观察

M3 的 `m3_refinement_drift` 在该 detector 下为 0，因为最终输出要么与 M0 分类一致，要么直接回退到 M0。这个结果说明 top-1 agreement 是一个强保守规则，但也意味着它几乎不接受 diffusion；后续必须配合更弱 diffusion strength 或 SNR-aware strength 才可能获得感知收益。

#### 失败案例

样例拼图：

- `outputs/EXP-S4-001/samples/snr_01db_original_m0_m1_m3final.png`
- `outputs/EXP-S4-001/samples/snr_07db_original_m0_m1_m3final.png`
- `outputs/EXP-S4-001/samples/snr_19db_original_m0_m1_m3final.png`

逐样本 detector 决策、final 输出路径、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-001/per_sample.csv`。

#### 复现备注

本实验不联网、不下载模型、不重新运行 diffusion，只读取已有 M0/M1 图像和 `EXP-S3-002` 的冻结分类器 CSV。LPIPS 使用已缓存 AlexNet 权重。输出目录存在时脚本会拒绝覆盖。

#### 下一步

新建实验 ID 做保守 diffusion strength validation 网格，例如 `strength <= 0.10` 和更少 steps，并把本 fallback 脚本接到新 M1/M2 输出上。只有当 M3 相比 blind diffusion 降低 Final-Failure、且相比 M0 保留可观感知收益时，才能进入正式 M3/Ours 结论。

### EXP-S4-002：SNR-aware low-strength diffusion validation

- 日期：2026-07-03
- 项目版本：N/A (local directory is not yet a git repo)
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / diffusion strength validation
- 方法：SNRAdaptiveDiffusionStrengthValidation
- 数据集：COCO2017 `val2017` subset export，每个 SNR 8 张图
- 数据 split / 样本 ID：`outputs/EXP-S4-002/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000007.png`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-002/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_snr_adaptive_diffusion_validation.py --device cuda:0`
- 关键源码：`scripts/s5_snr_adaptive_diffusion_validation.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-002/`
- 状态：完成；S5 validation，负/部分结果，不是完整 M3/Ours

#### 候选设置

| Candidate | Method | Strength schedule | Steps | Guidance |
|---|---|---|---:|---:|
| `fixed_0p05` | M1-LowStrengthFixedDiffusion | 1/4/7/13/19 dB: `0.05/0.05/0.05/0.05/0.05` | 15 | 1.0 |
| `snr_adaptive_0p10_to_0p05` | M2-SNRAdaptiveDiffusion | 1/4/7/13/19 dB: `0.10/0.08/0.06/0.05/0.05` | 15 | 1.0 |

两个 schedule 都满足 strength 随 SNR 升高不增加。failure handling 使用 `EXP-S4-001` 同类规则：若 `c(refined) == c(M0)` 则接受 refined，否则 fallback 到 M0。

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价；detector 不使用原图。

| Candidate | SNR(dB) | Strength | M0 PSNR | Refined PSNR | M3 PSNR | M0 LPIPS | Refined LPIPS | M3 LPIPS | Refined Failure | M3 Failure | Accept | False Accept | False Reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0p05 | 1 | 0.05 | 28.7285 | 25.1163 | 26.9112 | 0.1885 | 0.1989 | 0.1922 | 0.6250 | 0.3750 | 0.5000 | 0.1250 | 0.0000 |
| fixed_0p05 | 4 | 0.05 | 30.7464 | 26.0170 | 26.8029 | 0.1040 | 0.1367 | 0.1313 | 0.3750 | 0.2500 | 0.8750 | 0.2500 | 0.0000 |
| fixed_0p05 | 7 | 0.05 | 32.3475 | 26.5848 | 27.6615 | 0.0606 | 0.1089 | 0.1050 | 0.0000 | 0.2500 | 0.7500 | 0.0000 | 0.2500 |
| fixed_0p05 | 13 | 0.05 | 34.0785 | 26.9924 | 29.1365 | 0.0308 | 0.0871 | 0.0657 | 0.2500 | 0.0000 | 0.7500 | 0.0000 | 0.0000 |
| fixed_0p05 | 19 | 0.05 | 34.6217 | 27.0938 | 28.3947 | 0.0282 | 0.0889 | 0.0791 | 0.1250 | 0.0000 | 0.8750 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 1 | 0.10 | 28.7285 | 22.1567 | 26.2416 | 0.1885 | 0.2759 | 0.2244 | 0.6250 | 0.3750 | 0.3750 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 4 | 0.08 | 30.7464 | 22.7259 | 26.8134 | 0.1040 | 0.2180 | 0.1649 | 0.5000 | 0.2500 | 0.5000 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 7 | 0.06 | 32.3475 | 26.5599 | 27.6566 | 0.0606 | 0.1065 | 0.1034 | 0.0000 | 0.2500 | 0.7500 | 0.0000 | 0.2500 |
| snr_adaptive_0p10_to_0p05 | 13 | 0.05 | 34.0785 | 27.0258 | 29.1664 | 0.0308 | 0.0877 | 0.0660 | 0.2500 | 0.0000 | 0.7500 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 19 | 0.05 | 34.6217 | 27.1297 | 28.4295 | 0.0282 | 0.0888 | 0.0789 | 0.1250 | 0.0000 | 0.8750 | 0.0000 | 0.0000 |

#### 结果总结

相比 `EXP-S2-002` 的 `strength=0.25`，低强度和 SNR-aware schedule 的 semantic drift 明显缓和，高 SNR 下 fallback 后的 M3 Failure 可回到 M0 水平。但两个候选都没有获得有效视觉收益：即使 `strength=0.05`，refined PSNR 和 LPIPS 仍明显差于 M0，且高 SNR 下损伤更突出。

这说明当前 Stable Diffusion img2img 后处理并不只是 strength 过强的问题。VAE encode/decode、最小 denoise step 或 prompt-free generative prior 都可能对高保真 DeepJSCC 重建造成结构/纹理改写。该结果应记录为负/部分结果，不能包装为 M2 或 M3 的成功。

#### Semantic drift 观察

`snr_adaptive_0p10_to_0p05` 在 1/4 dB 使用更高 strength，语义 failure 并没有比 `fixed_0p05` 更好，图像质量反而更差。当前证据不支持“简单增大低 SNR diffusion strength”作为有效 SNR-aware 策略。semantic fallback 仍能压低 final failure，但如果 refined 图像本身没有视觉收益，fallback 只是在做风险控制，不构成主要贡献。

#### 失败案例

样例拼图位于：

- `outputs/EXP-S4-002/candidates/fixed_0p05/samples/`
- `outputs/EXP-S4-002/candidates/snr_adaptive_0p10_to_0p05/samples/`

每张拼图为 original / M0 / refined / M3-final 四行。逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-002/per_sample.csv`。

#### 复现备注

本实验使用已缓存的 `runwayml/stable-diffusion-v1-5` 和 AlexNet/LPIPS 权重，不下载模型。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。本地目录当前尚未初始化为 git 仓库，因此项目版本仍不能填写 commit；用户提供的 GitHub URL 已写入 config 和 metrics metadata。

#### 下一步

优先做 VAE/latent roundtrip 诊断，分离以下因素：

- SD VAE encode/decode 本身相对 M0 的失真。
- 最小 denoise step 在极低 strength 下是否仍改写结构。
- prompt-free prior 是否比 restoration-aware diffusion 更容易 hallucinate。

若 roundtrip 本身已显著损伤 PSNR/LPIPS，则第一版不应继续把通用 SD img2img 当作主正向 refinement，而应转向更保守的 restoration 模块或把 diffusion 结果仅作为负例和 failure handling 动机。

### EXP-S4-003：Stable Diffusion VAE roundtrip diagnostic

- 日期：2026-07-03
- 项目版本：N/A (local directory is not yet a git repo)
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / VAE bottleneck diagnostic
- 方法：SDVAERoundtripDiagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 8 张图
- 数据 split / 样本 ID：`outputs/EXP-S4-003/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000007.png`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-003/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sd_vae_roundtrip_eval.py --device cuda:0`
- 关键源码：`scripts/s5_sd_vae_roundtrip_eval.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-003/`
- 状态：完成；S5 VAE roundtrip 诊断，负/瓶颈确认，不是完整 M2/M3/Ours

#### 设置

- SD 组件：`runwayml/stable-diffusion-v1-5` 的 `vae` subfolder
- VAE latent：使用 `latent_dist.mode()`，deterministic roundtrip
- VAE scaling factor：0.18215
- UNet denoise：不运行
- diffusion prompt：不使用；本实验没有文本条件、没有 guidance、没有 denoising step
- 对照：
  - `M0 reconstruction vs original`
  - `M0-VAE roundtrip vs original`
  - `M0-VAE roundtrip vs M0 reconstruction`
  - `Original-VAE roundtrip vs original`

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。

| SNR(dB) | M0 PSNR | M0-VAE PSNR | Delta | M0 LPIPS | M0-VAE LPIPS | Delta | M0-VAE vs M0 PSNR | M0 Failure | M0-VAE Failure | M0-VAE Refinement Drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.7285 | 25.2433 | -3.4852 | 0.1885 | 0.1975 | +0.0090 | 25.6506 | 0.3750 | 0.6250 | 0.5000 |
| 4 | 30.7464 | 26.1930 | -4.5534 | 0.1040 | 0.1335 | +0.0295 | 26.7426 | 0.2500 | 0.3750 | 0.1250 |
| 7 | 32.3475 | 26.7550 | -5.5926 | 0.0606 | 0.1049 | +0.0443 | 27.4579 | 0.2500 | 0.1250 | 0.1250 |
| 13 | 34.0785 | 27.1925 | -6.8861 | 0.0308 | 0.0853 | +0.0545 | 27.9958 | 0.0000 | 0.2500 | 0.2500 |
| 19 | 34.6217 | 27.2957 | -7.3260 | 0.0282 | 0.0860 | +0.0578 | 28.1307 | 0.0000 | 0.1250 | 0.1250 |

Original-VAE roundtrip 相对原图在这 8 张样本上固定为 PSNR `26.8097` dB、LPIPS `0.0605`，pseudo failure 为 `0.1250`。这说明即使输入是干净原图，SD VAE 往返也会引入可观失真；当输入是高 SNR M0 时，该瓶颈会把 `34+` dB 的重建压到约 `27` dB。

#### 结果总结

该实验把 `EXP-S4-002` 中的质量下降拆开验证：不运行 UNet、不使用 prompt、不做任何 diffusion denoise，仅 SD VAE encode/decode 已足以解释大部分高保真损伤。M0-VAE 相对 M0 的 PSNR 损失随 SNR 升高变大，从 1 dB 的 `-3.4852` dB 扩大到 19 dB 的 `-7.3260` dB；LPIPS 也从 `+0.0090` 恶化到 `+0.0578`。

因此，当前通用 Stable Diffusion img2img 路线不是简单调低 `strength` 就能成为正向视觉增强。VAE roundtrip 本身已经破坏了 DeepJSCC high-SNR reconstruction 的细节和分类线索，后续若继续使用 diffusion，应优先考虑 restoration-aware 或 latent-free/像素域保守方法。

#### Semantic drift 观察

M0-VAE 不只是低层指标下降，也会改变冻结 AlexNet pseudo-label。All-subset 中，1 dB 的 M0-VAE pseudo Final-Failure 为 `0.6250`，高于 M0 的 `0.3750`；13/19 dB 中 M0 本身 failure 为 0，但 M0-VAE 分别引入 `0.2500/0.1250` 的 pseudo failure。该结果继续支持本项目主线：任何“看起来更自然”的 generative/latent 重建都必须接受 semantic drift 检查。

#### 失败案例

样例拼图位于：

- `outputs/EXP-S4-003/samples/snr_01db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_04db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_07db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_13db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_19db_original_m0_m0vae_originalvae.png`

每张拼图为 original / M0 / M0-VAE / original-VAE 四行。逐样本路径、top-1 pseudo-label 和一致性标记见 `outputs/EXP-S4-003/per_sample.csv`。

#### 复现备注

本实验使用已缓存的 `runwayml/stable-diffusion-v1-5` VAE 和 AlexNet/LPIPS 权重，不下载模型。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。本地目录运行时尚未初始化为 git 仓库，因此项目版本仍不能填写 commit；用户提供的 GitHub URL 已写入 config 和 metrics metadata。

#### 下一步

第一版不建议继续把通用 SD img2img 当作 M2/M3 正向主路线。更稳妥的推进方向是把 SD img2img 负结果和 VAE bottleneck 作为 semantic failure handling 的动机，同时探索更贴近 restoration 的保守模块；若仍使用 diffusion，需要优先验证无 VAE 高保真瓶颈的实现。

### EXP-S4-004：SNR-conditioned pixel residual refiner pilot attempt

- 日期：2026-07-03
- 项目版本：`401d4bdda6ff52602093e978ad8c1c34c6f939ac` + uncommitted local changes at run time
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free restoration pilot
- 方法：SNRConditionedPixelResidualRefinerPilot attempt
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：训练 `sample_000008.png` 到 `sample_000031.png`；评估计划为 `sample_000000.png` 到 `sample_000007.png`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-004/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --device cuda:0`
- 关键源码：`scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-004/`
- 状态：失败；训练完成后 CSV 写入失败，未生成最终 metrics，实验 ID 不复用

#### 失败原因

初版 `write_csv` 只用第一行 `train_history` 的字段作为 CSV header；但 `eval_mse` 和 `eval_psnr_db` 只在每 10 个 epoch 验证时出现，导致写入后续行时报错：

```text
ValueError: dict contains fields not in fieldnames: 'eval_mse', 'eval_psnr_db'
```

该失败发生在训练 80 epoch 和 checkpoint 写入之后、最终评估之前。输出目录保留了 `config.yaml`、`source_manifest.json`、`checkpoints/best.pt` 和 `checkpoints/latest.pt`。随后已修复 CSV 字段合并逻辑，并用新实验 ID `EXP-S4-005` 重新完整运行。

#### 复现备注

本实验不下载模型或数据，运行命令显式清空代理变量。由于这是失败实验，不能把 checkpoint 或中间训练 loss 包装成正式结果。

### EXP-S4-005：SNR-conditioned pixel residual refiner pilot

- 日期：2026-07-03
- 项目版本：`401d4bdda6ff52602093e978ad8c1c34c6f939ac` + uncommitted local changes at run time
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free restoration pilot
- 方法：SNRConditionedPixelResidualRefinerPilot
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：
  - train：`sample_000008.png` 到 `sample_000031.png`，每个 SNR 24 张，共 120 对 M0/original
  - eval：`sample_000000.png` 到 `sample_000007.png`，每个 SNR 8 张，共 40 对 M0/original
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- refiner checkpoint：`outputs/EXP-S4-005/checkpoints/best.pt`
- config：`outputs/EXP-S4-005/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --device cuda:0`
- 关键源码：`scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-005/`
- 状态：完成；S5 latent-free restoration pilot，小样本正向结果，不是最终 M2/M3/Ours

#### 方法设置

- 模型：小型 SNR-conditioned residual CNN
- 输入：`M0 reconstruction` + 1 通道 SNR map
- 输出：pixel-domain residual 后的 `x_refined`
- 初始化：最后一层零初始化，初始输出接近 M0
- residual gate：1/4/7/13/19 dB 使用 `0.12/0.10/0.08/0.05/0.04`，随 SNR 升高不增加
- 训练：80 epoch，batch size 8，128x128 random crop，MSE + 0.1 L1
- semantic failure handling：与 `EXP-S4-001` 类似，若 `c(refined) == c(M0)` 则接受，否则 fallback 到 M0；detector 不看原图

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。

| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Delta | M0 LPIPS | Refined LPIPS | Delta | M0 Failure | Refined Failure | M3 Failure | Accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 28.7285 | 29.1151 | +0.3866 | 0.1885 | 0.1703 | -0.0183 | 0.3750 | 0.2500 | 0.3750 | 0.8750 |
| 4 | 0.10 | 30.7464 | 30.9332 | +0.1868 | 0.1040 | 0.0995 | -0.0044 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 7 | 0.08 | 32.3475 | 32.4380 | +0.0905 | 0.0606 | 0.0607 | +0.0000 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 13 | 0.05 | 34.0785 | 34.2034 | +0.1248 | 0.0308 | 0.0299 | -0.0010 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 19 | 0.04 | 34.6217 | 34.7899 | +0.1682 | 0.0282 | 0.0254 | -0.0028 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

#### 结果总结

该 pilot 初步回答了 `EXP-S4-003` 后的关键问题：避开 Stable Diffusion VAE 后，保守 pixel-domain residual refinement 可以在不牺牲语义可靠性的情况下带来小幅质量收益。5 个 SNR 上 refined PSNR 均高于 M0，提升范围为 `+0.0905` 到 `+0.3866` dB；LPIPS 在 1/4/13/19 dB 改善，在 7 dB 基本持平。

语义侧没有出现 `EXP-S2-002` 那种系统性 drift。All-subset 下，13/19 dB 的 refined failure 仍为 0；1 dB refined failure 从 M0 的 `0.3750` 降到 `0.2500`，但经过 top-1 agreement fallback 后 M3 final failure 回到 M0 的 `0.3750`，说明当前 detector 对“修正了原错误分类”的情况较保守。

#### Semantic drift 观察

`refined_vs_m0_reconstruction` 的 PSNR 在 1/4/7/13/19 dB 分别为约 `40.57/45.16/48.06/49.36/48.37` dB，说明 residual 改动很小。除 1 dB 有 1 个样本改变 M0 top-1 外，其他 SNR 的 `refined_refinement_drift` 均为 0。与 SD img2img 的强 hallucination 相比，pixel residual 更符合 semantic drift control 的第一版方向。

#### 失败案例和样例

样例拼图位于：

- `outputs/EXP-S4-005/samples/snr_01db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_04db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_07db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_13db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_19db_original_m0_refined_m3final.png`

逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-005/per_sample.csv`。

#### 复现备注

本实验不联网、不下载模型或数据，只读取已有正式 M0 export 和本地 AlexNet/LPIPS 权重。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。运行时仓库 HEAD 是 `401d4bd`，但脚本和配置处于未提交状态；因此本记录额外列出脚本、配置和输出副本路径。

#### 下一步

将该 pilot 扩大到更稳定的 validation split：重新导出更多 COCO val M0 样本，训练/验证/测试三分，并比较 `M0`、`SD img2img negative M1`、`pixel residual M2` 和 `semantic fallback M3`。只有在更大 split 上稳定保持质量收益且不增加 semantic failure，才能把它作为第一版替代通用 SD img2img 的主路线。

### EXP-S4-006：SNR-conditioned pixel residual refiner validation

- 日期：2026-07-03
- 项目版本：`709f1c665f500e3f6a3dc71609267dd90789c005`
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free restoration validation
- 方法：SNRConditionedPixelResidualRefinerValidation
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：
  - 输入 export：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
  - train：`sample_000032.png` 到 `sample_000191.png`，每个 SNR 160 张，共 800 对 M0/original
  - eval：`sample_000192.png` 到 `sample_000255.png`，每个 SNR 64 张，共 320 对 M0/original
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- refiner checkpoint：`outputs/EXP-S4-006/checkpoints/best.pt`
- config：`outputs/EXP-S4-006/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --device cuda:0`
- 关键源码：`scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-006/`
- 状态：完成；S5 residual validation 正向结果，但仍需 detector error analysis；不是最终 M2/M3/Ours

#### 方法设置

- 模型：小型 SNR-conditioned residual CNN
- 输入：`M0 reconstruction` + 1 通道 SNR map
- 输出：pixel-domain residual 后的 `x_refined`
- 初始化：最后一层零初始化，初始输出接近 M0
- residual gate：1/4/7/13/19 dB 使用 `0.12/0.10/0.08/0.05/0.04`
- 训练：40 epoch，batch size 16，128x128 random crop，MSE + 0.1 L1
- semantic failure handling：若 `c(refined) == c(M0)` 则接受，否则 fallback 到 M0；detector 不看原图

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。`Refined Failure` 是 refined 相对原图 pseudo top-1 的 failure；`M3 Failure` 是 top-1 agreement fallback 后最终输出的 failure。

| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Refined Delta | M3 PSNR | M3 Delta | M0 LPIPS | Refined LPIPS | M3 LPIPS | M0 Failure | Refined Failure | M3 Failure | Accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 28.2390 | 29.3713 | +1.1323 | 28.5703 | +0.3313 | 0.1760 | 0.1005 | 0.1580 | 0.6250 | 0.5156 | 0.6250 | 0.3125 |
| 4 | 0.10 | 30.3021 | 31.0858 | +0.7837 | 30.6832 | +0.3812 | 0.1013 | 0.0672 | 0.0862 | 0.4375 | 0.3594 | 0.4375 | 0.5000 |
| 7 | 0.08 | 31.8137 | 32.3996 | +0.5859 | 32.1952 | +0.3815 | 0.0590 | 0.0452 | 0.0509 | 0.2656 | 0.3125 | 0.2656 | 0.7188 |
| 13 | 0.05 | 33.4944 | 34.0448 | +0.5504 | 33.9501 | +0.4557 | 0.0311 | 0.0256 | 0.0270 | 0.2656 | 0.2812 | 0.2656 | 0.8438 |
| 19 | 0.04 | 34.0518 | 34.6172 | +0.5654 | 34.5079 | +0.4561 | 0.0277 | 0.0196 | 0.0211 | 0.2812 | 0.2031 | 0.2812 | 0.8281 |

#### 结果总结

相比 `EXP-S4-005`，该实验使用更大的 fixed split。Pure refined 在所有 SNR 上均提升 PSNR，提升范围为 `+0.5504` 到 `+1.1323` dB，LPIPS 也全部降低。经过 top-1 agreement fallback 后，M3 final PSNR 仍在所有 SNR 上高于 M0，提升范围为 `+0.3313` 到 `+0.4561` dB，M3 LPIPS 也全部低于 M0。

语义侧的关键约束也满足：M3 final failure 在所有 SNR 上都没有高于 M0。但这不是说 detector 已经完善。1 dB 和 4 dB 下 accept rate 分别只有 `0.3125` 和 `0.5000`，说明低 SNR 下 top-1 agreement detector 很保守；7/13 dB 下 refined failure 略高于 M0，但 fallback 将 M3 failure 压回 M0。

#### Semantic drift 观察

`refined_refinement_drift` 在 1/4/7/13/19 dB 分别为 `0.6875/0.5000/0.2812/0.1562/0.1719`。这说明较大的 residual refiner 虽然带来更明显视觉收益，但也更容易改变冻结分类器 top-1；当前 M3 的价值正是把这些变化门控掉。后续不能把 pure refined 直接包装成最终方法，必须保留 drift detector/fallback，或者改进 detector 降低 false reject。

#### 失败案例和样例

样例拼图位于：

- `outputs/EXP-S4-006/samples/snr_01db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_04db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_07db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_13db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_19db_original_m0_refined_m3final.png`

逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-006/per_sample.csv`。后续应优先从 1/4 dB 的 false reject 和 false accept 样本中整理 detector failure gallery。

#### 派生 gate error analysis

已运行：

```bash
python3 scripts/s5_analyze_residual_gate_errors.py
```

输出：

```text
outputs/analysis/exp_s4_006_gate_error_analysis/summary.csv
outputs/analysis/exp_s4_006_gate_error_analysis/per_sample_with_case_type.csv
outputs/analysis/exp_s4_006_gate_error_analysis/index.json
outputs/analysis/exp_s4_006_gate_error_analysis/REPORT.md
outputs/analysis/exp_s4_006_gate_error_analysis/*/sheets/
outputs/analysis/exp_s4_006_gate_error_analysis/*/quads/
```

该分析不跑模型、不联网，只读取 `outputs/EXP-S4-006/per_sample.csv` 和已有 PNG。它把 top-1 agreement gate 的结果拆成四类：

- `protective_reject`：M0 与原图 pseudo-label 一致，refined 改变了 top-1，gate 拒绝 refined。
- `missed_semantic_repair`：M0 与原图 pseudo-label 不一致，refined 与原图 pseudo-label 一致，但 gate 因 refined 不等于 M0 而拒绝。
- `accepted_wrong_same_as_m0`：refined 与 M0 一致，但二者都不等于原图 pseudo-label。
- `rejected_both_wrong`：M0/refined 都不等于原图 pseudo-label，且二者互不一致。

| SNR(dB) | N | Accept | Protective Reject | Missed Repair | Accepted Wrong Same As M0 | Rejected Both Wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 20 (0.3125) | 9 (0.1406) | 16 (0.2500) | 5 (0.0781) | 19 (0.2969) |
| 4 | 64 | 32 (0.5000) | 7 (0.1094) | 12 (0.1875) | 3 (0.0469) | 13 (0.2031) |
| 7 | 64 | 46 (0.7188) | 7 (0.1094) | 4 (0.0625) | 6 (0.0938) | 7 (0.1094) |
| 13 | 64 | 54 (0.8438) | 3 (0.0469) | 2 (0.0312) | 10 (0.1562) | 5 (0.0781) |
| 19 | 64 | 53 (0.8281) | 2 (0.0312) | 7 (0.1094) | 9 (0.1406) | 2 (0.0312) |

关键解释：当前 gate 接受 refined 的条件是 `c(refined) == c(M0)`，因此在同一个冻结分类器口径下，M3 top-1 final failure 不会超过 M0 top-1 failure 是结构性保证。这是保守 gate 的优点，但还不能证明独立语义可靠性。分析显示 gate 保护了 28/320 个 M0-correct/refined-wrong 样本，同时错过了 41/320 个 refined 修复 M0 pseudo-label 的样本。下一版应考虑 top-k agreement、confidence margin 或 CLIP/caption 辅助，以减少 `missed_semantic_repair`，同时保留 `protective_reject`。

#### 派生 gate policy sweep

已运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_residual_gate_policies.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_gate_policy_sweep/topk_predictions.csv
outputs/analysis/exp_s4_006_gate_policy_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_gate_policy_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_gate_policy_sweep/metadata.json
outputs/analysis/exp_s4_006_gate_policy_sweep/REPORT.md
```

该分析不训练模型、不下载数据或权重，使用本地 AlexNet 权重重新计算 original/M0/refined 的 top-5。被扫的 gate policy 只使用 M0/refined 的预测结果做 receiver-side decision；original pseudo top-1 只用于离线评价。

全局关键结果：

| Policy | Final Failure | Delta Failure vs top1 | Final PSNR | Delta PSNR vs top1 | Missed Repair | Accepted Repair | Accepted New Error | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `top1_equal` | 0.3750 | +0.0000 | 31.9814 | +0.0000 | 41 | 0 | 0 | 0.6406 |
| `top1_equal_or_refined_conf_gain_ge_0p05` | 0.3188 | -0.0563 | 32.0966 | +0.1153 | 20 | 21 | 3 | 0.7563 |
| `top1_equal_or_refined_conf_gain_ge_0p10` | 0.3406 | -0.0344 | 32.0532 | +0.0719 | 28 | 13 | 2 | 0.7156 |
| `top1_equal_or_refined_conf_gain_ge_0p20` | 0.3563 | -0.0188 | 32.0037 | +0.0223 | 35 | 6 | 0 | 0.6656 |
| `refined_top1_in_m0_top5` | 0.3406 | -0.0344 | 32.1944 | +0.2130 | 8 | 33 | 22 | 0.8938 |
| `any_top5_overlap` | 0.3406 | -0.0344 | 32.2773 | +0.2960 | 2 | 39 | 28 | 0.9781 |

解释：top-5 overlap 类策略能大幅减少 missed repair 并提高 PSNR，但 accepted new error 也显著增加，语义风险偏大。当前最均衡候选是 `top1_equal_or_refined_conf_gain_ge_0p05`：它在 1/4 dB 上明显降低 final failure，在 7/13/19 dB 上不明显恶化；但它仍产生 3 个 accepted new error，因此只能作为下一轮 gate 设计候选，不能直接作为最终 M3。

#### 派生 confidence-gain gate auxiliary audit and candidate outputs

已运行辅助语义审计：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_residual_gate_aux_semantics.py --device cuda:0
```

已将候选 gate 的 final PNG 落盘：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_materialize_residual_gate_policy.py
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/per_sample_audit.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/summary.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/new_accepts.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/accepted_new_errors.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/galleries/
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/per_sample.csv
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/summary.csv
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/exports/
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/samples/
```

该审计仍保持 decision-time gate 为 receiver-side：候选策略只看 M0/refined 的冻结分类器预测和置信度，不看 original/caption。original 图像与 COCO captions 只用于离线辅助审计。审计使用本地 `outputs/cache/open_clip/ViT-B-32.pt` 和 `data/coco/annotations/captions_val2017.json`，不下载数据或权重。

全局关键结果：

| Metric | Value |
|---|---:|
| Candidate accept rate | 0.7563 |
| Newly accepted by candidate | 37 |
| Candidate final failure | 0.3188 |
| Baseline top-1 final failure | 0.3750 |
| Candidate delta PSNR vs top-1 | +0.1153 dB |
| Candidate delta PSNR vs M0 | +0.5164 dB |
| Candidate delta CLIP image-image vs top-1 | +0.0016 |
| Candidate delta caption CLIP vs top-1 | -0.0007 |
| Accepted repairs | 21 |
| Accepted new errors | 3 |

新增接受样本拆分：

| Subset | N | Delta PSNR | Delta CLIP | Delta caption | Aux both nonworse |
|---|---:|---:|---:|---:|---:|
| `new_accept_repair` | 21 | +1.0532 | +0.0205 | -0.0073 | 0.1429 |
| `new_accept_new_error` | 3 | +0.9838 | +0.0121 | -0.0058 | 0.0000 |
| `new_accept_both_wrong` | 13 | +0.9093 | +0.0038 | -0.0044 | 0.0769 |

解释：confidence-gain candidate 比原始 top-1 agreement gate 更积极，能把 missed repair 从 41 降到 20，并额外接受 21 个 pseudo-label repair；但它也引入 3 个 accepted new error。辅助语义信号是混合的：CLIP image-image 均值略升，但 caption CLIP 均值略降，且 3 个 accepted new error 都没有同时通过 image-image 与 caption 的 nonworse 检查。因此该策略已经可以作为下一轮 M3 候选输出进行视觉/held-out 审查，但不能直接登记为最终 M3/Ours。

#### 派生 held-out confidence-gain gate check

已运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_heldout_gate_check/per_sample.csv
outputs/analysis/exp_s4_006_heldout_gate_check/summary.csv
outputs/analysis/exp_s4_006_heldout_gate_check/new_accepts.csv
outputs/analysis/exp_s4_006_heldout_gate_check/accepted_new_errors.csv
outputs/analysis/exp_s4_006_heldout_gate_check/REPORT.md
outputs/analysis/exp_s4_006_heldout_gate_check/metadata.json
outputs/analysis/exp_s4_006_heldout_gate_check/exports/
outputs/analysis/exp_s4_006_heldout_gate_check/samples/
```

该复核不重训模型，只加载 `outputs/EXP-S4-006/checkpoints/best.pt`，在 `EXP-S4-006` 未使用的 `sample_000000.png` 到 `sample_000031.png` 上重新生成 refined、top-1 final 和 candidate final。该 split 对 `EXP-S4-006` 的 residual refiner 和 gate sweep 是 held-out，但仍属于同一个 COCO val export 和同一个 pseudo-label 评价口径，因此只能作为派生风险复核，不是最终 test 结论。

全局关键结果：

| Metric | Value |
|---|---:|
| Num images | 160 |
| Candidate accept rate | 0.7875 |
| Newly accepted by candidate | 19 |
| Candidate final failure | 0.2812 |
| Baseline top-1 final failure | 0.3250 |
| Candidate minus baseline failure | -0.0437 |
| Candidate final PSNR | 31.8609 dB |
| Candidate delta PSNR vs top-1 | +0.1007 dB |
| Candidate delta PSNR vs M0 | +0.5460 dB |
| Accepted repairs | 9 |
| Accepted new errors | 2 |

分 SNR 关键结果：

| SNR(dB) | M0 Failure | Refined Failure | Top-1 Failure | Candidate Failure | New Accept | Repair | New Error | Delta PSNR vs top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5312 | 0.3750 | 0.5312 | 0.4688 | 9 | 3 | 1 | +0.3132 |
| 4 | 0.4375 | 0.1875 | 0.4375 | 0.3750 | 6 | 3 | 1 | +0.1248 |
| 7 | 0.3750 | 0.2500 | 0.3750 | 0.2812 | 4 | 3 | 0 | +0.0654 |
| 13 | 0.1250 | 0.1875 | 0.1250 | 0.1250 | 0 | 0 | 0 | +0.0000 |
| 19 | 0.1562 | 0.1250 | 0.1562 | 0.1562 | 0 | 0 | 0 | +0.0000 |

解释：held-out 复核支持 confidence-gain candidate 的方向，尤其在 1/4/7 dB 能额外接受一批 repair 并降低 pseudo final failure；但仍出现 2 个 accepted new error，位于 1 dB 和 4 dB。`samples/accepted_new_error_review.png` 已固化这两个样本的 original / M0 / refined / top-1 final / candidate final 对照。当前结论是“候选 gate 可继续收紧”，不是“候选 gate 已通过”。

#### 派生 test-like confidence-gain gate check

已先扩展 M0 export：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 384 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384
```

然后运行：

```bash
python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv
outputs/analysis/exp_s4_006_testlike_gate_check/summary.csv
outputs/analysis/exp_s4_006_testlike_gate_check/new_accepts.csv
outputs/analysis/exp_s4_006_testlike_gate_check/accepted_new_errors.csv
outputs/analysis/exp_s4_006_testlike_gate_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_gate_check/metadata.json
outputs/analysis/exp_s4_006_testlike_gate_check/exports/
outputs/analysis/exp_s4_006_testlike_gate_check/samples/
```

该复核不重训模型，只加载 `outputs/EXP-S4-006/checkpoints/best.pt`，在新导出的 `sample_000256.png` 到 `sample_000319.png` 上重新生成 refined、top-1 final 和 candidate final。该 split 没有参与 `EXP-S4-006` 的 refiner 训练、验证或此前 gate sweep；但仍属于同一个 COCO val subset 和同一个 pseudo-label 评价口径，因此只能作为 test-like 派生风险复核，不是最终 test 结论。

全局关键结果：

| Metric | Value |
|---|---:|
| Num images | 320 |
| Candidate accept rate | 0.7063 |
| Newly accepted by candidate | 26 |
| Candidate final failure | 0.4313 |
| Baseline top-1 final failure | 0.4719 |
| Candidate minus baseline failure | -0.0406 |
| Candidate final PSNR | 32.2374 dB |
| Candidate delta PSNR vs top-1 | +0.0814 dB |
| Candidate delta PSNR vs M0 | +0.4927 dB |
| Accepted repairs | 17 |
| Accepted new errors | 4 |

分 SNR 关键结果：

| SNR(dB) | M0 Failure | Refined Failure | Top-1 Failure | Candidate Failure | New Accept | Repair | New Error | Delta PSNR vs top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6562 | 0.5000 | 0.6562 | 0.5938 | 11 | 6 | 2 | +0.2120 |
| 4 | 0.5469 | 0.4375 | 0.5469 | 0.4375 | 11 | 8 | 1 | +0.1443 |
| 7 | 0.4688 | 0.4062 | 0.4688 | 0.4375 | 2 | 2 | 0 | +0.0238 |
| 13 | 0.3281 | 0.2500 | 0.3281 | 0.3281 | 2 | 1 | 1 | +0.0269 |
| 19 | 0.3594 | 0.3125 | 0.3594 | 0.3594 | 0 | 0 | 0 | +0.0000 |

解释：test-like split 上方向仍复现，candidate final failure 比 top-1 gate 低 `0.0406`，PSNR 高 `+0.0814` dB，并额外接受 17 个 pseudo-label repair；但 accepted new error 增至 4 个。`samples/accepted_new_error_review.png` 显示其中既有真实语义漂移风险，也有 AlexNet pseudo-label 本身较吵的样本。结论仍应保守：raw confidence-gain gate 是有收益但不安全的候选，不能写成最终 M3。

#### 派生 confidence-gain CLIP veto sweep

已运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_conf_gain_clip_veto.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/per_sample_with_clip.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/joint_policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/galleries/
```

该派生分析读取 validation 的 `per_sample_audit.csv` 和 held-out 的 `per_sample.csv`，用本地 OpenCLIP ViT-B/32 只计算 receiver-side `CLIP(M0, refined)`。Original pseudo-label 仍只用于离线评价 final failure，不参与 veto 决策。

全局关键结果：

| Policy | Validation failure | Held-out failure | Validation repair | Held-out repair | Validation new error | Held-out new error | Sum delta PSNR vs top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `top1_equal` | 0.3750 | 0.3250 | 0 | 0 | 0 | 0 | +0.0000 |
| `top1_equal_or_refined_conf_gain_ge_0p05` | 0.3188 | 0.2812 | 21 | 9 | 3 | 2 | +0.2159 |
| `top1_equal_or_conf_gain_0p05_clip_m0_refined_ge_0p98` | 0.3719 | 0.3187 | 1 | 1 | 0 | 0 | +0.0073 |

解释：`CLIP(M0, refined) >= 0.98` 是当前扫描中能在 validation 和 held-out 同时清零 accepted new error、且不完全退回 top-1 的最保守阈值。它挡掉了 raw confidence-gain 的 5 个 accepted new error，但也挡掉了 28/30 个 repair，因此收益几乎被压平。这个结果说明单一 CLIP image-image veto 可作安全参考，但不够作为最终 M3；下一步需要 SNR-calibrated threshold、classifier ensemble 或 receiver-side risk predictor。

#### 派生 SNR-calibrated confidence-gain CLIP veto

已运行：

```bash
python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/calibrated_schedules.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/independent_threshold_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/monotonic_schedule_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/galleries/
```

该派生分析只读取 `outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/per_sample_with_clip.csv`，不训练模型、不联网、不重算 CLIP。阈值只在 validation split 上选择，再到 held-out split 上做风险复核。扫描网格包含 `no_veto`、`0.90/0.92/0.94/0.96/0.97/0.98/0.985/0.99/0.995` 和 `top1_only`；monotonic schedule 额外约束 `threshold(1 dB) >= threshold(4 dB) >= threshold(7 dB) >= threshold(13 dB) >= threshold(19 dB)`，对应低 SNR 语义控制不弱于高 SNR。

校准得到的 schedule：

| Policy | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB |
|---|---:|---:|---:|---:|---:|
| `fixed_clip_ge_0p98` | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| `snr_independent_calibrated` | 0.96 | no_veto | 0.98 | no_veto | no_veto |
| `snr_monotonic_calibrated` | 0.98 | 0.98 | 0.98 | no_veto | no_veto |

全局关键结果：

| Split | Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | `top1_equal` | 0.3750 | +0.0000 | 31.9814 | +0.0000 | 0 | 0 |
| validation | `raw_conf_gain` | 0.3187 | -0.0563 | 32.0966 | +0.1153 | 21 | 3 |
| validation | `fixed_clip_ge_0p98` | 0.3719 | -0.0031 | 31.9852 | +0.0038 | 1 | 0 |
| validation | `snr_independent_calibrated` | 0.3438 | -0.0312 | 32.0346 | +0.0533 | 10 | 0 |
| validation | `snr_monotonic_calibrated` | 0.3719 | -0.0031 | 31.9873 | +0.0059 | 1 | 0 |
| heldout | `top1_equal` | 0.3250 | +0.0000 | 31.7602 | +0.0000 | 0 | 0 |
| heldout | `raw_conf_gain` | 0.2812 | -0.0437 | 31.8609 | +0.1007 | 9 | 2 |
| heldout | `fixed_clip_ge_0p98` | 0.3187 | -0.0062 | 31.7637 | +0.0035 | 1 | 0 |
| heldout | `snr_independent_calibrated` | 0.3063 | -0.0187 | 31.7985 | +0.0383 | 4 | 1 |
| heldout | `snr_monotonic_calibrated` | 0.3187 | -0.0062 | 31.7637 | +0.0035 | 1 | 0 |

解释：independent per-SNR calibration 在 validation 上比全局 `0.98` 更有用，能在 0 accepted new error 条件下保留 10 个 repair；但它选择了 4 dB `no_veto`，既违反当前 SNR-aware semantic-control 的单调纪律，也在 held-out 上漏出 1 个 accepted new error。monotonic schedule 在 held-out 上安全，但只保留 1 个 repair，基本退回全局 `0.98` 的保守状态。因此，单一 `CLIP(M0, refined)` 标量阈值即使按 SNR 校准，也不足以作为最终 M3；后续应优先做 classifier ensemble 或 receiver-side risk predictor。

#### 派生 receiver-side confidence-gain risk rule sweep

已运行：

```bash
python3 scripts/s5_sweep_conf_gain_risk_rules.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_conf_gain_risk_rules.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/rule_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/selected_rule.json
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/galleries/
```

该派生分析只读取已有 validation/held-out 的 `CLIP(M0, refined)` CSV 和 M0/refined top-k classifier CSV，不训练模型、不联网、不重算 CLIP。规则只使用 receiver-side 特征：`CLIP(M0, refined)`、M0/refined top-5 overlap、M0 top-1 在 refined top-5 中的 rank、M0 top-1 margin、refined top-1 margin。Original pseudo-label 仍只用于 validation 规则选择和离线 held-out 风险复核。

选中的规则：

```text
baseline top-1 agreement 仍直接接受 refined。
对 confidence-gain 新增接受样本：
  要求 clip_sim_m0_refined >= 0.90；
  无 top-5 overlap 最小要求；
  若 m0_top1_rank_in_refined_top5 <= 2
     且 m0_top1_margin <= 0.07
     且 refined_top1_margin >= 0.05，
     则触发 shadow veto，回退 M0。
```

直觉：当 M0 的 top-1 label 在 refined 中仍是非常靠前的候选，而 M0 自身 top-1 margin 很弱、refined top-1 margin 又明显变强时，这类“分类边界被推过头”的样本更容易是假修复或新错。该规则把这个 shadow pattern 作为风险信号。

全局关键结果：

| Split | Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | `top1_equal` | 0.3750 | +0.0000 | 31.9814 | +0.0000 | 0 | 0 |
| validation | `raw_conf_gain` | 0.3187 | -0.0563 | 32.0966 | +0.1153 | 21 | 3 |
| validation | `fixed_clip_ge_0p98` | 0.3719 | -0.0031 | 31.9852 | +0.0038 | 1 | 0 |
| validation | `selected_risk_rule` | 0.3156 | -0.0594 | 32.0767 | +0.0953 | 19 | 0 |
| heldout | `top1_equal` | 0.3250 | +0.0000 | 31.7602 | +0.0000 | 0 | 0 |
| heldout | `raw_conf_gain` | 0.2812 | -0.0437 | 31.8609 | +0.1007 | 9 | 2 |
| heldout | `fixed_clip_ge_0p98` | 0.3187 | -0.0062 | 31.7637 | +0.0035 | 1 | 0 |
| heldout | `selected_risk_rule` | 0.2812 | -0.0437 | 31.8350 | +0.0748 | 7 | 0 |

解释：这是目前最强的 gate 候选。相比 raw confidence-gain，它在 held-out 上保留同样的 final failure 改善，同时把 2 个 accepted new error 清零；相比全局 `CLIP >= 0.98`，它在 held-out 上从 1 个 repair 提升到 7 个 repair，并保留 `+0.0748` dB PSNR vs top-1 gate。`galleries/selected_risk_rule/heldout_vetoed_candidate_new_errors.png` 已固化被挡掉的两个 held-out 新错；`heldout_accepted_repairs.png` 固化 7 个被保留的 repair。

限制：该规则仍在 COCO pseudo-label validation 上选择，held-out 也只是同一 COCO val export 的未用样本段，不是最终 test split。它可以作为下一版 M3 gate 候选，但不能直接写成最终结论。

#### 派生 selected risk-rule final PNG materialization

已运行：

```bash
python3 scripts/s5_materialize_risk_rule_policy.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_materialize_risk_rule_policy.py
```

输出：

```text
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/per_sample.csv
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/summary.csv
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/REPORT.md
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/metadata.json
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/exports/{validation,heldout}/snr_XXdb/final/
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/samples/
```

该派生流程只读取 `outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_decisions.csv`，筛选 `policy == selected_risk_rule` 的 480 条决策，并按 `accept_refined` 从已有 M0/refined PNG 复制 final 输出；不训练、不联网、不重算 CLIP 或分类器。`summary.csv` 同时写入 top-1 gate 的 per-sample reference，方便核对 final failure 和 PSNR 增量。

关键结果：

| Split | Images | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error | Shadow Veto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 320 | 0.3156 | -0.0594 | 32.0767 | +0.0953 | 19 | 0 | 5 |
| heldout | 160 | 0.2812 | -0.0437 | 31.8350 | +0.0748 | 7 | 0 | 5 |

说明：这是 risk-rule sweep 的 artifact 固化，不是新实验结论；作用是把当前最强 M3 gate 候选变成可复查的 final PNG/CSV/report，为后续正式 split 或更大 held-out 复核做准备。

#### 派生 selected risk-rule classifier ensemble audit

已检查代理变量，当前环境存在 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=http://127.0.0.1:17890`。本次按项目流量规则清空代理变量，从 PyTorch/torchvision 官方 model zoo 直连下载缺失的 ResNet18 和 MobileNetV3-Small ImageNet 权重，规模约 `44.7MB + 9.83MB`；未下载数据集或 diffusion 模型。

已运行：

```bash
python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --device cuda:0 --allow-download
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --device cuda:0 --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/per_model_per_sample.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/model_summary.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/vote_summary.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/REPORT.md
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/galleries/
```

该派生流程固定使用已经 materialize 的 `selected_risk_rule` 决策，不重新搜索 gate，也不把 ensemble 用作 receiver-side decision。它只用 AlexNet、ResNet18、MobileNetV3-Small 各自的 original top-1 pseudo-label 重新评价 M0/refined/selected-final 的 failure、repair 和 accepted-new-error 风险。

按分类器拆分：

| Classifier | Split | Selected Failure | Delta vs M0 | Repair | New Error |
|---|---|---:|---:|---:|---:|
| AlexNet | validation | 0.3156 | -0.0594 | 19 | 0 |
| AlexNet | heldout | 0.2812 | -0.0437 | 7 | 0 |
| ResNet18 | validation | 0.3688 | -0.0187 | 24 | 18 |
| ResNet18 | heldout | 0.4000 | -0.0062 | 8 | 7 |
| MobileNetV3-Small | validation | 0.4313 | -0.0563 | 28 | 10 |
| MobileNetV3-Small | heldout | 0.3562 | +0.0125 | 7 | 9 |

按样本投票：

| Split | Images | Any new-error vote | Majority new-error vote | Any repair vote | Majority repair vote |
|---|---:|---:|---:|---:|---:|
| validation | 320 | 26 | 2 | 64 | 7 |
| heldout | 160 | 15 | 1 | 17 | 4 |

解释：这个结果把边界说清楚了。`selected_risk_rule` 在 AlexNet pseudo-label 口径下确实是当前最强 gate 候选，但并非跨语义模型安全：ResNet18 和 MobileNetV3-Small 都能发现额外 accepted-new-error 风险，且有 3 个样本得到多数票 new-error。它仍可作为候选，但后续必须加入 ensemble-aware veto、辅助语义 veto 或更正式 split 复核，不能把它直接写成最终 M3。

#### 派生 ensemble-risk 二级 veto sweep

已运行：

```bash
python3 scripts/s5_sweep_ensemble_risk_veto.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_ensemble_risk_veto.py
```

输出：

```text
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/rule_candidates.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/selected_rule.json
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/metadata.json
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/REPORT.md
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/galleries/
```

该派生流程固定使用 `selected_risk_rule` 的 480 条决策和 classifier ensemble audit 的离线投票标签，不重训、不联网、不下载、不重算分类器。搜索阶段只在 validation 上用 ensemble 多数票 new-error 作为风险约束；规则本身只使用接收端可得特征，包括 refined top-1 margin、refined 相对 M0 的 confidence gain、M0 top-1 margin 和 selected-risk-rule 的接受类型。

选中的二级 veto：

```text
在 selected_risk_rule 已接受 refined 的样本上：
  若它是 new_accept_vs_top1 且 refined_top1_margin <= 0.005，则额外 veto；
  若它是 top1-equal accept，且 refined_conf_gain_vs_m0 <= 0.05，
     且 m0_top1_margin >= 0.10，则额外 veto。
```

关键结果：

| Split | Extra Veto | Remaining Majority New Error | Remaining Any New Error | Remaining Majority Repair | Remaining Any Repair | Delta PSNR vs selected |
|---|---:|---:|---:|---:|---:|---:|
| validation | 96 | 0 | 16 | 5 | 40 | -0.1834 dB |
| heldout | 58 | 0 | 8 | 4 | 14 | -0.2538 dB |

按分类器复核最终 failure：

| Split | Classifier | Candidate Failure | Delta vs selected | Repair | New Error |
|---|---|---:|---:|---:|---:|
| validation | AlexNet | 0.3187 | +0.0031 | 18 | 0 |
| validation | ResNet18 | 0.3719 | +0.0031 | 14 | 9 |
| validation | MobileNetV3-Small | 0.4688 | +0.0375 | 13 | 7 |
| heldout | AlexNet | 0.2812 | +0.0000 | 7 | 0 |
| heldout | ResNet18 | 0.3938 | -0.0062 | 5 | 3 |
| heldout | MobileNetV3-Small | 0.3313 | -0.0250 | 7 | 5 |

解释：该规则能把 `selected_risk_rule` 暴露出的 validation/held-out 多数票 new-error 从 `2/1` 清到 `0/0`，说明 ensemble 暴露的高置信风险样本可以被简单 receiver-side 特征部分捕捉。但代价很明显：额外 veto 数达到 validation/held-out `96/58`，多数票 repair 只剩 `5/4`，且 any-new-error 仍有 `16/8`。因此它是收紧 gate 的风险分析结果，不是最终 M3；后续更合理的方向是把这个二级 veto 当作 conservative safety upper-bound，再训练/选择更细的 receiver-side risk predictor 或扩展正式 split。

#### 派生 receiver-side risk score sweep

已运行：

```bash
python3 scripts/s5_sweep_receiver_risk_score.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_receiver_risk_score.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/score_candidates.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/selected_score.json
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/metadata.json
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/REPORT.md
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/galleries/
```

该派生流程固定使用 `selected_risk_rule` 的 480 条决策和 classifier ensemble audit 的离线投票标签，不重训、不联网、不下载、不重算分类器。它扫描 12 个透明 receiver-side risk score 模板和 1930 个 validation 阈值候选，目标是验证是否能用更少 extra veto 替代上一节的保守二级 veto。score 特征只来自接收端可得的 AlexNet/CLIP/top-k 派生量，如 `CLIP(M0, refined)`、top-5 overlap、refined top-1 是否偏离 M0 top-k、confidence gain 和 margin。

repair-pref validation 目标选中的分数：

```text
risk_score = low_top5_overlap + refined_top1_not_in_m0_safe_rank + low_clip
threshold = 0.444446
```

关键结果：

| Split | Extra Veto | Remaining Majority New Error | Remaining Any New Error | Remaining Majority Repair | Remaining Any Repair | Delta PSNR vs selected |
|---|---:|---:|---:|---:|---:|---:|
| validation | 48 | 0 | 17 | 4 | 36 | -0.1396 dB |
| heldout | 26 | 1 | 9 | 2 | 8 | -0.1581 dB |

按分类器复核最终 failure：

| Split | Classifier | Candidate Failure | Delta vs selected | Repair | New Error |
|---|---|---:|---:|---:|---:|
| validation | AlexNet | 0.3594 | +0.0437 | 5 | 0 |
| validation | ResNet18 | 0.3719 | +0.0031 | 17 | 12 |
| validation | MobileNetV3-Small | 0.4469 | +0.0156 | 18 | 5 |
| heldout | AlexNet | 0.3187 | +0.0375 | 1 | 0 |
| heldout | ResNet18 | 0.3938 | -0.0062 | 6 | 4 |
| heldout | MobileNetV3-Small | 0.3625 | +0.0063 | 3 | 6 |

解释：该 score 在 validation 上用更少 extra veto 清零多数票 new-error，但 held-out 漏掉 1 个多数票 new-error，即 19 dB `sample_000031.png`；该样本的 M0/refined AlexNet top-1 同为 `komondor`，且 top-k/CLIP 接收端分数很低风险，说明浅层 receiver-side score 很难覆盖所有跨模型语义风险。进一步查看 `score_candidates.csv`，若要求 validation 和 held-out 同时清零多数票 new-error，repair-pref 最好的 score 模板需要额外 veto validation/held-out `143/81` 张，PSNR held-out 相对 `selected_risk_rule` 回吐 `-0.3511` dB，比上一节的保守二级 veto 还重。因此这一步是负/部分结果：少 veto risk score 目前不够稳，不能作为最终 M3。

#### 派生 test-like frozen risk-rule check

已运行：

```bash
python3 scripts/s5_apply_testlike_risk_rules.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_apply_testlike_risk_rules.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_risk_rule_check/per_sample_with_clip.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_decisions.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_by_snr.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_risk_rule_check/metadata.json
outputs/analysis/exp_s4_006_testlike_risk_rule_check/exports/
outputs/analysis/exp_s4_006_testlike_risk_rule_check/galleries/
```

该派生流程固定使用已经在 validation/held-out 阶段选出的 `selected_risk_rule` 和保守 ensemble-risk veto，不在 test-like split 上重新搜索阈值。它读取 `outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv`，重新计算本地 `CLIP(M0, refined)`，并把 final PNG materialize 到 `outputs/analysis/exp_s4_006_testlike_risk_rule_check/exports/`。本次不联网、不下载，CLIP 权重来自本地 cache。

全局关键结果：

| Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error | New Accept | Vetoed Raw New Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `top1_equal` | 0.4719 | +0.0000 | 32.1560 | +0.0000 | 0 | 0 | 0 | 4 |
| `raw_conf_gain` | 0.4313 | -0.0406 | 32.2374 | +0.0814 | 17 | 4 | 26 | 0 |
| `fixed_clip_ge_0p98` | 0.4688 | -0.0031 | 32.1636 | +0.0076 | 2 | 1 | 3 | 3 |
| `selected_risk_rule` | 0.4437 | -0.0281 | 32.1995 | +0.0434 | 10 | 1 | 15 | 3 |
| `selected_risk_rule_plus_ensemble_veto` | 0.4437 | -0.0281 | 32.0092 | -0.1468 | 10 | 1 | 14 | 3 |

解释：冻结的 `selected_risk_rule` 在 test-like split 上仍有迁移收益：相比 raw confidence-gain，它把 accepted new error 从 4 降到 1，同时保留 10 个 pseudo-label repair 和 `+0.0434` dB PSNR vs top-1 gate。但它没有清零风险。剩余 accepted new error 是 13 dB `sample_000312.png`：original/M0 AlexNet top-1 为 `ear`，refined 为 `seat belt`，`CLIP(M0, refined)=0.9950`，M0 top-1 在 refined top-k 中 rank=3，因此旧 shadow-margin 规则和保守 ensemble veto 都没有触发。视觉样例显示该 case 也包含明显 pseudo-label 噪声，因此它应被记录为辅助语义风险，而不是最终真值错误。

保守 ensemble-risk veto 在 test-like 上没有降低 accepted new error 或 final failure，却额外 veto 93 张、PSNR 相比 `selected_risk_rule` 回吐 `-0.1902` dB，说明它作为 safety upper-bound 太保守，不能直接作为第一版 M3。当前结论进一步支持：浅层 receiver-side 标量/规则已经接近瓶颈，下一步应转向更正式语义标签/ensemble test-like 审计，或在 residual CNN 训练/选择阶段加入 semantic-risk-aware 约束。

#### 派生 test-like classifier-ensemble audit

已运行：

```bash
python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/per_model_per_sample.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/model_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/vote_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/REPORT.md
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/metadata.json
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/galleries/
```

该派生流程固定使用 test-like `selected_risk_rule` 决策，不重新搜索 gate，也不把 ensemble 用作 receiver-side decision。脚本从 `policy_decisions.csv` 中只抽取 `selected_risk_rule` 320 行，使用本地 AlexNet、ResNet18、MobileNetV3-Small 权重分别以各自 original top-1 pseudo-label 评价 M0/refined/selected-final 的 failure、repair 和 accepted-new-error 风险。本次不联网、不下载，正式运行前 dry-run 确认 3 个分类器权重均已在本地 cache。

按分类器拆分：

| Classifier | Split | Selected Failure | Delta vs M0 | Repair | New Error |
|---|---|---:|---:|---:|---:|
| AlexNet | testlike | 0.4437 | -0.0281 | 10 | 1 |
| ResNet18 | testlike | 0.4344 | -0.0563 | 31 | 13 |
| MobileNetV3-Small | testlike | 0.5406 | -0.0719 | 32 | 9 |

按样本投票：

| Split | Images | Any new-error vote | Majority new-error vote | Any repair vote | Majority repair vote |
|---|---:|---:|---:|---:|---:|
| testlike | 320 | 23 | 0 | 58 | 12 |

按 SNR 的 any-new-error vote 为 1/4/7/13/19 dB `3/4/6/6/4`，majority new-error vote 全部为 0。23 个 any-new-error 都只有单模型投票，其中 ResNet18 13 个、MobileNetV3-Small 9 个、AlexNet 1 个；AlexNet 的唯一风险仍是 13 dB `sample_000312.png`。

解释：test-like ensemble 审计比 validation/held-out 的跨模型结果更温和：没有 majority-vote accepted new error，说明 frozen `selected_risk_rule` 在 test-like 上没有暴露出明显多数票语义灾难；但 any-model new-error 仍有 23 张，且 ResNet18/MobileNetV3-Small 下 selected accepted new error 分别为 13/9 个。因此它只能说明当前 rule 有一定迁移性和辅助 repair 信号，不能说明跨模型安全。下一步更应该补带标签 clean-correct 评估或把 semantic-risk-aware 约束进入 residual CNN 训练/选择，而不是继续只在 AlexNet/CLIP/top-k 标量上调阈值。

#### 派生 test-like COCO object CLIP clean-correct eval

该派生流程不训练、不联网、不下载，读取 `outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_decisions.csv`、`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/source_manifest.json`、`data/coco/annotations/instances_val2017.json` 和本地 `outputs/cache/open_clip/ViT-B-32.pt`。它先用 COCO instance 面积得到 dominant object label，再用 OpenCLIP ViT-B/32 对 80 个 COCO object prompt 做 zero-shot 分类；只有 dominant label 面积占比满足阈值，且 original 的 CLIP top-1 与 dominant label 一致、prob/margin 过阈值的样本进入辅助 clean-correct 子集。

运行命令：

```bash
python3 scripts/s5_coco_object_clip_clean_eval.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_coco_object_clip_clean_eval.py --device cuda:0
```

输出路径：`outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/`

clean-correct 总表：

| Policy | Rows | Final Failure GT | Delta vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair GT | New Error GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top1_equal | 135 | 0.0815 | +0.0000 | 31.7925 | +0.0000 | 1 | 2 |
| raw_conf_gain | 135 | 0.0815 | +0.0000 | 31.8457 | +0.0533 | 1 | 2 |
| fixed_clip_ge_0p98 | 135 | 0.0815 | +0.0000 | 31.8042 | +0.0117 | 1 | 2 |
| selected_risk_rule | 135 | 0.0815 | +0.0000 | 31.8182 | +0.0257 | 1 | 2 |
| selected_risk_rule_plus_ensemble_veto | 135 | 0.0741 | -0.0074 | 31.6197 | -0.1727 | 0 | 0 |

解释：64 个 test-like 原图中有 55 个满足 dominant object 面积规则，其中 27 个 original 被 CLIP 判为 clean-correct，形成每个 policy 135 行统计。该辅助 GT-like 口径下，`selected_risk_rule` 没有比 top-1 gate 降低 final failure，也没有减少 GT-like accepted new error，只提供小幅 PSNR 增益；保守 ensemble veto 可把 GT-like new error 清零并稍降 final failure，但也清掉 repair 且 PSNR 低于 top-1。这个结果进一步确认当前浅层 gate 的语义保护和 restoration 收益存在硬 tradeoff。它比 ImageNet pseudo-label 更贴 COCO 物体标注，但仍依赖 CLIP zero-shot 和 dominant-object 假设，不能包装成最终监督真值指标。

#### 复现备注

`EXP-S4-006` 本体不联网、不下载模型或数据，只读取已有正式 M0 export 和本地 AlexNet/LPIPS 权重。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。本体 `summary.csv` 有 5 个 SNR 汇总行，`per_sample.csv` 有 320 个 eval 样本行，`train_history.csv` 有 40 个 epoch 行。

#### 下一步

围绕 `EXP-S4-006` 继续收敛 detector：`selected_risk_rule` final PNG、classifier ensemble audit、ensemble-risk 二级 veto sweep、receiver-side risk score sweep、raw confidence-gain test-like 复核、frozen risk-rule test-like 复核、test-like classifier-ensemble audit 和 COCO object CLIP clean-correct 辅助诊断都已完成。test-like 证据说明 raw confidence-gain 有 PSNR/repair 收益但会引入 accepted new error；`selected_risk_rule` 可挡掉其中 3/4 个 AlexNet new error，且在 test-like ensemble 下没有 majority-vote new error，但仍有 23 个 any-model accepted new-error vote；在 COCO-object clean-correct 口径下它仍有 2 个 GT-like new error。下一步应停止只在同一套浅层 receiver-side 标量上拧阈值，优先做真正带监督标签的 clean-correct 评估，或在 residual CNN 训练阶段加入 semantic-risk-aware 约束。当前证据显示 raw confidence-gain、全局 CLIP veto、SNR-calibrated scalar CLIP veto、当前 AlexNet-tuned selected rule、保守 ensemble-risk veto 和少 veto risk score 都不能直接定为第一版 M3。

### EXP-S4-007：SNR-conditioned pixel residual diffusion pilot

- 日期：2026-07-06
- 项目版本：`4f4eefb5f08096e5efdd57d6019b97683ea7648b`
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free residual diffusion design probe
- 方法：SNRConditionedPixelResidualDiffusionPilot
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：
  - 输入 export：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
  - train：`sample_000032.png` 到 `sample_000111.png`，每个 SNR 80 张，共 400 对 M0/original
  - eval：`sample_000192.png` 到 `sample_000207.png`，每个 SNR 16 张，共 80 对 M0/original
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；sampling seed 1234
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- diffusion checkpoint：`outputs/EXP-S4-007/checkpoints/best.pt`
- config：`outputs/EXP-S4-007/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_diffusion_pilot.py --device cuda:0`
- 关键源码：`scripts/s5_residual_diffusion_pilot.py`, `scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-007/`
- 状态：完成；负结果；不是最终 M2/M3/Ours

#### 方法设置

- 模型：小型 SNR-conditioned pixel residual DDPM，参数量 77,187
- 输入：noisy normalized residual + M0 reconstruction + SNR map + timestep map，共 8 通道
- 目标：学习 `(original - M0) / residual_gate` 后 clamp 到 `[-1, 1]` 的 residual
- diffusion：20 timesteps，linear beta `0.0001 -> 0.02`，epsilon prediction，deterministic DDIM sampling 20 steps
- residual gate：1/4/7/13/19 dB 使用 `0.12/0.10/0.08/0.05/0.04`
- 训练：20 epoch，batch size 16，128x128 random crop，epsilon loss + 0.1 x0 loss
- semantic failure handling：若 `c(refined) == c(M0)` 则接受，否则 fallback 到 M0；detector 不看原图

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。`Refined Failure` 是 refined 相对原图 pseudo top-1 的 failure；`M3 Failure` 是 top-1 agreement fallback 后最终输出的 failure。

| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Refined Delta | M3 PSNR | M3 Delta | M0 LPIPS | Refined LPIPS | M3 LPIPS | M0 Failure | Refined Failure | M3 Failure | Accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 28.6189 | 21.4555 | -7.1634 | 27.2033 | -1.4156 | 0.1893 | 0.5139 | 0.2472 | 0.6250 | 0.8750 | 0.6250 | 0.2500 |
| 4 | 0.10 | 30.6216 | 23.1373 | -7.4843 | 28.9598 | -1.6618 | 0.1138 | 0.4236 | 0.1785 | 0.5625 | 0.8125 | 0.5625 | 0.2500 |
| 7 | 0.08 | 32.0814 | 24.9932 | -7.0882 | 29.4795 | -2.6019 | 0.0673 | 0.3338 | 0.1641 | 0.4375 | 0.7500 | 0.4375 | 0.3750 |
| 13 | 0.05 | 33.6698 | 28.2494 | -5.4204 | 31.5131 | -2.1567 | 0.0335 | 0.1999 | 0.1142 | 0.2500 | 0.6250 | 0.2500 | 0.4375 |
| 19 | 0.04 | 34.1760 | 29.7543 | -4.4217 | 32.0758 | -2.1002 | 0.0284 | 0.1489 | 0.0969 | 0.1875 | 0.5000 | 0.1875 | 0.5625 |

#### 结果总结

该实验回答了“是否只要把 diffusion 挪到像素 residual 域就会变好”：当前朴素设计不成立。训练 loss 确实下降，eval epsilon loss 最低出现在 epoch 14，但最终 DDIM sampling 得到的 residual 噪声化很强，refined PSNR 在所有 SNR 上显著低于 M0，下降范围为 `-4.4217` 到 `-7.4843` dB；LPIPS 也全部变差。

top-1 agreement fallback 仍有语义保护作用：同一冻结 AlexNet 口径下，M3 final failure 在每个 SNR 上都回到 M0 failure。但这不是有效增强，因为 M3 final PSNR 仍比 M0 低 `-1.4156/-1.6618/-2.6019/-2.1567/-2.1002` dB，M3 LPIPS 也全部高于 M0。

#### Semantic drift 观察

Pure refined 的 pseudo failure 明显高于 M0：1/4/7/13/19 dB 分别为 `0.8750/0.8125/0.7500/0.6250/0.5000`。`refined_refinement_drift` 也很高，分别为 `0.7500/0.7500/0.6250/0.5625/0.4375`，说明随机残差采样即使有 M0/SNR/timestep conditioning，也容易改变冻结分类器 top-1。当前 gate 把这些变化大多回退，但代价是 final 图像仍被 accepted refined 样本拖低。

#### 失败案例和样例

样例拼图位于：

- `outputs/EXP-S4-007/samples/snr_01db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_04db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_07db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_13db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_19db_original_m0_refined_m3final.png`

逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-007/per_sample.csv`。

#### 复现备注

本实验不联网、不下载模型或数据，只读取已有正式 M0 export 和本地 AlexNet/LPIPS 权重。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`、`download_note: No model or data download is required`、`git_dirty_state: clean`。`summary.csv` 有 5 个 SNR 汇总行，`per_sample.csv` 有 80 个 eval 样本行，`train_history.csv` 有 20 个 epoch 行。

#### 下一步

不要把该 naive residual DDPM 作为正向 M2/M3 路线。若继续研究 diffusion，应改成 restoration-aware 的条件短链：从 M0 或 residual CNN 输出附近初始化，只做小幅 residual correction；或以 `EXP-S4-006` residual CNN 作为 mean / teacher，再训练低噪声 conditional diffusion。第一版论文闭环仍应优先收敛 `EXP-S4-006` 的 residual CNN + semantic gate。

### ANALYSIS-S6-004：Minimal Closure Report with Shrink M3

- 日期：2026-07-07
- 项目版本：`371833e` + uncommitted report script/config at run time
- 阶段：S6 minimal closure derived analysis
- 方法：MinimalClosureReportWithHeldoutShrinkM3
- 数据集：COCO2017 `val2017` subset outputs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB；M1 negative reference 仅覆盖 `[1, 7, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 输入：
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/metrics.json`
  - `outputs/EXP-S2-002/metrics.json`
  - `outputs/EXP-S4-006/summary.csv`
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；派生汇总，不训练、不推理、不下载

#### 核心表

| Method | Role | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | Status |
|---|---|---|---:|---:|---:|---|
| M0-DeepJSCC-HR | baseline | formal_coco512 | 0.0000 | N/A | N/A | usable M0 baseline |
| M1-BlindDiffusion-SDImg2Img | negative reference | exp_s2_002_16img_per_snr | -14.7485 | +0.3877 | N/A | failed due quality and semantic drift |
| M2-SNRConditionedPixelResidualRestoration | positive restoration anchor | exp_s4_006_eval | +0.7235 | -0.0274 | 0.3344 | positive quality, needs semantic handling |
| M3-ResidualRestorationTop1Fallback | conservative first M3 | exp_s4_006_eval | +0.4011 | -0.0104 | 0.3750 | safe conservative closure on pseudo-label metric |
| M3-ResidualRestorationTop1ShrinkFallback | stronger conservative M3 candidate | validation selected / frozen held-out/test-like | +0.4584 | -0.0153 | 0.3750 | best conservative M3 candidate so far; held-out/test-like PSNR delta +0.4689/+0.4552 and new error 0/0 |
| M3-SelectedRiskRuleCandidate | test-like candidate gate | testlike_policy | N/A | N/A | 0.4437 | not final; leaves AlexNet/GT-like risk |

#### 结果总结

该汇总把当前第一版闭环口径固定下来：M1 使用 SD img2img 空 prompt 是明确负结果，只作为 blind diffusion reference；M2 应写成 SNR-conditioned pixel residual restoration，是当前正向质量提升来源；M3 的保守第一版采用 top-1 semantic fallback，可以在 `EXP-S4-006` pseudo-label 口径下保证 final failure 不高于 M0，同时保留平均 `+0.4011` dB PSNR 和 `-0.0104` LPIPS 收益。

刷新后的报告新增 `M3-ResidualRestorationTop1ShrinkFallback`：validation-only schedule 选择 `1 dB alpha=0.5`、其余 SNR `alpha=0.75`，validation 平均 PSNR delta 为 `+0.4584` dB，LPIPS delta 为 `-0.0153`；冻结到 held-out 后，平均 PSNR delta 为 `+0.4689` dB，比 full-strength top-1 fallback 高 `+0.0236` dB，accepted new error 为 0；冻结到 test-like 后，平均 PSNR delta 为 `+0.4552` dB，比 full-strength top-1 fallback 高 `+0.0439` dB，accepted new error 为 0。因此它是当前最强保守 M3 候选，但仍是 pseudo-label/held-out/test-like 证据，不是监督标签安全证明。

`selected_risk_rule` 继续作为候选/消融：test-like AlexNet 口径下有 1 个 accepted new error，COCO-object clean-correct 口径下仍有 2 个 GT-like new error；保守 ensemble veto 可清 COCO-object new error，但 PSNR 相比 top-1 为 `-0.1727` dB，过于保守。

#### 复现备注

该流程只读已有本地 outputs，不重新运行模型或分类器。正式运行时清空代理变量，metadata 中记录 `proxy_environment_present: []`。生成文件包括 `REPORT.md`、6 个 CSV 和 4 张 figure：

- `outputs/analysis/minimal_closure_report/REPORT.md`
- `outputs/analysis/minimal_closure_report/method_closure_summary.csv`
- `outputs/analysis/minimal_closure_report/residual_per_snr_quality_semantics.csv`
- `outputs/analysis/minimal_closure_report/blind_diffusion_negative_reference.csv`
- `outputs/analysis/minimal_closure_report/residual_shrink_policy_tradeoff.csv`
- `outputs/analysis/minimal_closure_report/testlike_policy_tradeoff.csv`
- `outputs/analysis/minimal_closure_report/coco_object_clean_correct_tradeoff.csv`
- `outputs/analysis/minimal_closure_report/figures/`

#### 下一步

围绕这个闭环继续推进：优先把 residual strength / alpha 选择前移到 semantic-risk-aware residual training 或 validation model selection；若继续研究 diffusion，只做以 M2/refined/M0 附近初始化的短链 conditional residual correction。

### ANALYSIS-S6-005：EXP-S4-006 Held-Out Frozen Residual Shrink Schedule Check

- 日期：2026-07-07
- 项目版本：`371833e` + uncommitted generic split script/config at run time
- 阶段：S6 held-out derived analysis
- 方法：FrozenHeldoutResidualShrinkScheduleCheck
- 数据集：COCO2017 `val2017` held-out `sample_000000`-`sample_000031`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_residual_shrink_schedule.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_apply_residual_shrink_schedule.py`, `scripts/s6_residual_shrink_selection.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json`
  - `outputs/analysis/exp_s4_006_heldout_gate_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_heldout_gate_check/exports/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
- 输出路径：`outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/`
- 状态：完成；frozen schedule held-out 复核，不训练、不下载、不调参

#### 指标

| Policy | Delta PSNR vs M0 | Delta LPIPS vs M0 | Final Failure | Accept | Repair | Accepted New Error |
|---|---:|---:|---:|---:|---:|---:|
| top1_full_strength | +0.4454 | -0.0113 | 0.3250 | 0.6687 | 0 | 0 |
| validation_top1_shrink_schedule | +0.4689 | -0.0150 | 0.3250 | 0.7625 | 0 | 0 |
| always_full_strength | +0.6853 | -0.0223 | 0.2250 | 1.0000 | 26 | 10 |
| validation_always_m0_failure_constrained_schedule | +0.5292 | -0.0217 | 0.2375 | 0.8000 | 17 | 3 |

#### 结果总结

Validation 选出的 top-1 shrink schedule 在 held-out split 继续成立：平均 PSNR delta 从 full-strength top-1 fallback 的 `+0.4454` dB 提升到 `+0.4689` dB，LPIPS delta 从 `-0.0113` 改到 `-0.0150`，pseudo final failure 仍等于 M0，accepted new error 为 0。always-accept 两条路线仍有 10/3 个 accepted new error，不能作为最终 M3。

#### 复现备注

该流程只读取已有 held-out refined PNG 和 frozen validation schedule，不重新训练 residual refiner，不运行 diffusion，不下载模型或数据。metadata 中记录 `proxy_environment_present: []` 和 `split_name: held-out`。

#### 下一步

把 validation、held-out、test-like 三段 shrink 证据并入 minimal closure report；后续把 alpha/残差幅度约束前移到 residual CNN 训练或 validation model selection。

### ANALYSIS-S6-006：Residual Shrink M3 Artifact Gallery

- 日期：2026-07-07
- 项目版本：`c19cc0f` + uncommitted artifact-gallery script/config at run time
- 阶段：S6 derived artifact / failure-case organization
- 方法：ResidualShrinkM3ArtifactGallery
- 数据集：COCO2017 `val2017` validation、held-out、test-like residual shrink outputs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_residual_shrink_artifact_gallery_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_residual_shrink_gallery.py
python3 scripts/s6_make_residual_shrink_gallery.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_residual_shrink_gallery.py --overwrite
```

- 关键源码：`scripts/s6_make_residual_shrink_gallery.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv`
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv`
- 输出路径：`outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/`
- 状态：完成；只整理已有 CSV/PNG，不训练、不运行 diffusion、不重算分类器、不下载、不调参

#### 指标

| Split | M3 Delta PSNR | M3 Delta LPIPS | M3 New Error | Safe Accept | Protective Reject | Rejected Good | Always Full New Error | Always Constrained New Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | +0.4584 | -0.0153 | 0 | 183 | 17 | 34 | 28 | 19 |
| held-out | +0.4689 | -0.0150 | 0 | 102 | 6 | 19 | 10 | 3 |
| test-like | +0.4552 | -0.0152 | 0 | 156 | 13 | 44 | 25 | 12 |

#### 结果总结

该派生 artifact 把 validation、held-out、test-like 三段 residual shrink 证据合并到一个可引用目录。`M3-ResidualRestorationTop1ShrinkFallback` 在三段上 accepted new error 均为 0，同时提供 safe accept、protective reject、rejected good candidate 和 unsafe always-accept new-error 的样例 sheet。它进一步明确了当前 M3 的性质：保守质量增强，而不是冒险追求 repair 数。

Always-accept 仍作为负对照：full strength 在 validation/held-out/test-like 上分别有 28/10/25 个 accepted new error；validation-constrained always-accept 仍有 19/3/12 个 accepted new error，不能写成最终 M3。

#### 复现备注

正式运行时清空代理变量，metadata 中记录 `proxy_environment_present: []`。输出包括：

- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/REPORT.md`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/policy_summary.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/case_counts.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/case_index.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/samples/`

#### 下一步

把这些样例用于第一版 failure-case / reliability 小节；方法侧继续把 residual alpha/幅度控制前移到 residual CNN 训练、validation model selection 或短链 conditional residual diffusion。

### ANALYSIS-S6-007：Adaptive Residual Alpha Policy

- 日期：2026-07-07
- 项目版本：`fbcfe72` + uncommitted adaptive-alpha script/config at run time
- 阶段：S6 derived policy / residual strength control
- 方法：AdaptiveResidualAlphaPolicy
- 数据集：COCO2017 `val2017` validation、held-out、test-like residual alpha candidates
- 数据 split / 样本 ID：
  - validation：`sample_000192`-`sample_000255`，64 images/SNR
  - held-out：`sample_000000`-`sample_000031`，32 images/SNR
  - test-like：`sample_000256`-`sample_000319`，64 images/SNR
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_adaptive_residual_alpha_policy_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_adaptive_residual_alpha_policy.py
python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --device cuda:0
```

- 关键源码：`scripts/s6_apply_adaptive_residual_alpha_policy.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv`
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/candidates/`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/candidates/`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/candidates/`
- 输出路径：`outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/`
- 状态：完成；只读取已有 alpha candidate PNG、本地 AlexNet 和 LPIPS 权重，不训练、不运行 diffusion、不重新生成 residual、不下载、不在 held-out/test-like 上调参

#### 指标

| Split | Policy | Delta PSNR | Delta LPIPS | Failure Delta | Accept Rate | Mean Alpha | Repair | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | top1_full_strength | +0.4011 | -0.0104 | +0.0000 | 0.6406 | 1.0000 | 0 | 0 | 45 |
| validation | fixed_validation_top1_shrink_schedule | +0.4584 | -0.0153 | +0.0000 | 0.7438 | 0.7111 | 0 | 0 | 45 |
| validation | adaptive_max_top1_consistent_alpha | +0.5584 | -0.0189 | +0.0000 | 0.9062 | 0.8457 | 0 | 0 | 45 |
| validation | always_full_strength | +0.7235 | -0.0274 | -0.0406 | 1.0000 | 1.0000 | 41 | 28 | 4 |
| held-out | top1_full_strength | +0.4454 | -0.0113 | +0.0000 | 0.6687 | 1.0000 | 0 | 0 | 31 |
| held-out | fixed_validation_top1_shrink_schedule | +0.4689 | -0.0150 | +0.0000 | 0.7625 | 0.7131 | 0 | 0 | 31 |
| held-out | adaptive_max_top1_consistent_alpha | +0.5664 | -0.0174 | +0.0000 | 0.9187 | 0.8605 | 0 | 0 | 31 |
| held-out | always_full_strength | +0.6853 | -0.0223 | -0.1000 | 1.0000 | 1.0000 | 26 | 10 | 5 |
| test-like | top1_full_strength | +0.4113 | -0.0116 | +0.0000 | 0.6250 | 1.0000 | 0 | 0 | 70 |
| test-like | fixed_validation_top1_shrink_schedule | +0.4552 | -0.0152 | +0.0000 | 0.7063 | 0.7102 | 0 | 0 | 70 |
| test-like | adaptive_max_top1_consistent_alpha | +0.5691 | -0.0201 | +0.0000 | 0.8906 | 0.8482 | 0 | 0 | 70 |
| test-like | always_full_strength | +0.7180 | -0.0270 | -0.0906 | 1.0000 | 1.0000 | 54 | 25 | 16 |

Adaptive policy 的 per-SNR PSNR delta：

| Split | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB |
|---|---:|---:|---:|---:|---:|
| validation | +0.6850 | +0.5843 | +0.4802 | +0.5129 | +0.5294 |
| held-out | +0.6843 | +0.6055 | +0.4704 | +0.5143 | +0.5573 |
| test-like | +0.7739 | +0.6078 | +0.4638 | +0.4754 | +0.5246 |

#### 结果总结

`adaptive_max_top1_consistent_alpha` 在每个样本上从 `alpha=1.0/0.75/0.5/0.25` 中选择最大且 candidate top-1 与 M0 top-1 一致的 residual 强度，否则回退 M0。该规则不使用原图，只使用接收端已有 M0、alpha candidates 和冻结 AlexNet 的 top-1 一致性。

它在 validation/held-out/test-like 上把 PSNR delta 提升到 `+0.5584/+0.5664/+0.5691` dB，明显强于固定 per-SNR shrink schedule 的 `+0.4584/+0.4689/+0.4552` dB，并且在同一 AlexNet pseudo-label 口径下 accepted new error 保持 `0/0/0`。always-accept 仍然质量更高但有 `28/10/25` 个 new error，继续作为负对照。

需要特别记录的是：adaptive policy 没有产生 repair，且 missed repair 为 `45/31/70`。因此它是当前最强的保守质量增强候选，不是语义修复方法。下一步应把这种 per-sample alpha 选择前移到 residual CNN 的训练目标、validation model selection 或短链 conditional residual diffusion 的幅度控制里，而不是继续只做离线后验选择。

#### 复现备注

正式运行时清空代理变量，未下载模型或数据。metadata 记录 `proxy_environment_present: []`；由于脚本/config 在运行时尚未提交，`git_dirty_state` 为 `dirty`。输出包括：

- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/REPORT.md`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/metadata.json`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/samples/`

#### 下一步

把 adaptive alpha policy 写入 M3 方法候选：短期可作为 `M3-AdaptiveResidualAlphaTop1Fallback` 的派生方案；中期应训练一个 receiver-side alpha/risk predictor 或在 residual CNN 中加入 semantic-risk-aware amplitude loss，使方法不依赖离线枚举 alpha candidates。

### ANALYSIS-S6-008：Minimal Closure Report With Adaptive Alpha M3

- 日期：2026-07-07
- 项目版本：`bcfc1f1` + uncommitted closure-report script/config at run time
- 阶段：S6 derived closure report
- 方法：MinimalClosureReportWithAdaptiveAlphaM3
- 数据集：COCO2017 `val2017` existing outputs and analysis CSVs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_minimal_closure_report.py
python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 新增输入：`outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；只读取已有 metrics/CSV，不训练、不运行 diffusion、不重算分类器、不下载

#### 指标

| Method | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | New Error | Status |
|---|---|---:|---:|---:|---:|---|
| M3-ResidualRestorationTop1Fallback | validation | +0.4011 | -0.0104 | 0.3750 | 0 | conservative first closure |
| M3-ResidualRestorationTop1ShrinkFallback | validation / held-out / test-like | +0.4584 / +0.4689 / +0.4552 | -0.0153 / -0.0150 / -0.0152 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | fixed schedule candidate |
| M3-AdaptiveResidualAlphaTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5664 / +0.5691 | -0.0189 / -0.0174 / -0.0201 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | strongest conservative candidate |

#### 结果总结

本轮刷新把 `ANALYSIS-S6-007` 的 adaptive alpha policy 并入最小闭环报告。`outputs/analysis/minimal_closure_report/REPORT.md` 现在明确区分：

- `M3-ResidualRestorationTop1Fallback`：保守第一版闭环；
- `M3-ResidualRestorationTop1ShrinkFallback`：固定 per-SNR schedule 消融/备选；
- `M3-AdaptiveResidualAlphaTop1Fallback`：当前最强保守质量增强候选；
- `M3-SelectedRiskRuleCandidate`：有 repair 但仍有 new-error 风险，不能作为最终安全方法。

新增输出包括 `adaptive_residual_alpha_policy_tradeoff.csv` 和 `figures/adaptive_residual_alpha_policy_tradeoff.png`。报告仍保留 caveat：adaptive alpha 不使用原图，但还是后验枚举 alpha candidates 的 receiver-side policy，还不是带 learned amplitude/risk control 的 residual CNN。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。由于脚本/config 在运行时尚未提交，metadata 中 `git_dirty_state` 为 `dirty`。

#### 下一步

论文口径上可把 adaptive alpha 作为当前 M3 主候选；方法侧下一步应把该 per-sample alpha 选择前移到训练/模型选择流程，例如训练 receiver-side alpha predictor、把 residual amplitude loss 加入 residual CNN，或设计从 M0/refined 附近初始化的短链 conditional residual diffusion。

### ANALYSIS-S6-009：Two-Stage Residual Alpha Policy

- 日期：2026-07-07
- 项目版本：`9cacff5` + local script/config at run time
- 阶段：S6 deployability ablation
- 方法：TwoStageResidualAlphaPolicy
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha decisions
- 数据 split：validation `320` 行、held-out `160` 行、test-like `320` 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_two_stage_residual_alpha_policy_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_two_stage_residual_alpha_policy.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --device cuda:0
```

- 关键源码：`scripts/s6_apply_two_stage_residual_alpha_policy.py`
- 输入：
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
- 输出路径：`outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/`
- 状态：完成；只读取已有 adaptive alpha 决策表和 final 图，不训练、不运行 diffusion、不重算分类器、不下载；LPIPS 省略以避免外部权重加载

#### 策略

```text
full_then_fixed_schedule:
  first try top1_full_strength
  if alpha=1.0 candidate top-1 equals M0 top-1, accept full strength
  otherwise use fixed_validation_top1_shrink_schedule with the same top-1 gate
  otherwise fallback to M0
```

#### 指标

| Split | Delta PSNR | Final Failure Delta | Accept | Full Accept | Fallback Stage | Fallback Accept When Used | New Error | Missed Repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | +0.4831 | +0.0000 | 0.7469 | 0.6406 | 0.3594 | 0.2957 | 0 | 45 |
| held-out | +0.5009 | +0.0000 | 0.7625 | 0.6687 | 0.3312 | 0.2830 | 0 | 31 |
| test-like | +0.4875 | +0.0000 | 0.7250 | 0.6250 | 0.3750 | 0.2667 | 0 | 70 |

对比：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| `top1_full_strength` | +0.4011 | +0.4454 | +0.4113 | 0/0/0 |
| `fixed_validation_top1_shrink_schedule` | +0.4584 | +0.4689 | +0.4552 | 0/0/0 |
| `full_then_fixed_schedule` | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Two-stage policy 用最多两次 candidate 检查，质量上稳定优于 fixed schedule，但没有追上 exhaustive adaptive alpha。它的价值是证明可以把“残差强度控制”向更少候选、更接近接收端部署的策略压缩，同时保持同一 AlexNet pseudo-label 口径下 accepted new error 为 0。

该策略仍没有 repair，missed repair 仍为 `45/31/70`，因此仍是保守质量增强，不是语义修复。下一步如果继续这条线，应训练 receiver-side alpha predictor 或把 alpha/risk 控制并入 residual CNN，而不是继续增加后验枚举规则。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。本轮曾有一次 ad hoc 指标探针误触发 LPIPS/AlexNet 临时权重下载；下载进程已停止，`/tmp/alpha_twostage_cache` 已删除，未使用任何该探针结果。正式脚本默认不加载 LPIPS，避免再次触发外部权重加载。

### ANALYSIS-S6-010：Minimal Closure Report With Two-Stage Alpha Ablation

- 日期：2026-07-07
- 项目版本：`9cacff5` + local script/config at run time
- 阶段：S6 derived closure report
- 方法：MinimalClosureReportWithTwoStageAlphaAblation
- 数据集：COCO2017 `val2017` existing outputs and analysis CSVs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_minimal_closure_report.py scripts/s6_apply_two_stage_residual_alpha_policy.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 新增输入：`outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；只读取已有 metrics/CSV，不训练、不运行 diffusion、不重算分类器、不下载

#### 指标

| Method | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | New Error | Status |
|---|---|---:|---:|---:|---:|---|
| M3-ResidualRestorationTop1Fallback | validation | +0.4011 | -0.0104 | 0.3750 | 0 | conservative first closure |
| M3-ResidualRestorationTop1ShrinkFallback | validation / held-out / test-like | +0.4584 / +0.4689 / +0.4552 | -0.0153 / -0.0150 / -0.0152 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | fixed schedule candidate |
| M3-TwoStageResidualAlphaTop1Fallback | validation / held-out / test-like | +0.4831 / +0.5009 / +0.4875 | N/A | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | deployability ablation |
| M3-AdaptiveResidualAlphaTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5664 / +0.5691 | -0.0189 / -0.0174 / -0.0201 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | strongest conservative candidate |

#### 结果总结

本轮刷新把 `ANALYSIS-S6-009` 的 two-stage alpha 消融并入最小闭环报告。`outputs/analysis/minimal_closure_report/REPORT.md` 现在明确区分：

- `M3-AdaptiveResidualAlphaTop1Fallback`：当前最强保守质量增强候选；
- `M3-TwoStageResidualAlphaTop1Fallback`：少候选检查的部署折中，质量高于 fixed schedule 但低于 exhaustive adaptive alpha；
- `M3-ResidualRestorationTop1ShrinkFallback`：固定 per-SNR schedule 消融/备选；
- `M3-SelectedRiskRuleCandidate`：有 repair 但仍有 new-error 风险，不能作为最终安全方法。

新增输出包括 `two_stage_residual_alpha_policy_tradeoff.csv`。报告仍保留 caveat：two-stage alpha 的 LPIPS 被刻意省略，不能把空 LPIPS 项与其他策略的 LPIPS 数值横向比较。

#### 下一步

方法侧下一步不应继续堆后验策略，而应把 alpha 选择变成可学习或训练期约束：训练 receiver-side alpha/risk predictor，或在 residual CNN/短链 conditional residual diffusion 中加入 semantic-risk-aware residual amplitude 控制。

### ANALYSIS-S6-011：Receiver Alpha Predictor

- 日期：2026-07-09
- 项目版本：`4a466e8` + local script/config at run time
- 阶段：S6 learned deployability pilot
- 方法：ReceiverAlphaPredictor
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha decisions and candidate PNGs
- 数据 split：validation `320` 行用于训练，held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_receiver_alpha_predictor_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_receiver_alpha_predictor.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --device cuda:0
```

- 关键源码：`scripts/s6_train_receiver_alpha_predictor.py`
- 输入：
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - validation/held-out/test-like residual alpha candidate PNG roots
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_receiver_alpha_predictor/`
- 状态：完成；validation-only 训练小型 tabular predictor，不训练图像模型、不运行 diffusion、不重新生成 residual、不下载；LPIPS 省略以避免外部权重加载

#### 方法

Predictor 使用接收端可见特征：

- SNR 数值和 SNR one-hot；
- M0 top-1 confidence；
- full-strength candidate top-1 confidence、confidence delta/ratio、是否与 M0 top-1 一致；
- M0 图像均值/方差/edge proxy；
- full-strength residual 的 MAE/RMSE/P95/max/signed mean。

训练目标是 validation 上 `adaptive_max_top1_consistent_alpha` 的 `selected_alpha` pseudo target。评估时只对预测 alpha 的候选图运行冻结 AlexNet；若 candidate top-1 与 M0 top-1 不一致，则回退 M0。

#### 指标

| Split | Delta PSNR | Failure Delta | Accept | Target Alpha Acc | Pred Alpha <= Oracle | New Error | Missed Repair |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | +0.5584 | +0.0000 | 0.9062 | 1.0000 | 1.0000 | 0 | 45 |
| held-out | +0.5099 | +0.0000 | 0.8375 | 0.7500 | 0.8813 | 0 | 31 |
| test-like | +0.4871 | +0.0000 | 0.7906 | 0.7000 | 0.8563 | 0 | 70 |

对比：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| `fixed_validation_top1_shrink_schedule` | +0.4584 | +0.4689 | +0.4552 | 0/0/0 |
| `full_then_fixed_schedule` | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| `receiver_alpha_predictor_top1_fallback` | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Receiver alpha predictor 在 validation 上完全拟合 adaptive alpha pseudo target，并在 held-out 上略高于 two-stage；但 test-like 只与 two-stage 基本持平，仍明显低于 exhaustive adaptive alpha。它说明“学 alpha”方向有价值，但当前 tabular 特征不足以稳定复制 oracle adaptive alpha。由于最终仍用 top-1 consistency gate，accepted new error 维持 `0/0/0`，但 repair 仍为 0，missed repair 仍为 `45/31/70`。

结论：该结果应写成 learned deployability pilot，而不是当前最强 M3。下一步应把 alpha/risk 预测并入 residual CNN 训练或使用更强的 receiver-side 特征，而不是继续堆浅层后验规则。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。

### ANALYSIS-S6-012：Minimal Closure Report With Receiver Alpha Predictor

- 日期：2026-07-09
- 项目版本：`4a466e8` + local script/config at run time
- 阶段：S6 derived closure report
- 方法：MinimalClosureReportWithReceiverAlphaPredictor
- 数据集：COCO2017 `val2017` existing outputs and analysis CSVs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_minimal_closure_report.py scripts/s6_train_receiver_alpha_predictor.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 新增输入：`outputs/analysis/exp_s4_006_receiver_alpha_predictor/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；只读取已有 metrics/CSV，不训练、不运行 diffusion、不重算分类器、不下载

#### 指标

| Method | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | New Error | Status |
|---|---|---:|---:|---:|---:|---|
| M3-TwoStageResidualAlphaTop1Fallback | validation / held-out / test-like | +0.4831 / +0.5009 / +0.4875 | N/A | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | deployability ablation |
| M3-ReceiverAlphaPredictorTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5099 / +0.4871 | N/A | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | learned deployability pilot |
| M3-AdaptiveResidualAlphaTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5664 / +0.5691 | -0.0189 / -0.0174 / -0.0201 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | strongest conservative candidate |

#### 结果总结

本轮刷新把 `ANALYSIS-S6-011` 的 receiver alpha predictor 并入最小闭环报告。`outputs/analysis/minimal_closure_report/REPORT.md` 现在把 alpha-control 线拆成：

- `M3-AdaptiveResidualAlphaTop1Fallback`：当前最强保守质量增强候选；
- `M3-ReceiverAlphaPredictorTop1Fallback`：learned 部署 pilot，held-out 略优于 two-stage，但 test-like 未超过 two-stage；
- `M3-TwoStageResidualAlphaTop1Fallback`：少候选检查的非学习部署消融；
- `M3-ResidualRestorationTop1ShrinkFallback`：固定 schedule 消融/备选。

新增输出包括 `receiver_alpha_predictor_tradeoff.csv`。报告仍保留 caveat：receiver predictor 是 validation pseudo-target 训练结果，LPIPS 被省略，不能作为 supervised semantic proof。

#### 下一步

停止在浅层后验 alpha 规则上继续细调；下一步应进入训练侧：在 residual CNN 中加入 alpha/risk head，或设计短链 conditional residual diffusion，从模型内部学习何时放大/收缩 residual。

### ANALYSIS-S6-013：Alpha-Head Residual Refiner Pilot

- 日期：2026-07-09
- 项目版本：`a7076eb` + local script/config at run time
- 阶段：S6 training-side alpha-control exploration
- 方法：AlphaHeadResidualRefinerPilot
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha pseudo targets
- 数据 split：validation `320` 行用于训练 alpha head，held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_pilot_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/`
- 状态：完成；加载 `EXP-S4-006` residual CNN，默认冻结 refiner，仅训练 alpha head；不运行 diffusion、不下载、不加载 LPIPS

#### 方法

Alpha head 附着在 `EXP-S4-006` residual refiner 的 feature map 上。正式运行中：

- residual CNN 从 `outputs/EXP-S4-006/checkpoints/best.pt` 加载；
- `head/body/tail` 冻结，只训练 `alpha_head`；
- 训练目标是 validation 上 `adaptive_max_top1_consistent_alpha` 的 `selected_alpha` pseudo target；
- 评估时预测一个 alpha，生成 `M0 + alpha * (full_refined - M0)`，再用冻结 AlexNet top-1 consistency gate 决定接受或回退 M0。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Alpha Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4011 | +0.0000 | 0.6406 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.3846 | +0.0000 | 0.7312 | 0.6687 | 0 | 37 |
| held-out | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6687 |  | 0 | 26 |
| held-out | alpha_head_predicted_top1_fallback | +0.3808 | +0.0000 | 0.7438 | 0.6500 | 0 | 21 |
| test-like | full_strength_top1_fallback | +0.4113 | +0.0000 | 0.6250 |  | 0 | 54 |
| test-like | alpha_head_predicted_top1_fallback | +0.3623 | +0.0000 | 0.7094 | 0.5844 | 0 | 44 |

对比当前 alpha-control 线：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| `alpha_head_predicted_top1_fallback` | +0.3846 | +0.3808 | +0.3623 | 0/0/0 |
| `full_strength_top1_fallback` | +0.4011 | +0.4454 | +0.4113 | 0/0/0 |
| `full_then_fixed_schedule` | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| `receiver_alpha_predictor_top1_fallback` | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Alpha-head pilot 没有超过 full-strength top-1 fallback，也明显低于 two-stage、receiver predictor 和 exhaustive adaptive alpha。它的价值是把 alpha 控制第一次接进 residual refiner 模型内部，并暴露了当前训练设计的瓶颈：validation target 中 `alpha=1.0` 占 `205/320`，而 alpha head 预测 `alpha=1.0` 达到 `280/320`，说明普通 CE 在类别不平衡下偏向 majority alpha。held-out/test-like 也有同样倾向。

结论：这是训练侧方向的部分负结果，不进入 minimal closure 主表，也不能作为新 M3。下一步应尝试 inverse-frequency alpha loss、unfreeze/refiner joint fine-tune，或者直接设计 semantic-risk-aware residual amplitude loss，而不是只在冻结 feature 上训普通分类头。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `a7076eb + local script/config`。

### ANALYSIS-S6-014：Weighted Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`594db31` + local script/config at run time
- 阶段：S6 training-side alpha-control exploration
- 方法：WeightedAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha pseudo targets
- 数据 split：validation `320` 行用于训练 alpha head，held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/`
- 状态：完成；加载 `EXP-S4-006` residual CNN，冻结 refiner，仅训练 alpha head；启用 tempered inverse-frequency CE weights；不运行 diffusion、不下载、不加载 LPIPS

#### 方法

本实验直接验证上一版 alpha-head 的主要怀疑点：pseudo target 类别不均衡。validation 训练目标分布为 `alpha=0.0/0.25/0.5/0.75/1.0 = 30/34/26/25/205`，因此训练中使用：

- `class_weighting: inverse_frequency`
- `class_weight_power: 0.5`
- `class_weight_normalize_mean: true`

得到 class weights `[1.1132, 1.0457, 1.1958, 1.2195, 0.4259]`。其余结构与 `ANALYSIS-S6-013` 一致：冻结 residual CNN，只训练 alpha head；评估时预测一个 alpha 候选，再用 AlexNet top-1 consistency gate 保护输出。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Alpha Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4011 | +0.0000 | 0.6406 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.3851 | +0.0000 | 0.8094 | 0.6375 | 0 | 25 |
| held-out | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6687 |  | 0 | 26 |
| held-out | alpha_head_predicted_top1_fallback | +0.3506 | +0.0000 | 0.7875 | 0.5750 | 0 | 16 |
| test-like | full_strength_top1_fallback | +0.4113 | +0.0000 | 0.6250 |  | 0 | 54 |
| test-like | alpha_head_predicted_top1_fallback | +0.3166 | +0.0000 | 0.7562 | 0.4969 | 0 | 36 |

对比上一版普通 CE：

| Policy | validation | held-out | test-like | Target Alpha Acc | New Error |
|---|---:|---:|---:|---:|---:|
| unweighted alpha head | +0.3846 | +0.3808 | +0.3623 | 0.6687 / 0.6500 / 0.5844 | 0/0/0 |
| weighted alpha head | +0.3851 | +0.3506 | +0.3166 | 0.6375 / 0.5750 / 0.4969 | 0/0/0 |
| full-strength top-1 fallback | +0.4011 | +0.4454 | +0.4113 | N/A | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | N/A | 0/0/0 |

#### 结果总结

Weighted CE 把普通 CE 的 majority collapse 缓和了，但没有变成更好的 alpha policy。validation 上 unweighted 预测 `alpha=1.0` 为 `280/320`，weighted 降到 `223/320`，少数 alpha 预测明显增加；test-like 上 weighted 预测分布为 `0.0/0.25/0.5/0.75/1.0 = 64/21/5/21/209`，也比 unweighted 的 `39/5/0/9/267` 更分散。

问题是更分散不等于更优。weighted 版 accept rate 更高，accepted new error 仍为 0，但 held-out/test-like PSNR 明显低于 unweighted 和 full-strength top-1 fallback。这说明 alpha head 当前学到的是“少数类覆盖”，不是“何时某个 alpha 能带来最大质量收益且不引发 semantic drift”。冻结 residual feature 本身也可能没有足够信息区分 `0.25/0.5/0.75` 的边界。

结论：类别不均衡是症状之一，但不是主因。下一步不宜继续只调 CE 权重；应转向 benefit/risk-aware alpha 目标、联合微调 residual CNN，或在短链 conditional residual diffusion 中用 M0/refined 附近初始化并加入 identity/semantic-risk 约束。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `594db31 + local script/config`。

### ANALYSIS-S6-015：Benefit-Aware Alpha Predictor

- 日期：2026-07-09
- 项目版本：`050b0c2` + local script/config at run time
- 阶段：S6 receiver-side learned alpha-control exploration
- 方法：BenefitAwareAlphaPredictor
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练小型 predictor；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_benefit_alpha_predictor_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_receiver_alpha_predictor.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --config configs/s6_benefit_alpha_predictor_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --config configs/s6_benefit_alpha_predictor_exp_s4_006.yaml --device cuda:0 --overwrite
```

- 关键源码：`scripts/s6_train_receiver_alpha_predictor.py`
- 输入：
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - validation/held-out/test-like residual alpha candidate PNG roots
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_benefit_alpha_predictor/`
- 状态：完成；只训练小型 tabular predictor，不训练图像模型、不运行 diffusion、不下载、不加载 LPIPS

#### 方法

上一版 `ReceiverAlphaPredictor` 直接把 `adaptive_max_top1_consistent_alpha` 当 hard pseudo-label 分类。这个 follow-up 改为 utility soft labels：

- 对每个样本枚举 `alpha in [0.0, 0.25, 0.5, 0.75, 1.0]`；
- 若候选 alpha 的 AlexNet top-1 与 M0 top-1 一致，则 utility 为该候选相对 M0 的 PSNR delta；
- 若候选不满足 top-1 安全，则 utility 设为 `-2.0`；
- `alpha=0.0` 表示 fallback M0，utility 为 `0.0`；
- 用 temperature `0.20` 把 utility 转成 soft label 训练 predictor。

训练标签可用 validation 原图计算 PSNR，但 predictor 输入仍只包含接收端可见特征：SNR、M0/full candidate 的分类器置信度、full candidate 是否与 M0 top-1 一致，以及 M0 到 full candidate 的 residual 图像统计。评估时仍对预测 alpha 候选执行 top-1 fallback。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | benefit_alpha_predictor_top1_fallback | +0.5538 | +0.0000 | 0.8906 | 0.7188 | 0 | 45 |
| held-out | benefit_alpha_predictor_top1_fallback | +0.4474 | +0.0000 | 0.7562 | 0.3812 | 0 | 31 |
| test-like | benefit_alpha_predictor_top1_fallback | +0.4627 | +0.0000 | 0.7469 | 0.4250 | 0 | 70 |

对比当前 alpha-control 线：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| benefit-aware predictor | +0.5538 | +0.4474 | +0.4627 | 0/0/0 |
| receiver alpha predictor | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| full_then_fixed_schedule | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| adaptive_max_top1_consistent_alpha | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Benefit-aware 目标在 validation 上有效：PSNR delta `+0.5538` dB，几乎追上 exhaustive adaptive alpha 的 `+0.5584` dB。但它没有在 held-out/test-like 上迁移，分别只有 `+0.4474/+0.4627` dB，低于 two-stage 和上一版 receiver predictor。utility target 分布比原 adaptive pseudo target 更均衡，validation target 为 `0.0/0.25/0.5/0.75/1.0 = 30/34/35/115/106`，但 held-out/test-like target accuracy 只有 `0.3812/0.4250`。

结论：把 alpha 目标改成“安全前提下的质量收益”是更贴近问题的方向，但当前 tabular feature + 小 MLP 泛化不足。下一步不宜继续只换浅层 predictor loss；更合理的是把 benefit/risk 约束前移到 residual CNN joint fine-tune，或让模型内部特征直接预测 residual amplitude/risk。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 candidate PNG 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `050b0c2 + local script/config`。

### ANALYSIS-S6-016：Benefit-Aware Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`53b71b3` + local script/config at run time
- 阶段：S6 training-side residual alpha-control exploration
- 方法：BenefitAwareAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练 alpha head；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/`
- 状态：完成；冻结 residual CNN，仅训练 alpha head，不运行 diffusion、不下载、不加载 LPIPS

#### 方法

该实验复用 alpha-head residual refiner 流程，但把训练标签从 `adaptive_max_top1_consistent_alpha` hard pseudo target 换成上一轮 benefit predictor feature table 中的 `utility_target_alpha`。这些 utility target 使用 validation 原图构造：候选 alpha 必须满足 AlexNet top-1 与 M0 top-1 一致，安全候选按 PSNR delta 选最大收益，否则回退 M0。

模型输入和推理仍只使用接收端可见的 M0/SNR/refiner feature。评估阶段仍对 predicted-alpha candidate 使用冻结 AlexNet top-1 fallback，因此该实验是训练侧 alpha 控制探索，不是新的 M3 闭环。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4011 | +0.0000 | 0.6406 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.4251 | +0.0000 | 0.7812 | 0.5406 | 0 | 28 |
| held-out | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6687 |  | 0 | 26 |
| held-out | alpha_head_predicted_top1_fallback | +0.4192 | +0.0000 | 0.8000 | 0.4313 | 0 | 16 |
| test-like | full_strength_top1_fallback | +0.4113 | +0.0000 | 0.6250 |  | 0 | 54 |
| test-like | alpha_head_predicted_top1_fallback | +0.3530 | +0.0000 | 0.7406 | 0.4062 | 0 | 40 |

对比 alpha-control 线：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| benefit-aware alpha head | +0.4251 | +0.4192 | +0.3530 | 0/0/0 |
| unweighted alpha head | +0.3846 | +0.3808 | +0.3623 | 0/0/0 |
| weighted alpha head | +0.3851 | +0.3506 | +0.3166 | 0/0/0 |
| benefit-aware predictor | +0.5538 | +0.4474 | +0.4627 | 0/0/0 |
| receiver alpha predictor | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| full_then_fixed_schedule | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| adaptive_max_top1_consistent_alpha | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Benefit-aware alpha head 比普通/weighted alpha-head 有部分进展。validation 从 `+0.3846/+0.3851` 提到 `+0.4251` dB，held-out 从 `+0.3808/+0.3506` 提到 `+0.4192` dB，accepted new error 仍为 `0/0/0`。但是它没有超过 receiver predictor、two-stage policy 或 exhaustive adaptive alpha，test-like 也低于普通 alpha-head。

预测分布显示模型仍没有学到细粒度 alpha 边界：validation predicted alpha 为 `0.0/0.25/0.5/0.75/1.0 = 35/0/10/154/121`，而 target 为 `30/34/35/115/106`；test-like predicted 为 `50/1/12/113/144`，target 为 `35/35/30/123/97`。模型几乎不预测 `alpha=0.25`，说明冻结 residual feature + alpha classifier 仍主要学到粗粒度 fallback/strong-refine，而不是 utility target 中的收益/风险排序。

结论：benefit/risk 目标本身有价值，但只把标签换到冻结 alpha head 上还不够。下一步应优先 joint fine-tune residual CNN，或把 semantic-risk-aware residual amplitude loss 直接放进 residual restoration 训练，而不是继续只换 alpha 分类标签。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `53b71b3 + local script/config`。

### ANALYSIS-S6-017：Benefit-Aware Joint Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`901420f` + local script/config at run time
- 阶段：S6 training-side residual alpha-control exploration
- 方法：BenefitAwareJointAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于 joint fine-tune；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/`
- 状态：完成；解冻 residual CNN joint fine-tune，不运行 diffusion、不下载、不加载 LPIPS；负/诊断结果

#### 方法

该实验在 `ANALYSIS-S6-016` 的 benefit-aware alpha-head 基础上解冻 residual CNN，并新增训练损失：

- `soft_refiner_detach: false`：predicted soft-alpha reconstruction loss 反传到 residual CNN；
- `target_alpha_mse_weight: 100.0`：utility target alpha 对应的 refined 图像对 original 做 MSE；
- `full_mse_weight: 10.0`：保留一个弱 full-strength restoration anchor；
- `ce_weight: 0.30`：继续训练 alpha head 预测 utility alpha；
- `refiner_lr: 0.00005`，alpha head `lr: 0.001`。

评估阶段仍对 predicted-alpha candidate 使用冻结 AlexNet top-1 fallback。该实验测试的是“全量 unfreeze + benefit/risk alpha loss”是否能让 residual CNN 内部学到更好的 amplitude control。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.2530 | +0.0000 | 0.6656 |  | 0 | 45 |
| validation | alpha_head_predicted_top1_fallback | +0.3294 | +0.0000 | 0.8688 | 0.7719 | 0 | 17 |
| held-out | full_strength_top1_fallback | +0.2236 | +0.0000 | 0.7188 |  | 0 | 19 |
| held-out | alpha_head_predicted_top1_fallback | +0.2303 | +0.0000 | 0.8562 | 0.3875 | 0 | 8 |
| test-like | full_strength_top1_fallback | +0.1855 | +0.0000 | 0.6687 |  | 0 | 39 |
| test-like | alpha_head_predicted_top1_fallback | +0.1869 | +0.0000 | 0.8219 | 0.3719 | 0 | 21 |

对比 alpha-head 训练侧路线：

| Policy | validation | held-out | test-like | Target Acc | New Error |
|---|---:|---:|---:|---:|---:|
| frozen benefit alpha head | +0.4251 | +0.4192 | +0.3530 | 0.5406 / 0.4313 / 0.4062 | 0/0/0 |
| joint benefit alpha head | +0.3294 | +0.2303 | +0.1869 | 0.7719 / 0.3875 / 0.3719 | 0/0/0 |
| full-strength top-1 fallback before joint | +0.4011 | +0.4454 | +0.4113 | N/A | 0/0/0 |
| full-strength top-1 fallback after joint | +0.2530 | +0.2236 | +0.1855 | N/A | 0/0/0 |

#### 结果总结

Joint fine-tune 成功改善了 validation alpha 分类：target accuracy 从 frozen benefit alpha-head 的 `0.5406` 提升到 `0.7719`，预测分布也从几乎不用 `alpha=0.25` 变成 `0.0/0.25/0.5/0.75/1.0 = 28/24/23/127/118`。这说明解冻 shared feature 后，模型确实能更好地读出 utility alpha。

但图像 restoration anchor 被明显损伤。full-strength top-1 fallback 从原始 `+0.4011/+0.4454/+0.4113` dB 掉到 `+0.2530/+0.2236/+0.1855` dB；predicted-alpha final 也只有 `+0.3294/+0.2303/+0.1869` dB，低于 frozen benefit alpha-head。训练日志中 full MSE 从约 `0.000816` 升到约 `0.000872`，与最终 PSNR 下滑一致。

结论：benefit/risk 目标可以改善 alpha 分类，但全量 unfreeze 且 CE 仍占主导会破坏 residual restoration 表征。下一步应避免让分类目标直接改写 shared residual feature；更合理的是 partial fine-tune（只调 tail/amplitude/head）、更强 reconstruction-dominant objective、或在固定 residual feature 上学习单独的 amplitude/risk head。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `901420f + local script/config`。

### ANALYSIS-S6-018：Benefit-Aware Tail-Only Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`c69743a` + local script/config at run time
- 阶段：S6 training-side residual alpha-control exploration
- 方法：BenefitAwareTailAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/`
- 状态：完成；只训练 residual tail 与 alpha head，不运行 diffusion、不下载、不加载 LPIPS

#### 方法

该实验直接接续 `ANALYSIS-S6-017` 的负结果。全量 joint fine-tune 会破坏 shared residual feature，因此本轮只允许：

- `trainable_refiner_parts: [tail]`：冻结 head/body，只更新 residual tail；
- `alpha_head` 正常训练；
- `ce_weight: 0.05`：降低 alpha 分类目标对训练的主导性；
- `full_mse_weight: 100.0`：用 reconstruction-dominant loss 保护 full-strength restoration anchor；
- `soft_refiner_detach: false`、`soft_mse_weight: 25.0`、`target_alpha_mse_weight: 25.0`：让 predicted/target alpha reconstruction 对 tail 做温和幅度校准。

metadata 记录的可训练参数为：head `0/1776`、body `0/207840`、tail `1299/1299`、alpha head `3461/3461`。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6281 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.4749 | +0.0000 | 0.7531 | 0.5437 | 0 | 32 |
| held-out | full_strength_top1_fallback | +0.4820 | +0.0000 | 0.6687 |  | 0 | 27 |
| held-out | alpha_head_predicted_top1_fallback | +0.4552 | +0.0000 | 0.7937 | 0.4313 | 0 | 19 |
| test-like | full_strength_top1_fallback | +0.4259 | +0.0000 | 0.5938 |  | 0 | 53 |
| test-like | alpha_head_predicted_top1_fallback | +0.4061 | +0.0000 | 0.7312 | 0.4250 | 0 | 38 |

对比 alpha-head 训练侧路线：

| Policy | validation | held-out | test-like | Target Acc | New Error |
|---|---:|---:|---:|---:|---:|
| frozen benefit alpha head | +0.4251 | +0.4192 | +0.3530 | 0.5406 / 0.4313 / 0.4062 | 0/0/0 |
| full joint benefit alpha head | +0.3294 | +0.2303 | +0.1869 | 0.7719 / 0.3875 / 0.3719 | 0/0/0 |
| tail-only benefit alpha head | +0.4749 | +0.4552 | +0.4061 | 0.5437 / 0.4313 / 0.4250 | 0/0/0 |

#### 结果总结

Tail-only partial fine-tune 是训练侧正向阶段结果：它明显恢复了全量 joint 损伤的 restoration anchor，full-strength top-1 fallback 达到 `+0.4454/+0.4820/+0.4259` dB；predicted-alpha final 也达到 `+0.4749/+0.4552/+0.4061` dB，三段 accepted new error 均为 0。

该结果说明：上一轮失败不是 benefit/risk 目标本身无效，而是全量解冻让分类/target-alpha loss 改写了 shared residual feature。把可训练范围限制在 tail，并用 reconstruction-dominant loss 后，可以获得比冻结 benefit alpha-head 更好的泛化质量。

限制也明确：tail-only 仍低于 receiver predictor、two-stage policy 和后验 adaptive alpha，因此不能作为最终 M3。预测分布仍不使用 `alpha=0.25`（validation/held-out/test-like predicted counts for `0/0.25/0.5/0.75/1.0` 为 `27/0/9/159/125`、`20/0/0/66/74`、`42/0/8/123/147`），说明细粒度 alpha 边界还没有被学好。下一步若继续训练侧，应考虑显式 amplitude head、连续 alpha regression 或 validation model-selection loss，而不是单纯扩大解冻范围。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `c69743a + local script/config`。

### ANALYSIS-S6-019：Benefit-Aware Tail-Only Continuous-Alpha Residual Refiner

- 日期：2026-07-09
- 项目版本：`9b6f74a` + local script/config at run time
- 阶段：S6 training-side residual amplitude-control exploration
- 方法：BenefitAwareTailContinuousAlphaResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/`
- 状态：完成；只训练 residual tail 与 continuous alpha head，不运行 diffusion、不下载、不加载 LPIPS

#### 方法

该实验接续 `ANALYSIS-S6-018` 的限制：离散 alpha 分类仍几乎不预测 `alpha=0.25`。本轮保持 head/body 冻结、只训练 residual tail 和 alpha head，但把 alpha head 输出从 5 类 logits 改为单个连续 alpha：

- `model.alpha_mode: regression`：alpha head 输出经 `sigmoid` 映射到 `[0, 1]`；
- `trainable_refiner_parts: [tail]`：冻结 residual head/body，只微调 residual tail；
- `alpha_loss_weight: 0.20`，`regression_loss: smooth_l1`，`regression_beta: 0.10`；
- `full_mse_weight: 100.0`、`soft_mse_weight: 25.0`、`target_alpha_mse_weight: 25.0`：继续用 reconstruction-dominant loss 保护 restoration anchor；
- 评估时使用连续 predicted alpha 生成 candidate，再用同一冻结 AlexNet top-1 fallback 做 final decision。

metadata 记录的可训练参数为：head `0/1776`、body `0/207840`、tail `1299/1299`、alpha head `3201/3201`。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | Mean Alpha | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4463 | +0.0000 | 0.6281 |  |  | 0 | 41 |
| validation | continuous_alpha_top1_fallback | +0.5010 | +0.0000 | 0.7125 | 0.4188 | 0.7270 | 0 | 41 |
| held-out | full_strength_top1_fallback | +0.4824 | +0.0000 | 0.6687 |  |  | 0 | 27 |
| held-out | continuous_alpha_top1_fallback | +0.5049 | +0.0000 | 0.7438 | 0.3625 | 0.7381 | 0 | 23 |
| test-like | full_strength_top1_fallback | +0.4298 | +0.0000 | 0.5969 |  |  | 0 | 52 |
| test-like | continuous_alpha_top1_fallback | +0.5012 | +0.0000 | 0.7250 | 0.3469 | 0.7123 | 0 | 40 |

对比 learned / deployable alpha control：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| tail-only discrete alpha head | +0.4749 | +0.4552 | +0.4061 | 0/0/0 |
| two-stage alpha policy | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| receiver alpha predictor | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| tail-only continuous alpha head | +0.5010 | +0.5049 | +0.5012 | 0/0/0 |
| posterior adaptive alpha upper bound | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

连续 alpha 分布：

| Split | Mean | Min | Q1 | Median | Q3 | Max | Nearest Alpha Counts `0/0.25/0.5/0.75/1.0` |
|---|---:|---:|---:|---:|---:|---:|---|
| validation | 0.7270 | 0.2574 | 0.6490 | 0.7437 | 0.8281 | 0.9955 | `0/6/66/206/42` |
| held-out | 0.7381 | 0.3104 | 0.6218 | 0.7654 | 0.8650 | 0.9938 | `0/2/38/87/33` |
| test-like | 0.7123 | 0.1219 | 0.6072 | 0.7407 | 0.8379 | 0.9815 | `1/11/78/176/54` |

#### 结果总结

这是当前训练侧 amplitude-control 最明确的正向结果。连续 alpha head 在三段 split 上都保持 accepted new error 为 0，同时 PSNR delta 达到 `+0.5010/+0.5049/+0.5012` dB，明显超过离散 tail-only alpha head，并在 held-out/test-like 上达到或超过 two-stage policy 和 receiver predictor。

该结果说明，上一轮离散 alpha-head 的瓶颈很可能来自分类目标和离散候选表达，而不是 tail-only 微调本身。连续 alpha 的 nearest-class target accuracy 较低并不是直接负面信号，因为它没有强行复刻离散 utility label，而是在 `[0,1]` 上学到更平滑的幅度折中；test-like 最近 alpha 分布覆盖 `0.5/0.75/1.0`，不再完全跳过中间强度。

限制：该方法仍低于 posterior adaptive alpha upper bound，且本轮训练实验本身省略 LPIPS、classifier ensemble audit 和 COCO-object/CLIP 辅助诊断。后续 `ANALYSIS-S6-020` 已补 LPIPS 和三分类器 ensemble 审计，结论是它可以作为 learned deployable amplitude-control 的强候选，但仍不能直接写成最终 M3。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `9b6f74a + local script/config`。

### ANALYSIS-S6-020：Continuous-Alpha Tail Refiner LPIPS / Classifier-Ensemble Audit

- 日期：2026-07-09
- 项目版本：`3c8a0bd` + local script/config at run time
- 阶段：S6 derived perceptual and semantic robustness audit
- 方法：ContinuousAlphaTailRefinerPerceptualEnsembleAudit
- 数据集：COCO2017 `val2017` validation / held-out / test-like continuous-alpha outputs
- 数据 split：validation `320` 行，held-out `160` 行，test-like `320` 行；审计 continuous-alpha 与 full-strength top-1 fallback 两个 policy，共 `1600` 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_continuous_alpha_tail_refiner_audit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_audit_continuous_alpha_tail_refiner.py
python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --device cuda:0
```

- 关键源码：`scripts/s6_audit_continuous_alpha_tail_refiner.py`
- 输入：
  - `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/per_sample.csv`
  - `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/summary.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
  - `outputs/cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
  - `outputs/cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`
- 输出路径：`outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/`
- 状态：完成；只读取已有 PNG/CSV，不训练、不运行 diffusion、不调参

#### 指标

| Policy | Split | Delta PSNR | Delta LPIPS | AlexNet New Error | Any-Classifier New Error | Majority New Error | Any Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| continuous_alpha_top1_fallback | validation | +0.5010 | -0.0149 | 0 | 17 | 1 | 41 |
| continuous_alpha_top1_fallback | held-out | +0.5049 | -0.0149 | 0 | 9 | 0 | 11 |
| continuous_alpha_top1_fallback | test-like | +0.5012 | -0.0162 | 0 | 14 | 0 | 47 |
| full_strength_top1_fallback | validation | +0.4463 | -0.0097 | 0 | 19 | 1 | 38 |
| full_strength_top1_fallback | held-out | +0.4824 | -0.0106 | 0 | 12 | 1 | 7 |
| full_strength_top1_fallback | test-like | +0.4298 | -0.0098 | 0 | 20 | 0 | 51 |

#### 结果总结

LPIPS 证据支持 continuous-alpha：三段 split 上 final LPIPS delta 为 `-0.0149/-0.0149/-0.0162`，明显优于同 checkpoint full-strength top-1 fallback 的 `-0.0097/-0.0106/-0.0098`。这说明连续 alpha 的 PSNR 提升不是单纯牺牲感知质量换来的。

跨分类器审计给出更谨慎的边界。AlexNet source gate 下 continuous-alpha 仍保持 accepted new error `0/0/0`，但 ResNet18/MobileNetV3-Small 作为离线 pseudo reference 时，any-classifier new error 为 `17/9/14`，majority-vote new error 为 `1/0/0`。唯一 majority case 是 validation 4 dB `sample_000248.png`，由 MobileNetV3-Small 与 ResNet18 同时标为 accepted new error。相比 full-strength fallback，continuous-alpha 在 LPIPS、PSNR 和多数 split 的 ensemble 风险上更好，但仍不能声称跨模型完全安全。

结论：continuous-alpha tail refiner 是当前最强 learned training-side amplitude-control 候选；它可以进入下一轮方法设计依据，但不能直接升级为最终 M3。下一步应加入 semantic-risk-aware / ensemble-aware 训练或选择约束，或先做 labeled clean-correct subset 复核。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`，`lpips_error: null`。首次正式运行前脚本曾把 LPIPS `TORCH_HOME` 指向输出目录，触发临时 AlexNet 权重下载；该运行被中断、输出目录删除，脚本修正为使用项目本地 `outputs/cache/torch` 后重新正式运行，未使用中断结果。

### ANALYSIS-S6-002：EXP-S4-006 Residual Shrink Selection

- 日期：2026-07-07
- 项目版本：运行时基于 `20f9cc3d6d0444b3eee2a2ccab76bb04b9a18369` 之后的本轮新增脚本
- 阶段：S6 validation-only model-selection analysis
- 方法：ResidualShrinkSelection
- 数据集：COCO2017 `val2017` subset outputs from `EXP-S4-006`
- 数据 split / 样本 ID：`sample_000192`-`sample_000255`，5 个 SNR，共 320 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；本分析本身不使用随机采样
- checkpoint：不重新加载 JSCC/refiner checkpoint；读取 `EXP-S4-006` 已有 PNG
- config：`configs/s6_residual_shrink_selection_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_residual_shrink_selection.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --device cuda:0 --overwrite
```

- 关键源码：`scripts/s6_residual_shrink_selection.py`
- 输入：
  - `outputs/EXP-S4-006/per_sample.csv`
  - `outputs/EXP-S4-006/exports/snr_XXdb/refined/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/exports/original/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/exports/snr_XXdb/reconstruction/`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_residual_shrink_selection/`
- 状态：完成；派生分析，不训练、不运行 diffusion、不下载

#### 指标

| Policy | Alpha/Schedule | Mean Delta PSNR | Mean Delta LPIPS | Final Failure | Delta Failure | Accept | New Error |
|---|---|---:|---:|---:|---:|---:|---:|
| `top1_fallback_alpha` | 1.0 | +0.4011 | -0.0104 | 0.3750 | +0.0000 | 0.6406 | 0 |
| `top1_fallback_alpha` | validation-selected `[0.5,0.75,0.75,0.75,0.75]` | +0.4584 | -0.0153 | 0.3750 | +0.0000 | 0.7438 | 0 |
| `always_alpha` | 1.0 | +0.7235 | -0.0274 | 0.3344 | -0.0406 | 1.0000 | 28 |
| `selected_always_m0_failure_constrained_schedule` | validation-selected | +0.5505 | -0.0253 | 0.3281 | -0.0469 | 0.8000 | 19 |

#### 结果总结

缩放残差强度能提升保守 top-1 fallback 的质量/语义 tradeoff：validation-only per-SNR schedule 在不提高 pseudo final failure 的前提下，比 full-strength top-1 fallback 多 `+0.0573` dB PSNR，并进一步改善 LPIPS。选出的 schedule 为：1 dB 用 `alpha=0.5`，4/7/13/19 dB 用 `alpha=0.75`。

但是 always-accept 不能作为最终 M3。它的平均 final failure 低于 M0，是因为 repair 数量多于 new error；逐样本看仍有 19-28 个 accepted new error。该结果说明下一步应把 residual strength / alpha 控制放入训练或 validation model selection，而不是把 always-accept 包装成安全方法。

#### 复现备注

正式运行时清空代理变量，dry-run 记录 `proxy_environment_present: []`。输出包括：

- `outputs/analysis/exp_s4_006_residual_shrink_selection/REPORT.md`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/alpha_tradeoff.png`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/samples/`

#### 下一步

把该分析作为 M3 训练/选择设计依据：优先做 semantic-risk-aware residual CNN model selection 或在训练中加入残差幅度/语义风险约束；若继续 diffusion，只做从 M0/M2 附近初始化的短链 residual correction。

### ANALYSIS-S6-003：Frozen Residual Shrink Schedule Test-Like Check

- 日期：2026-07-07
- 项目版本：运行时基于 `7ef1753d` 之后的本轮新增脚本
- 阶段：S6 frozen schedule test-like analysis
- 方法：FrozenResidualShrinkScheduleCheck
- 数据集：COCO2017 `val2017` test-like outputs from `EXP-S4-006`
- 数据 split / 样本 ID：`sample_000256`-`sample_000319`，5 个 SNR，共 320 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；本分析本身不使用随机采样
- checkpoint：不重新加载 JSCC/refiner checkpoint；读取 test-like gate check 已有 PNG
- config：`configs/s6_testlike_residual_shrink_schedule_check_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_residual_shrink_schedule.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --device cuda:0
```

- 关键源码：`scripts/s6_apply_residual_shrink_schedule.py`, `scripts/s6_residual_shrink_selection.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json`
  - `outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_testlike_gate_check/exports/snr_XXdb/refined/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/exports/original/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/exports/snr_XXdb/reconstruction/`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/`
- 状态：完成；frozen schedule 复核，不训练、不调参、不运行 diffusion、不下载

#### 指标

| Policy | Mean Delta PSNR | Mean Delta LPIPS | Final Failure | Delta Failure | Accept | Repair | New Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m0` | 0.0000 | 0.0000 | 0.4719 | 0.0000 | 0.0000 | 0 | 0 |
| `top1_full_strength` | +0.4113 | -0.0116 | 0.4719 | +0.0000 | 0.6250 | 0 | 0 |
| `validation_top1_shrink_schedule` | +0.4552 | -0.0152 | 0.4719 | +0.0000 | 0.7063 | 0 | 0 |
| `always_full_strength` | +0.7180 | -0.0270 | 0.3812 | -0.0906 | 1.0000 | 54 | 25 |
| `validation_always_m0_failure_constrained_schedule` | +0.5555 | -0.0257 | 0.4031 | -0.0688 | 0.8000 | 34 | 12 |

#### 结果总结

Validation 上选出的 top-1 shrink schedule 在 test-like 上迁移成功：相对 full-strength top-1 fallback，PSNR 额外提升 `+0.0439` dB，LPIPS 进一步改善，同时 pseudo final failure 仍不高于 M0，accepted new error 为 0。分 SNR 看，固定 schedule 在 1/4/7/13/19 dB 的 PSNR delta 分别为 `+0.5087/+0.4268/+0.3769/+0.4499/+0.5137` dB。

Always-accept 路线继续不安全：full-strength always-accept 有 25 个 accepted new error，validation 的 always-constrained schedule 仍有 12 个 accepted new error。因此可写成证据的是“残差强度控制 + top-1 semantic fallback”提升了保守 M3 的质量，而不是 always-accept。

#### 复现备注

正式运行时清空代理变量，dry-run 记录 `proxy_environment_present: []`。输出包括：

- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/REPORT.md`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/metadata.json`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/samples/`

#### 下一步

把 `validation_top1_shrink_schedule` 作为当前更强的 conservative M3 候选，后续需要在带标签 clean-correct subset 或更正式 test split 上复核；训练侧则应考虑直接学习 SNR-aware residual amplitude / alpha，而不是只在输出后缩放。
