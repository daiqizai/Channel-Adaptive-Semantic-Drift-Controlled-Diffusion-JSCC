# scripts

存放可复现命令行脚本。

脚本命名建议：

- `run_s1_deepjscc_baseline.sh`
- `run_s2_blind_diffusion.sh`
- `run_s3_semantic_metrics.sh`
- `run_s4_adaptive_guidance.sh`

## 当前脚本

- `s1_deepjscc_smoke.py`：使用随机合成图像验证 checkpoint 加载、SNR 切换、重建输出和 PSNR 计算。
- `s1_deepjscc_mini_eval.py`：使用 CIFAR-10 test subset 运行 M0-DeepJSCC baseline，可用于 mini-eval 和 `EXP-S1-001` 正式 baseline。
- `train_deepjscc_highres.py`：训练 COCO-256 等高分辨率 DeepJSCC checkpoint；支持 `--dry-run` 检查训练代码路径；训练中遇到 NaN 会提前停止并保留 `best.pt`。
- `s2_deepjscc_highres_export.py`：加载高分辨率 DeepJSCC checkpoint，按 SNR sweep 评估并导出 `x_hat` PNG，供 diffusion refinement 使用。
- `s3_blind_diffusion_refine.py`：读取正式 M0-HR export，对 1/7/19 dB 小样本运行 `M1-BlindDiffusion` img2img refinement，保存 refined 图、PSNR/SSIM/MS-SSIM/LPIPS（可选）和样例拼图；当前需要本地 Stable Diffusion cache 或显式允许下载。
- `s4_clip_consistency_eval.py`：读取正式 M0 export 和 `EXP-S2-002` refined 输出，计算原图-M0、原图-M1、M0-M1 的 CLIP image cosine similarity，并保存 `metrics.json`、`per_sample.csv` 和 top failure case。
- `s4_make_clip_failure_gallery.py`：读取 `EXP-S3-001/per_sample.csv`，按 CLIP drop 生成 original/M0/M1 triptych、全局和分 SNR failure sheets、CSV/JSON 索引；不跑模型、不联网。
- `s4_classifier_consistency_eval.py`：读取正式 M0 export 和 `EXP-S2-002` refined 输出，用冻结 torchvision AlexNet/ImageNet 分类器比较 `c(original)`、`c(M0)`、`c(M1)` 的 pseudo-label consistency；保存 `metrics.json` 和 `per_sample.csv`。
- `s4_make_classifier_failure_gallery.py`：读取 `EXP-S3-002/per_sample.csv`，筛选 M0 匹配原图但 M1 不匹配的 pseudo-label drift case，生成 triptych、全局和分 SNR sheets、CSV/JSON 索引；不跑模型、不联网。
- `s4_coco_caption_clip_eval.py`：读取正式 M0 export、`EXP-S2-002` refined 输出和 COCO `captions_val2017.json`，计算 original/M0/M1 与对应人工 captions 的 CLIP image-text similarity；保存 `metrics.json`、`per_sample.csv` 和 `sample_metadata.json`。
- `s4_make_coco_caption_failure_gallery.py`：读取 `EXP-S3-003/per_sample.csv`，按 caption CLIP drop 生成 original/M0/M1 triptych、全局和分 SNR failure sheets、CSV/JSON/README 索引；不跑模型、不联网。
- `s4_summarize_m1_negative_result.py`：聚合 `EXP-S2-002` 图像指标、`EXP-S3-001` CLIP 诊断和 `EXP-S3-002` 分类器诊断，输出 M1 负结果的 `REPORT.md`、`summary.csv`、`summary.json`；不跑模型、不联网。
- `s4_make_project_progress_visual_summary.py`：聚合已有正式 metrics、辅助语义诊断和 failure gallery，生成项目阶段进度图、M0/M1 指标图、语义诊断图、代表性可视化拼图和 `REPORT.md`；派生分析，不跑模型、不联网。
- `s5_audit_residual_gate_aux_semantics.py`：读取 `EXP-S4-006` gate sweep 的 top-k 预测，使用本地 OpenCLIP 和 COCO captions 对 confidence-gain 候选 gate 做离线辅助语义审计。
- `s5_materialize_residual_gate_policy.py`：读取 `EXP-S4-006` gate sweep 结果，按指定候选 gate 将 final PNG 从已有 M0/refined 图像中复制落盘，并保存 summary、per-sample CSV 和样例拼图。
- `s5_residual_refiner_heldout_gate_eval.py`：加载 `EXP-S4-006` residual refiner checkpoint，在 held-out 或 test-like 样本段上重新生成 refined/top-1/candidate final，并复核 confidence-gain gate 的 repair 与 accepted new error。
- `s5_sweep_conf_gain_clip_veto.py`：读取 `EXP-S4-006` validation/held-out gate CSV，计算 receiver-side `CLIP(M0, refined)`，扫描 confidence-gain gate 的二级 veto 阈值并输出 policy summary、逐样本决策和 galleries。
- `s5_calibrate_conf_gain_clip_veto_by_snr.py`：读取已有 CLIP veto sweep CSV，在 validation 上校准 per-SNR CLIP veto schedule，并比较 independent 与 monotonic schedule 的 held-out 风险。
- `s5_sweep_conf_gain_risk_rules.py`：读取已有 CLIP/top-k CSV，在 validation 上搜索透明 receiver-side risk rules，用 held-out 复核 confidence-gain gate 的 accepted-new-error 风险。
- `s5_materialize_risk_rule_policy.py`：读取 risk-rule sweep 的 `policy_decisions.csv`，将 `selected_risk_rule` final PNG 从已有 M0/refined 图像复制落盘，并保存 summary、per-sample CSV、报告和样例 sheet。
- `s5_audit_risk_rule_classifier_ensemble.py`：固定 `selected_risk_rule` 决策，用 AlexNet/ResNet18/MobileNetV3-Small 等冻结 ImageNet 分类器离线审计 validation/held-out/test-like 的跨模型 semantic repair/new-error 风险。
- `s5_sweep_ensemble_risk_veto.py`：读取 selected risk-rule 决策和 classifier ensemble audit 投票，在 validation 上搜索透明 receiver-side 二级 veto，用 held-out 复核多数票 new-error 风险、repair 保留量和 PSNR 代价。
- `s5_sweep_receiver_risk_score.py`：读取 selected risk-rule 决策和 ensemble audit 投票，扫描多个透明 receiver-side risk score 模板及阈值，用 held-out 检查少 veto 风险分数是否会漏多数票 new-error。
- `s5_apply_testlike_risk_rules.py`：把 validation/held-out 阶段冻结的 `selected_risk_rule` 和保守 ensemble-risk veto 应用到 test-like split，重新计算本地 CLIP、输出决策表、final PNG 和风险样例。
- `s5_coco_object_clip_clean_eval.py`：读取 test-like gate 决策、COCO instance labels 和本地 OpenCLIP，按 dominant object label 构造辅助 clean-correct 子集，输出 policy-level GT-like semantic failure/repair/new-error 诊断。
- `s5_residual_diffusion_pilot.py`：读取正式 256 张/SNR M0 export，训练一个小型 SNR-conditioned pixel residual DDPM；避开 Stable Diffusion、text prompt 和 SD VAE，用同一 pseudo semantic fallback 口径评估 residual diffusion。
- `s6_make_minimal_closure_report.py`：只读取已有 metrics/CSV，聚合 M0/M1/M2/M3、residual shrink schedule、adaptive/two-stage/receiver-predictor residual alpha policy、test-like gate 和 COCO-object clean-correct 结果，生成最小闭环报告、CSV 和 tradeoff 图。
- `s6_residual_shrink_selection.py`：只读取 `EXP-S4-006` 已有 original/M0/refined PNG，生成 residual alpha shrink 候选并用冻结 AlexNet 与图像指标评估 always-accept、top-1 fallback 和 validation-only shrink schedule。
- `s6_apply_residual_shrink_schedule.py`：把 validation 阶段冻结的 residual shrink schedule 应用到 held-out/test-like split，不在目标 split 上重新选 alpha，输出 policy summary、逐样本决策、final PNG 和样例拼图。
- `s6_make_residual_shrink_gallery.py`：聚合 validation/held-out/test-like residual shrink CSV 和已有 PNG，生成 selected shrink M3 的 safe accept/protective reject/rejected good 样例，以及 unsafe always-accept new-error 负对照 gallery。
- `s6_apply_adaptive_residual_alpha_policy.py`：读取已有 residual alpha candidates，在 validation/held-out/test-like 上评估 per-sample 最大 top-1-consistent alpha 策略，并输出 policy summary、逐样本决策、metadata 和样例拼图。
- `s6_apply_two_stage_residual_alpha_policy.py`：读取 adaptive residual alpha 的已有 `per_sample.csv`，组合出 full-strength-then-fixed-schedule 的 two-stage 策略，并只计算无需外部权重的 PSNR/SSIM/MS-SSIM 与语义计数。
- `s6_train_receiver_alpha_predictor.py`：从 adaptive alpha 决策表和候选图提取接收端可见特征，在 validation 上训练小型 tabular alpha predictor，并在预测 alpha 后用 top-1 fallback 保护输出；支持 hard pseudo alpha CE 和 `utility_soft_labels` benefit/risk-aware 目标。
- `s6_train_alpha_head_residual_refiner.py`：加载 `EXP-S4-006` residual refiner checkpoint，冻结 residual CNN 并训练附着其上的 alpha head，用 adaptive alpha pseudo target 探索训练侧 residual amplitude control；支持可选 `training.class_weighting: inverse_frequency` 的 class-weighted CE follow-up。
- `run_s2_coco256_awgn_train.sh`：长任务脚本；负责断点续传 COCO2017 train/val、解压、检查图片数量，并启动 COCO-256 AWGN DeepJSCC GPU 训练。
- `prepare_image_symlink_split.py`：从一个图片目录按固定 seed 生成不重叠的 train/val 符号链接切分，用于 COCO-val pilot 等临时高分辨率训练。
