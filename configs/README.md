# configs

存放实验配置。

每个配置应显式记录：

- 数据集和分辨率
- 信道类型
- SNR
- CBR
- JSCC checkpoint
- diffusion 模型和 steps
- semantic guidance 设置
- 随机种子

## 当前配置

- `s1_deepjscc_cifar10_awgn.yaml`：CIFAR-10 sanity baseline。
- `s2_deepjscc_coco256_awgn.yaml`：COCO2017 256x256 高分辨率 DeepJSCC 重训配置。
- `s2_deepjscc_coco_val256_awgn_pilot.yaml`：使用 COCO `val2017` 固定 4500/500 切分的非正式高分辨率 pilot 训练配置。
- `s3_m1_blind_diffusion_coco256_awgn.yaml`：正式 COCO-256 M0 export 上的 `M1-BlindDiffusion` 小规模 refinement 配置，固定 1/7/19 dB、每个 SNR 16 张图、输入 checkpoint 为 `best.pt`。
- `s4_clip_consistency_m1_exp_s2_002.yaml`：对 `EXP-S2-002` 的 M1 refined 输出做 CLIP image-image consistency 辅助语义诊断，固定读取正式 M0 export 和本地 OpenAI CLIP ViT-B/32 权重。
- `s4_classifier_consistency_m1_exp_s2_002.yaml`：对 `EXP-S2-002` 的 M1 refined 输出做冻结 AlexNet/ImageNet pseudo-label consistency 辅助分类器诊断，固定读取正式 M0 export 和本地 torchvision AlexNet 权重。
- `s4_coco_caption_clip_m1_exp_s2_002.yaml`：对 `EXP-S2-002` 的 M1 refined 输出做 COCO caption CLIP image-text consistency 辅助语义诊断，固定读取正式 M0 export、本地 OpenAI CLIP ViT-B/32 权重和 `data/coco/annotations/captions_val2017.json`。
- `s5_residual_gate_aux_audit_exp_s4_006.yaml`：对 `EXP-S4-006` 的 confidence-gain 候选 gate 做 CLIP image-image 和 COCO caption CLIP 辅助审计，只用于离线风险复核。
- `s5_materialize_conf_gain_gate_exp_s4_006.yaml`：把 `EXP-S4-006` 中 `top1_equal_or_refined_conf_gain_ge_0p05` 候选 gate 的 final PNG 从已有 M0/refined 输出中落盘。
- `s5_residual_refiner_heldout_gate_exp_s4_006.yaml`：加载 `EXP-S4-006` residual refiner checkpoint，在未参与该实验 train/eval 的样本段上复核 confidence-gain 候选 gate。
- `s5_residual_refiner_testlike_gate_exp_s4_006.yaml`：加载同一 `EXP-S4-006` residual refiner checkpoint，在新导出的 `sample_000256`-`sample_000319` test-like 样本段上复核 confidence-gain 候选 gate，不用于调参。
- `s5_conf_gain_clip_veto_sweep_exp_s4_006.yaml`：对 `EXP-S4-006` confidence-gain gate 增加 receiver-side `CLIP(M0, refined)` 二级 veto 扫描，同时比较 validation 和 held-out 风险。
- `s5_conf_gain_clip_veto_snr_calibration_exp_s4_006.yaml`：基于已有 CLIP veto sweep CSV，在 validation 上校准 per-SNR `CLIP(M0, refined)` veto 阈值，并在 held-out 上复核风险。
- `s5_conf_gain_risk_rule_sweep_exp_s4_006.yaml`：基于已有 CLIP/top-k CSV，在 validation 上搜索 receiver-side confidence-gain risk rules，并在 held-out 上复核 accepted-new-error 风险。
- `s5_materialize_risk_rule_gate_exp_s4_006.yaml`：把 risk-rule sweep 选出的 `selected_risk_rule` final PNG 从已有 M0/refined 输出中落盘，并保存 summary、per-sample CSV 和样例 sheet。
- `s5_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml`：用多个冻结 torchvision ImageNet 分类器离线审计 `selected_risk_rule` 的跨模型 repair/new-error 风险。
- `s5_ensemble_risk_veto_sweep_exp_s4_006.yaml`：基于 classifier ensemble audit 的多数票 new-error 风险标签，在 validation 上搜索 `selected_risk_rule` 的 receiver-side 二级 veto，并在 held-out 上复核。
- `s5_receiver_risk_score_sweep_exp_s4_006.yaml`：扫描多个透明 receiver-side risk score 模板及阈值，评估能否比保守二级 veto 少误杀 repair，同时保持 ensemble majority new-error 安全。
- `s5_testlike_risk_rule_check_exp_s4_006.yaml`：把冻结的 `selected_risk_rule` 和保守 ensemble-risk veto 应用到 `sample_000256`-`sample_000319` test-like split，只做迁移复核不调参。
- `s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml`：用多个冻结 torchvision ImageNet 分类器离线审计 test-like `selected_risk_rule` 的跨模型 repair/new-error 风险。
- `s5_testlike_coco_object_clip_clean_eval_exp_s4_006.yaml`：用 COCO instance dominant object labels 和本地 OpenCLIP ViT-B/32，在 test-like split 上构造辅助 clean-correct 子集并复核各 gate policy 的 GT-like semantic failure/repair/new-error。
- `s5_residual_diffusion_pilot_coco256_awgn.yaml`：对正式 256 张/SNR M0 export 训练一个 latent-free pixel residual DDPM pilot，验证 diffusion 是否适合作为保守残差建模器。
- `s6_minimal_closure_report.yaml`：聚合 M0、M1、`EXP-S4-006` residual M2/M3、residual shrink schedule 和 test-like 语义审计，生成第一版最小论文闭环报告与 tradeoff 图。
- `s6_residual_shrink_selection_exp_s4_006.yaml`：在 `EXP-S4-006` 已有 M0/refined PNG 上做 residual-strength alpha shrink 派生分析，评估 `M0 + alpha*(refined-M0)` 与 top-1 fallback 的质量/语义 tradeoff。
- `s6_testlike_residual_shrink_schedule_check_exp_s4_006.yaml`：把 validation 上选出的 residual shrink schedule 冻结后应用到 `sample_000256`-`sample_000319` test-like split，复核其迁移质量和 accepted-new-error 风险。
