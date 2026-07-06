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
- `s5_conf_gain_clip_veto_sweep_exp_s4_006.yaml`：对 `EXP-S4-006` confidence-gain gate 增加 receiver-side `CLIP(M0, refined)` 二级 veto 扫描，同时比较 validation 和 held-out 风险。
- `s5_conf_gain_clip_veto_snr_calibration_exp_s4_006.yaml`：基于已有 CLIP veto sweep CSV，在 validation 上校准 per-SNR `CLIP(M0, refined)` veto 阈值，并在 held-out 上复核风险。
- `s5_conf_gain_risk_rule_sweep_exp_s4_006.yaml`：基于已有 CLIP/top-k CSV，在 validation 上搜索 receiver-side confidence-gain risk rules，并在 held-out 上复核 accepted-new-error 风险。
- `s5_materialize_risk_rule_gate_exp_s4_006.yaml`：把 risk-rule sweep 选出的 `selected_risk_rule` final PNG 从已有 M0/refined 输出中落盘，并保存 summary、per-sample CSV 和样例 sheet。
- `s5_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml`：用多个冻结 torchvision ImageNet 分类器离线审计 `selected_risk_rule` 的跨模型 repair/new-error 风险。
- `s5_residual_diffusion_pilot_coco256_awgn.yaml`：对正式 256 张/SNR M0 export 训练一个 latent-free pixel residual DDPM pilot，验证 diffusion 是否适合作为保守残差建模器。
