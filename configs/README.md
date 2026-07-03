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
