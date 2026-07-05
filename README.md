# Channel-Adaptive Semantic-Drift Controlled Diffusion-JSCC

本仓库研究：在不同信道条件下，用 diffusion 增强 DeepJSCC 图像恢复，同时把 semantic drift 作为核心评估目标。

## 文档

- `PROJECT.md`：项目定义、核心问题、假设和方法边界。
- `MILESTONES.md`：最小论文闭环、指标定义、阶段门槛和成功/失败判据。
- `AGENTS.md`：AI agent 和贡献者的协作规则。
- `PROGRESS.md`：当前阶段、已完成内容、下一步和开放决策。
- `EXPERIMENTS.md`：实验记录和结果索引。
- `LITERATURE.md`：相关工作、撞车风险和检索关键词。
- `README.md`：环境安装、运行命令和代码结构。

## 代码结构

```text
configs/          实验配置
data/             数据集说明或本地数据指针
references/       文献 PDF、BibTeX、阅读笔记或外部链接索引
src/              训练、推理、信道和评估代码
scripts/          可复现的命令行流程
outputs/          生成结果、指标和可视化样例
tests/            单元测试和 smoke test
third_party/      外部代码仓库或其路径说明
```

## 环境安装

当前已在用户 Python 环境中安装阶段1和后续研究常用依赖，CUDA 版 PyTorch 已可用，CIFAR-10 已下载到 `data/cifar10/`。

CPU 环境推荐安装命令：

```bash
python3 -m pip install --user --no-cache-dir --default-timeout 120 --retries 10 -r requirements-torch-cpu.txt
python3 -m pip install --user --no-cache-dir --default-timeout 120 --retries 10 -r requirements.txt
python3 -m pip install --user --no-cache-dir --default-timeout 120 --retries 10 -r requirements-research.txt
```

GPU 环境推荐安装命令，适用于当前 RTX 4090 D / CUDA driver 可见的机器：

```bash
python3 -m pip install --user --default-timeout 120 --retries 10 -r requirements-torch-cu128.txt
```

当前已验证的关键版本：

- Python：`3.10.12`
- torch：`2.11.0+cu128`
- torchvision：`0.26.0+cu128`
- numpy：`2.2.6`
- pillow：`12.2.0`
- diffusers：`0.38.0`
- transformers：`5.12.1`

GPU 备注：

- `nvidia-smi` 在非沙箱环境可见 RTX 4090 D，显存约 24GB。
- 非沙箱 Python 已验证 `torch.cuda.is_available()` 为 True，设备为 `NVIDIA GeForce RTX 4090 D`。
- 256x256 高分辨率训练脚本已通过 GPU dry-run，输出位于 `outputs/smoke/s2_deepjscc_coco256_train_gpu/`。

已知问题：

- 直接执行 `python3 -m pip install -r requirements.txt` 不会安装 PyTorch。
- 不建议无脑安装第三方仓库原始 `requirements.txt`，其中 `torchvison` 拼写错误，并且默认 PyPI 路线可能拉取很大的 CUDA 依赖。
- 2026-06-29 早前曾遇到 PyTorch 下载超时和 CPU-only 路线 hash mismatch；后续使用 `--user --no-cache-dir --default-timeout 120` 已成功安装。

下载流量规则：

- 大模型、大数据集、CUDA/PyTorch 等大文件下载默认走服务器直连，不走用户本机代理流量。
- 下载前先检查代理：

```bash
env | grep -i proxy
```

- 若存在代理变量，默认用清空代理变量的方式执行大下载：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy <download-command>
```

- 只有用户明确说可以使用代理/本机流量时，才允许通过代理下载大文件。

第一阶段建议先审计并尝试接入普通 DeepJSCC PyTorch baseline，再决定是否训练自己的 checkpoint。

候选 baseline：

1. `chunbaobao/Deep-JSCC-PyTorch`：第一优先候选，用于快速跑通普通 DeepJSCC baseline。
2. `mingyuyng/Dynamic_JSCC`：第二候选，用于理解 adaptive rate control。
3. `aprilbian/deepjscc-lplusplus`：第三候选，用于后续 channel-adaptive JSCC 对照。

## 第一阶段目标

构建 baseline 评估 pipeline：

1. 在一组 SNR 上运行 DeepJSCC 重建。
2. 加入 blind diffusion refinement。
3. 测量重建质量、感知质量和 semantic drift。
4. 加入 semantic guidance 和 SNR-aware diffusion strength。
5. 比较不同方法的 tradeoff 曲线。

## 最小闭环

本项目优先完成 `MILESTONES.md` 中定义的最小论文闭环：

- CIFAR-10 + AWGN 只作为 sanity baseline。
- COCO2017 `256x256` + AWGN 作为 diffusion 主实验闭环。
- CBR 固定为 `0.17`。
- SNR sweep 固定为 `[1, 4, 7, 13, 19]` dB。
- 对比 `M0-DeepJSCC`、`M1-BlindDiffusion`、`M2-SNRAdaptiveDiffusion`、`M3-Ours`。
- 用 semantic drift / final failure 约束 diffusion refinement 的语义可靠性。

完成该闭环前，不扩展到大型 DiT-JSCC 或复杂 adaptive JSCC 主线。

## 当前 smoke test

第三方 baseline 已浅克隆到：

```text
third_party/Deep-JSCC-PyTorch
```

已写好 smoke test：

```bash
python3 scripts/s1_deepjscc_smoke.py --device cpu --batch-size 2
```

说明：

- 该 smoke test 使用随机合成图像，不下载 CIFAR-10。
- 该 smoke test 只验证 checkpoint 加载、SNR 切换、重建输出和 PSNR 计算。
- smoke test 不是正式实验，不写入 `EXPERIMENTS.md`。
- 当前状态：已在 CPU 上通过 smoke test，输出位于 `outputs/smoke/s1_deepjscc/`。

## 当前 baseline

CIFAR-10 已下载到：

```text
data/cifar10/
```

真实 CIFAR-10 test subset mini-eval：

```bash
python3 scripts/s1_deepjscc_mini_eval.py --device cpu
```

首次下载数据集时使用：

```bash
python3 scripts/s1_deepjscc_mini_eval.py --device cpu --download
```

正式阶段1 baseline 已完成：

```bash
python3 scripts/s1_deepjscc_mini_eval.py --device cpu --num-samples 1024 --batch-size 64 --output-dir outputs/EXP-S1-001 --formal
```

输出位于：

```text
outputs/EXP-S1-001/
```

说明：CIFAR-10 图像为 32x32，`pytorch-msssim` 默认 MS-SSIM 要求边长大于 160，因此当前 S1 记录 PSNR/SSIM，MS-SSIM 留到高分辨率数据集或自定义设置后再启用。

## 高分辨率重训路线

CIFAR-10 只作为 sanity baseline。后续 diffusion 主路线需要重新训练或接入高分辨率 DeepJSCC checkpoint。

当前推荐主路线：

- 训练数据：COCO2017 `train2017`
- 验证数据：COCO2017 `val2017`
- 图像尺寸：`256x256`
- 信道：AWGN
- CBR：`0.17`
- 初始训练 SNR：`7` dB，后续扩展到 `[1, 4, 7, 13, 19]` dB

数据目录约定：

```text
data/coco/train2017/
data/coco/val2017/
```

当前 COCO2017 数据已就位：

```text
data/coco/train2017/  # 118287 images
data/coco/val2017/    # 5000 images
data/coco/annotations/ # COCO2017 captions / instances / keypoints JSON
```

COCO2017 官方 annotations 已通过服务器直连下载并验证：

```text
data/coco/annotations_trainval2017.zip  # 252907541 bytes, unzip -t OK
data/coco/annotations/captions_val2017.json
data/coco/annotations/instances_val2017.json
```

训练脚本：

```bash
python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco256_awgn.yaml --device cuda:0
```

数据准备并启动训练的长任务脚本：

```bash
scripts/run_s2_coco256_awgn_train.sh
```

该脚本中的 COCO 下载命令使用 `wget --no-proxy`，只让数据集下载直连，不影响 Codex 或其他命令使用当前代理环境。

如需临时覆盖数据路径，可使用：

```bash
python3 scripts/train_deepjscc_highres.py --train-root data/coco/val2017 --val-root data/coco/val2017 --device cuda:0
```

当前机器已验证可用 GPU。CPU dry-run 和 GPU dry-run 都已通过，GPU dry-run 命令：

```bash
python3 scripts/train_deepjscc_highres.py --dry-run --device cuda:0 --epochs 1 --batch-size 2 --num-workers 0 --max-train-batches 1 --max-val-batches 1 --output-dir outputs/smoke/s2_deepjscc_coco256_train_gpu
```

CPU dry-run 命令：

```bash
python3 scripts/train_deepjscc_highres.py --dry-run --device cpu --epochs 1 --batch-size 2 --num-workers 0 --max-train-batches 1 --max-val-batches 1
```

CPU dry-run 输出位于：

```text
outputs/smoke/s2_deepjscc_coco256_train/
```

GPU dry-run 输出位于：

```text
outputs/smoke/s2_deepjscc_coco256_train_gpu/
```

真实 COCO `val2017` 图像 GPU smoke 已通过，输出位于：

```text
outputs/smoke/s2_deepjscc_coco256_val2017_gpu/
```

在 `train2017` 下载较慢时，可用已完成的 `val2017` 做非正式高分辨率 pilot：

```bash
python3 scripts/prepare_image_symlink_split.py --source-root data/coco/val2017 --output-root data/coco_val_split --train-size 4500 --val-size 500 --seed 42 --overwrite
python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0
```

当前 pilot 训练输出位于：

```text
outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/
```

该 pilot 只用于高分辨率 JSCC checkpoint 和后续 diffusion 接口调试，不能替代正式 COCO `train2017/val2017` 主实验。

pilot checkpoint 的 M0-HR SNR sweep 和 `x_hat` 导出：

```bash
python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 32 --output-dir outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export
```

输出位于：

```text
outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/
```

其中 `exports/original/` 保存原图，`exports/snr_XXdb/reconstruction/` 保存各 SNR 的 DeepJSCC 重建图，可作为 `M1-BlindDiffusion` 的输入。

正式 COCO-256 训练已完成，但后段出现 NaN。可用 checkpoint 是：

```text
outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt
```

该 checkpoint 来自 epoch 73，验证指标约为 PSNR `31.5618` dB、SSIM `0.9054`。

不要使用：

```text
outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/latest.pt
```

`latest.pt` 来自 epoch 99，参数和指标均已 NaN。

正式 `M0-HR` SNR sweep 和 `x_hat` 导出位于：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/
```

该目录每个 SNR 只保存 32 张 PNG，主要用于复现 `EXP-S2-002` 到 `EXP-S4-005`。更大的 residual validation export 位于：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/
```

该目录仍在同一 512 张 COCO val subset 上评估 M0，但每个 SNR 保存前 256 张 PNG。复现命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 256 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256
```

后续 `M1-BlindDiffusion` 应优先读取：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/exports/
```

当前没有需要等待的 COCO 下载或训练 screen 会话。

## M1-BlindDiffusion 最小接口

当前 M1 配置：

```text
configs/s3_m1_blind_diffusion_coco256_awgn.yaml
```

该配置固定读取正式 M0 export：

```text
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/
```

并记录正式 DeepJSCC checkpoint：

```text
outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt
```

不要使用 `latest.pt`。

先做输入检查，不加载 diffusion 模型：

```bash
python3 scripts/s3_blind_diffusion_refine.py --dry-run
```

正式小规模 M1 运行命令：

```bash
python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download
```

默认设置：

- SNR：`[1, 7, 19]` dB
- 每个 SNR：16 张图
- diffusion：`runwayml/stable-diffusion-v1-5` img2img
- strength：`0.25`
- steps：`25`
- guidance scale：`1.0`
- 默认输出：`outputs/EXP-S2-002/`

输出包括：

```text
outputs/EXP-S2-002/exports/snr_XXdb/refined/
outputs/EXP-S2-002/samples/
outputs/EXP-S2-002/metrics.json
outputs/EXP-S2-002/source_manifest.json
```

`metrics.json` 会在相同 16 张图上报告 M0 reconstruction 和 M1 refined 相对原图的 PSNR、SSIM、MS-SSIM；LPIPS 若环境能初始化会一并写入，否则记录失败原因。

当前状态：

- `--dry-run` 已通过。
- `EXP-S2-001` 是早先环境阻塞记录，未创建输出目录。
- `EXP-S2-002` 已完成，输出位于 `outputs/EXP-S2-002/`。
- `runwayml/stable-diffusion-v1-5` 已缓存到 `outputs/cache/huggingface/`；LPIPS AlexNet 权重已缓存到 `outputs/cache/torch/`。
- 官方 `huggingface.co` 服务器直连在 2026-07-01 超时；`hf-mirror.com` 服务器直连可用。后续大下载仍必须清空代理变量，不走用户代理流量。

`EXP-S2-002` 结论：当前 `strength=0.25`、空 prompt、`guidance_scale=1.0` 的 blind SD img2img 是明显负结果。M1 在 1/7/19 dB 上 PSNR 和 MS-SSIM 大幅下降，LPIPS 也变差；样例图显示 hallucination 和 semantic drift 风险。后续不要把该设置包装成视觉提升。

## S4 Semantic Drift 初步诊断

当前已完成 `EXP-S3-001`：对 `EXP-S2-002` 的 M0 reconstruction 和 M1 refined 输出做 CLIP image-image consistency 辅助诊断。

配置：

```text
configs/s4_clip_consistency_m1_exp_s2_002.yaml
```

运行命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_clip_consistency_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S3-001/metrics.json
outputs/EXP-S3-001/per_sample.csv
outputs/EXP-S3-001/source_manifest.json
```

从 CLIP drop 指标整理 failure case gallery：

```bash
python3 scripts/s4_make_clip_failure_gallery.py
```

gallery 输出：

```text
outputs/EXP-S3-001/failure_cases/sheets/global_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/sheets/snr_01db_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/sheets/snr_07db_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/sheets/snr_19db_top_clip_drop.png
outputs/EXP-S3-001/failure_cases/triptychs/
outputs/EXP-S3-001/failure_cases/index.json
outputs/EXP-S3-001/failure_cases/global_top_clip_drop.csv
```

CLIP 权重缓存：

```text
outputs/cache/open_clip/ViT-B-32.pt
```

SHA256：

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

若缓存缺失，可用服务器直连下载 OpenAI 官方权重：

```bash
mkdir -p outputs/cache/open_clip
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy curl -L --fail --retry 3 --connect-timeout 20 -C - -o outputs/cache/open_clip/ViT-B-32.pt https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

结论：该指标不是最终分类一致性 semantic drift metric，但已经量化确认 M1 负结果。1/7/19 dB 下，M1 相对原图的 CLIP 相似度均显著低于 M0；所有 48 个样本中 M1 都低于 M0。failure gallery 已固化全局 top 12 和每个 SNR top 6 的 original/M0/M1 triptych；后续应补冻结分类器或 object-level 语义一致性指标。

当前也已完成 `EXP-S3-002`：冻结 ImageNet AlexNet 的 pseudo-label consistency 诊断。该实验不使用 COCO GT 标签，只比较 `c(original)`、`c(M0)` 和 `c(M1)`，因此仍是辅助诊断，不是最终 clean-correct 分类指标。

配置：

```text
configs/s4_classifier_consistency_m1_exp_s2_002.yaml
```

运行命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_classifier_consistency_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S3-002/metrics.json
outputs/EXP-S3-002/per_sample.csv
outputs/EXP-S3-002/source_manifest.json
```

从 pseudo-label drift 指标整理 classifier failure gallery：

```bash
python3 scripts/s4_make_classifier_failure_gallery.py
```

gallery 输出：

```text
outputs/EXP-S3-002/failure_cases/sheets/global_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/sheets/snr_01db_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/sheets/snr_07db_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/sheets/snr_19db_top_classifier_drift.png
outputs/EXP-S3-002/failure_cases/triptychs/
outputs/EXP-S3-002/failure_cases/index.json
outputs/EXP-S3-002/failure_cases/global_top_classifier_drift.csv
```

该脚本默认读取本地 AlexNet 权重：

```text
outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
```

结论：all-subset 中，M0 与原图 top-1 一致率在 1/7/19 dB 为 `0.5000/0.6875/0.9375`，M1 仅为 `0.1250/0.0625/0.1250`；在原图 top-1 confidence >= 0.30 的 subset 上，M0 为 `0.8889/1.0000/1.0000`，M1 为 `0.2222/0.1111/0.2222`。这进一步确认当前 blind diffusion 设置存在系统性 semantic drift。

将 M1 图像质量、CLIP 诊断和分类器诊断聚合成一个派生报告：

```bash
python3 scripts/s4_summarize_m1_negative_result.py
```

输出：

```text
outputs/analysis/m1_negative_result_summary/REPORT.md
outputs/analysis/m1_negative_result_summary/summary.csv
outputs/analysis/m1_negative_result_summary/summary.json
```

该报告不新增模型运行，只汇总已有实验。当前汇总结论：平均 PSNR delta M1-M0 为 `-14.7485` dB，平均 LPIPS delta 为 `+0.3877`，平均 CLIP drop 为 `0.2672`，分类器 all-subset M1 pseudo drift-origin 为 `0.8958`。

当前也已完成 `EXP-S3-003`：COCO caption CLIP image-text consistency 诊断。该实验使用 COCO `captions_val2017.json`，把导出的 `sample_XXXXXX.png` 反查到 COCO image id 和 5 条人工 captions，然后比较 original/M0/M1 与 captions 的 CLIP image-text 相似度。它仍是辅助语义诊断，不替代最终 clean-correct 冻结分类器指标。

配置：

```text
configs/s4_coco_caption_clip_m1_exp_s2_002.yaml
```

运行命令：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_coco_caption_clip_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S3-003/metrics.json
outputs/EXP-S3-003/per_sample.csv
outputs/EXP-S3-003/sample_metadata.json
outputs/EXP-S3-003/source_manifest.json
```

从 caption CLIP drop 指标整理 failure case gallery：

```bash
python3 scripts/s4_make_coco_caption_failure_gallery.py
```

gallery 输出：

```text
outputs/EXP-S3-003/failure_cases/sheets/global_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/sheets/snr_01db_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/sheets/snr_07db_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/sheets/snr_19db_top_caption_clip_drop.png
outputs/EXP-S3-003/failure_cases/triptychs/
outputs/EXP-S3-003/failure_cases/index.json
outputs/EXP-S3-003/failure_cases/global_top_caption_clip_drop.csv
```

结论：caption 语义诊断继续确认当前 blind diffusion 负结果。1/7/19 dB 下，M0 caption-max mean 为 `0.3306/0.3305/0.3263`，M1 为 `0.2816/0.2815/0.2877`；M1 caption-max 低于 M0 的比例为 `1.0000/0.8125/0.8125`。

## S5 Semantic Failure Handling Pilot

当前已完成 `EXP-S4-001`：基于 `EXP-S2-002` 的 M1 输出和 `EXP-S3-002` 的冻结分类器 CSV，评估一个最小 receiver-side fallback 规则。

配置：

```text
configs/s5_semantic_fallback_m1_exp_s2_002.yaml
```

规则：

- 若冻结 AlexNet 对 M1 refined 和 M0 reconstruction 的 top-1 预测一致，则接受 M1。
- 否则回退到 M0 reconstruction。
- detector 不使用 original 图像；original pseudo-label 只用于离线评价 Final-Failure。

先检查输入对齐：

```bash
python3 scripts/s5_semantic_fallback_eval.py --dry-run
```

运行 pilot：

```bash
python3 scripts/s5_semantic_fallback_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-001/metrics.json
outputs/EXP-S4-001/per_sample.csv
outputs/EXP-S4-001/REPORT.md
outputs/EXP-S4-001/exports/snr_XXdb/final/
outputs/EXP-S4-001/samples/snr_01db_original_m0_m1_m3final.png
outputs/EXP-S4-001/samples/snr_07db_original_m0_m1_m3final.png
outputs/EXP-S4-001/samples/snr_19db_original_m0_m1_m3final.png
```

结论：该 fallback 把 all-subset pseudo Final-Failure 从 M1 的 `0.8750/0.9375/0.8750` 降回 M0/M3 的 `0.5000/0.3125/0.0625`，false accept 和 false reject 在当前 48 个样本上均为 0。但它不是完整 M3/Ours，因为仍沿用固定 `strength=0.25` 的负结果 M1；少量 accepted M1 会拉低 PSNR 和 LPIPS。下一步应新建实验 ID 做 `strength <= 0.10` 的 SNR-aware validation 网格，再接这个 fallback 规则。

## S5 SNR-Aware Strength Validation

当前已完成 `EXP-S4-002`：在正式 COCO-256 M0 export 上运行低强度 diffusion validation，覆盖 `[1, 4, 7, 13, 19]` dB，每个 SNR 8 张图。

配置：

```text
configs/s5_snr_adaptive_diffusion_strength_validation.yaml
```

候选：

- `fixed_0p05`：所有 SNR 使用 `strength=0.05`。
- `snr_adaptive_0p10_to_0p05`：1/4/7/13/19 dB 使用 `0.10/0.08/0.06/0.05/0.05`，满足 strength 随 SNR 升高不增加。

先检查输入和 schedule：

```bash
python3 scripts/s5_snr_adaptive_diffusion_validation.py --dry-run
```

运行 validation。该命令应使用本地缓存；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_snr_adaptive_diffusion_validation.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-002/metrics.json
outputs/EXP-S4-002/summary.csv
outputs/EXP-S4-002/per_sample.csv
outputs/EXP-S4-002/REPORT.md
outputs/EXP-S4-002/candidates/fixed_0p05/
outputs/EXP-S4-002/candidates/snr_adaptive_0p10_to_0p05/
```

结论：低强度 diffusion 比 `strength=0.25` 语义更稳，但仍明显损伤图像质量。即使 `strength=0.05`，refined PSNR/LPIPS 仍显著差于 M0；fallback 可降低 final failure，但无法弥补 refined 图像本身的质量损伤。下一步应优先做 SD VAE/latent roundtrip 诊断，判断损伤来自 VAE 重编码、最小 denoise step，还是 prompt-free generative prior。

## S5 SD VAE Roundtrip Diagnostic

当前已完成 `EXP-S4-003`：只加载 Stable Diffusion v1.5 的 VAE，对正式 COCO-256 M0 export 做 encode/decode roundtrip，不运行 UNet denoise，不使用 prompt。

配置：

```text
configs/s5_sd_vae_roundtrip_coco256_awgn.yaml
```

先检查输入和样本对齐：

```bash
python3 scripts/s5_sd_vae_roundtrip_eval.py --dry-run
```

运行诊断。该命令应使用本地缓存；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sd_vae_roundtrip_eval.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-003/metrics.json
outputs/EXP-S4-003/summary.csv
outputs/EXP-S4-003/per_sample.csv
outputs/EXP-S4-003/REPORT.md
outputs/EXP-S4-003/exports/snr_XXdb/m0_vae_roundtrip/
outputs/EXP-S4-003/exports/snr_XXdb/original_vae_roundtrip/
outputs/EXP-S4-003/samples/
```

结论：SD VAE roundtrip 本身已经显著损伤高保真 M0。M0-VAE 相对 M0 的 PSNR 损失从 1 dB 的 `-3.4852` dB 扩大到 19 dB 的 `-7.3260` dB，LPIPS 也变差 `+0.0090` 到 `+0.0578`。这说明当前通用 Stable Diffusion img2img 路线不是简单调低 `strength` 就能变成有效视觉增强；后续应优先考虑 restoration-aware 或 latent-free/像素域保守模块，并继续记录 semantic drift。

## S5 Pixel Residual Restoration Pilot

当前已完成 `EXP-S4-005`：避开 Stable Diffusion 和 SD VAE，只在像素域训练一个小型 SNR-conditioned residual refiner。

配置：

```text
configs/s5_residual_refiner_pilot_coco256_awgn.yaml
```

切分：

- train：`sample_000008.png` 到 `sample_000031.png`，每个 SNR 24 张
- eval：`sample_000000.png` 到 `sample_000007.png`，每个 SNR 8 张

先检查输入和切分：

```bash
python3 scripts/s5_residual_refiner_pilot.py --dry-run
```

运行 pilot。该命令不下载模型或数据；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --device cuda:0
```

输出：

```text
outputs/EXP-S4-005/checkpoints/best.pt
outputs/EXP-S4-005/train_history.csv
outputs/EXP-S4-005/metrics.json
outputs/EXP-S4-005/summary.csv
outputs/EXP-S4-005/per_sample.csv
outputs/EXP-S4-005/REPORT.md
outputs/EXP-S4-005/exports/snr_XXdb/refined/
outputs/EXP-S4-005/exports/snr_XXdb/final/
outputs/EXP-S4-005/samples/
```

结论：这是小样本 pilot，不是最终 M2/M3/Ours，但方向明显比通用 SD img2img 更健康。1/4/7/13/19 dB 上 refined PSNR 相比 M0 分别提升 `+0.3866/+0.1868/+0.0905/+0.1248/+0.1682` dB；LPIPS 除 7 dB 基本持平外均改善；pseudo final failure 没有高于 M0。

注意：`EXP-S4-004` 是同一 pilot 的失败尝试，训练完成后因 `train_history.csv` 字段写入 bug 中断，保留在 `outputs/EXP-S4-004/`，不要复用该实验 ID。

## S5 Pixel Residual Restoration Validation

当前已完成 `EXP-S4-006`：使用更大的 M0 export 训练/验证同一个 SNR-conditioned residual refiner。

配置：

```text
configs/s5_residual_refiner_validation_coco256_awgn.yaml
```

切分：

- train：`sample_000032.png` 到 `sample_000191.png`，每个 SNR 160 张
- eval：`sample_000192.png` 到 `sample_000255.png`，每个 SNR 64 张

先检查输入和切分：

```bash
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --dry-run
```

运行 validation。该命令不下载模型或数据；如环境里有代理变量，建议清空代理变量运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --device cuda:0
```

输出：

```text
outputs/EXP-S4-006/checkpoints/best.pt
outputs/EXP-S4-006/train_history.csv
outputs/EXP-S4-006/metrics.json
outputs/EXP-S4-006/summary.csv
outputs/EXP-S4-006/per_sample.csv
outputs/EXP-S4-006/REPORT.md
outputs/EXP-S4-006/exports/snr_XXdb/refined/
outputs/EXP-S4-006/exports/snr_XXdb/final/
outputs/EXP-S4-006/samples/
```

结论：pure refined 在 1/4/7/13/19 dB 上 PSNR 分别提升 `+1.1323/+0.7837/+0.5859/+0.5504/+0.5654` dB，LPIPS 全部改善；经过 top-1 agreement fallback 后，M3 final PSNR 仍提升 `+0.3313/+0.3812/+0.3815/+0.4557/+0.4561` dB，且 pseudo final failure 未高于 M0。低 SNR 下 accept rate 较低，后续应做 detector error analysis，而不能把 pure refined 直接当最终方法。

## S5 Semantic Gate Error Analysis

当前已完成 `EXP-S4-006` 的派生 gate error analysis。该流程不跑模型、不联网，只读取：

```text
outputs/EXP-S4-006/per_sample.csv
outputs/EXP-S4-006/exports/
outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/exports/
```

运行：

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

核心结论：当前 gate 是 `c(refined) == c(M0)` 的 top-1 agreement，因此在同一个冻结分类器口径下，M3 final failure 不会超过 M0 是结构性保证；这还不是独立语义可靠性证明。分析中 `protective_reject` 有 28/320 个，说明 gate 确实阻止了一批 refined 改坏 pseudo-label 的情况；`missed_semantic_repair` 有 41/320 个，说明 gate 也拒绝了不少 refined 把 M0 pseudo-label 修回原图 pseudo-label 的样本。下一版 gate 应考虑 top-k、confidence margin 或 CLIP/caption 辅助，允许可信修复，同时保留保护性拒绝。

## S5 Semantic Gate Policy Sweep

当前已完成 `EXP-S4-006` 的派生 gate policy sweep。该流程不训练、不下载，只用本地 AlexNet 权重重新计算 original/M0/refined top-5，然后离线比较 receiver-side gate 策略。

运行：

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

核心结论：`top1_equal_or_refined_conf_gain_ge_0p05` 是当前最均衡的候选 gate。相对原始 top-1 agreement gate，全局 final failure 从 `0.3750` 降到 `0.3188`，final PSNR 提升 `+0.1153` dB，missed repair 从 `41` 降到 `20`；代价是 accepted new error 从 `0` 增到 `3`。top-5 overlap 类策略虽然 final PSNR 更高，但 accepted new error 明显更多，风险偏大。该 sweep 是 validation 派生分析，不能直接作为最终 M3 结论。

## 项目进度可视化汇总

可从已有 metrics、CSV 和 failure gallery 生成一套派生总览报告；该流程不跑训练、不跑 diffusion、不重新计算模型指标：

```bash
python3 scripts/s4_make_project_progress_visual_summary.py
```

输出：

```text
outputs/analysis/project_progress_visual_summary/REPORT.md
outputs/analysis/project_progress_visual_summary/summary.json
outputs/analysis/project_progress_visual_summary/coco256_m0_snr_sweep.csv
outputs/analysis/project_progress_visual_summary/m1_blind_diffusion_summary.csv
outputs/analysis/project_progress_visual_summary/figures/stage_progress.png
outputs/analysis/project_progress_visual_summary/figures/m0_snr_curves.png
outputs/analysis/project_progress_visual_summary/figures/m1_quality_metrics.png
outputs/analysis/project_progress_visual_summary/figures/m1_semantic_diagnostics.png
outputs/analysis/project_progress_visual_summary/figures/m1_negative_deltas.png
outputs/analysis/project_progress_visual_summary/figures/representative_visual_outputs.png
```

该报告适合快速查看当前项目进度、正式 M0 COCO-256 baseline、M1 负结果和已有 semantic drift failure case。
