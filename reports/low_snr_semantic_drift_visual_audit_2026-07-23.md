# 低 SNR 语义漂移定向可视化审计（2026-07-23）

## 结论先行

这轮专门修正了此前 top-LPIPS 审计几乎全落在 19 dB 的选择偏差。我们只在 1 dB 的 192 个既有样本键中筛选“LPIPS 尚可，但冻结 T_cls、三个跨模型分类器或 CLIP 一致性出现异常”的候选，再对其中 15 个 source 去重样本进行人工核查；随后固定这 15 张图，在 −3/−5 dB 做不训练的范围外压力重放。

本轮没有观察到“重建清晰可信，但主体、物体或场景意义与原图不符”的语义漂移。观察到的主要失败模式是：S33 pure JSCC 在信道继续恶化时先出现涂抹、重影和假色，随后变成不可用的模糊色块；SGD-JSCC paper upper bound 的输出仍清晰且主体/场景保持一致，虽然 −5 dB 个别图存在明显 patch 接缝和小细节变化。这不是 SGD 与 S33 的公平胜负结论：SGD 使用作者权重、额外 edge 码率和免费完美 captions，只作为生成式方法的论文协议上界。

## 审计问题

本轮只回答一个定性问题：低 SNR 下的失败更接近以下哪一种？

1. **重建失败**：图像明显变糊、变噪、出现重影或假色，观看者知道它坏了。
2. **语义漂移**：图像仍清晰、自然、可信，但主体身份、物体、数量、动作或场景意义被生成模型改写，观看者可能被“骗过”。

分类器、CLIP 和 LPIPS 只用于寻找值得看的异常样本，最终类别由原图与重建图逐张人工对照确定。自动分类失败不直接等于语义漂移。

## 1 dB 定向筛选合同

- Population：冻结的 Imagenette policy-dev 64 图 × 3 channel seeds，在 `1 dB` 共 192 个键；official validation 未访问。
- 方法：冻结 S33 strong-B0 与既有 SGD-JSCC paper-upper 结果。S33 对入选键按历史 canonical noise 合同推理重放，历史 PSNR 最大绝对误差为 `0.0 dB`；没有训练或调参。
- 感知质量过滤：一般候选要求至少一种方法的 LPIPS 位于该方法 1 dB 分布的前 60%，且两种方法均不差于各自第 90 百分位；T_cls failure 候选只要求两种方法均不差于第 90 百分位，以免真正异常被一般排序淹没。
- 异常信号：冻结 T_cls failure、AlexNet/ResNet-18/MobileNetV3-small 相对原图 top-1 不一致票数、CLIP ViT-B/32 原图—重建余弦相似度低尾。
- 去偏：先保留实际 T_cls failure 层，再按联合异常分数填充；每个 source image 最多一条，共 15 张。192 个键中过滤后候选 84 个，实际 T_cls failure 候选 5 个，最终覆盖其中 3 个不同 source。

所用 CLIP checkpoint 已在本机，未联网下载。第一次尝试因 PyTorch 2.11 对本地 TorchScript checkpoint 的默认 `weights_only=True` 不兼容而 fail-closed，失败目录完整保留；最终有效运行显式把该受信任本地 checkpoint 按旧兼容模式加载。

## 人工核查结果

人工标签固定为：`faithful`、`reconstruction_failure_blur_noise`、`semantic_drift_clear_wrong`、`uncertain`。

| SNR | 方法 | faithful | 重建失败：糊/噪/重影/假色 | 语义漂移：清晰但错 | uncertain |
|---:|---|---:|---:|---:|---:|
| 1 dB | S33 pure JSCC | 8 | 7 | 0 | 0 |
| 1 dB | SGD diffusion upper bound | 15 | 0 | 0 | 0 |
| −3 dB stress | S33 pure JSCC | 1 | 14 | 0 | 0 |
| −3 dB stress | SGD diffusion upper bound | 15 | 0 | 0 | 0 |
| −5 dB stress | S33 pure JSCC | 0 | 15 | 0 | 0 |
| −5 dB stress | SGD diffusion upper bound | 15 | 0 | 0 | 0 |

1 dB 入选 15 张的描述性均值如下。由于这是异常定向、source 去重的选择样本，不是随机 population，禁止把均值外推为整体性能：S33=`28.398905 dB / 0.173764 LPIPS`，SGD=`26.217593 dB / 0.088304 LPIPS`。入选集里两者各有 2 个冻结 T_cls failures；S33/SGD 的三分类器 mismatch votes 分别为 37/12，CLIP 低尾分别为 8/3，但人工核查的 clear-wrong 均为 0。这说明自动异常信号适合作为召回工具，却不足以单独给语义漂移定罪。

−3/−5 dB 使用同一批冻结 source、同一共同 base seed `20260748`，不根据压力结果重新选图。描述性均值为：

| SNR | 方法 | PSNR | LPIPS | T_cls failures / 15 |
|---:|---|---:|---:|---:|
| −3 dB | S33 | 18.094805 | 0.420703 | 7 |
| −3 dB | SGD upper | 24.441469 | 0.126476 | 2 |
| −5 dB | S33 | 8.952718 | 0.821213 | 12 |
| −5 dB | SGD upper | 22.819949 | 0.161639 | 2 |

这些数值只描述这 15 张定向样本的压力表现，不用于方法排名。尤其是 SGD 在 −5 dB 的若干输出出现四象限接缝或局部颜色/纹理变化；它们是可见重建伪影，但主体、物体和场景仍与原图一致，因此没有标成“清晰但错”。

## 可视化

- 1 dB 定向异常样本：`outputs/analysis/ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-003/low_snr_semantic_risk_top15_reviewed.png`
- −3 dB 范围外压力：`outputs/analysis/ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001/stress_snr_-3_reviewed.png`
- −5 dB 范围外压力：`outputs/analysis/ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001/stress_snr_-5_reviewed.png`

颜色含义：绿色=`faithful`，蓝色=`reconstruction failure`，红色=`semantic drift / clear-but-wrong`，紫色=`uncertain`。本轮没有红色或紫色样本。

## 应该怎样解释

本轮支持的结论很窄但清楚：**在这批低 SNR 异常候选里，S33 的危险更像“明显坏掉”，而不是“清晰地说谎”；SGD paper upper 在完美文本与边缘条件帮助下保持了可感知真实性，也没有观察到语义改写。** 因为“明显坏掉”不会欺骗观看者，而“清晰但错”会，这一区分对 semantic reliability 是有意义的。

本轮不支持以下说法：

- 不支持“SGD 没有 hallucination 风险”。15 张定向样本仍太小，且 perfect caption/edge 本来就在强力约束生成空间。
- 不支持“SGD 胜过 S33”。两者训练数据、训练 SNR、总码率和 side information 合同不一致。
- 不支持用 −3/−5 dB 结果描述训练范围内的总体能力。它们是明确的 out-of-range stress，且 source 来自 1 dB 异常筛选。
- 不支持用 T_cls、跨模型分类器或 CLIP 的单次不一致直接定义语义漂移；本轮已有多个视觉 false positive。

## 公平性边界

S33 是 COCO 上从零训练、原生 `16,384 real`、离散 `[1,4,7,13,19] dB` 训练的 pure JSCC。这里的 SGD 是作者发布权重的 paper protocol upper bound：其 JSCC 训练来自 ImageNet、固定 `10 dB`；main image branch `16,384 real`，active edge 另占 `3,328 real`，四个 captions 按论文协议免费、完美传输。最低未保护计费总量至少 `21,856 real`，比 S33 的严格总预算高 `33.40%`。因此所有图和数值只用于失败模式审计，不作胜负。

## 复现入口与产物

1 dB 有效分析：

```bash
python3 scripts/low_snr_semantic_drift_visual_audit.py \
  --config configs/low_snr_semantic_drift_visual_audit.yaml \
  --stage prepare --device cuda:0

# 填写 manual_review.json 后
python3 scripts/low_snr_semantic_drift_visual_audit.py \
  --config configs/low_snr_semantic_drift_visual_audit.yaml \
  --stage finalize
```

范围外压力历史执行顺序：

```bash
python3 scripts/low_snr_out_of_range_stress.py --stage prepare-s33 --device cuda:0

.venv-sgdjscc/bin/python scripts/external_sgdjscc_common_pilot.py \
  --config outputs/analysis/ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001/sgd_configs/sgd_stress_resolved.yaml \
  --run

python3 scripts/low_snr_out_of_range_stress.py --stage assemble

# 填写 manual_review.json 后
python3 scripts/low_snr_out_of_range_stress.py --stage finalize
```

有效输出分别为：

- `outputs/analysis/ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-003/`
- `outputs/analysis/ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001/`

失败/草稿目录 `ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-001-FAILED-CLIP-TORCHSCRIPT-LOAD` 与 `ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-002` 均按纪律保留，不得当作最终结果。
