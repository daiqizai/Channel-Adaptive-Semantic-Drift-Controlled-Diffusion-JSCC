# RDD-P0 生成式重建分布偏移结果

日期：2026-07-30
实验ID：`ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001`
预注册：`reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md`（在任何分布指标产生前冻结）
状态：完成

## 一句话结论

按预注册三分判据，本轮结果是 **①②同时成立 → "存在可识别的定向偏移"**。但②的方向**不是**"生成式重建偏向自己的生成先验"，而是所有方法（含纯判别式）都偏离源分布；且 C3 对照证明可识别指纹**不是生成先验特有**。因此：

> **现有生成式 JSCC 的重建确实存在可识别的、非源分布的偏移，但当前证据不支持"偏移由生成先验定向导致"这一更强解释。RDD 的 deception 项在本项目语境下有实证立足点，但立足点比预期弱，且必须与"任何有损 JSCC 都偏离源分布"这一平凡事实区分开。**

## 前提修正（重要）

用户原始设计为"CLIC2020 test 428 图 × S33/DiffJSCC/SGD"。只读核验证明该设计**在现有输出上不可执行**：`paper_idea1b/A1_DISCRIMINATIVE_RESULT.md:19` 明确记录"DiffJSCC、SGD 和 refiner 未加载"，CLIC 重建目录只有 S33 + 两条 Swin 臂。三方法唯一共存总体是 **64 图 Imagenette policy-dev @256²**（5 SNR × 3 seeds = 每方法 960 行，每 (method,SNR) 单元 n=192）。

用户确认改为：主实验用该共享总体（并新增 author-JSCC 第四臂），另加 CLIC-428 判别式补充。未授权新的 CLIC 生成式推理，A2 仍未授权。

## 数据完整性与验证门（全部通过）

四臂均来自现有输出，无新增生成式推理：

| 臂 | 先验 | 恢复方式 | 验证 |
|---|---|---|---|
| `s33_strong` | 无 | 冻结 checkpoint 精确重放 | max\|ΔPSNR\|=**0.0 dB**，960/960 noise SHA 校验通过 |
| `author_jscc` | 无 | S30 montage 面板1 | max\|ΔPSNR\|=**5.46e-06 dB** |
| `diffjscc` | SD 2.1 | S30 montage 面板2 | max\|ΔPSNR\|=**3.98e-06 dB** |
| `sgd_jscc` | MDTv2 (DiT) | S20 montage tile 裁切 | max\|Δ\|=0.0385 dB、median=0.0030 dB |

- **SGD 源 tile 与 DiffJSCC 源面板逐字节相同：0 mismatch / 64。** 证明两条链共享同一总体，跨方法分布比较合法。
- SGD 的 0.0385 dB 残差与 S34C 记录的 uint8-截断 vs float-张量口径差一致，不是裁切错误。
- 每臂 960 图，共 3,840；参考集各 64 图，共 10 组。

**测量链独立交叉验证**：本轮 CLIC 管线在 7 个与 A1 重叠的单元上复现 A1 冻结值至 **ΔFID < 0.008**、**ΔKID < 3e-6**（残差来自 KID 子集 RNG 与 PNG 重编码）。

## 统计功效边界（事前声明，不可忽略）

共享总体每单元 **n=192**，2048 维 Inception 协方差秩亏，FID 有正偏。**KID 为主判据，FID 必报作次要**（沿用 S34C 的"KID 主、FID 必报"先例）。CLIC 补充 n=428，功效充足。

## 参考分布三角（各参考集自身离 real 多远）

n=64，256²：

| 参考集 | FID vs real | KID vs real |
|---|---:|---:|
| `resample_512` | 2.4985 | −0.008108 |
| `blur_s0p5` | 4.1150 | −0.008093 |
| `jpeg_q70` | 12.6422 | −0.007112 |
| `vae_sgd` | 18.7427 | −0.007035 |
| `vae_sd21` | 19.1215 | −0.007354 |
| `blur_s1` | 30.7117 | −0.004916 |
| `jpeg_q30` | 33.1255 | −0.005083 |
| `blur_s1p5` | 64.0663 | +0.002614 |
| `blur_s2` | 96.0044 | +0.010456 |

两个 VAE 往返 PSNR：`vae_sd21`=**28.015 dB**、`vae_sgd`=**30.773 dB**，均为忠实往返。

## 核心结果：criterion ②（KID 意义下最近的参考分布）

| 臂 | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB | real 的排名（逐档） |
|---|---|---|---|---|---|---|
| `s33_strong` | blur_s1p5 | blur_s1 | blur_s1 | blur_s1 | blur_s1 | **9,10,9,9,9 / 10** |
| `author_jscc` | blur_s2 | blur_s1p5 | blur_s1 | blur_s1 | blur_s1 | 9,10,10,8,6 / 10 |
| `diffjscc` | **vae_sd21** | **vae_sd21** | **vae_sd21** | vae_sgd | resample_512 | 4,4,4,4,4 / 10 |
| `sgd_jscc` | **vae_sd21** | **vae_sd21** | **vae_sgd** | vae_sd21 | **vae_sgd** | 6,6,5,6,4 / 10 |

②在 116 个 (arm, SNR, reference) 组合上成立，其中**强②仅 12 个，且全部是判别式臂 → blur**（`s33_strong→blur_s1p5/blur_s2` 7 个，`author_jscc→blur_s1p5/blur_s2` 5 个）。生成臂的②全部为**弱②**。

**关键定性结构（这是本轮真正的发现）：**

1. **两个判别式臂偏向平滑分布。** real 在 10 个候选中排名 9–10 / 10，即"重建离源分布几乎是最远的"。最近的总是某档高斯模糊，这与 MSE 训练导致低通化的预期完全一致。
2. **两个生成臂偏向 VAE 往返分布。** 最近参考几乎总是 `vae_sd21` 或 `vae_sgd`，且 real 排名（DiffJSCC 恒为 4/10，SGD 为 4–6/10）总体好于判别式臂（`s33_strong` 恒为 9–10/10；`author_jscc` 在 1–7 dB 为 9–10，13/19 dB 回升到 8/6）。即生成臂**同时**比判别式臂更接近源分布、又可测地带有 VAE 先验痕迹。
3. **但方向性归因不成立。** `diffjscc` 与 `sgd_jscc` 都最常偏向 `vae_sd21`，而 `sgd_jscc` 的先验是 MDTv2/DiT 而非 SD 2.1。若偏移真由"各自的生成先验"定向导致，SGD 应系统性偏向 `vae_sgd`。实际是两个 VAE 参考集互相接近（FID 18.74 vs 19.12），无法区分。**因此只能说"偏向某种 VAE-latent 往返痕迹"，不能说"偏向各自的先验"。**

## 核心结果：criterion ①（指纹分类）

GroupKFold(5)，按 **source image** 分组（防止靠认图作弊）；bootstrap 10,000 次，按 source-image cluster。

| 设置 | 臂数 | 随机 | logreg 准确率 [95% CI] | hgb 准确率 [95% CI] | CI 排除随机 |
|---|---:|---:|---|---|---|
| C0 4臂全图 | 4 | 25.0% | **0.8396** [0.7984, 0.8776] | 0.8346 [0.7945, 0.8724] | 是 |
| S 3臂全图（原始要求） | 3 | 33.3% | **0.9059** [0.8715, 0.9378] | 0.8816 [0.8372, 0.9236] | 是 |
| C1 中心128裁切 | 4 | 25.0% | 0.7852 [0.7438, 0.8258] | 0.7839 [0.7414, 0.8227] | 是 |
| C2 降采样128 | 4 | 25.0% | 0.7102 [0.6542, 0.7635] | 0.6190 [0.5703, 0.6677] | 是 |
| **C3 仅两判别式臂** | 2 | 50.0% | **0.8693** [0.8214, 0.9120] | 0.8182 [0.7714, 0.8630] | 是 |

①**明确成立**：3 臂 90.6%（随机 33.3%），4 臂 84.0%（随机 25%），CI 下界远高于随机。

**逐臂 recall（C0 logreg）**：`s33_strong` 0.821、`author_jscc` 0.760、`diffjscc` **0.970**、`sgd_jscc` 0.807。DiffJSCC 几乎完全可分（混淆矩阵中仅 29/960 被错分）。混淆主要发生在两个判别式臂之间（99+119=218 例），符合"两者都是 MSE 类低通输出"。

**最有区分力的特征/频段**：`dct_hi_cv`（8×8 DCT 高频能量的变异系数）、`rps_b09`/`rps_b11`（径向功率谱**高频段**）、`hp_mad`（高通残差平均绝对值）、`grad_mean`。即**区分信息集中在高频**。C2 把图降到 128² 后准确率从 0.840 掉到 0.710（logreg）/0.619（hgb），进一步确认高频是主要载体。

**C3 是本轮最重要的否证。** 两个都没有生成先验的判别式臂之间，仅用轻量频域统计就能以 **86.9%**（随机 50%）区分。所以"可识别指纹"**不是生成先验特有的**，而是任何 JSCC 实现（架构/训练/量化差异）都会留下。这按预注册的事前声明，**直接削弱**把①解释为"先验导致分布偏移"的力度。

## CLIC-428 判别式补充（高功效，纠正性结果）

n=428，原生分辨率，仅判别式臂（`s33_strong`/Base-SA/CM-SA）。

CLIC 参考三角远比 256² 紧凑：`resample_2x` FID=0.0062、`blur_s0p5`=0.0509、`blur_s1`=0.6466、`blur_s2`=6.5824、`jpeg_q30`=8.7941。

KID 最近参考与 real 排名：

| 臂 | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB |
|---|---|---|---|---|---|
| `s33_strong` | jpeg_q30 (3/8) | jpeg_q30 (3/8) | jpeg_q30 (3/8) | jpeg_q70 (2/8) | jpeg_q70 (2/8) |
| `swin_official_base_sa` | jpeg_q30 (3/8) | jpeg_q30 (3/8) | jpeg_q70 (2/8) | **real (1/8)** | **real (1/8)** |
| `swin_capacity_matched_sa` | jpeg_q30 (3/8) | jpeg_q70 (3/8) | **real (1/8)** | **real (1/8)** | **real (1/8)** |

CLIC 共 17 个②命中（strong 2 / weak 15），**全部指向 JPEG 参考集，没有一个指向 blur**。

**这与 256² 结果方向不同，必须如实记录：**

- 在 n=428、原生高分辨率下，判别式重建在中高 SNR **最接近 real**（Swin 两臂在 7–19 dB real 排名 1/8），并没有系统性偏向平滑分布。
- 256² 上"偏向 blur"的强②很可能被两件事放大：其一 256² 是 `Resize(256)+CenterCrop` 后的低分辨率域，模糊参考与重建的差距被压缩；其二 n=192 下 KID/FID 噪声更大。
- 因此 **256² 的强②不应外推为"判别式 JSCC 普遍偏向平滑分布"**；在高功效高分辨率设置下该结论明显减弱。
- 仅剩的稳定 CLIC 现象是低 SNR（1–4 dB）下所有判别式臂都比 real 更接近 JPEG 参考集，且 `s33_strong` 在全部五档都未把 real 排到第 1（best 为 2/8），弱于两条 Swin 臂——与 A1 已冻结的"S33 劣于 Swin"方向一致，但本轮不改变 A1 的质量胜负结论。

## 预注册判据渲染

- ① **成立**：3 臂 0.9059 [0.8715, 0.9378] vs 随机 0.3333；4 臂 0.8396 [0.7984, 0.8776] vs 0.25。CI 下界均远超随机。
- ② **成立**：116 个组合命中，含 12 个强②。
- 因此正式判定为三分中的第一档：**"存在可识别偏移"**。

**但必须同时记录三条限定，否则该判定会被误读：**

1. **①非先验特有**：C3 用两个无生成先验的臂达到 86.9%（随机 50%）。指纹反映的是"实现差异"，不等于"生成先验导致的分布偏移"。
2. **②非先验定向**：SGD（DiT 先验）最常偏向 `vae_sd21`（SD 先验代理），与"偏向各自先验"预期不符；两个 VAE 参考集彼此过近（FID 18.74/19.12），不可区分。
3. **强②不稳健**：12 个强②全在 256² 判别式臂 → blur；CLIC n=428 高分辨率下该方向消失（②全部转向 JPEG，且中高 SNR real 常为最近）。

## 对 RDD 方向的判断

**支持继续的部分**：重建分布确实可测地偏离源分布，且偏离方向不是随机的——生成臂带 VAE-latent 往返痕迹，判别式臂在低分辨率/低 SNR 带低通与压缩类痕迹。RDD 把"重建分布匹配到某个非源目标 P_Y"作为一等公民，与这些观测相容。生成臂 real 排名（DiffJSCC 恒 4/10、SGD 4–6/10）总体好于判别式臂（`s33_strong` 恒 9–10/10），说明生成先验在 256² 上实际**减小**了与源分布的距离，而不是把重建推离源分布。

**不支持的部分（必须写进任何后续论文）**：本轮**没有**证据表明现有方法存在"无意的、由生成先验定向导致的 deception"。观测到的偏移大部分可由平凡机制解释：有损压缩必然偏离源分布、不同实现留不同高频指纹、低分辨率域评测放大平滑差异。若要立"现有方法已无意实现 deception"的论点，当前证据**不足**。

**下一步若继续，最小必要条件**：
1. 让两个先验代理**可区分**（当前 FID 18.74 vs 19.12 太近）。例如用各自完整生成链的无条件/弱条件采样作代理，而不是 VAE 往返。
2. 在**原生分辨率、足够 n** 上重做生成臂（需授权 A2 或等价的生成式 CLIC 推理），因为 256² 与 CLIC-428 的结论方向已经不一致。
3. 指纹检验需加入**跨实现对照**（同先验不同实现 / 同实现不同先验），才能把"先验痕迹"与"实现痕迹"分离。C3 已证明二者当前混在一起。

## 边界

- SGD 全程 non-ranking paper upper，本轮只做分布分析，未做质量胜负。
- 未训练任何生成模型；指纹分类器为轻量分析工具（31 维手工特征 + logreg/HGB），不进入任何方法主链。
- official Imagenette validation 继续封存。
- 不改变 A1 已冻结的 S33-vs-SwinJSCC 质量结论。
- 全程离线、无下载。`cleanfid` 的 Inception 权重使用 A0 已冻结本地副本（`95,607,719` bytes，与 A0 契约 `expected_bytes` 精确一致），通过软链接注入 cleanfid 缓存路径，未联网获取。

## 工程记录（含失败，按项目规则保留）

- **失败并已保留**：首轮 `vae_sgd` 参考集使用未归一化 latent 直接 decode，往返 PSNR 仅 **12.55 dB**、FID vs real=**273.79**，肉眼可见色彩崩坏与块状伪影。根因是 SGD 解码器在训练/推理中始终接收功率归一化 latent（`third_party/SGDJSCC/inference_config.py:151` 为 `decode(normalize(...))`，`through_channel` 同样归一化），直接 decode 后验均值属于分布外输入。修正后 PSNR **30.773 dB**、FID **18.74**。失败产物保留于 `outputs/analysis/ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001/failed/`，未删除、未覆盖。该 bug 若未发现会静默污染整个 (b) 参考集并使 SGD 的②结论完全错误。
- 静态核出并在运行前修掉的问题：`ldm.AutoencoderKL` 需 `lossconfig`；其 `loss.*` 键在 `strict=False` 下报 missing 会误触发 fail-closed（改为只对非 `loss.*` 关键键中止）；KID `max_subset_size` 需钉在实际 n（A1 惯例）而非固定 1000；`build_feature_extractor` 需 `use_dataparallel=False` 以对齐 A1。
- 两个 VAE 的 API 分歧已显式处理：LDM `encode→posterior`、`decode→Tensor`；SGD `encode→AutoencoderKLOutput(latent_dist)`、`decode→[Tensor]`。SGD 的 `forward()` 会注入 AWGN，故只调用 `encode`/`decode`，参考集不含信道噪声。
- `pytorch_lightning` 在主环境缺失，VAE 阶段改用项目既有 `.venv-sgdjscc`（pl 2.4.0 / diffusers / einops），该环境本就是为加载这两个作者 checkpoint 建立的。未新增安装、未联网。
- DiffJSCC checkpoint 因内嵌 Lightning 对象无法以 `weights_only=True` 加载；改为先校验 SHA-256 再加载，实测 `ae1e6df0…dec579` 与 S30 契约一致。SGD checkpoint SHA 记录为 `455cb603…1915fe`。
- 两个 VAE 均为 strict 兼容加载：missing_critical=0、unexpected=0。

## 产物

- `outputs/analysis/ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001/`
  - `arms/{s33_strong,author_jscc,diffjscc,sgd_jscc}/` 各 960 图
  - `reference_sets/` 10 组各 64 图
  - `distribution_metrics_matrix.csv` / `.jsonl`（200 行 = 4 臂 × 5 SNR × 10 参考）
  - `reference_triangle.json`、`criterion2_hits.json`
  - `fingerprint_report.json`（5 设置 × 2 分类器 + 混淆矩阵 + 置换重要性）
  - `clic_distribution_metrics_matrix.csv`、`clic_reference_triangle.json`、`clic_criterion2_hits.json`
  - `build_arms_report.json`、`s33_replay_report.json`、`build_vae_references_report.json`
  - `failed/`（未归一化 SGD VAE 的失败参考集与其派生指标）
- 脚本：`scripts/rdd_p0_build_arms.py`、`rdd_p0_replay_s33.py`、`rdd_p0_build_references.py`、`rdd_p0_build_vae_references.py`、`rdd_p0_distribution_metrics.py`、`rdd_p0_fingerprint.py`、`rdd_p0_clic_complement.py`
