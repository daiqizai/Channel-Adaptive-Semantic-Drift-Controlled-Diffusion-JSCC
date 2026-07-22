# DiffJSCC 外部严格对比阶段结果（2026-07-21）

## 一句话结论

**更强的 JSCC 前端给出了高保真端点，固定 diffusion 把它推向更好的感知端点，却会随 SNR 升高从“净修复”转为“净语义风险”；本项目合理的主线不是放弃 diffusion，而是把强 JSCC backbone 与可校准的生成风险控制结合起来。当前方法仍明显强于自己的 B1，但没有战胜官方 DiffJSCC 系统，更没有战胜其纯 JSCC 前端。**

## 阶段判定

- S30 完整协议通过：`64` 张冻结 Imagenette policy-development 图像、`3` 个 channel seed、`5` 个 SNR，共 `960/960` 行；唯一键、共同噪声、实测 latent、功率归一化、finite 指标和输出数量均通过。
- current 相对 DiffJSCC 最终输出是明确的 fidelity/perception Pareto：
  - PSNR `+0.625280 dB`，source-image cluster 95% CI `[+0.423123,+0.824753]`；
  - MS-SSIM `+0.008258`，CI `[+0.004583,+0.011950]`；
  - LPIPS `+0.051861`，CI `[+0.041360,+0.063002]`，即 current 的感知距离更差；
  - failure `29 vs 23`，差值 CI `[-0.017708,+0.034375]`，证据不足以宣布任一方语义显著更好。
- 更重要的是，作者的纯 JSCC 前端相对 current 在三项质量指标上都显著更好：
  - current−author-JSCC PSNR `-1.762457 dB`，CI `[-1.938592,-1.601835]`；
  - MS-SSIM `-0.014035`，CI `[-0.015697,-0.012622]`；
  - LPIPS `+0.023742`，CI `[+0.019926,+0.027543]`；
  - failure `29 vs 22`，但差值 CI `[-0.009375,+0.029167]` 仍跨零。
- 因此，S30 的预注册 verdict 是 **`PARETO_OR_INCONCLUSIVE`**。不能写成“当前方法战胜 DiffJSCC”，更不能写成“战胜所有论文方法”。

这不是坏消息：它把此前模糊的“可能还缺 idea”收敛为两个可测问题——**主干能力不足**和**生成端点的风险/收益切换**。

## 复现合同

### 官方资产

- DiffJSCC 源码固定在 commit `13aeb62451b872ce41ceba132c9c30a9ca172c53`。
- OpenImage C16 checkpoint：`9,859,655,693` bytes，SHA-256 `ae1e6df0b706d09857cfa02d399f94cc171d8d0ce44f851d96cb032bd7dec579`。
- checkpoint 包含 OpenCLIP text encoder、DeepJSCC、ControlNet、UNet、VAE 和 spatial condition encoder；作者设计上排除 `blip_model.*`。
- 外部 BLIP2 固定为 `Salesforce/blip2-opt-2.7b` revision `59a1ef6c1e5117b3f65523d1c6066825bcf315e3`：
  - shard 1：`9,996,328,120` bytes，SHA-256 `b81228c9ac1b3dee1731ee71d51fe3b2c34f915019c44c25a793b51300ae24fc`；
  - shard 2：`4,982,879,016` bytes，SHA-256 `536bd73b8f1de7d94f503b23fea2eaa4f7f3ea5f74f8f874fcb21d6df1555a19`。
- preload 只允许缺失上述 BLIP2 键：实际 `1,248` 个 allowed missing key，其余 missing/unexpected 均为 `0`；随后由精确 BLIP2 权重补齐。

### 总体、信道与采样

- 总体：S20 冻结的 `64` 张 Imagenette `policy_dev`、`T_cls` clean-correct 图像；未访问 official Imagenette validation。
- SNR：`[1,4,7,13,19] dB`；seeds：`[20260748,20260749,20260750]`。
- 每个 `(sample, seed, SNR)` 使用 S20/S28 同一个 `19,712` 维 canonical CPU-float32 标准正态向量；DiffJSCC 使用其前 `16,384` 个实坐标。
- 作者推理保持 `100` steps、control strength `1.0`、CFG `1.0`、wavelet color fix、无 intermediate MSE guidance、batch size `1`。
- BLIP2 caption 在接收端从带噪 JSCC 重建生成，不是发送端 side information，传输符号记 `0`。

### 码率边界

- 作者 512×512 处理网格的 C16 latent 为 `16×32×32=16,384 real`，即 `8,192` 次复信道使用。
- 相对本项目原始 256×256 源，CBR 为 `1/24=0.0416667`。
- current/B1 使用 `19,712 real`，即 `9,856` 次复使用、CBR `0.0501302`。
- DiffJSCC 使用项目预算的 `83.1169%`，少用 `3,328 real`。这是**同预算上限内**的合法对照，不是 exact-rate match；该差异对 DiffJSCC 不利，不能用来解释 current 的优势。
- DiffJSCC 公开权重训练 SNR 为 `[0,14] dB`，所以 `19 dB` 必须单列为外推。

## 总体结果

指标方向：PSNR/MS-SSIM 越高越好，LPIPS/failure 越低越好。

| 方法 | 图像链路实符号 | 文本计费 | PSNR | MS-SSIM | LPIPS | `T_cls` failures |
|---|---:|---|---:|---:|---:|---:|
| author-JSCC（DiffJSCC 前端） | 16,384 | receiver caption 尚未使用 | **29.986135** | **0.963092** | 0.128342 | **22** |
| DiffJSCC 最终输出 | 16,384 | 0 | 27.598398 | 0.940799 | 0.100223 | 23 |
| current（S28） | 19,712 | 0 | 28.223678 | 0.949057 | 0.152084 | 29 |
| B1 | 19,712 | 0 | 28.124602 | 0.946698 | 0.159396 | 35 |
| SGD paper upper（S20/S28 上下文） | 19,712 | caption 免费；严格最低另需 2,144 real | 27.740368 | 0.952973 | **0.072101** | 25 |

SGD 行来自完全相同 64 图、3 seeds、5 SNR 和 canonical noise 的既有 S20/S28 结果，可用于位置参照；但其 caption 未计费，最低严格总量超项目预算 `10.88%`，不能据此做严格端到端排名。描述性配对中，DiffJSCC−SGD 的 PSNR 为 `-0.141970 dB`，但 source-cluster CI `[-0.401143,+0.156250]` 跨零；SGD 的 MS-SSIM 和 LPIPS 更好，failure `25 vs 23` 的差异同样不确定。因此 current、DiffJSCC、SGD 三者都不是全轴赢家。

## diffusion 相对自身强 JSCC 前端做了什么

总体 DiffJSCC−author-JSCC：

| 指标 | 均值差 | source-image cluster 95% CI | 解释 |
|---|---:|---:|---|
| PSNR | `-2.387737 dB` | `[-2.500458,-2.272485]` | 明确保真损失 |
| MS-SSIM | `-0.022293` | `[-0.025937,-0.019037]` | 明确结构保真损失 |
| LPIPS | `-0.028119` | `[-0.040291,-0.016685]` | 明确感知改善 |
| failure rate | `+0.001042` | `[-0.015625,+0.018750]` | 总体语义净差异不显著 |

语义事件账本为 `10 new / 9 repair`，涉及 `4` 个 new-error source cluster 和 `3` 个 repair source cluster。只看总 failure 会把两个方向相反的机制抵消掉，因此必须分 SNR 报告：

| SNR | current−author PSNR | author / Diff / current failures | Diff−author PSNR | Diff−author LPIPS | Diff new / repair |
|---:|---:|---:|---:|---:|---:|
| 1 dB | `-0.826968` | `10 / 8 / 9` | `-1.643623` | `-0.043531` | `1 / 3` |
| 4 dB | `-1.143650` | `8 / 6 / 7` | `-1.976385` | `-0.032917` | `2 / 4` |
| 7 dB | `-1.536525` | `3 / 4 / 5` | `-2.229119` | `-0.028428` | `3 / 2` |
| 13 dB | `-2.486298` | `1 / 4 / 5` | `-2.588677` | `-0.025307` | `3 / 0` |
| 19 dB† | `-2.818847` | `0 / 1 / 3` | `-3.500882` | `-0.010411` | `1 / 0` |

† 19 dB 超出作者训练区间，只是外推稳定性结果。

这个表给出了本项目最有价值的新证据：

1. 1/4 dB 下 diffusion 分别是净 `2` 次修复，说明低 SNR 生成式先验确实有用；
2. 7 dB 变为 `3 new / 2 repair`，风险开始超过收益；
3. 13 dB 是 `3 new / 0 repair`，并且三个 seed 各自都出现 `1 new / 0 repair`；
4. 19 dB 继续为纯新增风险，但属于外推，不能单独概括作者方法。

因此，“信道好就完全不用 diffusion”仍然过于粗糙；更准确的命题是：**随观测可靠性提高，启用生成端点所需的证据必须变强。** SNR 是先验条件，不应是唯一开关。

## 定性案例

所有拼图顺序均为“原图 | author-JSCC | DiffJSCC”。

- 低 SNR 修复：[dog-show, 1 dB, seed 20260748](../outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/images/4681bd263977f702.png)。author-JSCC 把狗和人物明显模糊，caption 正确识别 dog show，diffusion 恢复出清晰狗形态，`T_cls` 从错误变为正确。这是生成式先验真正有价值的例子。
- 13 dB 新错：[chainsaw/excavator, seed 20260750](../outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/images/b1eede16ac0601e5.png)。caption 为 “a man is standing next to a large excavator”；输出视觉真实且更锐利，但任务类别是 chainsaw。diffusion 强化了 excavator 场景，弱化了小型 chainsaw 判别信息，`T_cls` 从正确变为 garbage-truck 类错误。
- 另一个 13 dB 新错：[ice-sculpture chainsaw, seed 20260748](../outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/images/a4bc4a464ad34c12.png)。肉眼看并非荒谬幻觉，仍触发分类语义失败。这提醒我们：semantic drift 不一定是显眼的“凭空造物”，也可能是生成过程把任务所需的小线索变弱。

定性图只能解释机制，不能替代冻结 `T_cls` 的统计；反过来，单一 `T_cls` 也不能代表所有语义任务。最终论文还需要检测/分割或多评价器复核。

## 系统代价

- author-JSCC 平均 `5.787 ms/图`；
- receiver BLIP2 caption 平均 `113.600 ms/图`；
- 100-step diffusion 平均 `5,115.503 ms/图`；
- 完整 DiffJSCC 平均 `5,237.712 ms/图`；
- peak allocated VRAM `14,927.42 MiB`。

S28 的 current 运行时字段主要是已缓存 B0/diffusion 后的融合后处理，不包含完整 DeepJSCC 与 diffusion，因此不能拿 `3.27 ms` 与上述 `5.24 s` 直接算倍数。

## 对当前项目水平的判断

### 已经成立的部分

- current 相对自己的 B1 仍是可靠提升：S28 已有 `+0.099085 dB / -0.007314 LPIPS`、failure `35→29`。
- S19/S27 已在新 COCO population 证明 matched diffusion 含 B0-only 路径没有的互补信息，且 exact-B1 high-SNR fallback 能稳定工作。
- S30 进一步给出外部机制证据：固定强 diffusion 的 repair/new-error 比例随 SNR 从有利转为不利。

### 尚未成立的部分

- current 没有超过 DiffJSCC 最终输出的 LPIPS 或 failure 点估计；failure 差异统计不显著。
- current 在 PSNR、MS-SSIM、LPIPS 三个质量指标上都落后于 author-JSCC 前端；这不是小模块能掩盖的差距。
- 尚未有同一终点同时达到“强 backbone 保真 + 生成感知收益 + semantic new-error 上界”。
- DiT-JSCC 仍无可运行作者实现，其他外部论文也未全部按同样合同复现；不能声称“已战胜所有论文方法”。

综合评价：**目前是一个有明确研究问题、已有机制证据和强内部结果的项目，但还不是可以按 SOTA 全面优势投稿的最终系统。** S30 反而让文章逻辑更扎实：贡献不应包装成“加模块”，而应围绕“生成端点相对可靠观测的风险—收益相变及其可校准控制”。

## 建议的下一阶段主线

本报告只给建议，不擅自改写 `PROJECT.md` / `MILESTONES.md` 主线。

### P0：把强 JSCC backbone 纳入项目合同

1. 冻结 author-JSCC 为新的 `J_strong` 外部保真端点，保持 `16,384 real`、同噪声、无文本 side information。
2. 在新的 train/cal population 上生成 `J_strong` cache，先验证 latent、功率、码率和分辨率；不直接复用只适配旧 B0 的 S19 fusion checkpoint。
3. 由于第三方源码仓库缺 license，论文主方法最好采用清洁重实现/自训练的兼容强 backbone；官方权重继续作为不可修改的外部参照。

### P1：形成两个明确端点，而不是继续堆小模块

- `J`：强 JSCC 保真端点；
- `G`：由 receiver-visible caption/structure/SNR 条件形成的生成端点；
- 控制器的任务不是“预测一个看起来合理的 alpha”，而是回答：**本行是否有足够证据允许离开 `J`？**

### P2：把“不确定域”定义成可校准风险

仅使用接收端可见特征：SNR、`J/G` 多尺度差异、caption 与 `J` 的兼容度、结构边缘保持、冻结多评价器分歧、生成多样本一致性。离线在独立 `cls_train/cls_cal` 上学习或校准：

\[
P(\text{new semantic error}\mid \text{receiver-visible evidence}).
\]

只有在 new-error 上置信界低于冻结阈值、且预测感知收益为正时才允许 `G`；否则 exact fallback 到 `J`。SNR 进入先验，但不能单独决定开关。S30 的 `1/4→7→13 dB` repair/new-error 转折可以用于提出假设，不能直接拿当前 64 图事后硬编码阈值。

### P3：最小、可发表的下一闭环

1. `J_strong` vs `G` 两端点同图同噪声 cache；
2. capacity-matched risk controller 与仅 SNR gate、仅感知 gate、无风险 gate 三个消融；
3. 主端点：PSNR/MS-SSIM/LPIPS、`new/repair vs J`、source-cluster failure CI、new-error tail upper bound；
4. 训练/校准/测试 source image 严格隔离；
5. 最终用全新 population 一次性审计，official Imagenette validation 继续封存到真正 final protocol。

这个方向保留了 diffusion 的上限，也正面吸收了 S30 暴露的 backbone 差距。它比“好信道不用 diffusion”更有理论和实验内容，也比继续在旧 B1 上加一个小模块更接近一篇完整文章。

## 产物与哈希

- 预注册：`reports/diffjscc_external_comparison_preregistration_2026-07-21.md`
- preflight：`outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-PREFLIGHT-001/`
- checkpoint audit：`outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-CHECKPOINT-AUDIT-001/`
- smoke：`outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-SMOKE-001/`
- 第一 seed：`outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-FIRST-SEED-001/`
- 完整结果：`outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/`
- `per_sample.csv` SHA-256：`549720b804df6cfd87a7035ba37be096b0cf8e683634ebb7f93feff47c49b6e2`
- runner `summary.json` SHA-256：`6547e3b35480a8e1132f9b943fcca9b8889bc72e8bab9901fee5d72894cb137d`
- 最终派生 `post_analysis_v3.json` SHA-256：`87c2ffcb2a699c4f39c1ab92ae88c8d515ed406ad5d808427626e8829fc2aa1f`
- 派生脚本 snapshot SHA-256：`1d5521c270d2218c993263d7c72df57d2629faea116a4a9f52cdc346a538510c`
- checkpoint audit summary SHA-256：`1c3a0ef130f348a7377db1b0270e9cdaa9ace2ccaa38422a3f76fb15273c4eae`

独立一致性审计再次确认 960 行/唯一键/共同噪声/caption/符号数/图片数/状态全部通过；`git diff --check` 通过。第一次全套单测调用因漏写 `PYTHONPATH=src` 产生 3 个 import error，其余 109 项执行；按仓库标准入口重跑后 `122/122` 全部通过。该调用错误保留记录，不解释为实验方法失败。

历史输出禁止覆盖；复跑必须使用新 analysis ID。
