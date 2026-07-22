# UInt2 预留感知 diffusion-JSCC 阶段结果

日期：2026-07-14。实际里程碑归属：`MILESTONES.md` 的 S5 adaptive-control validation；文件和旧输出中的 `S15` 只是历史实验标签，不表示新增项目阶段。

## 结论先行

本轮没有放弃 diffusion。相反，已经把固定总码率系统的质量下限推进到五个 SNR 相对 paired unpunctured M2 全部正增益，并证明预留感知 B1 的训练收益在同输入配对比较中显著成立。

但当前方法仍不能晋级为 semantic-safe M3。冻结三路路由在新 AWGN seed `20260728` 上得到与 M2 相同的 aggregate failure，质量增益稳定，但 system new-error cluster 上界和逐 SNR failure gate 失败。阶段结论是：**diffusion restoration 已稳定，剩余瓶颈是少数样本的三候选语义决策，而不是 diffusion 是否保留。** official Imagenette validation 继续封存。

## 1. UInt2 固定码率链路

把 sender probability payload 从 UInt4 降为 UInt2：10 类×2 bit、BPSK×4，共 80 个实符号；总预算仍为 65536 个实符号、CBR `1/6`，图像 latent 使用其余 65456 个符号。payload 与图像使用同一次 AWGN 调用，接收端在 DeepJSCC decoder 前擦除 payload 位置。

在已暴露 policy-dev / seed `20260727`、旧 S13 B1 上：

- payload vector exact `99.7043%`，source top-1 agreement `98.3949%`；
- final−paired-M2 PSNR `+0.071845 dB`，cluster bootstrap 95% CI `[+0.066098,+0.077531]`；
- final−paired-M2 LPIPS `-0.002577`，95% CI `[-0.002836,-0.002323]`；
- 五个 SNR 的 PSNR 增益均为正；
- paired M2/final primary failure `61→60`，但 system new/repair 为 `7/8`，new-error cluster upper95 `0.7766%>0.5%`，严格 verdict 为 NEGATIVE。

这一步解决了 UInt4 版本低/中 SNR 相对 M2 的 PSNR 负增益，但没有解决 rare semantic tail。

## 2. 预留感知 B1

### 2.1 COCO reserved cache

用 local COCO train2017 SHA-ranked 2000 train + 200 validation、五个 SNR 生成 80-symbol reservation/erase cache；固定 balanced BPSK 只保持符号数和功率，payload 位在 decoder 前全部擦除，不使用 COCO 标签或 Imagenette classifier。source manifest SHA-256 为 `51b7efe4...e8de`。

同一批样本相对原 unpunctured S13 B0 的平均 reservation penalty 为 `-0.04833 dB`；逐 SNR 为 `-0.02203/-0.03312/-0.04589/-0.06593/-0.07470 dB`。这确认 post-hoc reservation 的影响虽小但系统性存在。

### 2.2 微调与公平配对比较

从 S13 B1 checkpoint 以 `lr=1e-4` 微调 3 epochs，best 为 epoch 2；新 checkpoint SHA-256：

`57aa528345b90b06a3daadd1069b27d320534a0124769d46118b760fbbc85495`

随后把旧 B1 与新 B1 放到完全相同的 200 张×5 SNR reserved inputs，输出统一量化为 8-bit PNG 后逐样本比较：

| SNR | new−old PSNR | paired bootstrap 95% CI | new−old LPIPS |
|---:|---:|---:|---:|
| 1 | +0.11514 | [+0.10304,+0.12998] | -0.003108 |
| 4 | +0.10223 | [+0.09165,+0.11468] | -0.002731 |
| 7 | +0.09737 | [+0.08711,+0.10951] | -0.001982 |
| 13 | +0.09747 | [+0.08759,+0.10793] | -0.000391 |
| 19 | +0.10170 | [+0.09202,+0.11175] | -0.000196 |
| image-cluster aggregate | **+0.10278** | **[+0.09338,+0.11400]** | **-0.001682** |

aggregate LPIPS 的 95% CI 为 `[-0.002022,-0.001333]`。预注册判据全部通过，说明收益来自 reservation-aware B1 本身，而不是只与受损 B0 比较造成的假象。

## 3. 接回冻结 diffusion 完整链

保持 S14 diffusion、三步 received-latent posterior correction、UInt2 payload 和 controller 全部不变，只替换 B1。

seed `20260727` 的 final−paired-M2 PSNR/LPIPS 为 `+0.073967 dB/-0.002633`，质量 CI 均严格通过；但新 B1 同时把 paired M2 failure 从旧实验的 61 降到 59，而旧二路 final 仍为 60，system new/repair 变为 `7/6`，严格 verdict 仍为 NEGATIVE。

7 个新增事件中：

- 6 个被 controller 拒绝后选择 anchor，但 raw 和 posterior 实际都保持正确；
- 其中 5 个同时出现 recovered sender top-1 与 G_gate(anchor) top-1 不一致；
- 另 1 个是接受 posterior 后错误。

这把失败定位为“拒绝后的错误候选回退”，而不是普遍的 diffusion false accept。

## 4. 冻结三路路由与新 seed 复核

冻结自然规则：通过 triplet 时输出 posterior；拒绝且 recovered source 与 anchor top-1 不一致时输出 diffusion raw；其他拒绝输出 anchor。规则不使用原图、标签、T_cls 或新阈值。

### 4.1 seed20260727 离线选择结果

从已保存 CSV 复算：paired M2/final primary failure `59→56`，system new/repair `2/5`，new-error upper95 `0.3718%`；五 SNR PSNR 全为正，aggregate PSNR/LPIPS `+0.06536/-0.002595`。但是 failure cluster bootstrap CI 为 `[-0.001571,+0.000393]`，上界未严格小于 0，因此按完整统计层仍为 NEGATIVE development result，不能当晋级证据。

### 4.2 预注册 seed20260728 结果

在读取任何新行前冻结 config、规则和 gates，只换独立 AWGN seed `20260728`：

| SNR | paired M2 failure | final failure | final−M2 PSNR | final−M2 LPIPS |
|---:|---:|---:|---:|---:|
| 1 | 34 | 36 | +0.02106 | -0.003341 |
| 4 | 19 | 18 | +0.01619 | -0.003089 |
| 7 | 9 | 8 | +0.01516 | -0.002228 |
| 13 | 2 | 2 | +0.09593 | -0.002003 |
| 19 | 3 | 2 | +0.18065 | -0.002039 |

aggregate final−M2 PSNR `+0.065798 dB`，image-cluster 95% CI `[+0.060055,+0.071703]`；LPIPS `-0.002540`，95% CI `[-0.002805,-0.002286]`。质量收益在新信道 realization 下完整复现。

primary aggregate M2/final failure 为 `62→62`，system new/repair rows 为 `4/4`；new error 来自 4 个 image clusters，eligible 1690，upper95 `0.5408%`，略高于冻结的 `0.5%`。1 dB failure `34→36`，逐 SNR gate 失败；failure-delta bootstrap CI 为 `[-0.001179,+0.001179]`。因此预注册 verdict 是 **NEGATIVE**。

4 个新错误的形态也不再能由单一 fallback 解决：两个样本的 anchor/raw/posterior 全错，一个 raw 错而 posterior 对但规则选择 raw，一个 raw 对而被接受的 posterior 错。继续在这个 seed 上补布尔规则只会过拟合。

## 5. 当前判断与下一步

1. **保留 diffusion。** 两个 seed、五档 SNR 的质量/感知收益都稳定且 CI 明确，当前不是“不用 diffusion 上限低”的路线。
2. **冻结物理链路。** UInt2 payload、总 CBR、预留感知 B1、S14 diffusion 与三步 posterior correction 暂不再按 policy-dev outcome 改动。
3. **下一开发对象是三候选 semantic decision layer。** 应在与 outcome audit 分离的 `cls_train/cls_cal` 上，用真实 WNID 监督训练 anchor/raw/posterior 三路选择或 risk-to-abstain；输入只允许 recovered sender distribution、G_gate/G_aux 的 candidate distributions、SNR 与 data-consistency，new error 采用高代价。不能再在 seed20260728 上扫阈值。
4. **必须换 image population 做一次性审计。** 仅更换更多 policy-dev channel seeds 已不能证明图像泛化；在训练/校准和统计功效冻结前，official validation 不解封。
5. 若分离监督的三路 decision layer 仍失败，再单独预注册 reservation-aware diffusion fine-tune；它会改变 generative module，不能混在本轮路由调试中悄悄进行。

## 资产与复现

- reserved exporter/config：`scripts/s15_export_coco_uint2_reserved_c8.py`、`configs/s15_coco_uint2_reserved_c8_export_pilot.yaml`
- B1 fine-tune/config：`scripts/s5_residual_refiner_pilot.py`、`configs/s15_uint2_reservation_aware_b1_finetune_pilot.yaml`
- B1 paired comparison：`scripts/s15_compare_reservation_aware_b1.py`
- sender audit：`scripts/pc_imagenette_sender_inbudget_awgn_audit.py`
- offline router analysis：`scripts/pc_analyze_mismatch_raw_routing.py`
- seed20260728 prereg/config：`reports/reservation_aware_fallback_routing_seed20260728_preregistration_2026-07-14.md`、`configs/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_seed20260728.yaml`
- 关键输出：`outputs/EXP-S15-001/`、`outputs/analysis/s15_reservation_aware_b1_paired_comparison/`、`outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_seed20260728/`

本轮没有联网或下载；大任务均清空 proxy 环境运行；没有读取 Imagenette official validation。
