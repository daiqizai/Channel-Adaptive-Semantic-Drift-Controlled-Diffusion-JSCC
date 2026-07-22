# SNR-Conditioned Identity Envelope 预注册（2026-07-15）

## 研究问题

`EXP-S17-LATDIFF-002` 与 `EXP-S17-LATDIFF-004-DECODER` 在两个不重叠 holdout 上重复得到同一结构：channel-matched diffusion 在 1/4/7 dB 提升明显，但 13/19 dB 的 latent MSE 改善仍会转化为轻微 PSNR 过修复。decoder-aware loss 已显著优于同训练预算 control，却没有自动让 correction 在高 SNR 收敛到严格零。

本阶段只检验一个新变量：**在 frozen decoder-aware DDIM correction 外加入随信道质量提高而不增、并在高 SNR 允许严格恒等的 correction envelope，能否保留低/中 SNR diffusion 收益，同时消除 13/19 dB PSNR 回吐。**

本阶段不训练或微调任何网络，不改 step matching、DDIM steps、信道、码率、payload reservation、decoder-aware loss、B1 或分类器，也不加入 measurement consistency/文本/结构条件。它是 frozen diffusion 的 channel-adaptive strength/control 层。

## 冻结支路

- exact-rate DeepJSCC SHA-256：`bca5b67a...bb606`；
- decoder-aware latent diffusion SHA-256：`5b708117...5d98f`；
- frozen B1 SHA-256：`7a295976...b7615a`；
- AWGN：`sigma²=P/(2*gamma)`，SNR `[1,4,7,13,19]`；
- 总预算：19,712 real symbols，其中 19,632 image coordinates、80 reserved payload coordinates；
- matched state：`alpha_channel=2*gamma/(2*gamma+1)`；
- sampler：6-step deterministic DDIM，`measurement_blend=0`。

设接收活动 codeword 为 `y`，frozen decoder-aware DDIM 输出为 `z_diff`，最终输入 decoder 的 codeword 为：

`z_g = y + g(SNR) * (z_diff-y)`。

因此 `g=0` 精确退化为同噪声 B0，`g=1` 是完整 decoder-aware diffusion；reserved 与 inactive coordinates 始终为零。envelope 不增加参数、时延主项或信道符号。

## 预注册候选族

令 `alpha_min=alpha_channel(1 dB)`、`r(alpha)=(1-alpha)/(1-alpha_min)`。候选在五个固定 SNR 上均为非增：

1. `full`：`g=1`，只作 frozen baseline，不参与 adaptive policy 选择；
2. `smooth_p0p25`：`g=r^0.25`；
3. `smooth_p0p5`：`g=r^0.5`；
4. `smooth_p1`：`g=r`；
5. `smooth_p2`：`g=r^2`；
6. `hard_identity_7db`：SNR `<=7 dB` 时 `g=1`，SNR `>=13 dB` 时 `g=0`。

最后一个候选不是看新数据后增加：前两个独立 S17 holdout 已重复显示前三档为正、后两档为负，因此在本轮任何新 population 指标前冻结为可证伪的硬 identity policy。平滑候选检验是否能用连续 envelope 取得更好 tradeoff。

## 新 population

旧 exact-rate cache 的 10,000 train + 1,000 validation 图均视为已使用。本阶段从本地 COCO train2017 剩余图像中：

- 用旧 manifest 的 source path 与 source SHA-256 双重排除全部 11,000 图；
- 继续排除 COCO val2017 同名图；
- 以 `SHA256("20260738:" + relative_source_path)` 排序剩余图；
- 前 256 张冻结为 `selection`，后 256 张冻结为一次性 `holdout`；
- 统一 `Resize(256)+CenterCrop(256)` 并保存 8-bit PNG；
- population materialize 后先冻结 manifest SHA，再运行任何 channel evaluation。

selection channel seed=`20260739`；holdout channel seed=`20260740`。两个 role 图像、sample ID 和 channel seed 均不重叠。official Imagenette validation 继续封存。

## Selection 与一次性 holdout

selection 对所有预注册 adaptive candidates 计算逐 SNR mean PSNR delta vs B0。冻结策略按以下字典序选择：

1. `nonnegative_snr_count` 最大；这里 `delta>=-1e-9` 记 nonnegative；
2. mean PSNR delta vs B0 最大；
3. mean LPIPS delta vs B0 最小（仅当前两项严格相同才使用）；
4. 按配置中的候选顺序取前者。

策略 JSON、输入 manifest/config/checkpoint 哈希写入后，才允许打开 holdout。不得根据 holdout 换 exponent、cutoff 或 tie-break。holdout 同时报告所有冻结候选用于消融，但正式 outcome 只看 selection 选中的 policy。

## 指标与成功判据

必须报告 PSNR、MS-SSIM、LPIPS、活动 latent MSE，以及 AlexNet pseudo new/repair 和 AlexNet/ResNet18/MobileNetV3-Small 多数票 pseudo new/repair；COCO pseudo label 仍不替代监督 semantic-tail 审计。

阶段 PASS 需要全部满足：

1. selected envelope 相对 full diffusion 的 mean PSNR 差 95% image-cluster bootstrap CI 下界 `>0`；
2. selected 相对 B0 mean PSNR `>0`，至少 3/5 SNR 严格为正，且 5/5 SNR 不为负；
3. selected mean LPIPS 不差于 B0；
4. 在 1/4/7 dB 上，selected 保留 full diffusion 至少 95% 的 aggregate PSNR delta；
5. selected AlexNet pseudo new error 不多于 repair，且不多于 full diffusion；
6. selected 多数票 pseudo new error 不多于 repair，且不多于 full diffusion；
7. 码率、noise SHA、sample/SNR keys 完整一致。

若 selected 使用 `g=0` 的档位，则 PSNR delta 精确为 0 属于预期 identity success，不要求人为制造正增益。B1 继续作为强 deterministic reference，但“超过 B1”不是本控制层的阶段判据。

## 预定输出

- 配置：`configs/s18_snr_identity_envelope.yaml`
- population：`outputs/eval/s18_fresh_coco_identity_envelope_population/`
- selection：`outputs/analysis/ANALYSIS-S18-IDENTITY-SELECTION-001/`
- holdout：`outputs/analysis/ANALYSIS-S18-IDENTITY-HOLDOUT-001/`
- bootstrap：`outputs/analysis/ANALYSIS-S18-IDENTITY-BOOTSTRAP-001/`
- 中文报告：`reports/snr_identity_envelope_stage_result_2026-07-15.md`

本阶段最多证明 frozen channel-matched diffusion 的 SNR-adaptive identity control 是否有效，不授权声称已超过 B1/SGD-JSCC、已完成 semantic-safe M3，或可在已暴露 S17 validation 上继续调策略。

## Population 冻结记录（任何 selection 输出前）

按预注册 rank/exclusion 规则已从本地 118,287 张 train2017 中排除旧 11,000 source，剩余候选 107,287 张；materialize 512 张后 path overlap=`0`、SHA-256 overlap=`0`，selection/holdout 各 256 张。新 source manifest SHA-256 为 `c467d2ccadd94242f51ff683f09d7b43d91a07d5edc538d06c06f2b6d93a8bed`。该哈希已在任何 selection channel/quality 输出前写入正式配置。

## Policy 冻结记录（任何 holdout 输出前）

selection 1,280 rows 已按冻结字典序选出 `hard_identity_7db`：它是唯一 `nonnegative_snr_count=5` 的候选，mean PSNR/LPIPS delta vs B0 为 `+0.194020/-0.036302`；1/4/7 dB strength=`1`，13/19 dB strength=`0`。平滑候选中 `p=0.5` 虽 mean PSNR delta 更高 `+0.199099 dB`，但 13/19 dB 仍分别为 `-0.007737/-0.003579 dB`，按预注册第一优先级未入选。selected policy SHA-256 为 `c31d68533bf4e470d585ff5d279e948a71dca3ead1964902647393b2f37d05eb`，selection CSV SHA-256 为 `e226d871184333495a2e94709f89388d4dc56987afb776d249572745f1e7f995`；均已在任何 holdout 输出前冻结。
