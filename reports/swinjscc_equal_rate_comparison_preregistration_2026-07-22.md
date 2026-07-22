# S34A：S33 Strong 与 SwinJSCC 严格等设定对比预注册

日期：2026-07-22
状态：**1-batch smoke 已通过；用户现只授权 Base-SA-12ep + CM-SA-12ep equal-budget 双臂及 epoch 9--12 收敛检查，extension 未获授权。**
范围：只增加 SwinJSCC 外部骨干基线；S34B 内部消融、S35 diffusion 和 S36 official Imagenette validation 均不在本轮执行范围内。

## 1. 唯一验收问题

在相同 COCO 数据、原生 `16,384 real`、相同训练预算和相同冻结评估合同下，S33 strong 相对 SwinJSCC 是：

- **显著超过**；
- **在 `0.10 dB` margin 下追平/非劣**；
- **劣于**；
- 或因 LPIPS、MS-SSIM、semantic failure 与 PSNR 冲突而只能称 **Pareto**。

本轮不得用官方在 DIV2K/CLIC、不同码率或不同 SNR 上训练的 checkpoint 直接参与优劣排名。official Imagenette validation 继续封存。

## 2. 官方资产与复现边界

- 论文：Ke Yang 等，*SwinJSCC: Taming Swin Transformer for Deep Joint Source-Channel Coding*，arXiv:2308.09361v2，后发表于 IEEE TCCN（DOI `10.1109/TCCN.2024.3424842`）。
- 官方仓库：`https://github.com/semcomm/SwinJSCC`；本轮静态审计的 `main` commit 为 `a6d0e6da53548976acbe9317839a077ef31f190f`。
- 官方代码公开且可拉取；浅层源码归档实测约 `22 KB` zip / `128 KB` 解压。仓库根目录未发现 `LICENSE`，因此论文复现可做，但不得把源码重新分发许可写成已确认。
- README 给出公开 Google Drive 权重入口；权重是作者原数据/码率/SNR 合同的复现辅助资产，**不用于本轮训练初始化或正式排名**。当前服务器清空代理后直连 Google Drive 超时，因此尚未把“链接公开”升级为“本机已完整下载并校验 SHA”。
- 官方 README 建议 Python 3.8、PyTorch 1.9，并警告 `w/ SA & RA` 在 PyTorch `>1.12` 推理可能不一致。本轮固定码率选择 `w/_SA` 而非 `w/_SAandRA`；仍须在正式训练前完成当前 PyTorch 端口的固定张量数值、梯度与 checkpoint round-trip 审计。

正式报告必须明确称为：**official-source SwinJSCC architecture retrained under the project contract**，不能称为“直接复现作者论文表格数值”。

## 3. 为什么选择 `w/_SA` 与 `C=64`

- 本项目码率固定，不需要 Rate ModNet。`SwinJSCC_w/_SA` 保留论文的 Channel ModNet，同时没有动态 rate mask 和 mask side-information 歧义。
- 对 `256x256 RGB`，官方四级下采样输出 `C×16×16` 个实坐标。取 `C=64` 时原生输出严格为 `64×16×16=16,384 real`，对应 `8,192 complex channel uses` 和 source-relative CBR `1/24`。
- 禁止 padding、截断、重复发送、稀疏 mask 冒充 exact rate；side information 记为 0。

## 4. 已确认的保守双臂参数合同

为了同时回答“是否还是官方 Base”与“是否只是参数量差异”两类审稿问题，建议训练两臂：

1. **SwinJSCC-B-SA（官方 Base 架构）**：官方 `embed_dims=[128,192,256,320]`、encoder depths=`[2,2,6,2]`、decoder 镜像、`C=64`、SA-only；静态实测 `28,182,512` trainable parameters，比 S33 的 `31,028,163` 少 `9.17%`。
2. **SwinJSCC-CM-SA（参数匹配 control）**：只把第三 Swin stage 的 encoder/decoder depth 从 `6` 增到 `8`，其余官方模块、宽度、head、window 和 Channel ModNet 不变；静态实测 `31,348,752` trainable parameters，比 S33 多 `1.03%`，比约 `31.289M` 的 author-JSCC 多约 `0.19%`。该臂必须标为 capacity-matched official-code variant，不能冒充论文原版 Base。

保守总判定以 S33 面对两臂中更强的一个为准：只有 S33 对两臂均满足相应条件时，才允许写“相对 SwinJSCC 显著超过/非劣”；若只胜官方 Base、但败给 CM，只能写“胜官方 Base，但容量匹配后优势不成立”。若用户只授权单臂，推荐先跑 `SwinJSCC-CM-SA`，但论文中必须保留“非原始 Base 深度”的限定。

## 5. 与官方默认合同的逐项对齐

| 项目 | 官方发布示例/代码 | 本轮冻结方向 | 原因 |
|---|---|---|---|
| 训练数据 | DIV2K + CLIC 混合目录 | 与 S33 相同 COCO2017 train2017 | 消除训练域差异 |
| 输入处理 | 固定 `RandomCrop(256)` | 与 S33 相同 `RandomResizedCrop scale=[0.6,1.0]` + horizontal flip | 完全复用 S33 数据合同 |
| 码率 | HR 示例常用 `C=96` 或多 C | 固定 `C=64` | 原生 exact `16,384 real` |
| 模型 | `w/o`、`w/_SA`、`w/_RA`、`w/_SAandRA` | `w/_SA` | 固定 rate，仅保留 SNR adaptation；无 rate-mask side information |
| 训练 SNR | 示例 `[1,4,7,10,13]`；源码每 batch 随机一个 SNR | `[1,4,7,13,19]`，每图离散均匀采样 | 与 S33 一致；不得写连续 SNR |
| 功率归一化 | 源码默认按整个 batch 统计功率 | 每图独立单位实维平均功率 | 与 S33 和逐 key canonical channel 一致 |
| AWGN | 官方复信道实现内部随机采样 | paired-real half-variance，`var(real)=1/(2·10^(SNR/10))` | 与 S33/author 的冻结物理口径一致 |
| 优化 | 官方 Adam `1e-4`、极大 epoch 上限，未给项目可直接照搬的停止规则 | 随机初始化、FP32、S33 同 4+8 epoch、同 AdamW/LR/selection | 等数据暴露、等 optimizer-step 预算，避免给任一方法额外训练 |
| checkpoint 选择 | 官方训练脚本周期保存 | COCO 固定 val512 五档 aggregate PSNR，MS-SSIM tie-break | 不接触 policy-dev 或 official validation 选模型 |
| 评估 | 作者 Kodak/CLIC、内部随机噪声 | 冻结 64 图×3 seeds×5 SNR、同 canonical noise coordinates | 与 S33/author 的 policy-dev 定位完全一致 |
| 指标 | PSNR/MS-SSIM 为主 | PSNR、LPIPS、MS-SSIM、`T_cls` failure/new-error/repair | 保留本项目 semantic-drift 纪律 |

实现时允许对官方代码做的修改仅限：数据 adapter、逐图 SNR 向量化、逐图功率归一化、外部 canonical-noise 注入、训练/记录/checkpoint 适配，以及预注册的 CM depth。Swin block、Channel ModNet 计算和解码拓扑不得另行改进。每项 patch 必须保存 diff/SHA，并用单元测试证明 exact symbols、同噪声、有限梯度和可恢复 checkpoint。

## 6. 冻结训练合同

- 数据、train/val manifest、增强、seed 与 S33 相同；不得据 SwinJSCC 结果重抽样。
- 每臂随机初始化；不加载官方权重，也不从 S33/author-JSCC warm start。
- FP32；effective batch size=`32`。若 24 GB 显存不能容纳 microbatch 32，只允许用预注册的 gradient accumulation 保持 effective batch 和 optimizer-step 数不变，并记录 microbatch。
- 与 S33 相同两阶段共 12 epochs：主阶段 4 epochs，AdamW，LR `2e-4→2e-6` cosine、warmup 1,000 optimizer steps；主阶段 best 只加载 model state，fresh optimizer/scheduler/scaler 续训 8 epochs，LR `5e-5→1e-6` cosine。
- loss 仅 MSE；checkpoint 按五档 COCO val512 aggregate PSNR、再按 MS-SSIM 选择。LPIPS、semantic label、S32 policy-dev 和 official validation 不参与选择。
- 若 non-finite、码率、功率、noise SHA 或 output-dir 合同失败，fail-closed 保留失败目录；禁止覆盖重跑。
- 连续 `Uniform[1,19]`、MS-SSIM/perceptual loss、延长 epoch、官方 checkpoint warm start 均不在本预注册内。

## 7. 冻结评估与统计合同

- population：S33 使用的同一 64 张 known Imagenette policy-dev 图；这不是 independent final test。
- channel seeds：`[20260748,20260749,20260750]`；SNR：`[1,4,7,13,19] dB`。
- 每个 Swin 臂 960 行；S33 复用冻结的 960 行，不重新选 checkpoint。
- 每个 key 先生成冻结的完整 `19,712-D external-common-v1` canonical standard-normal noise，并验证完整 SHA，再给 S33/Swin 共同使用相同前 `16,384` 个实坐标。
- 主指标口径沿用 S33 的 floor-uint8；同时保留必要的 raw-float audit，禁止混用作者内部 PSNR 口径排名。
- 报 aggregate 和 per-SNR：PSNR、LPIPS、MS-SSIM、semantic failure；并报 S33 相对每个 Swin 臂的 new-error/repair。
- 差值以 `S33−Swin` 定义：PSNR/MS-SSIM 正值有利，LPIPS/failure-rate 负值有利。95% CI 使用按 source image 聚类、10,000 次 bootstrap；同一图的 SNR 与 channel seed 不当作独立样本。
- 输出必须含 config/source/checkpoint/per-sample/noise SHA、parameter/FLOP ledger、训练时间、推理时间、peak VRAM 和失败样例。

## 8. 预注册判定

对每个 Swin 臂分别用 aggregate `PSNR(S33)-PSNR(Swin)` 判定：

- 双侧 95% CI 下界 `>0 dB`：S33 **显著超过**该臂；
- CI 下界位于 `(-0.10,0] dB`：S33 在 `0.10 dB` margin 下 **追平/非劣**该臂；
- CI 下界 `<-0.10 dB`：S33 **劣于**该臂；
- 下界恰为 `-0.10 dB`：边界不确定，原样报告。

若 PSNR 通过但 LPIPS、MS-SSIM 或 semantic failure 的 CI 显著反向，只能写 Pareto，不能写全面超过。13/19 dB 必须独立呈现；五档聚合不得遮蔽高 SNR 边界。

双臂总判定使用更保守结果：

- 对 Base 和 CM 均显著超过，才称“相对 SwinJSCC（含容量匹配 control）显著超过”；
- 两臂均至少非劣、但未均显著超过，称“追平/非劣”；
- 任一臂 CI 下界 `<-0.10 dB`，总判定为“存在 SwinJSCC 对照下劣于”，并分臂解释；
- 任何二级指标冲突均降为 Pareto。

## 9. 时间预算与启动 gate

基于 S33 在 RTX 4090 D 上纯训练 `3.1 h`、Swin shifted-window attention 的更低吞吐和可能需要 gradient accumulation，当前不做训练 smoke 的保守估计为：

- 代码适配、静态/CPU/GPU smoke、exact-rate/noise/checkpoint 审计：`2--4 h`；
- 单个 Swin 12-epoch 臂：`8--14 h` 纯训练；
- 单臂五档 COCO validation、960-key 评估、bootstrap 与报告：`1--3 h`；
- 推荐双臂串行：总 wall-clock 约 `20--34 h`，正常情况下 `1--2` 天得到明确结果；只有 CM 单臂约 `11--21 h`。

该估计须用不产生正式指标的 1-batch GPU smoke 校正；smoke 不能改动冻结训练预算。

用户已于 2026-07-22 确认：

1. 接受推荐的 **官方 Base + 31.349M capacity-matched** 双臂；
2. 接受两臂都严格使用 S33 的 **FP32 4+8 epoch / equal optimizer-step** 合同，且不因结果延长这两个 equal-budget 臂。

同时增加一项公平性要求：12 epochs 不能被预先当作 SwinJSCC 已收敛。先运行双臂 equal-budget 训练并检查冻结 COCO val512 曲线；即使某臂在 epoch 12 仍明显上升，本轮也只把它报告为 extension trigger，**不得自动延训**。extension 是否执行、执行多少 epochs，必须等 equal-budget 结果报给用户后由用户另行授权。official Imagenette validation 仍封存。

## 10. 12-epoch 收敛 gate 与 fully-converged 条件臂

### 10.1 当前能下的判断

1-batch smoke 只能验证代码、显存和单步耗时，**不能判断优化是否收敛**。官方论文没有报告可复用的明确 epoch 数；官方源码把 `tot_epoch` 写成 `10,000,000` 的开放上限，并每 100 epochs 保存一次。论文明确报告 DIV2K 上每个训练 step model 用单张 RTX 3090 约四天，而且 SNR-adaptive 模型采用“先训练非 ModNet 参数/基础模型，再训练带 Channel ModNet 全模型”的分阶段方式。因此，在看到本项目实际 val 曲线前，12 epochs 的状态冻结为 **unknown**，不得写“已收敛”。

### 10.2 “12 epochs 明显未收敛”的预注册判据

两条 equal-budget 臂每个 epoch 都在同一 COCO val512、同五档 SNR、同固定验证噪声下记录 aggregate PSNR/MS-SSIM。以 1-based epoch 编号计，只有以下三项同时成立才触发“明显未收敛”：

1. 目前最佳 aggregate PSNR 出现在 epoch 11 或 12；
2. epoch 9--12 的 aggregate PSNR 普通最小二乘斜率至少为 `+0.01 dB/epoch`；
3. `PSNR(epoch 12)-PSNR(epoch 9) >= +0.03 dB`。

该 gate 故意要求“最近仍稳定上升”，避免把单点波动误判成未收敛。未触发时只能写“没有明显未收敛证据”，不能证明全局最优或复现了作者四天训练。train loss 作为诊断同时报告，但 gate 只依赖冻结验证 PSNR，防止训练损失下降而泛化不再改善时继续消耗算力。

### 10.3 fully-converged extension（本轮未授权）

- epoch 9--12 gate 在本轮只产生 `triggered/not_triggered` 诊断，不产生训练权限。
- 当前每臂授权上限严格为总计 12 epochs；训练脚本必须拒绝 epoch 13 及以后，也不得预建 extension 输出目录。
- equal-budget 结果、收敛曲线和 trigger 状态报给用户后，是否延训、延训哪些臂及延到多少 epochs，均由用户另行决定。此前讨论过的 60-epoch 上限不是当前授权，不能执行。

### 10.4 两种最终结论必须分开

- **Equal-budget 结论**：S33 对 Base-SA-12ep 与 CM-SA-12ep，二者取对 S33 更不利的结果。
- **充分训练结论**：本轮不产生；只有用户看完 equal-budget 结果后另行授权 extension 才会建立。
- 只有 S33 在 equal-budget 和充分训练两种口径下都按既定 CI/margin 规则不输，才允许写不带训练预算限定的“超过/非劣 SwinJSCC”。若只在 equal-budget 胜出，则写“同等训练预算下超过”；若对充分训练 Swin 仅追平，则写“以更简单卷积结构达到相当性能”。

## 11. 1-batch smoke 校时结果

运行 ID：`SMOKE-S34A-SWINJSCC-CALIBRATION-001`。两臂各使用同一个真实 COCO microbatch=8，在 RTX 4090 D / PyTorch `2.11.0+cu128` 上执行一次 FP32 forward、backward、gradient clip 和 AdamW step；不运行 epoch、不产生方法质量结论。

| 检查 | Base-SA | CM-SA |
|---|---:|---:|
| 参数量 | 28,182,512 | 31,348,752 |
| latent | `[8,256,64]` | `[8,256,64]` |
| 每图实符号 | 16,384 | 16,384 |
| 输出 | `[8,3,256,256]` | `[8,3,256,256]` |
| 最大单位功率误差 | `1.1921e-7` | `1.1921e-7` |
| gradient norm | `1.70264` | `2.62209` |
| peak allocated VRAM | `8.988 GiB` | `9.563 GiB` |
| peak reserved VRAM | `9.748 GiB` | `10.396 GiB` |
| 单 microbatch 时间 | `0.5366 s` | `0.1788 s` |

两臂的 finite、exact-rate、power 与 checkpoint strict round-trip 全部 PASS。另在 scalar 7 dB、同权重和同输入下，对项目 adapter 与官方原始 encoder/decoder forward 做直接数值对照，最大绝对差分别为 `0/0`；因此逐图 SNR vectorization 在退化为 scalar 时不改变官方 SA 运算。

Base 是同一进程首臂，`0.5366 s` 包含 CUDA cold-start；CM 是 warm-cache，禁止据两点声称 Base 比 CM 慢。正式合同据显存冻结 `microbatch=8 + accumulation=4 = effective batch 32`。以 warm CM 的线性累计估算，单臂 12 epochs 纯训练约 9 小时；加入 cold-start、每 epoch 五档 val512、checkpoint 和 I/O 后按 `10--14 h/臂` 规划，equal-budget 双臂约 `20--28 h`。本轮没有任何 extension 时间预算或运行权限。

结果文件：`outputs/smoke/EXP-S34A-SWINJSCC-CALIBRATION-001/smoke_result.json`，SHA-256=`010d9befe939f3c9288755888c34857e68bcfcbeb7135fd0f9f685ef944cacb3`。official Imagenette validation 未访问，formal train output 未创建。
