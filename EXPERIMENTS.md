# 实验记录

## 方法终止与项目冻结（非实验记录）

- 日期：2026-08-03
- 类型：静态证据审计与项目决策；**不是新实验，不分配 EXP/ANALYSIS ID**
- 最终状态：`ENGINEERING_STOP`
- 最终判定：原始完整联合优势主张未建立，停止继续投入方法开发
- 解释层：`reports/METHOD_TERMINATION_REPORT_2026-08-03.md`
- 证据登记：`audit/CLAIM_REGISTRY.csv`
- 本轮执行：没有训练、推理、评测、下载或 official validation 访问；没有新增或覆盖任何实验输出
- 冻结项：S35R-P1、新 diffusion refiner、matched B1/M2/envelope、semantic gate/controller/fusion、CVaR 模型、Swin extension、S34B/S34C 长版、A2/S36 及大量旧实验复跑均不启动
- 历史保护：下方实验、负结果、失败目录、机器 `verdict.json` 和预注册全部原样保留；其中旧“下一步/待授权”只表示当时时点，不再构成执行许可
- 结论边界：各子假设分别为局部支持、被反驳、尚未建立或工程停止；不得将本项目终止写成所有相关方法已被普遍反驳

## CVaR 候选方向二：Rayleigh matched mean-training 归因闭环（方向终结）

### EXP-CVAR-P1-RAYLEIGH-MATCHED-MEAN-001 + ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001/-002

- 日期：2026-07-31 / 2026-08-01
- 当前状态：完成；**判定 `END-CVAR`，候选方向二正式结束，未训练任何 CVaR 模型**
- 授权：用户在 P0 `NO-GO` 后授权一次独立归因闭环——若消除尾部则彻底结束 CVaR，只有匹配均值训练后仍残留显著条件尾部才进入 CVaR
- 预注册：`reports/cvar_p1_rayleigh_matched_preregistration_2026-07-31.md`（训练输出前冻结）
- 结果报告：`reports/cvar_p1_rayleigh_matched_result_2026-07-31.md`
- 目的：P0 无法区分「均值目标掩盖风险」与「train/test 信道错配」。本闭环去掉两重错配（模型没见过衰落、有效 SNR 跌出条件嵌入范围），只保留均值目标，再测残余尾部
- **预注册把用户的两种结局补成三种**：额外增加能力门槛，防止退化模型（输出糊平均图）因尾部天然小被误读为「尾部已消除」；全部阈值提前数值化；并修正 P0 报告 §4 披露的归因口径缺陷（归因只在触发档评估，不再跨五档 `all(...)`）
- 训练：初始化自冻结 S33B（SHA `2daad9e7…`，原文件未改）；block fading `h~CN(0,1)` 逐图 + ZF `ε=0`；encoder 条件 = **标称 SNR**（无反馈，发端不可能知道 `h`）；decoder 条件 = **真实有效 SNR 不 clamp**；纯 MSE；COCO train2017 全量 `256×256`；6 epoch = `22,182` steps；FP32；lr `5e-5` cosine
- 训练结果：`92.5 min`，`246 ms/batch`，峰值 `12.34 GiB`；验证聚合逐 epoch `27.6263/27.5082/27.8484/27.9127/28.0181/28.1409 dB`，最后增量 `+0.12 dB` 已收敛；best SHA `4a52028480c7317c7084c7922af7d22e216b3798613036b278131122e44dbc20`
- 评测：与 P0 **完全相同**的脚本与总体——同 200 图、同 64 realization、同 5 SNR、同 `base_seed=20260731`、同四配对 arm；主 arm 事前固定为 `rayleigh_effective_csi`（训练匹配的部署方式），不用 P0 的数据依赖规则
- **能力门槛 PASS**：聚合 `28.4707 dB` vs 要求 `≥27.9585 dB` 为 `+0.5122 dB`；逐档 `+1.2154/+0.7581/+0.4713/+0.1164/−0.0003 dB`，最差退化 `+0.0003 dB` 即无退化。预注册 §5.2 担心的「过度保守变糊」未出现
- **尾部大幅收缩但未消失**：同 arm 同图同 realization，mean PSNR `21.48/23.38/24.54/25.08/25.31 → 25.77/27.23/28.38/30.06/30.92 dB`；`median−p10` `7.98/7.33/5.14/2.72/2.73 → 4.18/3.99/3.66/2.34/0.95 dB`；`worst10-mean` `12.04/14.21/16.68/20.73/21.69 → 19.82/21.60/23.11/26.30/28.89 dB`；`outage(<24dB)` `0.631/0.488/0.401/0.368/0.343 → 0.292/0.173/0.110/0.041/0.021`；`CVaR-10 MSE` 降 `4.4~5.1×`
- 失效模式质变：同图同 realization（`000000013004`, `r54`），worst-10%（`|h|²=0.0385`）从 P0 的 `16.25 dB` 彩色噪声变为 P1 的 `23.95 dB` **内容清晰可辨的模糊**；worst（`|h|²=0.0003`）`8.40 → 14.07 dB`
- 幅度条款 PASS（1/4/7/13 dB 四档 `≥2.0 dB`），**归因条款 FAIL**——决定性一环：触发档信道方差占比 `0.546/0.500/0.445/0.286`（P0 低中三档为 `0.80/0.75/0.67`），即残余尾部里**图像内容难度已追平甚至超过信道随机性**。CVaR 立论基础是「同一图像在不同信道实现下的尾部」，逐图 CVaR 对图像间难度差异不敏感，故无从发力
- **边缘性双重检验并如实披露**：4 dB 占比 `0.499945` 距门槛仅 `5.5e-5`；2,000 次 image-cluster bootstrap 得 `P(占比≥0.5)=0.542`、95% CI `[0.4525,0.5547]` 跨越门槛，确为掷硬币。但独立 seed `20260802`（全新 realization，另 256,000 行）复现同样判定 `END-CVAR`，占比 `0.540/0.493/0.430/0.267`；且即便 4 dB 判通过，7 dB `0.44` 与 13 dB `0.29` 仍明确失败，**结论不依赖该点**
- 顺带产物：本模型是合格的 Rayleigh block-fading channel-adaptive JSCC 基线，即任务书 §7.1 的 `Repeated-fading mean control`。但按 `MILESTONES.md` Rayleigh 属 AWGN 最小闭环之后的扩展项，**不自动进入主线**，未做语义评估
- 明确不做：不训练 CVaR-10/20/worst-one；不把匹配训练的 `+3.8~5.6 dB` 包装成方法贡献（只是修正信道错配，是应有对照）；不把残余 `4.18 dB` 尾部写成「CVaR 仍有机会」
- 失败保留：外部 VLLM 进程占用 `20.8/24 GiB` 导致首次诊断在 32 行处 CUDA OOM，目录保留为 `ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001_failed_oom_20260801`；重跑 `realization_chunk` 由 32 降为 8
- **chunk 非逐比特不变**（实测 8 vs 2，按 `(arm,image,snr,realization)` 键比较）：`|h|²` 与 decoder SNR 逐比特相同，`max|ΔPSNR|=8.3e-4 dB`、`max|ΔMSE|=4.4e-7`、`max|ΔLPIPS|=1.7e-4`，为 GPU kernel 非确定性，比 `2.0 dB` 门槛低四个数量级；**行顺序随 chunk 改变**，跨运行比较一律按键而非按位置（P0 用 32、P1 用 8）
- 局限：单一 recipe（continuation、单预算、单 lr）；深衰落 `|h|²≈1e-4` 时有效 SNR ≈ `nominal−40 dB` 信息论不可恢复，任何模型都有物理残余尾部下界；阈值 `0.5` 是判断值非理论推导值（`0.44`/`0.29` 距门槛足够远故不敏感，`0.4999` 敏感）；无语义指标；单 backbone/单码率 `1/24`/block fading/ZF `ε=0`
- 新增代码：`scripts/cvar_p1_train_rayleigh_matched.py`、`scripts/cvar_p1_attribution_verdict.py`、`configs/cvar_p1_rayleigh_matched_mean_training.yaml`、`configs/cvar_p1_matched_tail_risk_diagnostic.yaml`、`configs/cvar_p1_matched_tail_risk_seed_replication.yaml`；`tail_risk.apply_block_fading_channel` 扩展为支持逐样本 SNR 张量（向后兼容，新增 2 项单测）
- 验证：全仓 `142/142` unittest 通过，`py_compile` 通过，40 组最差案例全部通过重放校验

## CVaR 候选方向二：条件信道尾部风险诊断

### ANALYSIS-CVAR-P0-TAIL-RISK-001

- 日期：2026-07-31
- 当前状态：完成；只读诊断，无训练、无 checkpoint 选择、无下载、无 official validation 访问
- 来源任务书：`候选二_CVaR尾部风险JSCC_Codex实验任务书.md`（P0 审计 → P1 dry-run → P2 诊断 → P3 判定）
- 预注册：`reports/cvar_p0_tail_risk_preregistration_2026-07-31.md`（在任何正式统计前冻结）
- 结果报告：`reports/cvar_p0_tail_risk_result_2026-07-31.md`
- 目的：在投入 CVaR 训练前，先验证「均值训练模型对同一图像重复采样信道时，最差 10% 是否明显差于中位数」
- checkpoint：冻结 S33B，SHA `2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`，脚本内强制校验
- 总体：COCO val2017 中与 S33 的 512 图 checkpoint-selection 子集**不相交**的 SHA 排序前 200 图，`Resize(256)+CenterCrop(256)`
- 规模：200 图 × 64 realization × 5 SNR `[1,4,7,13,19]` × 4 arm = **256,000 行**，`576.1 s`，RTX 4090 D
- 信道：block fading `y=hx+n`，`h~CN(0,1)` 逐图一个复系数，接收端已知 `h`，ZF 均衡 `ε=0`，发端无 `h`（无反馈）；沿用仓库 SNR 口径（每复信道使用 Es/N0，每实坐标方差 `P/(2γ)`）
- **四 arm 设计（对任务书的关键补充）**：任务书只规定单一 Rayleigh arm，但本仓 backbone 是 SNR-conditioned 的，且 ZF 均衡后等价于有效 SNR `γ|h|²` 的 AWGN，故「喂 decoder 什么 SNR」是会完全改变结论的自由变量。四 arm 共享同一 encoder 前向与同一条标准正态噪声，逐 realization 严格配对：`awgn_control`（`h=1`）、`rayleigh_nominal_csi`、`rayleigh_effective_csi`、`rayleigh_effective_csi_clamped`（clamp 到训练范围 `[1,19]`）
- 主 arm 选择为 **tail-blind** 规则：逐 SNR 取平均 PSNR 最高的 Rayleigh arm，对一切尾部统计量盲，避免 outcome-based selection；结果五档全部选出 `rayleigh_nominal_csi`
- 实现校验：`awgn_control` 由 `h=1` 实现，单测 `test_unit_gain_reduces_to_the_existing_awgn_path` 证明与既有 `complex_awgn_from_standard_normal` 逐元素相等；`test_equalized_noise_variance_matches_effective_snr` 固定均衡后噪声方差等于 `P/(2γ|h|²)`
- **发现 1：AWGN 下不存在条件尾部风险。** `median−p10` 五档仅 `0.11/0.09/0.07/0.04/0.02 dB`；信道方差占总方差 `0.001/0.001/0.000/0.000/0.000`；`CVaR-10 MSE/mean MSE ≤ 1.03×`。项目当前主线信道上 CVaR 无优化对象
- **发现 2：Rayleigh 下尾部很大，但主因是分布外错配。** `rayleigh_nominal_csi` 的 `median−p10` 为 `10.06/8.05/5.76/2.79/1.01 dB`，`mean−worst10-mean` 为 `11.14/10.44/8.97/5.15/2.50 dB`，`outage(<24dB)` 为 `0.353/0.227/0.149/0.055/0.026`，`CVaR-10 MSE/mean` 为 `5.67/6.02/5.85/4.34/2.65×`；信道方差占比 `0.801/0.747/0.670/0.436/0.186`，Spearman(PSNR,`|h|²`) `0.748/0.685/0.620/0.385/0.187`
- **接收端喂入真实有效 SNR 反而五档全面变差**：mean PSNR `nominal` vs `effective` 为 `24.56/21.48`、`26.47/23.38`、`27.91/24.54`、`29.94/25.08`、`30.92/25.31`；clamp 版介于两者之间。说明条件嵌入无法表示深衰落有效 SNR（`[1,19] dB` 训练，深衰落可达 `−20 dB` 量级）
- 最差案例：40 组 `原图|median|worst-10%|worst` 全部通过重放校验（`|ΔPSNR|<0.01 dB`）。典型 `snr1dB_000000013004`：median（`|h|²=0.627`）`31.08 dB` 视觉良好，worst-10%（`|h|²=0.0385`）`16.25 dB` 已语义崩塌，worst（`|h|²=0.0003`）`8.40 dB` 纯噪声，同图跨度 `24.71 dB`
- **判定：`NO-GO`。** 四项 GO 条件通过 3 项（`median−p10≥2dB` 的 SNR 点有 4 个、`mean−worst10≥1dB`、outage 不可忽略），归因项未通过
- **预注册缺陷已主动披露**（结果报告 §4）：归因统计量在预注册中标为「必须报告，不作为 gate」，实现中却被写成 gate，阈值 `0.5` 未预先数值化，且用了跨五档 `all(...)` 而第 1 项 GO 条件只要求两个 SNR 点。读法 A（脚本字面）=`NO-GO`，读法 B（与第 1 项同口径，仅在 1/4/7/13 dB 评估）=`GO`。两种读法均如实报告，未改脚本取有利结果，`verdict.json` 保留原始输出；两种读法都不改变上述实质结论
- 结论：**不以当前形式启动 P4/P5。** 任务书 §10 要求 CVaR 必须打败 `Repeated-fading mean control`，该对照不存在且很可能自己吸收大部分已测尾部。建议的下一个实验是更便宜的那个——在 Rayleigh 上用正确有效 SNR 条件训练均值基线，再用同一诊断脚本重测；仍有 `≥2 dB` 尾部才值得测 CVaR
- 局限：S33B 从未在衰落上训练，无法完全分离「风险不敏感」与「信道错配」；无语义指标；单 backbone/单码率 `1/24`/单 block-fading/ZF `ε=0`；未做 bootstrap CI（go/no-go 阶段只报点估计）
- 新增代码：`src/cadsd_jscc/tail_risk.py`、`scripts/cvar_p0_diagnose_tail_risk.py`、`scripts/cvar_p0_analyze_tail_risk.py`、`scripts/cvar_p0_export_worst_cases.py`、`configs/cvar_p0_tail_risk_diagnostic.yaml`、`tests/test_tail_risk.py`
- 既有信道/训练/评测代码**零改动**；`external_common.py` 未修改（realization 噪声复用 `canonical_standard_normal(base_seed, f"{id}|r{k}", snr, 16384)`）
- 验证：新增单测 18/18，全仓 `140/140` unittest 通过，`py_compile` 通过

## RDD-P0 生成式重建分布偏移

### ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001

- 日期：2026-07-30
- 当前状态：完成；纯分析实验，无训练、无下载、无 official validation 访问
- 目的：在借用 rate-distortion-deception (RDD, arXiv 2607.25997) 框架前，先验证现有生成式 JSCC 重建是否已存在"无意的、可识别的分布偏移"
- 前提修正：原设计要求 CLIC2020 × S33/DiffJSCC/SGD 不可执行——`paper_idea1b/A1_DISCRIMINATIVE_RESULT.md:19` 记录"DiffJSCC、SGD 和 refiner 未加载"，CLIC 重建只有 S33+两条 Swin。三方法唯一共存总体为 64 图 Imagenette policy-dev @256²
- 主总体：64 图 × 5 SNR `[1,4,7,13,19]` × 3 seeds `[20260748,20260749,20260750]`；每方法 960 行，每 (method,SNR) 单元 n=192
- 四臂：`s33_strong`（无先验，冻结 checkpoint 精确重放）、`author_jscc`（无先验，S30 montage 面板1，免费获得的第二判别式对照）、`diffjscc`（SD 2.1，面板2）、`sgd_jscc`（MDTv2/DiT，S20 montage tile）
- 参考集（10 组各 64 图）：`real`、`vae_sd21`、`vae_sgd`、`resample_512`、`blur_s{0.5,1,1.5,2}`、`jpeg_q{30,70}`
- 指标：`cleanfid`；**KID 主、FID 必报**（n=192 下 2048 维协方差秩亏、FID 正偏，沿用 S34C 先例）；每 (arm,SNR,reference) 单独计算，不跨 SNR/方法混合
- 验证门全部通过：S33 重放 max\|ΔPSNR\|=`0.0 dB` 且 960/960 noise SHA 校验；author-JSCC 面板=`5.46e-06 dB`；DiffJSCC 面板=`3.98e-06 dB`；SGD=`0.0385 dB`（median `0.0030`，属 S34C 已记录 uint8/float 口径差）
- **SGD 源 tile 与 DiffJSCC 源面板逐字节相同：0 mismatch/64**，证明两链共享同一总体、跨方法比较合法
- 测量链交叉验证：本轮 CLIC 管线在 7 个与 A1 重叠单元复现 A1 冻结值至 ΔFID<`0.008`、ΔKID<`3e-6`
- ② 结果：116 个 (arm,SNR,ref) 命中，强② 仅 12 个且**全部是判别式臂→blur**；生成臂全部弱②。判别式臂偏向 blur_s1/s1p5/s2（`s33_strong` real 排名五档恒为 9–10/10；`author_jscc` 为 9,10,10,8,6），生成臂偏向 vae_sd21/vae_sgd（DiffJSCC real 排名恒 4/10、SGD 为 6,6,5,6,4）
- ② 的方向性归因**不成立**：`sgd_jscc` 先验是 MDTv2/DiT 但最常偏向 `vae_sd21`；两 VAE 参考集彼此过近（FID `18.74` vs `19.12`）不可区分
- ① 结果（GroupKFold(5) 按 source image 分组，bootstrap 10,000 次 source-cluster CI）：3 臂 logreg=`0.9059 [0.8715,0.9378]`（随机 0.3333）；4 臂=`0.8396 [0.7984,0.8776]`（随机 0.25）；CI 下界均远超随机
- 逐臂 recall（C0 logreg）：`0.821/0.760/0.970/0.807`；DiffJSCC 几乎完全可分（仅 28/960 错分）；混淆主要在两判别式臂之间
- 最有区分力特征：`dct_hi_cv`、`rps_b09`、`rps_b11`、`hp_mad`、`grad_mean` → **区分信息集中在高频**；C2 降到 128² 后准确率 `0.840→0.710`(logreg)/`0.619`(hgb)
- **C3 关键否证**：两个均无生成先验的判别式臂之间，轻量频域统计即达 `0.8693 [0.8214,0.9120]`（随机 0.5）。故可识别指纹**不是生成先验特有**，按预注册事前声明削弱"先验导致偏移"的解释
- CLIC-428 补充（n=428、原生分辨率、仅判别式臂）：17 个②命中全部指向 JPEG，**无一指向 blur**；Swin 两臂在 7–19 dB real 排名 `1/8`（最接近 real）；`s33_strong` 五档 best 均为 `2/8`，未把 real 排第一。因此 256² 的"偏向 blur"强②**不可外推**到高分辨率高功效设置
- 预注册判据渲染：①②同时成立 → **"存在可识别偏移"**，但必须同载三条限定（①非先验特有、②非先验定向、强②不稳健）
- 失败并保留：首轮 `vae_sgd` 用未归一化 latent 直接 decode，往返 PSNR 仅 `12.55 dB`、FID vs real=`273.79`、肉眼色彩崩坏。根因是 SGD 解码器始终接收功率归一化 latent（`inference_config.py:151`）。修正后 `30.773 dB`/FID `18.74`。失败产物保留于 `outputs/analysis/ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001/failed/`，未删除未覆盖。该 bug 若未发现会静默污染整个 (b) 参考集
- 工程：VAE 阶段改用既有 `.venv-sgdjscc`（主环境缺 `pytorch_lightning`），未新增安装；DiffJSCC ckpt 内嵌 Lightning 对象无法 `weights_only=True`，改为先校验 SHA `ae1e6df0…dec579`（与 S30 契约一致）再加载；SGD ckpt SHA=`455cb603…1915fe`；两 VAE 均 missing_critical=0/unexpected=0；cleanfid Inception 使用 A0 冻结本地副本（`95,607,719` bytes，与 A0 `expected_bytes` 精确一致）软链接注入，未联网
- 边界：SGD 全程 non-ranking paper upper，只做分布分析不做质量胜负；不改变 A1 已冻结的 S33-vs-Swin 结论；不训练任何生成模型；official Imagenette validation 继续封存
- 预注册：`reports/rdd_p0_distribution_shift_preregistration_2026-07-30.md`
- 配置：`configs/rdd_p0_distribution_shift.yaml`
- 结果：`reports/rdd_p0_distribution_shift_result_2026-07-30.md`
- 产物：`outputs/analysis/ANALYSIS-RDD-P0-DISTRIBUTION-SHIFT-001/`

## paper idea1b A1 判别式主表

### SMOKE-IDEA1B-A1-DISCRIMINATIVE-001 / ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001

- 日期：2026-07-23
- 当前状态：完成；Kodak1080行、CLIC6420行、全指标7500行与15组FID/KID均闭合；保守 verdict 为S33劣于Swin
- 方法：冻结S33、Swin official Base-SA、Swin capacity-matched CM-SA
- 数据：smoke为Kodak一张+最大2048×2048 CLIC一张；正式Kodak 24×5SNR×3seed，CLIC428×5SNR×1seed
- 码率：共同256 tile、共同padding与noise，每tile exact `16,384 real`；逐图actual CBR严格相同
- 指标：PSNR/MS-SSIM/LPIPS/DISTS、OpenCLIP image cosine；CLIC FID/KID；source-cluster 10,000次bootstrap CI
- 禁止：DiffJSCC、SGD、refiner、official Imagenette validation
- 预注册：`paper_idea1b/A1_DISCRIMINATIVE_PREREGISTRATION.md`
- 配置：`paper_idea1b/configs/a1_discriminative_benchmark.yaml`
- smoke结果：Kodak三臂wall=`42.6/42.6/44.0 ms`；最大2048² CLIC=`189.7/439.5/464.4 ms`；peak reserved=`1.21/2.20/2.21 GiB`；最大功率误差=`2.38e-7`
- 码率结果：Kodak actual CBR=`1/24`；CLIC范围=`0.041667–0.063210`、均值=`0.045472`；三臂逐图actual-rate/noise PASS
- Kodak aggregate：S33−Base PSNR=`+0.0477 dB [−0.0537,+0.1612]`，追平/非劣但LPIPS/DISTS显著更差；S33−CM=`−0.2003 [−0.3116,−0.0846]`，劣于
- CLIC aggregate：S33−Base=`−0.2631 dB [−0.3211,−0.2074]`；S33−CM=`−0.4909 [−0.5513,−0.4352]`；五档均劣于，LPIPS/DISTS/CLIP/FID/KID总体也弱
- 指标smoke：attempt1因PyTorch2.6/OpenCLIP TorchScript `weights_only`兼容性失败并保留；attempt2通过，最大图DISTS约`795 ms`、peak reserved=`6.37 GiB`
- 中断记录：全量指标在6210/7500处遭外部终止；断点续跑后越过原位置并完成，无缺失或覆盖
- 语义边界：Kodak/CLIC只报告原图—重建CLIP连续相似度，不报告事后阈值化的监督失败率；official validation未访问
- 结果：`paper_idea1b/A1_DISCRIMINATIVE_RESULT.md`、`paper_idea1b/outputs/ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001/summary.json`

## paper idea1b Gate A0

### GATE-A0-BENCHMARK-SETUP-001

- 日期：2026-07-23
- 状态：完成；基础设施/identity实验，不作任何方法质量排名；A1未授权
- 数据：Kodak 24 + CLIC2020 test 428，共452张RGB，无内容重复
- 下载：显式清空proxy、服务器直连；CLIC Mobile/Professional官方包SHA=`2025f07a...aa732 / 857df244...52884`；Kodak mirror archive SHA=`44e2569b...00223`并逐文件通过官方字节数表
- 冻结引用：S33 checkpoint `2daad9e7...5bfb`、canonical noise `01978a77...6d22`、S34D aggregate `7fdeb1ff...f931`，均原地复核、不搬移/复制
- manifests：source=`452`、method-rate=`2,260`、S33/Swin tiles=`20,018`、SGD released patches=`882,675`
- 公平合同：方法使用原生冻结处理；S33/Swin可tile，DiffJSCC whole-frame，SGD author patch；padding/overlap/sender-side信息实际计费；receiver-only先验只计计算
- 码率预检：Kodak S33/Swin=`98,304 real/图, CBR=1/24`；DiffJSCC native whole-frame公式=`CBR 1/96`，是under-budget且待A1 runtime instrumentation，不能称exact-rate；SGD caption未计费、non-ranking
- identity结果：PSNR=`120 dB`、MS-SSIM=`1`、LPIPS=`0`、DISTS=`5.96e-8–1.19e-7`；CLIC self-FID=`−4.5057e-5`、self-KID=`−0.00205318`
- 失败记录：attempt1错误要求PSNR=∞且设`|FID|≤1e-5`，已保留；按共享PSNR 120 dB clamp和clean-fid浮点残差修订判据后通过
- 训练/方法推理/official Imagenette validation：均未执行
- 输出：`paper_idea1b/outputs/GATE-A0-BENCHMARK-SETUP-001/`

## S35R 新主线预注册

### ANALYSIS-S35R-P0-SGD-ADAPTIVE-COST-001 / EXP-S35R-P1-LIGHT-RECEIVER-REFINER-001

- 日期：2026-07-23
- P0 状态：完成；不训练、不下载、official validation 不访问
- P1 状态：只预注册；one-batch smoke 与正式训练均未获授权
- P0 输入：S34D SGD 80条逐图计时，RTX 4090D、PyTorch 2.1、batch=1、五档各16张相同图
- P0 核心审计：`alpha_bar_channel=2γ/(2γ+1)` 只决定 continuous trajectory endpoint；actual evaluation count 以作者 sampler 的点数、循环和 final prediction逐项核算
- P1 backbone：冻结 S33 `16,384 real`，checkpoint SHA=`2daad9e7...5bfb`；refiner额外通信符号=0
- P1 generator：三尺度48/96/192 residual U-Net，SNR-FiLM，零初始化输出，部署参数目标2M–6M；训练期 conditional PatchGAN 不计部署参数
- P1 loss：`1.0 LPIPS + 5.0 MSE + 0.01 hinge GAN + 0.5 L1(refined,S33 anchor)`
- P1 selection：冻结 COCO val512，PSNR差 `>−0.10 dB` 的候选中选 LPIPS最低；policy-dev不选点
- P1 go/no-go：冻结64图×3 seed×5 SNR，10,000次 source-cluster bootstrap；LPIPS显著改善、PSNR非劣、semantic failure不显著上升三项同时满足才继续
- 预注册：`reports/s35r_p0_sgd_adaptive_cost_preregistration_2026-07-23.md`、`reports/s35r_p1_light_receiver_refiner_preregistration_2026-07-23.md`
- P0 结果：五档均为50次 denoiser evaluation；端到端均值=`2044.877/2043.783/2044.802/2044.636/2045.410 ms`
- P0 fixed floor：BLIP2+MuGE=`1069.933 ms/图`，约占总延迟 `52.33%`；五档合计均值范围仅 `0.440 ms`
- P0 解释：`alpha_bar_channel`/learned CSI 改 continuous trajectory endpoint，不改变 released `diffusion_step=50` 的调用数；完整结果见 `reports/s35r_p0_sgd_adaptive_cost_result_2026-07-23.md`

## ID 规则

每个实验必须有唯一 ID。

格式如下：

- `EXP-S1-001`：阶段1，DeepJSCC baseline
- `EXP-S2-001`：阶段3，Blind diffusion refinement
- `EXP-S3-001`：阶段4，Semantic drift metric
- `EXP-S4-001`：阶段5，Channel-adaptive semantic guidance
- `EXP-S5-001`：阶段6，完整实验

即使实验失败，也不能复用 ID。

## 实验索引

| ID | 日期 | 项目版本 | 方法 | 数据集 | 信道 | SNR | CBR | 指标 | 状态 | 输出路径 |
|---|---|---|---|---|---|---|---|---|---|---|
| GATE-A0-BENCHMARK-SETUP-001 | 2026-07-23 | config SHA `37ffc8c7...f6bd`；旧资产只读SHA复核 | method-native processing/rate manifest + metric identity sanity；无方法推理 | Kodak 24 + CLIC2020 test 428 | 无信道推理 | N/A | 逐图actual ledger；S33 Kodak=`1/24`；Diff native formula=`1/96`；SGD caption unpriced | PSNR/MS-SSIM/LPIPS/DISTS identity；CLIC self-FID/KID | 完成（452张RGB无重复；全部identity checks PASS；attempt1过严判据失败保留；A1未授权；official val sealed） | `paper_idea1b/outputs/GATE-A0-BENCHMARK-SETUP-001/` |
| ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001 | 2026-07-23 | frozen S33/DiffJSCC/SGD checkpoints；measurement-only | same-entry batch1 receiver latency decomposition + DiffJSCC 100/50/25/10/4-step curve | frozen policy-dev；latency 16×5；quality 64×1seed×5 | canonical AWGN | `[1,4,7,13,19]` dB | historical contracts retained；S33/Diff 16,384 real；SGD ≥21,856 real | wall/core component ms、LPIPS/PSNR/MS-SSIM/failure、unique params、profiled FLOPs lower bound | 完成（Diff 25-step 最低过 LPIPS gate=`1458.5ms`；同 Torch S33=`8.833ms`，慢165×；25-step failure显著增加；无训练/下载；official val sealed） | `outputs/analysis/ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001/` |
| ANALYSIS-S34C-LITE-RATE-TRANSPARENCY-001 | 2026-07-23 | config input snapshot SHA `a3b5cb8f...8bb6`；script SHA `87d1ad4a...5c6a`；existing S33/S30/S20/S28 frozen artifacts only | read-only unified rate/side-info/prior/metric ledger；no global ranking | frozen 64 images×3 seeds×5 SNR，960 rows/method | reuse existing canonical AWGN results | `[1,4,7,13,19]` dB | S33/DiffJSCC 16,384 real；SGD minimum 21,856 real（+33.40%） | PSNR/MS-SSIM/LPIPS/T_cls failure、per-SNR、source-cluster 95% CI；FID/KID unavailable | 完成（audit PASS；S33/Diff exact-rate fidelity–perception Pareto；SGD non-ranking paper upper；无训练/推理/下载；official val sealed） | `outputs/analysis/ANALYSIS-S34C-LITE-RATE-TRANSPARENCY-001/` |
| PLAN-S34C-FAIR-GENERATIVE-REPRODUCTION-001 | 2026-07-23 | DiffJSCC `13aeb624...`；SGDJSCC `2188acc0...`；S33 checkpoint `2daad9e7...5bfb` | official-code DiffJSCC COCO/five-SNR retrain + approximate SGD released-component exact-total-rate adaptation | train COCO train2017；paired audit frozen 64×3×5；planned sealed COCO val 2048 perception holdout | canonical paired-real AWGN | `[1,4,7,13,19]` dB | strict total `16,384 real` / `8,192 complex` / `1/24` | planned PSNR/MS-SSIM/LPIPS/FID/KID/T_cls failure + source-cluster 95% CI | **用户在任何执行前暂停**；未 smoke/训练/创建输出；等待轻量版后再决定 | 尚未创建；保留合同 `configs/s34c_fair_generative_reproduction_preregistration.yaml` |
| ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-003 | 2026-07-23 | config SHA `b71d5c09...dcba2`；script SHA `b78391bc...183f`；S33 checkpoint `2daad9e7...5bfb` | 1 dB LPIPS 可接受域内按 T_cls / AlexNet / ResNet18 / MobileNetV3 / CLIP 异常分层选 15 source；S33 exact replay + SGD 既有重建 + 人工三分类 | frozen policy-dev 64 图×3 seeds 的 1 dB 共 192 键；official val sealed | canonical AWGN；S33 历史同噪声重放 | 1 dB | 沿用历史各自合同；SGD 仅 paper upper，不作公平排名 | LPIPS、PSNR、T_cls、跨模型 top-1、CLIP cosine、人工 faithful/重建失败/clear-wrong | 完成（候选 84；S33 replay PSNR max error `0.0 dB`；S33=`8/7/0`，SGD=`15/0/0`；无训练） | `outputs/analysis/ANALYSIS-LOW-SNR-SEMANTIC-DRIFT-AUDIT-003/` |
| ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001 | 2026-07-23 | config SHA `09dc7fe8...3393`；script SHA `5216cb4a...adf3`；S33 checkpoint `2daad9e7...5bfb` | 固定上述 15 source、不按压力结果回选；S33 与 SGD paper-upper 在共同 seed 下 −3/−5 dB 推理重放并人工审阅 | 同一 15 张低 SNR 异常候选 source；official val sealed | canonical AWGN / SGD official step matching | −3、−5 dB（明确 out-of-range stress） | S33 16,384 real；SGD main+edge 19,712 real 且 captions 免费，不作排名 | PSNR、LPIPS、T_cls failure、人工 faithful/重建失败/clear-wrong | 完成（−3 dB S33=`1/14/0`、SGD=`15/0/0`；−5 dB S33=`0/15/0`、SGD=`15/0/0`；无训练） | `outputs/analysis/ANALYSIS-LOW-SNR-OUT-OF-RANGE-STRESS-001/` |
| ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004 | 2026-07-23 | config SHA `3286cbcf...e643`；script SHA `83c72f99...7dbf`；S33 checkpoint `2daad9e7...5bfb` | SGD formal montage crop + frozen S33 canonical-noise inference replay；各方法 LPIPS 升序、source 去重 top-15 | frozen S20 Imagenette policy-dev 64×3 seeds×5 SNR 的既有逐样本记录 | canonical AWGN（只重放 S33 已选键） | SGD: 14×19dB+1×7dB；S33: 14×19dB+1×13dB | 沿用各自历史合同；不作公平排名 | LPIPS 排序、历史 PSNR replay、T_cls failure、人工 subject/object/scene fidelity | 完成（无训练；S33 replay PSNR max error `0.0 dB`；SGD/S33 semantic mismatch 均 `0/15`；minor change=`1/3`；official val 未访问） | `outputs/analysis/ANALYSIS-TOP-LPIPS-SEMANTIC-VISUAL-AUDIT-004/` |
| EXP-S34A-SWINJSCC-BASE/CM-EQUAL-BUDGET-001 | 2026-07-22 | config SHA `a209af08...676d`；initial train script SHA `9e30dd10...71ca`；resume RNG-device patch SHA 见 resume event；official source `a6d0e6d...90f` | official Base-SA 28.18M + capacity-matched CM-SA 31.35M，FP32 4+8ep equal-budget | COCO train2017 full / 与 S33 相同固定 val2017 512 | canonical paired-real AWGN | per-image discrete `[1,4,7,13,19]` dB | exact 16,384 real（8,192 complex；CBR 1/24） | train MSE；逐 epoch五档 PSNR/MS-SSIM；epoch 9--12 收敛 gate | 完成（双臂 12/12，best 均 epoch12；Base=`29.100812 dB`，CM=`29.322195 dB`；两臂均触发 extension gate，但未授权/未执行） | `outputs/train/EXP-S34A-SWINJSCC-BASE-SA-EQUAL-BUDGET-001/`、`outputs/train/EXP-S34A-SWINJSCC-CM-SA-EQUAL-BUDGET-001/` |
| ANALYSIS-S34A-SWINJSCC-EQUAL-BUDGET-COMPARISON-001 | 2026-07-22 | config SHA `af7a01e5...133a`；final evaluator SHA `c6223cec...3b4`；checkpoint SHA Base=`d645e156...f75`、CM=`751ef505...160` | frozen S33 vs official Base-SA / capacity-matched CM-SA | frozen S20 Imagenette policy-dev 64×3 seeds×5 SNR | canonical paired-real AWGN，共用 16,384-D noise prefix | `[1,4,7,13,19]` dB | exact 16,384 real（8,192 complex；CBR 1/24） | PSNR/MS-SSIM/LPIPS/T_cls failure、new/repair、source-cluster 95% CI、0.10 dB gate | 完成（PASS 1,920/1,920；S33−Base `+0.173947 dB [0.078178,0.265733]` 显著；S33−CM `−0.065902 [−0.168886,0.025307]` 未过非劣 gate；总 verdict PARETO；official val 未访问） | `outputs/external_baselines/ANALYSIS-S34A-SWINJSCC-EQUAL-BUDGET-COMPARISON-001/` |
| SMOKE-S34A-SWINJSCC-CALIBRATION-001 | 2026-07-22 | config snapshot SHA `9c05d62d...cae4`；script SHA `19eca7bc...ce8`；official source `a6d0e6d...90f` | official-source SwinJSCC Base-SA 28.18M + capacity-matched SA 31.35M project adapter | COCO train2017 first real microbatch（8 images/arm） | canonical paired-real AWGN adapter | per-image `[1,4,7,13,19]` dB cycling | exact 16,384 real（8,192 complex；CBR 1/24） | finite forward/backward, exact symbols, per-image power, checkpoint round-trip, time, VRAM | 完成（双臂 PASS；peak reserved `9.75/10.40 GiB`；systems-only，不作质量或收敛结论） | `outputs/smoke/EXP-S34A-SWINJSCC-CALIBRATION-001/` |
| EXP-S31-STRONG-JSCC-001 | 2026-07-21 | config SHA `a880490e...`；原 script SHA `f215f97b...` | clean-room 31.12M native-rate SNR-conditioned strong JSCC | COCO train2017 full / val2017 frozen 512 | AWGN | train/eval [1, 4, 7, 13, 19] dB | exact 19,712 real（9,856 complex；0.05013） | MSE, PSNR, MS-SSIM, normalized-power error | 失败并保留（epoch4 batch418 AMP gradient overflow；此前 best epoch3 `28.0448 dB/0.958405`，SHA `8e8f3b7b...`） | `outputs/train/EXP-S31-STRONG-JSCC-001/` |
| EXP-S31B-STRONG-JSCC-FP32-001 | 2026-07-21 | local changes；config/script snapshot SHA 见 metadata | frozen S31 epoch3 model-only init + FP32 continuation | COCO full；错误 seed 会改变 frozen val512 | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | pre-validation contract audit | 配置合同失败并主动中止（0 history/validation rows；错误在结果前发现） | `outputs/train/EXP-S31B-STRONG-JSCC-FP32-001/` |
| EXP-S31B-STRONG-JSCC-FP32-002 | 2026-07-21 | config SHA `825304f6...`；script SHA `a9d03af8...` | frozen S31 epoch3 model-only init + FP32 stable continuation | same COCO full / identical frozen val512；external population sealed during selection | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | MSE, PSNR, MS-SSIM, power/finite audit | 完成（8/8 finite；best epoch7 `29.360583 dB/0.967330`，SHA `2f8972a9...`） | `outputs/train/EXP-S31B-STRONG-JSCC-FP32-002/` |
| ANALYSIS-S32-STRONG-JSCC-COMPARISON-001 | 2026-07-21 | config SHA `b9766fd7...`；script SHA `ca931872...` | frozen strong-JSCC vs author-JSCC/DiffJSCC/current/B1 | frozen S20 Imagenette policy-dev 64×3 seeds×5 SNR | AWGN | [1, 4, 7, 13, 19] dB | strong/current/B1 19,712 real；author/DiffJSCC 16,384 real | PSNR, MS-SSIM, LPIPS, T_cls failure, source-cluster CI, float/uint8 sensitivity | 完成（PASS 960/960；strong 聚合三质量轴显著优于 author-JSCC；与完整 DiffJSCC 为 fidelity/perception Pareto） | `outputs/external_baselines/ANALYSIS-S32-STRONG-JSCC-COMPARISON-001/` |
| SMOKE-S31-STRONG-JSCC-001 | 2026-07-21 | local changes | full-size strong JSCC one-update GPU smoke | synthetic train 8 / val 4 | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | forward/backward, finite metrics, normalized-power error | 完成（systems-only；max power error `1.19e-7`，质量数值不作结论） | `outputs/smoke/EXP-S31-STRONG-JSCC-001/` |
| SMOKE-S31B-STRONG-JSCC-FP32-001 | 2026-07-21 | local changes | frozen S31 init + full-size FP32 one-update smoke | synthetic train 8 / val 4 | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | init SHA, forward/backward, finite metrics, normalized-power error | 完成（systems-only；max power error `2.38e-7`，质量数值不作结论） | `outputs/smoke/EXP-S31B-STRONG-JSCC-FP32-001/` |
| SMOKE-S31B-STRONG-JSCC-FP32-002 | 2026-07-21 | local changes | corrected-seed frozen S31 init + full-size FP32 one-update smoke | synthetic train 8 / val 4 | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | init SHA, forward/backward, finite metrics, normalized-power error | 完成（systems-only；max power error `1.19e-7`，质量数值不作结论） | `outputs/smoke/EXP-S31B-STRONG-JSCC-FP32-002/` |
| ANALYSIS-S31-DIFFJSCC-RESIZE-ROUNDTRIP-001 | 2026-07-21 | local changes | DiffJSCC 256→128→256 bicubic source roundtrip audit | frozen S20 Imagenette policy-dev 64 images | N/A | N/A | N/A | PSNR, MS-SSIM | 完成（mean `45.5015 dB/0.999635`；仅隔离输入缩放损失） | `outputs/analysis/ANALYSIS-S31-DIFFJSCC-RESIZE-ROUNDTRIP-001/` |
| EXP-S21-B1AGF-001/002 | 2026-07-20 | abf117b + local changes | B1-anchored learned-gate output fusion | fresh COCO 5000 train / 256 selection；holdout sealed | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, LPIPS, gate/injection | 失败并停止（learned gate 塌零；holdout 未访问） | `outputs/EXP-S21-B1AGF-001/`、`outputs/EXP-S21-B1AGF-002/` |
| EXP-S21-B1AR-003 | 2026-07-20 | abf117b + local changes | Fixed-gate bounded B1 output residual | same fresh development population；holdout sealed | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, LPIPS, injection saturation | 失败并停止（第3轮达到 envelope 上限；holdout 未访问） | `outputs/EXP-S21-B1AR-003/` |
| ANALYSIS-S21-CONVEX-SELECTION-004 | 2026-07-20 | abf117b + local changes | Monotonic B1/diffusion pixel convex envelope | fresh COCO 256 selection；holdout sealed | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, LPIPS, feasible policy | 完成（120 个候选仅全零 B1 可行；holdout 未访问） | `outputs/analysis/ANALYSIS-S21-CONVEX-SELECTION-004/` |
| EXP-S22-B1FI-001 | 2026-07-20 | abf117b + local changes | Frozen-B1 zero-conv matched-diffusion feature injection | fresh COCO 5000 train / 256 selection；holdout sealed | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, LPIPS, exact anchor, feature injection | 完成（LPIPS 明显改善但 PSNR 小幅退化；选择 epoch0；holdout 未访问） | `outputs/EXP-S22-B1FI-001/` |
| EXP/ANALYSIS-S23-B1FS-001 | 2026-07-20 | abf117b + local changes | One-epoch frozen-B1 diffusion feature direction + preregistered global shrink | fresh COCO 5000 train / 256 selection / 256 independent holdout | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR/LPIPS cluster CI, exact fallback, pseudo new/repair | 完成（5/5 checks；非零安全合并，但 PSNR 效应很小） | `outputs/EXP-S23-B1FS-001/`、`outputs/analysis/ANALYSIS-S23-B1FS-*` |
| ANALYSIS-S24-RECENT-PROGRESS-SUMMARY-001 | 2026-07-20 | local changes | Frozen S19/S20/S23 derived metric aggregation + receiver-postprocessor microbenchmark | S19/S23 independent COCO holdouts；S20 independent Imagenette policy-dev | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real；SGD caption 另计 | PSNR, MS-SSIM, LPIPS, semantic new/repair, cluster CI, rate, params, scoped latency | 完成（只汇总已知冻结结果；不选模型、不调 holdout） | `outputs/analysis/ANALYSIS-S24-RECENT-PROGRESS-SUMMARY-001/` |
| ANALYSIS-S25-B1FA-HEADROOM-001 | 2026-07-20 | local changes | Frozen S23 per-sample amplitude oracle headroom diagnostic | exposed S23 selection 256×5；holdout not accessed | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, MS-SSIM, LPIPS, majority new/repair, cluster CI, alpha distribution | 完成（明确负结果；safe oracle 仅比 fixed 高 `0.001365 dB`，关闭 controller 路线） | `outputs/analysis/ANALYSIS-S25-B1FA-HEADROOM-001/` |
| ANALYSIS-S26-S19-XF-REPLICATION-001 | 2026-07-20 | local changes | Frozen S19 low-SNR fusion/control + exact-B1 high-SNR route | S21/S23 COCO holdout 256×5；no target selection | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, MS-SSIM, LPIPS, majority new/repair, exact fallback, cluster CI | 完成（9/9 PASS；相对 B1 `+0.09327 dB/-0.00766 LPIPS`） | `outputs/analysis/ANALYSIS-S26-S19-XF-REPLICATION-001/` |
| ANALYSIS-S27-S19-XF-FRESH-001 | 2026-07-21 | local changes | Frozen S19 low-SNR fusion/control + exact-B1 high-SNR pristine replication | fresh COCO 512×5；path/SHA overlap 0；no selection | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | PSNR, MS-SSIM, LPIPS, majority new/repair, exact fallback, cluster CI | 完成（9/9 PASS；相对 B1 `+0.09266 dB/-0.00792 LPIPS`） | `outputs/analysis/ANALYSIS-S27-S19-XF-FRESH-001/` |
| ANALYSIS-S28-CURRENT-VS-SGD-001 | 2026-07-21 | local changes | Frozen current method vs B1/control/SGD-JSCC paper upper | frozen S20 Imagenette policy-dev 64×3 seeds×5 SNR | AWGN | [1, 4, 7, 13, 19] dB | current/B1 exact 19,712 real；SGD captions unmetered | PSNR, MS-SSIM, LPIPS, T_cls failure/new/repair, cluster CI, rate audit | 完成（B1 增益全通过；对 SGD 为 fidelity/perception Pareto；原技术 verdict 因 batch 浮点阈值保留 NEGATIVE） | `outputs/external_baselines/ANALYSIS-S28-CURRENT-VS-SGD-001/` |
| ANALYSIS-S29-S28-B1-EXACT-BATCH-001 | 2026-07-21 | local changes | S28 B1 original-batch numerical contract audit | same frozen S20 64×3×5 | AWGN | [1, 4, 7, 13, 19] dB | exact 19,712 real | per-row noise/prediction/failure/PSNR/MS-SSIM/LPIPS exact replay | 完成（6/6 PASS；960 行全部零误差） | `outputs/analysis/ANALYSIS-S29-S28-B1-EXACT-BATCH-001/` |
| ANALYSIS-S30-DIFFJSCC-CHECKPOINT-AUDIT-001 | 2026-07-21 | local changes | Official DiffJSCC OpenImage C16 checkpoint integrity/module audit | N/A（推理前资产审计） | N/A | N/A | checkpoint latent contract=16,384 real | SHA-256, tensor keys/numel/dtype, critical module prefixes | 完成（PASS；六类核心模块齐全，确认只排除 BLIP2） | `outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-CHECKPOINT-AUDIT-001/` |
| ANALYSIS-S30-DIFFJSCC-PREFLIGHT-001 | 2026-07-21 | local changes | Official DiffJSCC source/checkpoint/BLIP2/runtime/population/noise/rate preflight | frozen S20 Imagenette policy-dev 64 图合同 | AWGN | [1, 4, 7, 13, 19] dB | DiffJSCC 16,384 real；project ceiling 19,712 real | exact hashes, 960 noise keys, rate ledger | 完成（PASS；双权重、源码、总体和噪声全部闭合） | `outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-PREFLIGHT-001/` |
| ANALYSIS-S30-DIFFJSCC-SMOKE-001 | 2026-07-21 | local changes | Official DiffJSCC exact-BLIP2 100-step end-to-end smoke | frozen population first image×1 seed×1 dB | AWGN | 1 dB | 16,384 real | PSNR, MS-SSIM, LPIPS, T_cls, caption, runtime, VRAM | 完成（PASS；1/1 row，仅可执行性结论） | `outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-SMOKE-001/` |
| ANALYSIS-S30-DIFFJSCC-FIRST-SEED-001 | 2026-07-21 | local changes | Official DiffJSCC first-seed staged comparison | frozen S20 Imagenette policy-dev 64×1 seed×5 SNR | AWGN | [1, 4, 7, 13, 19] dB | 16,384 real vs current/B1 19,712 real ceiling | four-arm quality, T_cls new/repair, cluster CI, systems | 完成（PASS 320/320；Pareto early result，不作最终 claim） | `outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-FIRST-SEED-001/` |
| ANALYSIS-S30-DIFFJSCC-COMPARISON/POST-003 | 2026-07-21 | local changes | Official DiffJSCC full comparison + range-separated post analysis | frozen S20 Imagenette policy-dev 64×3 seeds×5 SNR | AWGN | [1, 4, 7, 13, 19] dB | 16,384 real；project ceiling 19,712 real | PSNR, MS-SSIM, LPIPS, T_cls failure/new/repair, source-cluster CI, runtime, VRAM | 完成（960/960；`PARETO_OR_INCONCLUSIVE`；发现强 author-JSCC backbone 差距） | `outputs/external_baselines/ANALYSIS-S30-DIFFJSCC-COMPARISON-001/` |
| EXP-S1-001 | 2026-06-29 | N/A (not a project git repo) | M0-DeepJSCC | CIFAR-10 test subset, 1024 images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM | 完成 | `outputs/EXP-S1-001/` |
| EXP-S2HR-001 | 2026-06-30 | N/A (not a project git repo) | M0-DeepJSCC-HR-pilot | COCO2017 val split pilot, 4500 train / 500 val | AWGN | 7 dB | 0.17 | MSE, PSNR, SSIM | 完成（非正式 pilot） | `outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/` |
| EXP-S2HR-002 | 2026-06-30 | N/A (not a project git repo) | M0-DeepJSCC-HR-pilot export | COCO2017 val split pilot, 500 val | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成（非正式 pilot） | `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/` |
| EXP-S2HR-003 | 2026-07-01 | N/A (not a project git repo) | M0-DeepJSCC-HR formal train | COCO2017 train2017 / val2017 | AWGN | 7 dB | 0.17 | MSE, PSNR, SSIM | 完成（best 可用，latest NaN） | `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/` |
| EXP-S2HR-004 | 2026-07-01 | N/A (not a project git repo) | M0-DeepJSCC-HR formal export | COCO2017 val2017 subset, 512 images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成 | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/` |
| EXP-S2HR-005 | 2026-07-03 | 8678e4f | M0-DeepJSCC-HR formal export 256 saved images | COCO2017 val2017 subset, 512 eval / 256 exported images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成（供 residual validation 使用） | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/` |
| EXP-S2HR-006 | 2026-07-06 | 3bcf825 | M0-DeepJSCC-HR formal export 384 saved images | COCO2017 val2017 subset, 512 eval / 384 exported images | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | MSE, PSNR, SSIM, MS-SSIM, inference time | 完成（供 test-like split 复核使用） | `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/` |
| EXP-S2-001 | 2026-07-01 | N/A (not a project git repo) | M1-BlindDiffusion preflight/run attempt | COCO2017 val2017 export subset, 16 images/SNR planned | AWGN | [1, 7, 19] dB | 0.17 | 未生成 | 阻塞（模型权重缺失；提权下载/GPU 运行被拒绝） | 未创建 |
| EXP-S2-002 | 2026-07-01 | N/A (not a project git repo) | M1-BlindDiffusion | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, diffusion time | 完成（负结果） | `outputs/EXP-S2-002/` |
| EXP-S3-001 | 2026-07-02 | N/A (not a project git repo) | CLIP image-image consistency diagnostic | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | CLIP cosine similarity, CLIP drop rate | 完成（辅助语义诊断；负结果） | `outputs/EXP-S3-001/` |
| EXP-S3-002 | 2026-07-02 | N/A (not a project git repo) | Frozen classifier pseudo-label consistency diagnostic | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | Pseudo-label prediction consistency, pseudo drift-origin, refinement drift | 完成（辅助分类器诊断；负结果） | `outputs/EXP-S3-002/` |
| EXP-S3-003 | 2026-07-02 | N/A (not a project git repo) | COCO caption CLIP text consistency diagnostic | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | COCO caption CLIP image-text similarity, caption drop rate | 完成（辅助 caption 语义诊断；负结果） | `outputs/EXP-S3-003/` |
| EXP-S4-001 | 2026-07-03 | N/A (not a project git repo) | M3-PseudoClassifierFallbackPilot | COCO2017 val2017 export subset, 16 images/SNR | AWGN | [1, 7, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo Final-Failure, accept/reject, false accept/reject | 完成（S5 fallback pilot；非完整 M3） | `outputs/EXP-S4-001/` |
| EXP-S4-002 | 2026-07-03 | N/A (local directory is not yet a git repo) | SNRAdaptiveDiffusionStrengthValidation | COCO2017 val2017 export subset, 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject | 完成（S5 strength validation；负/部分结果） | `outputs/EXP-S4-002/` |
| EXP-S4-003 | 2026-07-03 | N/A (local directory is not yet a git repo) | SD VAE roundtrip diagnostic | COCO2017 val2017 export subset, 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure | 完成（S5 VAE 诊断；负/瓶颈确认） | `outputs/EXP-S4-003/` |
| EXP-S4-004 | 2026-07-03 | 401d4bd + uncommitted local changes at run time | SNR-conditioned pixel residual refiner pilot attempt | COCO2017 val2017 export subset, train 24 images/SNR, eval planned 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | training loss only; final metrics not written | 失败（CSV 写入字段 bug；保留输出，不复用） | `outputs/EXP-S4-004/` |
| EXP-S4-005 | 2026-07-03 | 401d4bd + uncommitted local changes at run time | SNR-conditioned pixel residual refiner pilot | COCO2017 val2017 export subset, train 24 images/SNR, eval 8 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject | 完成（S5 latent-free restoration pilot；正向小样本结果） | `outputs/EXP-S4-005/` |
| EXP-S4-006 | 2026-07-03 | 709f1c6 | SNR-conditioned pixel residual refiner validation | COCO2017 val2017 export subset, train 160 images/SNR, eval 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject | 完成（S5 residual validation；正向但需 detector error analysis） | `outputs/EXP-S4-006/` |
| EXP-S4-007 | 2026-07-06 | 4f4eefb | SNR-conditioned pixel residual diffusion pilot | COCO2017 val2017 export subset, train 80 images/SNR, eval 16 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo drift/failure, accept/reject, sampling time | 完成（S5 residual diffusion pilot；负结果） | `outputs/EXP-S4-007/` |
| EXP-S4-008 | 2026-07-10 | abf117b + local script/config | SGD-inspired edge-conditioned pixel residual refiner validation | COCO2017 val2017 export subset, train 160 images/SNR, eval 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, accept/new-error | 完成（S5 edge/structure-conditioned residual validation；LPIPS 省略以避免下载） | `outputs/EXP-S4-008/` |
| EXP-S4-009 | 2026-07-10 | abf117b + local script/config; exact SHA in ANALYSIS-S6-026 | Capacity-matched large no-edge residual refiner | COCO2017 val2017 export subset, train 160 images/SNR, eval 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo failure, matched ablation | 完成（`EXP-S4-008` 的 `64×6/60 epoch` no-edge control） | `outputs/EXP-S4-009/` |
| EXP-S4-010 | 2026-07-10 | abf117b + local script/config; exact SHA in ANALYSIS-S6-026 | Capacity-matched small edge residual refiner | COCO2017 val2017 export subset, train 160 images/SNR, eval 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo failure, matched ablation | 完成（`EXP-S4-006` 的 `48×5/40 epoch` edge arm） | `outputs/EXP-S4-010/` |
| ANALYSIS-S6-002 | 2026-07-07 | 20f9cc3 + local script | ResidualShrinkSelection | COCO2017 val2017 `EXP-S4-006` eval outputs, 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, accept/new-error | 完成（派生分析；validation-only；不训练不下载） | `outputs/analysis/exp_s4_006_residual_shrink_selection/` |
| ANALYSIS-S6-003 | 2026-07-07 | 7ef1753 + local script | FrozenResidualShrinkScheduleCheck | COCO2017 val2017 test-like `sample_000256`-`sample_000319`, 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, accept/new-error | 完成（frozen schedule 复核；不调参不训练不下载） | `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/` |
| ANALYSIS-S6-004 | 2026-07-07 | 371833e + local script | MinimalClosureReportWithHeldoutShrinkM3 | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 held-out/test-like shrink M3；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-005 | 2026-07-07 | 371833e + local script | FrozenHeldoutResidualShrinkScheduleCheck | COCO2017 val2017 held-out `sample_000000`-`sample_000031`, 32 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, accept/new-error | 完成（frozen schedule held-out 复核；不调参不训练不下载） | `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/` |
| ANALYSIS-S6-006 | 2026-07-07 | c19cc0f + local script | ResidualShrinkM3ArtifactGallery | COCO2017 val2017 validation/held-out/test-like residual shrink outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | policy summary, case counts, safe accept/protective reject/new-error galleries | 完成（派生 artifact；不训练不下载不调参） | `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/` |
| ANALYSIS-S6-007 | 2026-07-07 | fbcfe72 + local script | AdaptiveResidualAlphaPolicy | COCO2017 val2017 validation/held-out/test-like residual alpha candidates | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, pseudo final failure, selected alpha, accept/new-error | 完成（派生 policy；不训练不下载不调参） | `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/` |
| ANALYSIS-S6-008 | 2026-07-07 | bcfc1f1 + local script/config | MinimalClosureReportWithAdaptiveAlphaM3 | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink/adaptive-alpha tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 adaptive alpha M3；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-009 | 2026-07-07 | 9cacff5 + local script/config | TwoStageResidualAlphaPolicy | COCO2017 val2017 validation/held-out/test-like adaptive alpha decisions | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, two-stage accept/fallback, accepted new error | 完成（派生 policy；不重分类；不训练不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/` |
| ANALYSIS-S6-010 | 2026-07-07 | 9cacff5 + local script/config | MinimalClosureReportWithTwoStageAlphaAblation | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink/adaptive/two-stage alpha tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 two-stage alpha 消融；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-011 | 2026-07-09 | 4a466e8 + local script/config | ReceiverAlphaPredictor | COCO2017 val2017 validation/held-out/test-like adaptive alpha decisions and candidate PNGs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, target alpha accuracy, accept/new-error | 完成（validation-only tabular predictor；不训练图像模型不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_receiver_alpha_predictor/` |
| ANALYSIS-S6-012 | 2026-07-09 | 4a466e8 + local script/config | MinimalClosureReportWithReceiverAlphaPredictor | COCO2017 val2017 existing outputs and analysis CSVs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | method summary, residual shrink/adaptive/two-stage/predictor alpha tradeoff, pseudo semantic failure, accepted new error | 完成（派生汇总；纳入 receiver predictor；不训练不下载） | `outputs/analysis/minimal_closure_report/` |
| ANALYSIS-S6-013 | 2026-07-09 | a7076eb + local script/config | AlphaHeadResidualRefinerPilot | COCO2017 val2017 validation/held-out/test-like adaptive-alpha pseudo targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, target alpha accuracy, accept/new-error | 完成（冻结 residual CNN，仅训练 alpha head；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/` |
| ANALYSIS-S6-014 | 2026-07-09 | 594db31 + local script/config | WeightedAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like adaptive-alpha pseudo targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, target alpha accuracy, accept/new-error | 完成（冻结 residual CNN，仅训练 class-weighted alpha head；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/` |
| ANALYSIS-S6-015 | 2026-07-09 | 050b0c2 + local script/config | BenefitAwareAlphaPredictor | COCO2017 val2017 validation/held-out/test-like alpha candidates | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（validation-derived safe-PSNR utility soft labels；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_benefit_alpha_predictor/` |
| ANALYSIS-S6-016 | 2026-07-09 | 53b71b3 + local script/config | BenefitAwareAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（冻结 residual CNN，仅训练 benefit-aware alpha head；不运行 diffusion 不下载；LPIPS 省略） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/` |
| ANALYSIS-S6-017 | 2026-07-09 | 901420f + local script/config | BenefitAwareJointAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（解冻 residual CNN joint fine-tune；不运行 diffusion 不下载；LPIPS 省略；负/诊断结果） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/` |
| ANALYSIS-S6-018 | 2026-07-09 | c69743a + local script/config | BenefitAwareTailAlphaHeadResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, utility target accuracy, accept/new-error | 完成（只微调 residual tail + alpha head；不运行 diffusion 不下载；LPIPS 省略；训练侧正向阶段结果） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/` |
| ANALYSIS-S6-019 | 2026-07-09 | 9b6f74a + local script/config | BenefitAwareTailContinuousAlphaResidualRefiner | COCO2017 val2017 validation/held-out/test-like benefit utility alpha targets | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, continuous alpha, accept/new-error | 完成（只微调 residual tail + continuous alpha head；不运行 diffusion 不下载；LPIPS 省略；训练侧正向突破） | `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/` |
| ANALYSIS-S6-020 | 2026-07-09 | 3c8a0bd + local script/config | ContinuousAlphaTailRefinerPerceptualEnsembleAudit | COCO2017 val2017 validation/held-out/test-like continuous-alpha outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, AlexNet/ResNet18/MobileNetV3-Small pseudo final failure, ensemble new-error votes | 完成（派生审计；不训练不运行 diffusion；本地 LPIPS 与分类器权重；强候选但非跨模型完全安全） | `outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/` |
| ANALYSIS-S6-021 | 2026-07-10 | abf117b + local script/config | EdgeResidualShrinkSelection | COCO2017 val2017 `EXP-S4-008` eval outputs, 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, accept/new-error | 完成（validation-only 派生分析；不训练不下载；LPIPS 省略） | `outputs/analysis/exp_s4_008_edge_residual_shrink_selection/` |
| ANALYSIS-S6-022 | 2026-07-10 | abf117b + local script/config | EdgeResidualRefinerHeldoutGateCheck | COCO2017 val2017 held-out `sample_000000`-`sample_000031`, 32 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, confidence-gain gate accepted new error | 完成（held-out 复核；宽松 confidence-gain gate 漏 new error） | `outputs/analysis/exp_s4_008_edge_heldout_gate_check/` |
| ANALYSIS-S6-023 | 2026-07-10 | abf117b + local script/config | EdgeResidualRefinerTestlikeGateCheck | COCO2017 val2017 test-like `sample_000256`-`sample_000319`, 64 images/SNR | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, confidence-gain gate accepted new error | 完成（test-like 复核；宽松 confidence-gain gate 漏 new error） | `outputs/analysis/exp_s4_008_edge_testlike_gate_check/` |
| ANALYSIS-S6-024 | 2026-07-10 | abf117b + local script/config | FrozenEdgeResidualShrinkScheduleHeldoutCheck | COCO2017 val2017 held-out edge residual outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, accepted new error | 完成（validation schedule frozen；held-out new error 0；LPIPS 省略） | `outputs/analysis/exp_s4_008_edge_heldout_residual_shrink_schedule_check/` |
| ANALYSIS-S6-025 | 2026-07-10 | abf117b + local script/config | FrozenEdgeResidualShrinkScheduleTestlikeCheck | COCO2017 val2017 test-like edge residual outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, pseudo final failure, accepted new error | 完成（validation schedule frozen；test-like new error 0；LPIPS 省略） | `outputs/analysis/exp_s4_008_edge_testlike_residual_shrink_schedule_check/` |
| ANALYSIS-S6-026 | 2026-07-10 | abf117b + local script/config; SHA256 manifest | EdgeCapacityFactorialAblation | `EXP-S4-006/008/009/010` aligned validation PNGs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | paired raw/M3 PSNR, 10k cluster bootstrap, pseudo semantic counts, provenance hashes | 完成（2×2 matched ablation；edge quality effect CI 排除 0） | `outputs/analysis/exp_s4_006_008_009_010_edge_capacity_ablation/` |
| ANALYSIS-S6-027 | 2026-07-10 | abf117b + local script/config; SHA256 manifest | MatchedLargeEdgeCrossSplitPairedAudit | validation/held-out/test-like/fresh-holdout | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | paired raw PSNR, per-SNR direction, 10k cluster bootstrap, pseudo semantic deltas | 完成（四段 CI 均排除 0；fresh-holdout 未用于调参） | `outputs/analysis/exp_s4_008_009_matched_edge_holdout_audit/` |
| ANALYSIS-S6-028 | 2026-07-10 | abf117b + local script/config | MonotonicEdgeResidualShrinkSchedule | validation/held-out/test-like/fresh-holdout | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | PSNR, SSIM, MS-SSIM, LPIPS, monotonic effective strength, pseudo failure | 完成（四段 PSNR > +0.56 dB、LPIPS 均改善） | `outputs/analysis/exp_s4_008_edge_monotonic_*` |
| ANALYSIS-S6-029 | 2026-07-10 | abf117b + local script/config; input/script SHA256 | EdgeMonotonicPolicyClassifierEnsembleAudit | validation/held-out/test-like/fresh-holdout fixed policy outputs | AWGN | [1, 4, 7, 13, 19] dB | 0.17 | AlexNet/ResNet18/MobileNetV3-Small failure, new-error/repair votes | 完成（majority new error `1/1/0/3`；强质量候选但非跨模型完全安全） | `outputs/analysis/exp_s4_008_edge_monotonic_policy_ensemble_audit/` |
| EXP-S10-001 | 2026-07-12 | local script/config | Matched-rate short-chain residual-shift diffusion pilot | S7 frozen split, train 160 / eval 64 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=6+c=2`) | PSNR, LPIPS, pseudo repair/new-error, sampling time | 完成（LPIPS 微改善但 semantic risk gate 失败；当前版本不晋级） | `outputs/EXP-S10-001/` |
| EXP-S11-001 | 2026-07-12 | local config + existing refiner implementation | P0 B1 `c8 + same-capacity receiver refiner` | COCO fixed split, train 160 / eval 64 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | PSNR, LPIPS, pseudo repair/new-error, latency | 完成（强公平 reference backend） | `outputs/EXP-S11-001/` |
| ANALYSIS-S11-001 | 2026-07-12 | local script/config | P0 B1 vs B3 paired fairness audit | COCO eval 64 images × 5 SNR | AWGN | [1, 4, 7, 13, 19] dB | matched 1/6 | paired PSNR bootstrap, LPIPS, params, latency, pseudo events | 完成（B3 structure-increment gate 失败） | `outputs/analysis/s11_p0_b1_b3_paired_comparison/` |
| EXP-S12-001 | 2026-07-12 | local script/config | B1-anchored semantic-preserving short-chain diffusion | COCO fixed split, train 160 / eval 64 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | PSNR, LPIPS, incremental pseudo new-error/repair, latency | 完成（质量/感知改善，semantic-risk gate 失败） | `outputs/EXP-S12-001/` |
| EXPORT-S13-001 | 2026-07-12 | local script/config | Deterministic train2017 c8 scale-up cache | COCO train2017, 10k train + 1k validation | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | PSNR, manifest/hash integrity, latency | 完成（55k reconstruction cache） | `outputs/eval/s13_coco_train2017_c8_scaleup_10k_1k/` |
| EXP-S13-001 | 2026-07-12--13 | local config + existing refiner implementation | Scale-up B1 receiver-structure residual anchor | COCO train2017 internal split, train 10k / eval 1k per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | PSNR, LPIPS, pseudo new-error/repair, latency | 完成（全部预注册 anchor gate 通过） | `outputs/EXP-S13-001/` |
| EXP-S14-001 | 2026-07-13 | local script/config | Train2017-scale B1-anchored 6-step diffusion | COCO train2017 internal split, train 10k / eval 1k per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | PSNR, LPIPS, incremental new-error/repair, latency | 完成（risk gate 通过，quality/perception gate 失败） | `outputs/EXP-S14-001/` |
| ANALYSIS-PC-001 | 2026-07-13 | local script/config | Frozen received-latent posterior correction pilot | COCO train2017 unused SHA-rank block, 64 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | received-latent consistency, PSNR, LPIPS, incremental pseudo new-error/repair | 完成（全部预注册 feasibility/promotion gates 通过） | `outputs/analysis/s15_received_latent_posterior_pilot/` (legacy artifact path) |
| ANALYSIS-PC-002 | 2026-07-13 | local script/config | Frozen posterior correction independent replication | COCO train2017 unused SHA-rank block, 256 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | per-image latent consistency, PSNR, LPIPS, 3-classifier new-error/repair | 完成（质量/一致性复现；cross-model semantic gates 失败） | `outputs/analysis/s16_posterior_consistency_independent_replication/` (legacy artifact path) |
| ANALYSIS-PC-003 | 2026-07-13 | local script/config | Posterior correction plus frozen AlexNet agreement fallback | COCO train2017 new unused SHA-rank block, 256 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | coverage, PSNR, LPIPS, 3-classifier new-error/repair | 完成（风险显著降低但 cross-model semantic gates 仍失败） | `outputs/analysis/s17_posterior_consistency_failure_handling/` (legacy artifact path) |
| ANALYSIS-PC-CTRL-001 | 2026-07-13 | local script/config | Two-model posterior consensus controller with held-out classifier audit | COCO train2017 new unused SHA-rank block, 256 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | coverage, PSNR, LPIPS, controller-ensemble and holdout new-error/repair | 完成（controller ensemble 表面安全；holdout gate 失败） | `outputs/analysis/pc_controller_holdout_audit/` |
| ANALYSIS-PC-GT-001 | 2026-07-13 | local script/config | COCO dominant-object OpenCLIP clean-correct audit | COCO train2017 new unused SHA-rank block, 512 images per SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | COCO-label clean subset, object failure/new-error/repair, PSNR, LPIPS | 完成（净 failure 改善；object new-error gate 失败） | `outputs/analysis/pc_coco_object_clip_audit/` |
| ANALYSIS-PC-SUP-001 | 2026-07-13 | local script/config | Scratch-classifier supervised posterior policy-dev audit | Imagenette policy-dev, 1894 images per SNR; official val sealed | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | WNID supervised failure/new-error/repair, PSNR, LPIPS, consistency | 完成（aggregate supervised 改善；7 dB per-SNR new-error gate 失败） | `outputs/analysis/pc_imagenette_supervised_policy_dev/` |
| ANALYSIS-PC-RISK-001 | 2026-07-13 | local script/config | Frozen scratch `G_gate` posterior fallback follow-up | Imagenette policy-dev, 1894 images per SNR; official val sealed | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | evaluator-separated WNID failure/new-error, coverage, PSNR, LPIPS | 完成（failure/new-error 优于旧 controller；7 dB strict tail gate 仍失败） | `outputs/analysis/pc_imagenette_scratch_gate_policy_dev/` |
| ANALYSIS-PC-RISK-REP-001 | 2026-07-13 | local script/config | Frozen scratch-gated posterior multi-channel-seed replication | Imagenette policy-dev, 1894 images × 3 new seeds × 5 SNR; official val sealed | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | seed/SNR consistency, WNID failure/new-error, image-cluster tail UCB, PSNR, LPIPS | 完成（failure/quality 稳定改善；new-error 与 cluster-UCB gates 失败） | `outputs/analysis/pc_imagenette_scratch_gate_policy_dev_multiseed/` |
| TRAIN-PC-AUX-001 | 2026-07-14 | local script/config | Scratch EfficientNet-B0 auxiliary receiver-risk classifier | Imagenette `cls_train/cls_cal`; policy-dev and official val excluded | N/A | N/A | N/A | cal macro top-1, temperature scaling, split/checkpoint integrity | 完成（cal macro top-1 `0.90270`；feature-extractor gate passed） | `outputs/analysis/imagenette_scratch_risk_classifier/G_aux/` |
| ANALYSIS-PC-RISK-FEAT-001 | 2026-07-14 | local script/config | Receiver-visible continuous risk-feature extraction | Imagenette policy-dev, 1894 images × 3 exposed seeds × 5 SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | 43 receiver features, separated teacher targets, row/hash regression | 完成（28,410 rows；integrity pass；development only） | `outputs/analysis/pc_imagenette_receiver_risk_features_multiseed/` |
| ANALYSIS-PC-RISK-CTRL-DEV-001 | 2026-07-14 | local script/config | Six-feature empirical-percentile receiver-risk controller development | Existing 3-seed policy-dev feature table | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | rejection, failure/new-error, cluster UCB, retained PSNR/LPIPS | 完成（development gates pass；threshold frozen） | `outputs/analysis/pc_imagenette_receiver_risk_controller_dev/` |
| ANALYSIS-PC-RISK-SEED-AUDIT-001 | 2026-07-14 | local script/config | Frozen receiver-risk controller new-channel-seed audit | Imagenette policy-dev, 1894 images × seed 20260725 × 5 SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | frozen-score rejection, supervised failure/new-error, cluster UCB, PSNR/LPIPS | 完成（NEGATIVE；2 new errors missed，11 repairs rejected） | `outputs/analysis/pc_imagenette_receiver_risk_seed_20260725_audit/` |
| ANALYSIS-PC-RISK-FAIL-001 | 2026-07-14 | local script/config | Exact replay of receiver-risk missed errors and rejected repairs | seed 20260725 的 2 个 missed new-error + 11 个 rejected repair | AWGN | [1, 4, 7] dB | 1/6 (`c=8`) | exact replay, probability diagnostics, failure panels, pairwise AUC | 完成（确认 top-1 checksum 无效；full-probability source risk 有开发信号） | `outputs/analysis/pc_receiver_risk_failure_cases_seed20260725/` |
| ANALYSIS-PC-SENDER-DEV-001 | 2026-07-14 | local script/config | Noiseless 80-bit sender full-probability zero veto feasibility | Imagenette policy-dev, 1894 images × seed 20260725 × 5 SNR | AWGN + noiseless description | [1, 4, 7, 13, 19] dB | 图像 c=8 + 额外 80 raw bits（仅可达性） | supervised failure/new-error, PSNR/LPIPS, accept rate | 完成（开发可达性 POSITIVE；不具码率公平性） | `outputs/analysis/pc_imagenette_sender_aux_fullprob_zero_veto_dev/` |
| ANALYSIS-PC-SENDER-RATE-DEV-001 | 2026-07-14 | local script/config | In-budget analog 10D probability R16 zero veto | Imagenette policy-dev, 1894 images × seed 20260725 × 5 SNR | shared AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`; 160/65536 payload symbols) | payload recovery, masked consistency, failure/new-error, PSNR/LPIPS | 完成（NEGATIVE；1 dB per-SNR new-error gate 失败；决策噪声敏感） | `outputs/analysis/pc_imagenette_sender_aux_inbudget_awgn_zero_veto_dev/` |
| ANALYSIS-PC-SENDER-DIGITAL-DEV-001 | 2026-07-14 | local script/config | In-budget UInt4+BPSK×4 sender probability zero veto | Imagenette policy-dev, 1894 images × seed 20260725 × 5 SNR | shared AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`; 40 bits/160 symbols) | BER/vector exact rate, supervised tail risk, PSNR/LPIPS | 完成（development POSITIVE；全部门槛通过） | `outputs/analysis/pc_imagenette_sender_aux_uint4_bpsk_inbudget_awgn_zero_veto_dev/` |
| ANALYSIS-PC-SENDER-DIGITAL-SEED-AUDIT-REF-001 | 2026-07-14 | local script/config | New-seed unpunctured c8 reference generation | Imagenette policy-dev, 1894 images × seed 20260726 × 5 SNR | AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`) | reference raw/posterior/final, supervised outcomes, PSNR/LPIPS | 完成（只作 frozen audit reference） | `outputs/analysis/pc_imagenette_sender_aux_seed20260726_unpunctured_reference/` |
| ANALYSIS-PC-SENDER-DIGITAL-SEED-AUDIT-001 | 2026-07-14 | local script/config | Frozen UInt4+BPSK×4 sender controller new-seed audit | Imagenette policy-dev, 1894 images × seed 20260726 × 5 SNR | shared AWGN | [1, 4, 7, 13, 19] dB | 1/6 (`c=8`; 40 bits/160 symbols) | BER/vector exact rate, supervised failure/new-error, cluster UCB, PSNR/LPIPS | 完成（NEGATIVE；编码迁移但 final new-error `5>3`） | `outputs/analysis/pc_imagenette_sender_aux_uint4_bpsk_seed20260726_audit/` |

`项目版本` 优先填写 git commit。若当前项目目录不是 git 仓库，填写 `N/A (not a project git repo)`，并在单实验记录中写明 config、脚本和关键源码路径。

## 指标要求

### 图像质量

- PSNR
- MS-SSIM
- LPIPS
- FID，可选

### 语义可靠性

- Classification accuracy
- Prediction consistency
- Semantic drift rate
- Semantic failure rate
- Detector accept / reject rate，若使用 failure detector
- CLIP similarity，可选

### 系统开销

- Diffusion steps
- Inference time
- 参数量
- FLOPs

## 单实验模板

### EXP-Sx-000：标题

- 日期：
- 项目版本：
- 第三方 commit：
- 阶段：
- 方法：
- 数据集：
- 数据 split / 样本 ID：
- 信道：
- SNR：
- CBR：
- 随机种子：
- checkpoint：
- config：
- 运行命令：
- 关键源码：
- 输出路径：
- 状态：

#### 指标

- PSNR：
- MS-SSIM：
- LPIPS：
- FID：
- Classification accuracy：
- Prediction consistency：
- Semantic drift rate：
- Semantic failure rate：
- Detector accept rate：
- Detector reject rate：
- CLIP similarity：
- Diffusion steps：
- Inference time：
- 参数量：
- FLOPs：

#### 结果总结

-

#### Semantic drift 观察

-

#### 失败案例

-

#### 复现备注

-

#### 下一步

-

## 正式实验要求

正式实验必须满足：

- 使用唯一 `EXP-*` ID 和唯一输出目录。
- 保存 config 副本、metrics 文件和样例图。
- 记录项目版本；如果没有项目 git commit，必须记录脚本、配置和关键源码路径。
- 记录第三方 baseline commit。
- 记录数据 split、随机种子、checkpoint、SNR、CBR 和信道模型。
- smoke test 不写入正式实验索引，但可以写入 `PROGRESS.md`。

### EXP-S1-001：DeepJSCC CIFAR-10 AWGN baseline

- 日期：2026-06-29
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S1 DeepJSCC baseline
- 方法：M0-DeepJSCC
- 数据集：CIFAR-10 test subset, 1024 images
- 数据 split / 样本 ID：`outputs/EXP-S1-001/subset_indices.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；subset seed 42
- checkpoint：`third_party/Deep-JSCC-PyTorch/out/checkpoints/CIFAR10_8_13.0_0.17_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl`
- config：`outputs/EXP-S1-001/config.yaml`
- 运行命令：`python3 scripts/s1_deepjscc_mini_eval.py --device cpu --num-samples 1024 --batch-size 64 --output-dir outputs/EXP-S1-001 --formal`
- 关键源码：`scripts/s1_deepjscc_mini_eval.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S1-001/`
- 状态：完成

#### 指标

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM |
|---:|---:|---:|---:|---|
| 1 | 0.004698 | 23.5428 | 0.8216 | N/A |
| 4 | 0.002464 | 26.3794 | 0.8927 | N/A |
| 7 | 0.001371 | 28.9857 | 0.9350 | N/A |
| 13 | 0.000584 | 32.8612 | 0.9696 | N/A |
| 19 | 0.000390 | 34.7994 | 0.9785 | N/A |

- PSNR：见上表
- MS-SSIM：未计算；CIFAR-10 为 32x32，`pytorch-msssim` 默认 4 次下采样要求图像边长大于 160
- LPIPS：未计算，后续接入 perceptual metric 时补
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用
- Inference time：未单独统计
- 参数量：未单独统计
- FLOPs：未单独统计

#### 结果总结

M0-DeepJSCC baseline 在固定 CIFAR-10 test subset 上跑通。PSNR 和 SSIM 随 SNR 升高单调提升，可作为后续 M1/M2/M3 的 pre-diffusion 对照。

#### Semantic drift 观察

本实验不包含 diffusion refinement，尚未统计 semantic drift。下一步需要冻结 `T_cls` 并实现 classifier consistency 指标。

#### 失败案例

本实验仅保存每个 SNR 的样例对比图，尚未整理 semantic failure case。

#### 复现备注

当前项目目录不是 git 仓库，因此项目版本记为 `N/A (not a project git repo)`。正式复现依赖第三方 commit、配置副本、脚本路径和固定 subset indices。

#### 下一步

实现 semantic drift metric 的最小版本，或开始接入 M1-BlindDiffusion 的后处理接口。

### EXP-S2HR-001：DeepJSCC COCO-val 256x256 AWGN pilot

- 日期：2026-06-30
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC pilot
- 方法：M0-DeepJSCC-HR-pilot
- 数据集：COCO2017 `val2017` 固定切分，4500 train / 500 val
- 数据 split / 样本 ID：`data/coco_val_split/split_manifest.json`
- 信道：AWGN
- SNR：7 dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/checkpoints/best.pt`
- config：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/config.yaml`
- 运行命令：`python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0`
- 关键源码：`scripts/train_deepjscc_highres.py`, `scripts/prepare_image_symlink_split.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/`
- 状态：完成；非正式 pilot，不替代 COCO2017 train/val 主实验

#### 指标

- PSNR：26.6647 dB
- MS-SSIM：未计算；当前训练脚本记录 SSIM
- SSIM：0.7837
- MSE：0.002548
- LPIPS：未计算
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用
- Inference time：未单独统计
- 参数量：未单独统计
- FLOPs：未单独统计

#### 结果总结

使用已下载的 COCO2017 `val2017` 生成不重叠 4500/500 pilot split，并训练 50 epoch 得到可用的 256x256 DeepJSCC checkpoint。该 checkpoint 适合后续 high-res inference、diffusion refinement 接口和样例流程调试。

#### Semantic drift 观察

本实验不包含 diffusion refinement，尚未统计 semantic drift。样例图显示重建能保留主要物体和场景结构，但细节明显模糊，适合作为后续 diffusion semantic drift 控制的调试输入。

#### 失败案例

尚未整理。

#### 复现备注

这是非正式 pilot 实验，训练和验证都来自 COCO2017 `val2017` 的固定不重叠切分。正式论文主实验仍必须等待 COCO2017 `train2017` 下载完成后重新训练。

#### 下一步

用该 checkpoint 调试 high-res inference/export 和 M1-BlindDiffusion 接口；COCO2017 `train2017` 完成后重新训练正式 `M0-HR`。

### EXP-S2HR-002：DeepJSCC COCO-val 256x256 pilot SNR sweep export

- 日期：2026-06-30
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC pilot export
- 方法：M0-DeepJSCC-HR-pilot export
- 数据集：COCO2017 `val2017` pilot validation split, 500 images
- 数据 split / 样本 ID：`data/coco_val_split/split_manifest.json`; evaluated paths copied to `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco_val256_awgn_snr7_cbr017_pilot/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/config.yaml`
- 运行命令：`python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco_val256_awgn_pilot.yaml --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 32 --output-dir outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/`
- 状态：完成；非正式 pilot，不替代 COCO2017 train/val 主实验

#### 指标

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.003426 | 25.1263 | 0.7096 | 0.8991 | 0.7205 |
| 4 | 0.002837 | 26.0905 | 0.7563 | 0.9280 | 0.1874 |
| 7 | 0.002547 | 26.6680 | 0.7836 | 0.9441 | 0.1840 |
| 13 | 0.002333 | 27.1678 | 0.8064 | 0.9572 | 0.1849 |
| 19 | 0.002279 | 27.3030 | 0.8125 | 0.9607 | 0.1830 |

- LPIPS：未计算
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用

#### 结果总结

pilot checkpoint 在高分辨率 COCO-val validation split 上完成 SNR sweep。PSNR、SSIM 和 MS-SSIM 随 SNR 升高整体提升。脚本同时导出 32 张原图和各 SNR 的 DeepJSCC 重建图，用于后续 `M1-BlindDiffusion` 输入。

#### Semantic drift 观察

本实验只导出 pre-diffusion `x_hat`，尚未统计 semantic drift。低 SNR 样例显示纹理和边缘更模糊，但主要物体/场景仍可辨认，适合作为 diffusion hallucination 风险测试输入。

#### 失败案例

尚未整理。

#### 复现备注

这是非正式 pilot export。第一组 SNR 的 inference time 包含 CUDA warmup，计时仅供粗略参考。正式论文主实验仍需等待 COCO2017 `train2017` 完成后重新训练和评估。

#### 下一步

读取 `outputs/eval/s2_deepjscc_coco_val256_awgn_pilot_m0_export/exports/snr_XXdb/reconstruction/` 接入 `M1-BlindDiffusion`，并开始记录 refinement 后的视觉指标和初步 semantic drift。

### EXP-S2HR-003：DeepJSCC COCO2017 256x256 AWGN formal train

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal train
- 方法：M0-DeepJSCC-HR formal train
- 数据集：COCO2017 `train2017` / `val2017`
- 数据 split / 样本 ID：`configs/s2_deepjscc_coco256_awgn.yaml`，验证集使用 config 中固定 val subset
- 信道：AWGN
- SNR：7 dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/config.yaml`
- 运行命令：`python3 scripts/train_deepjscc_highres.py --config configs/s2_deepjscc_coco256_awgn.yaml --device cuda:0`
- 关键源码：`scripts/train_deepjscc_highres.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/`
- 状态：完成；`best.pt` 可用，`latest.pt` 不可用

#### 指标

- best epoch：73
- best val MSE：0.0008254946042143274
- best val PSNR：31.56180403754115 dB
- best val SSIM：0.9054122059606016
- latest epoch：99
- latest metrics：NaN
- latest 参数：NaN，不可用于后续实验
- LPIPS：未计算
- FID：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算

#### 结果总结

COCO2017 `train2017` 和 `val2017` 已完整就位，正式 COCO-256 DeepJSCC 训练产出了可用 `best.pt`。训练在 epoch 0-88 指标有限，epoch 89-99 出现 NaN，因此本实验的正式 baseline 必须使用 epoch 73 的 `best.pt`，不能使用 `latest.pt` 或最终 `metrics.json` 中的 NaN final。

#### Semantic drift 观察

本实验只训练 pre-diffusion DeepJSCC，不包含 refinement，因此尚未统计 semantic drift。

#### 失败案例

epoch 89 后训练发散为 NaN。已在训练脚本中增加非有限 loss/metrics 防护，后续重训会提前停止并用 best checkpoint 评估 final metrics。

#### 复现备注

当前项目目录不是 git 仓库，因此项目版本记为 `N/A (not a project git repo)`。训练日志见 `outputs/logs/s2_coco256_awgn_train.direct.screen.log`。后续论文主实验和 diffusion 输入应固定使用 `outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`。

#### 下一步

基于 `best.pt` 跑正式 SNR sweep/export，并将导出的 `x_hat` 输入 `M1-BlindDiffusion`。

### EXP-S2HR-004：DeepJSCC COCO2017 256x256 formal SNR sweep export

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal export
- 方法：M0-DeepJSCC-HR formal export
- 数据集：COCO2017 `val2017` subset, 512 images
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/config.yaml`
- 运行命令：`python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 32 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/`
- 状态：完成

#### 指标

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0018173862 | 28.0189655945 | 0.8090499612 | 0.9363910668 | 0.6834 |
| 4 | 0.0011532078 | 30.0470464826 | 0.8700513527 | 0.9622469177 | 0.1861 |
| 7 | 0.0008255254 | 31.5589745864 | 0.9054089159 | 0.9763763894 | 0.1969 |
| 13 | 0.0005807463 | 33.1954004802 | 0.9348068793 | 0.9876335945 | 0.1824 |
| 19 | 0.0005199769 | 33.7264324129 | 0.9425818466 | 0.9905498993 | 0.1869 |

- LPIPS：未计算
- FID：未计算
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- Detector accept rate：不适用
- Detector reject rate：不适用
- CLIP similarity：未计算
- Diffusion steps：不适用

#### 结果总结

正式 `best.pt` 在 COCO2017 val subset 上完成 SNR sweep。PSNR、SSIM 和 MS-SSIM 随 SNR 增加稳定提升，7 dB 结果与 best checkpoint 训练记录一致。导出目录包含 32 张 `exports/original/` 原图，以及每个 SNR 下 32 张 `exports/snr_XXdb/reconstruction/` 重建图，可直接作为 `M1-BlindDiffusion` 的输入。

#### Semantic drift 观察

本实验只导出 pre-diffusion `x_hat`，尚未统计 semantic drift。下一阶段需要比较 DeepJSCC 原始重建、blind diffusion refinement 和 semantic-controlled refinement 的分类一致性或 CLIP consistency。

#### 失败案例

尚未整理。当前低 SNR 样例应优先用于观察 diffusion 是否把主体语义修偏。

#### 复现备注

第一组 SNR 的 inference time 包含 CUDA warmup，计时仅供粗略参考。该实验是后续正式 high-resolution diffusion 实验的 M0 输入来源，优先级高于 COCO-val pilot export。

#### 下一步

读取 `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/exports/snr_XXdb/reconstruction/`，实现 `M1-BlindDiffusion` 的最小可复现后处理与 LPIPS/semantic drift 指标。

### EXP-S2HR-005：DeepJSCC COCO2017 256x256 formal SNR sweep export 256 saved images

- 日期：2026-07-03
- 项目版本：`8678e4f`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal export
- 方法：M0-DeepJSCC-HR formal export, 256 saved images per SNR
- 数据集：COCO2017 `val2017` subset, 512 images evaluated, first 256 images exported
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 256 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
- 状态：完成；support export for `EXP-S4-006`

#### 指标

该实验仍在同一 512 张 COCO val subset 上评估 M0，因此 MSE/PSNR/SSIM/MS-SSIM 与 `EXP-S2HR-004` 的主指标一致；区别是保存的 PNG 从 32 张/SNR 扩大到 256 张/SNR。

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0018173862 | 28.0189655945 | 0.8090499612 | 0.9363910668 | 0.6802 |
| 4 | 0.0011532078 | 30.0470464826 | 0.8700513527 | 0.9622469177 | 0.1960 |
| 7 | 0.0008255254 | 31.5589745864 | 0.9054089159 | 0.9763763894 | 0.1961 |
| 13 | 0.0005807463 | 33.1954004802 | 0.9348068793 | 0.9876335945 | 0.1942 |
| 19 | 0.0005199769 | 33.7264324129 | 0.9425818466 | 0.9905498993 | 0.1953 |

#### 结果总结

本实验用于给 residual restoration validation 提供更大的固定 M0 PNG 输入，不代表新的 M0 模型。输出包含 `exports/original/sample_000000.png` 到 `sample_000255.png`，以及每个 SNR 对应的 `exports/snr_XXdb/reconstruction/`。后续 `EXP-S4-006` 使用其中 `sample_000032` 到 `sample_000191` 训练，`sample_000192` 到 `sample_000255` 验证。

#### Semantic drift 观察

本实验只导出 pre-refinement `x_hat`，未额外统计 semantic drift。语义可靠性在后续 `EXP-S4-006` 中统计。

#### 复现备注

本实验不下载数据或模型，只读取本地 COCO、已有 `best.pt` checkpoint 和第三方 DeepJSCC 代码。运行命令显式清空代理变量；输出目录是新目录，不覆盖旧 32 张 formal export。

### EXP-S2HR-006：DeepJSCC COCO2017 256x256 formal SNR sweep export 384 saved images

- 日期：2026-07-06
- 项目版本：`3bcf82525ca6760a66d3b9dfa4d846ec275451e7`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S2-HR High-resolution DeepJSCC formal export
- 方法：M0-DeepJSCC-HR formal export, 384 saved images per SNR
- 数据集：COCO2017 `val2017` subset, 512 images evaluated, first 384 images exported
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/source_manifest.json`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 384 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384`
- 关键源码：`scripts/s2_deepjscc_highres_export.py`, `src/cadsd_jscc/deepjscc_adapter.py`, `src/cadsd_jscc/datasets.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/`
- 状态：完成；support export for `EXP-S4-006` test-like split check

#### 指标

该实验仍在同一 512 张 COCO val subset 上评估 M0，因此 MSE/PSNR/SSIM/MS-SSIM 与 `EXP-S2HR-004` 和 `EXP-S2HR-005` 的主指标一致；区别是保存的 PNG 从 256 张/SNR 扩大到 384 张/SNR。

| SNR(dB) | MSE | PSNR(dB) | SSIM | MS-SSIM | Inference ms/image |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.0018173862 | 28.0189655945 | 0.8090499612 | 0.9363910668 | 0.6610 |
| 4 | 0.0011532078 | 30.0470464826 | 0.8700513527 | 0.9622469177 | 0.1956 |
| 7 | 0.0008255254 | 31.5589745864 | 0.9054089159 | 0.9763763894 | 0.1937 |
| 13 | 0.0005807463 | 33.1954004802 | 0.9348068793 | 0.9876335945 | 0.1929 |
| 19 | 0.0005199769 | 33.7264324129 | 0.9425818466 | 0.9905498993 | 0.1961 |

#### 结果总结

本实验用于给 `EXP-S4-006` 的更正式 test-like gate 复核提供额外 M0 PNG 输入，不代表新的 M0 模型。输出包含 `exports/original/sample_000000.png` 到 `sample_000383.png`，以及每个 SNR 对应的 `exports/snr_XXdb/reconstruction/`。后续 test-like 复核使用 `sample_000256` 到 `sample_000319`，该样本段不与 `EXP-S4-006` 的 train `sample_000032`-`sample_000191` 或 eval `sample_000192`-`sample_000255` 重叠。

#### Semantic drift 观察

本实验只导出 pre-refinement `x_hat`，未额外统计 semantic drift。语义可靠性在后续 `EXP-S4-006` test-like gate 复核中统计。

#### 复现备注

本实验不下载数据或模型，只读取本地 COCO、已有 `best.pt` checkpoint 和第三方 DeepJSCC 代码。运行命令显式清空代理变量；输出目录是新目录，不覆盖旧 256 张 formal export。

### EXP-S2-001：M1-BlindDiffusion preflight/run attempt

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S3 Blind Diffusion
- 方法：M1-BlindDiffusion
- 数据集：COCO2017 `val2017` subset export，计划每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`configs/s3_m1_blind_diffusion_coco256_awgn.yaml`
- 运行命令：
  - `python3 scripts/s3_blind_diffusion_refine.py --dry-run`
  - `python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download`
  - `python3 scripts/s3_blind_diffusion_refine.py --device cpu`
- 关键源码：`scripts/s3_blind_diffusion_refine.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：未创建；`outputs/EXP-S2-001/` 不存在
- 状态：阻塞 / 未生成正式结果

#### 指标

- PSNR：未生成
- MS-SSIM：未生成
- LPIPS：未生成
- Diffusion steps：计划值 25，未执行
- Inference time：未生成

#### 结果总结

已完成 M1 脚本、配置和输入样本对齐 preflight。dry-run 确认正式 M0 export 中 1/7/19 dB 各有 16 张匹配样本可用，且 checkpoint 指向 `best.pt` 而非 `latest.pt`。

正式 diffusion 运行未完成：提权命令因审批层拒绝，无法使用 GPU 和网络下载 Stable Diffusion 权重；local-only CPU 命令在 `local_files_only=true` 下报错，原因是 `runwayml/stable-diffusion-v1-5` 不在本地 Hugging Face cache。

#### Semantic drift 观察

未生成 refinement 图像，不能报告 semantic drift 或视觉提升。

#### 失败案例

本次失败属于环境/模型权重阻塞，不是方法结果失败。不能把该尝试写成 M1 的有效实验。

#### 复现备注

后续若用户显式允许下载并使用 GPU，使用当前配置默认输出 `outputs/EXP-S2-002/`，避免复用本次失败 ID。也可以预先把 diffusion 权重放入 `outputs/cache/huggingface/` 后去掉 `--allow-download` 运行。

#### 下一步

获得 Stable Diffusion img2img 权重和 GPU 运行许可后，运行 `python3 scripts/s3_blind_diffusion_refine.py --device cuda:0 --allow-download`，生成 refined 图、`metrics.json` 和样例图。

### EXP-S2-002：M1-BlindDiffusion COCO-256 small-scale refinement

- 日期：2026-07-01
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S3 Blind Diffusion
- 方法：M1-BlindDiffusion
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S2-002/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S2-002/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy HF_ENDPOINT=https://hf-mirror.com python3 scripts/s3_blind_diffusion_refine.py --device cuda:0`
- 关键源码：`scripts/s3_blind_diffusion_refine.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S2-002/`
- 状态：完成；负结果

#### 指标

| SNR(dB) | M0 PSNR(dB) | M1 PSNR(dB) | M0 SSIM | M1 SSIM | M0 MS-SSIM | M1 MS-SSIM | M0 LPIPS | M1 LPIPS | Diffusion ms/image |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.1746 | 16.2229 | 0.8107 | 0.3204 | 0.9398 | 0.5421 | 0.1747 | 0.5025 | 107.27 |
| 7 | 31.8274 | 16.7812 | 0.9088 | 0.3795 | 0.9779 | 0.5843 | 0.0542 | 0.4600 | 82.42 |
| 19 | 34.1357 | 16.8880 | 0.9470 | 0.4065 | 0.9915 | 0.5959 | 0.0254 | 0.4549 | 81.57 |

- Diffusion steps：25
- Strength：0.25
- Guidance scale：1.0
- Prompt：空字符串
- LPIPS：成功计算，AlexNet 权重缓存到 `outputs/cache/torch/`
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算
- Semantic failure rate：未计算
- CLIP similarity：已由 `EXP-S3-001` 作为辅助语义诊断计算

#### 结果总结

当前固定强度 blind SD img2img 不是有效 refinement。相对 M0，M1 在所有 SNR 下 PSNR、SSIM、MS-SSIM 大幅下降，LPIPS 也显著变差。高 SNR 下 M0 已很接近原图，但 blind diffusion 仍强行改写结构，说明该设置不适合作为正向视觉增强。

#### Semantic drift 观察

尚未用冻结分类器或 CLIP 计算正式 semantic drift 指标，但样例图已经显示明显 hallucination / semantic drift 风险：甜甜圈纹理被改成不稳定的杂乱结构，花瓶和花朵被重新生成，狗和车内猫/座椅场景出现主体和背景结构错乱。该结果应作为后续 semantic control / failure handling 的负例动机，不能包装成提升。

#### 失败案例

样例拼图：

- `outputs/EXP-S2-002/samples/snr_01db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_07db_original_reconstruction_refined.png`
- `outputs/EXP-S2-002/samples/snr_19db_original_reconstruction_refined.png`

这些图的第三行均为 M1 refined 输出，显示 diffusion 对主体结构的强烈改写。

#### 复现备注

大模型下载按用户要求走服务器直连，不走 `127.0.0.1:17890` 代理。官方 `huggingface.co` 服务器直连在本机超时，改用 `HF_ENDPOINT=https://hf-mirror.com`。由于 diffusers 多线程下载在 UNet 大文件上不稳定，本次用临时 range downloader 补齐 `unet/diffusion_pytorch_model.safetensors`，并把完整 blob 链接回 `outputs/cache/huggingface/`。该下载过程不改变实验方法，实际运行时脚本使用 local cache。

#### 下一步

实现 semantic drift / CLIP consistency 的初步评估，并把当前样例整理为 failure case。若继续探索 M1，应新建实验 ID，先做更低 `strength` 的 validation 小网格，不能覆盖本实验输出。

### EXP-S3-001：M1-BlindDiffusion CLIP image consistency diagnostic

- 日期：2026-07-02
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S4 Semantic drift metric
- 方法：CLIP image-image consistency diagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S3-001/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S3-001/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_clip_consistency_eval.py --device cuda:0`
- 关键源码：`scripts/s4_clip_consistency_eval.py`, `scripts/s4_make_clip_failure_gallery.py`
- 输出路径：`outputs/EXP-S3-001/`
- 状态：完成；辅助语义诊断，负结果

#### 指标

| SNR(dB) | CLIP sim(original, M0) | CLIP sim(original, M1) | CLIP drop M0-M1 | M1 lower than M0 | Drop >= 0.10 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9022 | 0.6619 | 0.2402 | 1.0000 | 0.9375 |
| 7 | 0.9587 | 0.6867 | 0.2720 | 1.0000 | 1.0000 |
| 19 | 0.9848 | 0.6954 | 0.2895 | 1.0000 | 1.0000 |

- CLIP backbone：OpenAI CLIP `ViT-B/32` via `open_clip`
- CLIP checkpoint：`outputs/cache/open_clip/ViT-B-32.pt`
- CLIP checkpoint SHA256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- PSNR / MS-SSIM / LPIPS：本实验不重复计算；见 `EXP-S2-002`
- Classification accuracy：未计算
- Prediction consistency：未计算
- Semantic drift rate：未计算正式阈值版；当前以 CLIP drop rate 作辅助诊断
- Semantic failure rate：未计算正式阈值版
- Detector accept rate：不适用
- Detector reject rate：不适用

#### 结果总结

CLIP image-image consistency 进一步确认 `EXP-S2-002` 的 blind diffusion refinement 明显不可靠。所有 48 个样本中，M1 refined 相对原图的 CLIP cosine similarity 都低于 M0 reconstruction；7 dB 和 19 dB 下所有样本的 drop 都大于 0.10。高 SNR 下 M0 已非常接近原图，但 M1 仍把图像改写到 CLIP 空间显著远离原图的位置。

#### Semantic drift 观察

该实验不是最终的分类一致性 semantic drift 指标，但它把视觉样例中的 hallucination 风险量化出来：M0 的 CLIP 相似度随 SNR 升高从 0.9022 增至 0.9848，而 M1 基本停留在 0.66 到 0.70，说明 blind diffusion 并没有利用高 SNR 下更可靠的 JSCC 重建，反而引入额外语义漂移。

#### 失败案例

每个 SNR 的 top failure case 已写入 `outputs/EXP-S3-001/metrics.json`，逐样本指标见 `outputs/EXP-S3-001/per_sample.csv`。按 CLIP drop 排名前列的样本包括：

- 1 dB：`sample_000004.png`, `sample_000013.png`, `sample_000000.png`
- 7 dB：`sample_000005.png`, `sample_000009.png`, `sample_000013.png`
- 19 dB：`sample_000013.png`, `sample_000004.png`, `sample_000008.png`

已用 `scripts/s4_make_clip_failure_gallery.py` 从 `per_sample.csv` 生成 failure case gallery：

- 全局 top sheet：`outputs/EXP-S3-001/failure_cases/sheets/global_top_clip_drop.png`
- 分 SNR sheets：`outputs/EXP-S3-001/failure_cases/sheets/snr_01db_top_clip_drop.png`, `outputs/EXP-S3-001/failure_cases/sheets/snr_07db_top_clip_drop.png`, `outputs/EXP-S3-001/failure_cases/sheets/snr_19db_top_clip_drop.png`
- triptych 目录：`outputs/EXP-S3-001/failure_cases/triptychs/`
- 索引：`outputs/EXP-S3-001/failure_cases/index.json`, `outputs/EXP-S3-001/failure_cases/global_top_clip_drop.csv`

gallery 共包含 18 个不重复 triptych：全局 top 12 和每个 SNR top 6。抽查全局最大失败样本 `snr_19db/sample_000013.png` 时，M0 与原图接近，但 M1 refined 出现明显主体纹理和背景结构改写，符合 CLIP drop 0.4026 的诊断。

#### 复现备注

open_clip 3.3.0 对 `ViT-B-32/openai` 默认优先尝试 Hugging Face Hub；本机服务器直连 `huggingface.co` HEAD 请求超时，因此本实验改为从 OpenAI 官方 URL 直连下载 `ViT-B-32.pt` 到项目缓存，并在配置中显式使用本地权重路径。该 `.pt` 是 TorchScript archive，PyTorch 2.6+ 默认 `weights_only=True` 会拒绝加载；由于文件来源为 OpenAI 官方 URL 且 SHA256 校验完全匹配，本实验配置中对该权重设置 `weights_only: false`。

#### 下一步

补充更正式的 semantic drift metric，例如冻结分类器 prediction consistency 或 object-level / CLIP text consistency。后续如果继续试更保守的 blind diffusion strength，应先用本诊断脚本和 failure gallery 筛查是否仍然破坏语义。

### EXP-S3-002：M1-BlindDiffusion frozen classifier pseudo-label consistency diagnostic

- 日期：2026-07-02
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S4 Semantic drift metric
- 方法：Frozen classifier pseudo-label consistency diagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S3-002/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S3-002/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_classifier_consistency_eval.py --device cuda:0`
- 关键源码：`scripts/s4_classifier_consistency_eval.py`, `scripts/s4_make_classifier_failure_gallery.py`
- 输出路径：`outputs/EXP-S3-002/`
- 状态：完成；辅助分类器诊断，负结果

#### 指标

All-subset，使用原图 ImageNet top-1 作为 pseudo-label：

| SNR(dB) | M0 matches original top-1 | M1 matches original top-1 | M0 pseudo drift-origin | M1 pseudo drift-origin | M1 refinement drift |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5000 | 0.1250 | 0.5000 | 0.8750 | 0.8750 |
| 7 | 0.6875 | 0.0625 | 0.3125 | 0.9375 | 0.9375 |
| 19 | 0.9375 | 0.1250 | 0.0625 | 0.8750 | 0.8750 |

原图 top-1 confidence >= 0.30 的 pseudo-clean subset：

| SNR(dB) | subset n | M0 matches original top-1 | M1 matches original top-1 | M0 pseudo drift-origin | M1 pseudo drift-origin |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 0.8889 | 0.2222 | 0.1111 | 0.7778 |
| 7 | 9 | 1.0000 | 0.1111 | 0.0000 | 0.8889 |
| 19 | 9 | 1.0000 | 0.2222 | 0.0000 | 0.7778 |

- Frozen classifier：torchvision AlexNet `IMAGENET1K_V1`
- Classifier checkpoint：`outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- Classification accuracy：未计算；COCO GT 标签未用于本实验
- Prediction consistency：见上表的 matches original top-1
- Semantic drift rate：未计算正式 clean-correct 版本；当前为 pseudo-label drift diagnostic
- Semantic failure rate：未计算正式版本
- Detector accept rate：不适用
- Detector reject rate：不适用

#### 结果总结

冻结分类器诊断与 CLIP 诊断一致：M0 随 SNR 升高越来越保持原图分类器 top-1，而 M1 refined 在所有 SNR 下都明显偏离原图 pseudo-label。尤其在原图置信度 >= 0.30 的 subset 上，7 dB 和 19 dB 的 M0 一致率为 1.0，但 M1 只有 0.1111 和 0.2222，说明 blind diffusion 会在高质量 DeepJSCC 输入上仍然强行改写语义线索。

#### Semantic drift 观察

该实验比 CLIP 更接近 `MILESTONES.md` 中要求的冻结分类器路线，但仍不是最终 clean-correct 指标，因为 COCO 图像没有使用分类 GT，ImageNet AlexNet top-1 只能作为 pseudo-label。它适合作为当前 M1 负结果的第二条证据，以及后续 failure detector / fallback 规则的调试信号。

#### 失败案例

`outputs/EXP-S3-002/metrics.json` 中保存了每个 SNR 的 top failure cases，筛选条件为 M0 与原图 top-1 一致但 M1 不一致。典型样本包括：

- 1 dB：`sample_000002.png`，原图/M0 为 `Pomeranian`，M1 为 `shoe shop`
- 7 dB：`sample_000002.png`，原图/M0 为 `Pomeranian`，M1 为 `dogsled`
- 19 dB：`sample_000002.png`，原图/M0 为 `Pomeranian`，M1 为 `gondola`
- 19 dB：`sample_000015.png`，原图/M0 为 `broccoli`，M1 为 `indigo bunting`

逐样本预测见 `outputs/EXP-S3-002/per_sample.csv`。

已用 `scripts/s4_make_classifier_failure_gallery.py` 从 `per_sample.csv` 生成 classifier failure case gallery：

- 全局 top sheet：`outputs/EXP-S3-002/failure_cases/sheets/global_top_classifier_drift.png`
- 分 SNR sheets：`outputs/EXP-S3-002/failure_cases/sheets/snr_01db_top_classifier_drift.png`, `outputs/EXP-S3-002/failure_cases/sheets/snr_07db_top_classifier_drift.png`, `outputs/EXP-S3-002/failure_cases/sheets/snr_19db_top_classifier_drift.png`
- triptych 目录：`outputs/EXP-S3-002/failure_cases/triptychs/`
- 索引：`outputs/EXP-S3-002/failure_cases/index.json`, `outputs/EXP-S3-002/failure_cases/global_top_classifier_drift.csv`

gallery 共包含 18 个不重复 triptych：全局 top 12 和每个 SNR top 6。抽查全局最大失败样本 `snr_19db/sample_000002.png` 时，原图和 M0 均被分类为 `Pomeranian`，M1 refined 被分类为 `gondola`，图像中主体结构也明显被破坏。

#### 复现备注

本实验没有联网下载。AlexNet ImageNet 权重已由 LPIPS/torch cache 路径提供，脚本在 `--allow-download` 未开启时会要求 `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth` 已存在。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。

#### 下一步

确定正式 semantic drift 主指标的语义模型选择：若继续 COCO 主线，优先考虑 object detector / CLIP-text / caption-based consistency；若需要严格分类 clean-correct 统计，可引入带 ImageNet 标签的 Imagenette/ImageNet subset 作为补充，而不是把当前 pseudo-label 诊断包装成最终指标。

### EXP-S3-003：M1-BlindDiffusion COCO caption CLIP text consistency diagnostic

- 日期：2026-07-02
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S4 Semantic drift metric
- 方法：COCO caption CLIP image-text consistency diagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S3-003/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- COCO 标注：`data/coco/annotations/captions_val2017.json`，来自官方 `annotations_trainval2017.zip`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S3-003/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s4_coco_caption_clip_eval.py --device cuda:0`
- 关键源码：`scripts/s4_coco_caption_clip_eval.py`, `scripts/s4_make_coco_caption_failure_gallery.py`
- 输出路径：`outputs/EXP-S3-003/`
- 状态：完成；辅助 caption 语义诊断，负结果

#### 指标

使用每张 COCO val 图的 5 条人工 caption，计算每张图像与其 caption 集合的 CLIP image-text cosine similarity；表中 `caption-max` 表示取 5 条 caption 中最高相似度。

| SNR(dB) | Original caption-max | M0 caption-max | M1 caption-max | Drop M0-M1 | M1 max lower than M0 | Drop >= 0.05 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.3292 | 0.3306 | 0.2816 | 0.0490 | 1.0000 | 0.6250 |
| 7 | 0.3292 | 0.3305 | 0.2815 | 0.0490 | 0.8125 | 0.5000 |
| 19 | 0.3292 | 0.3263 | 0.2877 | 0.0386 | 0.8125 | 0.3125 |

caption-mean 辅助结果：

| SNR(dB) | M0 caption-mean | M1 caption-mean | Drop M0-M1 | M1 mean lower than M0 | Drop >= 0.05 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.3054 | 0.2568 | 0.0486 | 0.9375 | 0.6250 |
| 7 | 0.3063 | 0.2559 | 0.0504 | 0.8125 | 0.5000 |
| 19 | 0.3022 | 0.2605 | 0.0417 | 0.8125 | 0.3125 |

- CLIP backbone：OpenAI CLIP `ViT-B/32` via `open_clip`
- CLIP checkpoint：`outputs/cache/open_clip/ViT-B-32.pt`
- CLIP checkpoint SHA256：`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`
- Classification accuracy：未计算；COCO caption 不是分类 GT
- Prediction consistency：未计算冻结分类器版本；见 `EXP-S3-002`
- Semantic drift rate：未计算正式 clean-correct 版本；当前为 caption CLIP drop diagnostic
- Semantic failure rate：未计算正式版本
- Detector accept rate：不适用
- Detector reject rate：不适用

#### 结果总结

COCO caption image-text consistency 与 CLIP image-image 和冻结分类器 pseudo-label 诊断方向一致：M0 reconstruction 与原图 captions 的对齐基本保持在 original 附近，而 M1 refined 在所有 SNR 下都明显下降。尤其 1 dB 下 16/16 个样本的 M1 caption-max 低于 M0；7 dB 和 19 dB 下也有 13/16 个样本低于 M0。

#### Semantic drift 观察

该实验把 COCO 主数据集的人工 captions 接入了 S4 诊断，解决了此前只有 image-image CLIP 或 ImageNet pseudo-label 的局限。它仍是辅助指标，不能替代 `MILESTONES.md` 要求的正式 clean-correct 冻结分类器统计，但能更直接说明 blind diffusion 会把图像从 COCO caption 描述的语义内容上拉开。

#### 失败案例

`outputs/EXP-S3-003/metrics.json` 中保存了每个 SNR 的 top caption drop cases。典型样本包括：

- 1 dB：`sample_000002.png`，caption 为小狗，caption-max drop 0.0957
- 7 dB：`sample_000008.png`，caption 为 car / clock / flowers，caption-max drop 0.1198
- 19 dB：`sample_000003.png`，caption 为 car 中的黑猫，caption-max drop 0.0951

已用 `scripts/s4_make_coco_caption_failure_gallery.py` 从 `per_sample.csv` 生成 caption failure case gallery：

- 全局 top sheet：`outputs/EXP-S3-003/failure_cases/sheets/global_top_caption_clip_drop.png`
- 分 SNR sheets：`outputs/EXP-S3-003/failure_cases/sheets/snr_01db_top_caption_clip_drop.png`, `outputs/EXP-S3-003/failure_cases/sheets/snr_07db_top_caption_clip_drop.png`, `outputs/EXP-S3-003/failure_cases/sheets/snr_19db_top_caption_clip_drop.png`
- triptych 目录：`outputs/EXP-S3-003/failure_cases/triptychs/`
- 索引：`outputs/EXP-S3-003/failure_cases/index.json`, `outputs/EXP-S3-003/failure_cases/global_top_caption_clip_drop.csv`, `outputs/EXP-S3-003/failure_cases/README.md`

gallery 共包含全局 top 12 和每个 SNR top 6 的 triptych。抽查全局最大失败样本 `snr_07db/sample_000008.png` 时，原图/M0 都保留 car、clock 和 flowers 场景，M1 refined 出现明显纹理和结构改写。

#### 复现备注

本实验没有联网下载模型权重，使用已缓存的 OpenAI CLIP 权重。COCO annotations 下载发生在实验前，来源为 `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`，大小 252907541 bytes；下载时清空代理变量并使用服务器直连，`unzip -t` 验证无错误。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。

#### 下一步

结合 `EXP-S3-001`、`EXP-S3-002` 和 `EXP-S3-003` 设计最小 semantic failure handling：优先实现一个可复现的 fallback 规则，统计 detector accept/reject 和 Final-Failure，再进入 M3/Ours。

### EXP-S4-001：M3 pseudo-classifier semantic fallback pilot

- 日期：2026-07-03
- 项目版本：N/A (not a project git repo)
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / Semantic Failure Handling pilot
- 方法：M3-PseudoClassifierFallbackPilot
- 数据集：COCO2017 `val2017` subset export，每个 SNR 16 张图
- 数据 split / 样本 ID：`outputs/EXP-S4-001/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000015.png`
- 信道：AWGN
- SNR：`[1, 7, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-001/config.yaml`
- 运行命令：`python3 scripts/s5_semantic_fallback_eval.py --device cuda:0`
- 关键源码：`scripts/s5_semantic_fallback_eval.py`, `src/cadsd_jscc/metrics.py`, `scripts/s4_classifier_consistency_eval.py`
- 输入实验：M0 export `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/`；M1 output `outputs/EXP-S2-002/`；classifier CSV `outputs/EXP-S3-002/per_sample.csv`
- 输出路径：`outputs/EXP-S4-001/`
- 状态：完成；S5 fallback pilot，不是完整 M3/Ours

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价；detector 决策只看 `c(M1) == c(M0)`，不使用原图。

| SNR(dB) | Accept | Reject | M0 PSNR | M1 PSNR | M3 PSNR | M0 LPIPS | M1 LPIPS | M3 LPIPS | M0 Final-Failure | M1 Final-Failure | M3 Final-Failure | False Accept | False Reject |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1250 | 0.8750 | 28.1746 | 16.2229 | 26.8313 | 0.1747 | 0.5025 | 0.2123 | 0.5000 | 0.8750 | 0.5000 | 0.0000 | 0.0000 |
| 7 | 0.0625 | 0.9375 | 31.8274 | 16.7812 | 30.9141 | 0.0542 | 0.4600 | 0.0782 | 0.3125 | 0.9375 | 0.3125 | 0.0000 | 0.0000 |
| 19 | 0.1250 | 0.8750 | 34.1357 | 16.8880 | 32.0135 | 0.0254 | 0.4549 | 0.0733 | 0.0625 | 0.8750 | 0.0625 | 0.0000 | 0.0000 |

Pseudo-clean subset，原图 top-1 confidence >= 0.30：

| SNR(dB) | subset n | Accept | M0 Final-Failure | M1 Final-Failure | M3 Final-Failure | M3 Prediction-Consistency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 0.2222 | 0.1111 | 0.7778 | 0.1111 | 0.8889 |
| 7 | 9 | 0.1111 | 0.0000 | 0.8889 | 0.0000 | 1.0000 |
| 19 | 9 | 0.2222 | 0.0000 | 0.7778 | 0.0000 | 1.0000 |

- Detector：frozen AlexNet top-1 agreement between M0 and M1
- Final output：若 `c(M1) == c(M0)`，输出 M1；否则 fallback 到 M0
- Diffusion steps：沿用 `EXP-S2-002` 的 25 steps
- Strength：沿用 `EXP-S2-002` 的 0.25
- Guidance scale：沿用 `EXP-S2-002` 的 1.0
- Prompt：空字符串
- CLIP similarity：本实验不重新计算；见 `EXP-S3-001` 和 `EXP-S3-003`

#### 结果总结

该 pilot 验证了最小 semantic failure handling 的可复现流程：在不看原图的接收端规则下，detector 拒绝大多数会改变冻结分类器 top-1 的 M1 refined 输出，使 M3 pseudo Final-Failure 回到 M0 水平。相对 M1，M3 在 all-subset 上将 Final-Failure 分别降低 `0.3750/0.6250/0.8125`。

但这不是完整 M3/Ours：底层 diffusion 仍是 `EXP-S2-002` 的固定强度负结果。少量 accepted M1 虽然没有造成 pseudo-label failure，却仍降低 PSNR、MS-SSIM 和 LPIPS，因此 fallback 只能控制语义风险，不能把一个过强的 blind diffusion 设置变成视觉增强。

#### Semantic drift 观察

M3 的 `m3_refinement_drift` 在该 detector 下为 0，因为最终输出要么与 M0 分类一致，要么直接回退到 M0。这个结果说明 top-1 agreement 是一个强保守规则，但也意味着它几乎不接受 diffusion；后续必须配合更弱 diffusion strength 或 SNR-aware strength 才可能获得感知收益。

#### 失败案例

样例拼图：

- `outputs/EXP-S4-001/samples/snr_01db_original_m0_m1_m3final.png`
- `outputs/EXP-S4-001/samples/snr_07db_original_m0_m1_m3final.png`
- `outputs/EXP-S4-001/samples/snr_19db_original_m0_m1_m3final.png`

逐样本 detector 决策、final 输出路径、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-001/per_sample.csv`。

#### 复现备注

本实验不联网、不下载模型、不重新运行 diffusion，只读取已有 M0/M1 图像和 `EXP-S3-002` 的冻结分类器 CSV。LPIPS 使用已缓存 AlexNet 权重。输出目录存在时脚本会拒绝覆盖。

#### 下一步

新建实验 ID 做保守 diffusion strength validation 网格，例如 `strength <= 0.10` 和更少 steps，并把本 fallback 脚本接到新 M1/M2 输出上。只有当 M3 相比 blind diffusion 降低 Final-Failure、且相比 M0 保留可观感知收益时，才能进入正式 M3/Ours 结论。

### EXP-S4-002：SNR-aware low-strength diffusion validation

- 日期：2026-07-03
- 项目版本：N/A (local directory is not yet a git repo)
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / diffusion strength validation
- 方法：SNRAdaptiveDiffusionStrengthValidation
- 数据集：COCO2017 `val2017` subset export，每个 SNR 8 张图
- 数据 split / 样本 ID：`outputs/EXP-S4-002/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000007.png`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-002/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_snr_adaptive_diffusion_validation.py --device cuda:0`
- 关键源码：`scripts/s5_snr_adaptive_diffusion_validation.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-002/`
- 状态：完成；S5 validation，负/部分结果，不是完整 M3/Ours

#### 候选设置

| Candidate | Method | Strength schedule | Steps | Guidance |
|---|---|---|---:|---:|
| `fixed_0p05` | M1-LowStrengthFixedDiffusion | 1/4/7/13/19 dB: `0.05/0.05/0.05/0.05/0.05` | 15 | 1.0 |
| `snr_adaptive_0p10_to_0p05` | M2-SNRAdaptiveDiffusion | 1/4/7/13/19 dB: `0.10/0.08/0.06/0.05/0.05` | 15 | 1.0 |

两个 schedule 都满足 strength 随 SNR 升高不增加。failure handling 使用 `EXP-S4-001` 同类规则：若 `c(refined) == c(M0)` 则接受 refined，否则 fallback 到 M0。

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价；detector 不使用原图。

| Candidate | SNR(dB) | Strength | M0 PSNR | Refined PSNR | M3 PSNR | M0 LPIPS | Refined LPIPS | M3 LPIPS | Refined Failure | M3 Failure | Accept | False Accept | False Reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_0p05 | 1 | 0.05 | 28.7285 | 25.1163 | 26.9112 | 0.1885 | 0.1989 | 0.1922 | 0.6250 | 0.3750 | 0.5000 | 0.1250 | 0.0000 |
| fixed_0p05 | 4 | 0.05 | 30.7464 | 26.0170 | 26.8029 | 0.1040 | 0.1367 | 0.1313 | 0.3750 | 0.2500 | 0.8750 | 0.2500 | 0.0000 |
| fixed_0p05 | 7 | 0.05 | 32.3475 | 26.5848 | 27.6615 | 0.0606 | 0.1089 | 0.1050 | 0.0000 | 0.2500 | 0.7500 | 0.0000 | 0.2500 |
| fixed_0p05 | 13 | 0.05 | 34.0785 | 26.9924 | 29.1365 | 0.0308 | 0.0871 | 0.0657 | 0.2500 | 0.0000 | 0.7500 | 0.0000 | 0.0000 |
| fixed_0p05 | 19 | 0.05 | 34.6217 | 27.0938 | 28.3947 | 0.0282 | 0.0889 | 0.0791 | 0.1250 | 0.0000 | 0.8750 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 1 | 0.10 | 28.7285 | 22.1567 | 26.2416 | 0.1885 | 0.2759 | 0.2244 | 0.6250 | 0.3750 | 0.3750 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 4 | 0.08 | 30.7464 | 22.7259 | 26.8134 | 0.1040 | 0.2180 | 0.1649 | 0.5000 | 0.2500 | 0.5000 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 7 | 0.06 | 32.3475 | 26.5599 | 27.6566 | 0.0606 | 0.1065 | 0.1034 | 0.0000 | 0.2500 | 0.7500 | 0.0000 | 0.2500 |
| snr_adaptive_0p10_to_0p05 | 13 | 0.05 | 34.0785 | 27.0258 | 29.1664 | 0.0308 | 0.0877 | 0.0660 | 0.2500 | 0.0000 | 0.7500 | 0.0000 | 0.0000 |
| snr_adaptive_0p10_to_0p05 | 19 | 0.05 | 34.6217 | 27.1297 | 28.4295 | 0.0282 | 0.0888 | 0.0789 | 0.1250 | 0.0000 | 0.8750 | 0.0000 | 0.0000 |

#### 结果总结

相比 `EXP-S2-002` 的 `strength=0.25`，低强度和 SNR-aware schedule 的 semantic drift 明显缓和，高 SNR 下 fallback 后的 M3 Failure 可回到 M0 水平。但两个候选都没有获得有效视觉收益：即使 `strength=0.05`，refined PSNR 和 LPIPS 仍明显差于 M0，且高 SNR 下损伤更突出。

这说明当前 Stable Diffusion img2img 后处理并不只是 strength 过强的问题。VAE encode/decode、最小 denoise step 或 prompt-free generative prior 都可能对高保真 DeepJSCC 重建造成结构/纹理改写。该结果应记录为负/部分结果，不能包装为 M2 或 M3 的成功。

#### Semantic drift 观察

`snr_adaptive_0p10_to_0p05` 在 1/4 dB 使用更高 strength，语义 failure 并没有比 `fixed_0p05` 更好，图像质量反而更差。当前证据不支持“简单增大低 SNR diffusion strength”作为有效 SNR-aware 策略。semantic fallback 仍能压低 final failure，但如果 refined 图像本身没有视觉收益，fallback 只是在做风险控制，不构成主要贡献。

#### 失败案例

样例拼图位于：

- `outputs/EXP-S4-002/candidates/fixed_0p05/samples/`
- `outputs/EXP-S4-002/candidates/snr_adaptive_0p10_to_0p05/samples/`

每张拼图为 original / M0 / refined / M3-final 四行。逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-002/per_sample.csv`。

#### 复现备注

本实验使用已缓存的 `runwayml/stable-diffusion-v1-5` 和 AlexNet/LPIPS 权重，不下载模型。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。本地目录当前尚未初始化为 git 仓库，因此项目版本仍不能填写 commit；用户提供的 GitHub URL 已写入 config 和 metrics metadata。

#### 下一步

优先做 VAE/latent roundtrip 诊断，分离以下因素：

- SD VAE encode/decode 本身相对 M0 的失真。
- 最小 denoise step 在极低 strength 下是否仍改写结构。
- prompt-free prior 是否比 restoration-aware diffusion 更容易 hallucinate。

若 roundtrip 本身已显著损伤 PSNR/LPIPS，则第一版不应继续把通用 SD img2img 当作主正向 refinement，而应转向更保守的 restoration 模块或把 diffusion 结果仅作为负例和 failure handling 动机。

### EXP-S4-003：Stable Diffusion VAE roundtrip diagnostic

- 日期：2026-07-03
- 项目版本：N/A (local directory is not yet a git repo)
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / VAE bottleneck diagnostic
- 方法：SDVAERoundtripDiagnostic
- 数据集：COCO2017 `val2017` subset export，每个 SNR 8 张图
- 数据 split / 样本 ID：`outputs/EXP-S4-003/source_manifest.json`；样本名为 `sample_000000.png` 到 `sample_000007.png`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-003/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sd_vae_roundtrip_eval.py --device cuda:0`
- 关键源码：`scripts/s5_sd_vae_roundtrip_eval.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-003/`
- 状态：完成；S5 VAE roundtrip 诊断，负/瓶颈确认，不是完整 M2/M3/Ours

#### 设置

- SD 组件：`runwayml/stable-diffusion-v1-5` 的 `vae` subfolder
- VAE latent：使用 `latent_dist.mode()`，deterministic roundtrip
- VAE scaling factor：0.18215
- UNet denoise：不运行
- diffusion prompt：不使用；本实验没有文本条件、没有 guidance、没有 denoising step
- 对照：
  - `M0 reconstruction vs original`
  - `M0-VAE roundtrip vs original`
  - `M0-VAE roundtrip vs M0 reconstruction`
  - `Original-VAE roundtrip vs original`

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。

| SNR(dB) | M0 PSNR | M0-VAE PSNR | Delta | M0 LPIPS | M0-VAE LPIPS | Delta | M0-VAE vs M0 PSNR | M0 Failure | M0-VAE Failure | M0-VAE Refinement Drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.7285 | 25.2433 | -3.4852 | 0.1885 | 0.1975 | +0.0090 | 25.6506 | 0.3750 | 0.6250 | 0.5000 |
| 4 | 30.7464 | 26.1930 | -4.5534 | 0.1040 | 0.1335 | +0.0295 | 26.7426 | 0.2500 | 0.3750 | 0.1250 |
| 7 | 32.3475 | 26.7550 | -5.5926 | 0.0606 | 0.1049 | +0.0443 | 27.4579 | 0.2500 | 0.1250 | 0.1250 |
| 13 | 34.0785 | 27.1925 | -6.8861 | 0.0308 | 0.0853 | +0.0545 | 27.9958 | 0.0000 | 0.2500 | 0.2500 |
| 19 | 34.6217 | 27.2957 | -7.3260 | 0.0282 | 0.0860 | +0.0578 | 28.1307 | 0.0000 | 0.1250 | 0.1250 |

Original-VAE roundtrip 相对原图在这 8 张样本上固定为 PSNR `26.8097` dB、LPIPS `0.0605`，pseudo failure 为 `0.1250`。这说明即使输入是干净原图，SD VAE 往返也会引入可观失真；当输入是高 SNR M0 时，该瓶颈会把 `34+` dB 的重建压到约 `27` dB。

#### 结果总结

该实验把 `EXP-S4-002` 中的质量下降拆开验证：不运行 UNet、不使用 prompt、不做任何 diffusion denoise，仅 SD VAE encode/decode 已足以解释大部分高保真损伤。M0-VAE 相对 M0 的 PSNR 损失随 SNR 升高变大，从 1 dB 的 `-3.4852` dB 扩大到 19 dB 的 `-7.3260` dB；LPIPS 也从 `+0.0090` 恶化到 `+0.0578`。

因此，当前通用 Stable Diffusion img2img 路线不是简单调低 `strength` 就能成为正向视觉增强。VAE roundtrip 本身已经破坏了 DeepJSCC high-SNR reconstruction 的细节和分类线索，后续若继续使用 diffusion，应优先考虑 restoration-aware 或 latent-free/像素域保守方法。

#### Semantic drift 观察

M0-VAE 不只是低层指标下降，也会改变冻结 AlexNet pseudo-label。All-subset 中，1 dB 的 M0-VAE pseudo Final-Failure 为 `0.6250`，高于 M0 的 `0.3750`；13/19 dB 中 M0 本身 failure 为 0，但 M0-VAE 分别引入 `0.2500/0.1250` 的 pseudo failure。该结果继续支持本项目主线：任何“看起来更自然”的 generative/latent 重建都必须接受 semantic drift 检查。

#### 失败案例

样例拼图位于：

- `outputs/EXP-S4-003/samples/snr_01db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_04db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_07db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_13db_original_m0_m0vae_originalvae.png`
- `outputs/EXP-S4-003/samples/snr_19db_original_m0_m0vae_originalvae.png`

每张拼图为 original / M0 / M0-VAE / original-VAE 四行。逐样本路径、top-1 pseudo-label 和一致性标记见 `outputs/EXP-S4-003/per_sample.csv`。

#### 复现备注

本实验使用已缓存的 `runwayml/stable-diffusion-v1-5` VAE 和 AlexNet/LPIPS 权重，不下载模型。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。本地目录运行时尚未初始化为 git 仓库，因此项目版本仍不能填写 commit；用户提供的 GitHub URL 已写入 config 和 metrics metadata。

#### 下一步

第一版不建议继续把通用 SD img2img 当作 M2/M3 正向主路线。更稳妥的推进方向是把 SD img2img 负结果和 VAE bottleneck 作为 semantic failure handling 的动机，同时探索更贴近 restoration 的保守模块；若仍使用 diffusion，需要优先验证无 VAE 高保真瓶颈的实现。

### EXP-S4-004：SNR-conditioned pixel residual refiner pilot attempt

- 日期：2026-07-03
- 项目版本：`401d4bdda6ff52602093e978ad8c1c34c6f939ac` + uncommitted local changes at run time
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free restoration pilot
- 方法：SNRConditionedPixelResidualRefinerPilot attempt
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：训练 `sample_000008.png` 到 `sample_000031.png`；评估计划为 `sample_000000.png` 到 `sample_000007.png`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- config：`outputs/EXP-S4-004/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --device cuda:0`
- 关键源码：`scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-004/`
- 状态：失败；训练完成后 CSV 写入失败，未生成最终 metrics，实验 ID 不复用

#### 失败原因

初版 `write_csv` 只用第一行 `train_history` 的字段作为 CSV header；但 `eval_mse` 和 `eval_psnr_db` 只在每 10 个 epoch 验证时出现，导致写入后续行时报错：

```text
ValueError: dict contains fields not in fieldnames: 'eval_mse', 'eval_psnr_db'
```

该失败发生在训练 80 epoch 和 checkpoint 写入之后、最终评估之前。输出目录保留了 `config.yaml`、`source_manifest.json`、`checkpoints/best.pt` 和 `checkpoints/latest.pt`。随后已修复 CSV 字段合并逻辑，并用新实验 ID `EXP-S4-005` 重新完整运行。

#### 复现备注

本实验不下载模型或数据，运行命令显式清空代理变量。由于这是失败实验，不能把 checkpoint 或中间训练 loss 包装成正式结果。

### EXP-S4-005：SNR-conditioned pixel residual refiner pilot

- 日期：2026-07-03
- 项目版本：`401d4bdda6ff52602093e978ad8c1c34c6f939ac` + uncommitted local changes at run time
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free restoration pilot
- 方法：SNRConditionedPixelResidualRefinerPilot
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：
  - train：`sample_000008.png` 到 `sample_000031.png`，每个 SNR 24 张，共 120 对 M0/original
  - eval：`sample_000000.png` 到 `sample_000007.png`，每个 SNR 8 张，共 40 对 M0/original
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- refiner checkpoint：`outputs/EXP-S4-005/checkpoints/best.pt`
- config：`outputs/EXP-S4-005/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --device cuda:0`
- 关键源码：`scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-005/`
- 状态：完成；S5 latent-free restoration pilot，小样本正向结果，不是最终 M2/M3/Ours

#### 方法设置

- 模型：小型 SNR-conditioned residual CNN
- 输入：`M0 reconstruction` + 1 通道 SNR map
- 输出：pixel-domain residual 后的 `x_refined`
- 初始化：最后一层零初始化，初始输出接近 M0
- residual gate：1/4/7/13/19 dB 使用 `0.12/0.10/0.08/0.05/0.04`，随 SNR 升高不增加
- 训练：80 epoch，batch size 8，128x128 random crop，MSE + 0.1 L1
- semantic failure handling：与 `EXP-S4-001` 类似，若 `c(refined) == c(M0)` 则接受，否则 fallback 到 M0；detector 不看原图

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。

| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Delta | M0 LPIPS | Refined LPIPS | Delta | M0 Failure | Refined Failure | M3 Failure | Accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 28.7285 | 29.1151 | +0.3866 | 0.1885 | 0.1703 | -0.0183 | 0.3750 | 0.2500 | 0.3750 | 0.8750 |
| 4 | 0.10 | 30.7464 | 30.9332 | +0.1868 | 0.1040 | 0.0995 | -0.0044 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 7 | 0.08 | 32.3475 | 32.4380 | +0.0905 | 0.0606 | 0.0607 | +0.0000 | 0.2500 | 0.2500 | 0.2500 | 1.0000 |
| 13 | 0.05 | 34.0785 | 34.2034 | +0.1248 | 0.0308 | 0.0299 | -0.0010 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| 19 | 0.04 | 34.6217 | 34.7899 | +0.1682 | 0.0282 | 0.0254 | -0.0028 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |

#### 结果总结

该 pilot 初步回答了 `EXP-S4-003` 后的关键问题：避开 Stable Diffusion VAE 后，保守 pixel-domain residual refinement 可以在不牺牲语义可靠性的情况下带来小幅质量收益。5 个 SNR 上 refined PSNR 均高于 M0，提升范围为 `+0.0905` 到 `+0.3866` dB；LPIPS 在 1/4/13/19 dB 改善，在 7 dB 基本持平。

语义侧没有出现 `EXP-S2-002` 那种系统性 drift。All-subset 下，13/19 dB 的 refined failure 仍为 0；1 dB refined failure 从 M0 的 `0.3750` 降到 `0.2500`，但经过 top-1 agreement fallback 后 M3 final failure 回到 M0 的 `0.3750`，说明当前 detector 对“修正了原错误分类”的情况较保守。

#### Semantic drift 观察

`refined_vs_m0_reconstruction` 的 PSNR 在 1/4/7/13/19 dB 分别为约 `40.57/45.16/48.06/49.36/48.37` dB，说明 residual 改动很小。除 1 dB 有 1 个样本改变 M0 top-1 外，其他 SNR 的 `refined_refinement_drift` 均为 0。与 SD img2img 的强 hallucination 相比，pixel residual 更符合 semantic drift control 的第一版方向。

#### 失败案例和样例

样例拼图位于：

- `outputs/EXP-S4-005/samples/snr_01db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_04db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_07db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_13db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-005/samples/snr_19db_original_m0_refined_m3final.png`

逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-005/per_sample.csv`。

#### 复现备注

本实验不联网、不下载模型或数据，只读取已有正式 M0 export 和本地 AlexNet/LPIPS 权重。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。运行时仓库 HEAD 是 `401d4bd`，但脚本和配置处于未提交状态；因此本记录额外列出脚本、配置和输出副本路径。

#### 下一步

将该 pilot 扩大到更稳定的 validation split：重新导出更多 COCO val M0 样本，训练/验证/测试三分，并比较 `M0`、`SD img2img negative M1`、`pixel residual M2` 和 `semantic fallback M3`。只有在更大 split 上稳定保持质量收益且不增加 semantic failure，才能把它作为第一版替代通用 SD img2img 的主路线。

### EXP-S4-006：SNR-conditioned pixel residual refiner validation

- 日期：2026-07-03
- 项目版本：`709f1c665f500e3f6a3dc71609267dd90789c005`
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free restoration validation
- 方法：SNRConditionedPixelResidualRefinerValidation
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：
  - 输入 export：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
  - train：`sample_000032.png` 到 `sample_000191.png`，每个 SNR 160 张，共 800 对 M0/original
  - eval：`sample_000192.png` 到 `sample_000255.png`，每个 SNR 64 张，共 320 对 M0/original
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- refiner checkpoint：`outputs/EXP-S4-006/checkpoints/best.pt`
- config：`outputs/EXP-S4-006/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_residual_refiner_validation_coco256_awgn.yaml --device cuda:0`
- 关键源码：`scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-006/`
- 状态：完成；S5 residual validation 正向结果，但仍需 detector error analysis；不是最终 M2/M3/Ours

#### 方法设置

- 模型：小型 SNR-conditioned residual CNN
- 输入：`M0 reconstruction` + 1 通道 SNR map
- 输出：pixel-domain residual 后的 `x_refined`
- 初始化：最后一层零初始化，初始输出接近 M0
- residual gate：1/4/7/13/19 dB 使用 `0.12/0.10/0.08/0.05/0.04`
- 训练：40 epoch，batch size 16，128x128 random crop，MSE + 0.1 L1
- semantic failure handling：若 `c(refined) == c(M0)` 则接受，否则 fallback 到 M0；detector 不看原图

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。`Refined Failure` 是 refined 相对原图 pseudo top-1 的 failure；`M3 Failure` 是 top-1 agreement fallback 后最终输出的 failure。

| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Refined Delta | M3 PSNR | M3 Delta | M0 LPIPS | Refined LPIPS | M3 LPIPS | M0 Failure | Refined Failure | M3 Failure | Accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 28.2390 | 29.3713 | +1.1323 | 28.5703 | +0.3313 | 0.1760 | 0.1005 | 0.1580 | 0.6250 | 0.5156 | 0.6250 | 0.3125 |
| 4 | 0.10 | 30.3021 | 31.0858 | +0.7837 | 30.6832 | +0.3812 | 0.1013 | 0.0672 | 0.0862 | 0.4375 | 0.3594 | 0.4375 | 0.5000 |
| 7 | 0.08 | 31.8137 | 32.3996 | +0.5859 | 32.1952 | +0.3815 | 0.0590 | 0.0452 | 0.0509 | 0.2656 | 0.3125 | 0.2656 | 0.7188 |
| 13 | 0.05 | 33.4944 | 34.0448 | +0.5504 | 33.9501 | +0.4557 | 0.0311 | 0.0256 | 0.0270 | 0.2656 | 0.2812 | 0.2656 | 0.8438 |
| 19 | 0.04 | 34.0518 | 34.6172 | +0.5654 | 34.5079 | +0.4561 | 0.0277 | 0.0196 | 0.0211 | 0.2812 | 0.2031 | 0.2812 | 0.8281 |

#### 结果总结

相比 `EXP-S4-005`，该实验使用更大的 fixed split。Pure refined 在所有 SNR 上均提升 PSNR，提升范围为 `+0.5504` 到 `+1.1323` dB，LPIPS 也全部降低。经过 top-1 agreement fallback 后，M3 final PSNR 仍在所有 SNR 上高于 M0，提升范围为 `+0.3313` 到 `+0.4561` dB，M3 LPIPS 也全部低于 M0。

语义侧的关键约束也满足：M3 final failure 在所有 SNR 上都没有高于 M0。但这不是说 detector 已经完善。1 dB 和 4 dB 下 accept rate 分别只有 `0.3125` 和 `0.5000`，说明低 SNR 下 top-1 agreement detector 很保守；7/13 dB 下 refined failure 略高于 M0，但 fallback 将 M3 failure 压回 M0。

#### Semantic drift 观察

`refined_refinement_drift` 在 1/4/7/13/19 dB 分别为 `0.6875/0.5000/0.2812/0.1562/0.1719`。这说明较大的 residual refiner 虽然带来更明显视觉收益，但也更容易改变冻结分类器 top-1；当前 M3 的价值正是把这些变化门控掉。后续不能把 pure refined 直接包装成最终方法，必须保留 drift detector/fallback，或者改进 detector 降低 false reject。

#### 失败案例和样例

样例拼图位于：

- `outputs/EXP-S4-006/samples/snr_01db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_04db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_07db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_13db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-006/samples/snr_19db_original_m0_refined_m3final.png`

逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-006/per_sample.csv`。后续应优先从 1/4 dB 的 false reject 和 false accept 样本中整理 detector failure gallery。

#### 派生 gate error analysis

已运行：

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

该分析不跑模型、不联网，只读取 `outputs/EXP-S4-006/per_sample.csv` 和已有 PNG。它把 top-1 agreement gate 的结果拆成四类：

- `protective_reject`：M0 与原图 pseudo-label 一致，refined 改变了 top-1，gate 拒绝 refined。
- `missed_semantic_repair`：M0 与原图 pseudo-label 不一致，refined 与原图 pseudo-label 一致，但 gate 因 refined 不等于 M0 而拒绝。
- `accepted_wrong_same_as_m0`：refined 与 M0 一致，但二者都不等于原图 pseudo-label。
- `rejected_both_wrong`：M0/refined 都不等于原图 pseudo-label，且二者互不一致。

| SNR(dB) | N | Accept | Protective Reject | Missed Repair | Accepted Wrong Same As M0 | Rejected Both Wrong |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 20 (0.3125) | 9 (0.1406) | 16 (0.2500) | 5 (0.0781) | 19 (0.2969) |
| 4 | 64 | 32 (0.5000) | 7 (0.1094) | 12 (0.1875) | 3 (0.0469) | 13 (0.2031) |
| 7 | 64 | 46 (0.7188) | 7 (0.1094) | 4 (0.0625) | 6 (0.0938) | 7 (0.1094) |
| 13 | 64 | 54 (0.8438) | 3 (0.0469) | 2 (0.0312) | 10 (0.1562) | 5 (0.0781) |
| 19 | 64 | 53 (0.8281) | 2 (0.0312) | 7 (0.1094) | 9 (0.1406) | 2 (0.0312) |

关键解释：当前 gate 接受 refined 的条件是 `c(refined) == c(M0)`，因此在同一个冻结分类器口径下，M3 top-1 final failure 不会超过 M0 top-1 failure 是结构性保证。这是保守 gate 的优点，但还不能证明独立语义可靠性。分析显示 gate 保护了 28/320 个 M0-correct/refined-wrong 样本，同时错过了 41/320 个 refined 修复 M0 pseudo-label 的样本。下一版应考虑 top-k agreement、confidence margin 或 CLIP/caption 辅助，以减少 `missed_semantic_repair`，同时保留 `protective_reject`。

#### 派生 gate policy sweep

已运行：

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

该分析不训练模型、不下载数据或权重，使用本地 AlexNet 权重重新计算 original/M0/refined 的 top-5。被扫的 gate policy 只使用 M0/refined 的预测结果做 receiver-side decision；original pseudo top-1 只用于离线评价。

全局关键结果：

| Policy | Final Failure | Delta Failure vs top1 | Final PSNR | Delta PSNR vs top1 | Missed Repair | Accepted Repair | Accepted New Error | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `top1_equal` | 0.3750 | +0.0000 | 31.9814 | +0.0000 | 41 | 0 | 0 | 0.6406 |
| `top1_equal_or_refined_conf_gain_ge_0p05` | 0.3188 | -0.0563 | 32.0966 | +0.1153 | 20 | 21 | 3 | 0.7563 |
| `top1_equal_or_refined_conf_gain_ge_0p10` | 0.3406 | -0.0344 | 32.0532 | +0.0719 | 28 | 13 | 2 | 0.7156 |
| `top1_equal_or_refined_conf_gain_ge_0p20` | 0.3563 | -0.0188 | 32.0037 | +0.0223 | 35 | 6 | 0 | 0.6656 |
| `refined_top1_in_m0_top5` | 0.3406 | -0.0344 | 32.1944 | +0.2130 | 8 | 33 | 22 | 0.8938 |
| `any_top5_overlap` | 0.3406 | -0.0344 | 32.2773 | +0.2960 | 2 | 39 | 28 | 0.9781 |

解释：top-5 overlap 类策略能大幅减少 missed repair 并提高 PSNR，但 accepted new error 也显著增加，语义风险偏大。当前最均衡候选是 `top1_equal_or_refined_conf_gain_ge_0p05`：它在 1/4 dB 上明显降低 final failure，在 7/13/19 dB 上不明显恶化；但它仍产生 3 个 accepted new error，因此只能作为下一轮 gate 设计候选，不能直接作为最终 M3。

#### 派生 confidence-gain gate auxiliary audit and candidate outputs

已运行辅助语义审计：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_residual_gate_aux_semantics.py --device cuda:0
```

已将候选 gate 的 final PNG 落盘：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_materialize_residual_gate_policy.py
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/per_sample_audit.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/summary.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/new_accepts.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/accepted_new_errors.csv
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_gate_aux_audit/galleries/
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/per_sample.csv
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/summary.csv
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/exports/
outputs/analysis/exp_s4_006_conf_gain_gate_candidate_outputs/samples/
```

该审计仍保持 decision-time gate 为 receiver-side：候选策略只看 M0/refined 的冻结分类器预测和置信度，不看 original/caption。original 图像与 COCO captions 只用于离线辅助审计。审计使用本地 `outputs/cache/open_clip/ViT-B-32.pt` 和 `data/coco/annotations/captions_val2017.json`，不下载数据或权重。

全局关键结果：

| Metric | Value |
|---|---:|
| Candidate accept rate | 0.7563 |
| Newly accepted by candidate | 37 |
| Candidate final failure | 0.3188 |
| Baseline top-1 final failure | 0.3750 |
| Candidate delta PSNR vs top-1 | +0.1153 dB |
| Candidate delta PSNR vs M0 | +0.5164 dB |
| Candidate delta CLIP image-image vs top-1 | +0.0016 |
| Candidate delta caption CLIP vs top-1 | -0.0007 |
| Accepted repairs | 21 |
| Accepted new errors | 3 |

新增接受样本拆分：

| Subset | N | Delta PSNR | Delta CLIP | Delta caption | Aux both nonworse |
|---|---:|---:|---:|---:|---:|
| `new_accept_repair` | 21 | +1.0532 | +0.0205 | -0.0073 | 0.1429 |
| `new_accept_new_error` | 3 | +0.9838 | +0.0121 | -0.0058 | 0.0000 |
| `new_accept_both_wrong` | 13 | +0.9093 | +0.0038 | -0.0044 | 0.0769 |

解释：confidence-gain candidate 比原始 top-1 agreement gate 更积极，能把 missed repair 从 41 降到 20，并额外接受 21 个 pseudo-label repair；但它也引入 3 个 accepted new error。辅助语义信号是混合的：CLIP image-image 均值略升，但 caption CLIP 均值略降，且 3 个 accepted new error 都没有同时通过 image-image 与 caption 的 nonworse 检查。因此该策略已经可以作为下一轮 M3 候选输出进行视觉/held-out 审查，但不能直接登记为最终 M3/Ours。

#### 派生 held-out confidence-gain gate check

已运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_heldout_gate_check/per_sample.csv
outputs/analysis/exp_s4_006_heldout_gate_check/summary.csv
outputs/analysis/exp_s4_006_heldout_gate_check/new_accepts.csv
outputs/analysis/exp_s4_006_heldout_gate_check/accepted_new_errors.csv
outputs/analysis/exp_s4_006_heldout_gate_check/REPORT.md
outputs/analysis/exp_s4_006_heldout_gate_check/metadata.json
outputs/analysis/exp_s4_006_heldout_gate_check/exports/
outputs/analysis/exp_s4_006_heldout_gate_check/samples/
```

该复核不重训模型，只加载 `outputs/EXP-S4-006/checkpoints/best.pt`，在 `EXP-S4-006` 未使用的 `sample_000000.png` 到 `sample_000031.png` 上重新生成 refined、top-1 final 和 candidate final。该 split 对 `EXP-S4-006` 的 residual refiner 和 gate sweep 是 held-out，但仍属于同一个 COCO val export 和同一个 pseudo-label 评价口径，因此只能作为派生风险复核，不是最终 test 结论。

全局关键结果：

| Metric | Value |
|---|---:|
| Num images | 160 |
| Candidate accept rate | 0.7875 |
| Newly accepted by candidate | 19 |
| Candidate final failure | 0.2812 |
| Baseline top-1 final failure | 0.3250 |
| Candidate minus baseline failure | -0.0437 |
| Candidate final PSNR | 31.8609 dB |
| Candidate delta PSNR vs top-1 | +0.1007 dB |
| Candidate delta PSNR vs M0 | +0.5460 dB |
| Accepted repairs | 9 |
| Accepted new errors | 2 |

分 SNR 关键结果：

| SNR(dB) | M0 Failure | Refined Failure | Top-1 Failure | Candidate Failure | New Accept | Repair | New Error | Delta PSNR vs top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5312 | 0.3750 | 0.5312 | 0.4688 | 9 | 3 | 1 | +0.3132 |
| 4 | 0.4375 | 0.1875 | 0.4375 | 0.3750 | 6 | 3 | 1 | +0.1248 |
| 7 | 0.3750 | 0.2500 | 0.3750 | 0.2812 | 4 | 3 | 0 | +0.0654 |
| 13 | 0.1250 | 0.1875 | 0.1250 | 0.1250 | 0 | 0 | 0 | +0.0000 |
| 19 | 0.1562 | 0.1250 | 0.1562 | 0.1562 | 0 | 0 | 0 | +0.0000 |

解释：held-out 复核支持 confidence-gain candidate 的方向，尤其在 1/4/7 dB 能额外接受一批 repair 并降低 pseudo final failure；但仍出现 2 个 accepted new error，位于 1 dB 和 4 dB。`samples/accepted_new_error_review.png` 已固化这两个样本的 original / M0 / refined / top-1 final / candidate final 对照。当前结论是“候选 gate 可继续收紧”，不是“候选 gate 已通过”。

#### 派生 test-like confidence-gain gate check

已先扩展 M0 export：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s2_deepjscc_highres_export.py --config configs/s2_deepjscc_coco256_awgn.yaml --checkpoint outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt --device cuda:0 --snrs 1,4,7,13,19 --batch-size 16 --num-workers 4 --export-count 384 --output-dir outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384
```

然后运行：

```bash
python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_heldout_gate_eval.py --config configs/s5_residual_refiner_testlike_gate_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv
outputs/analysis/exp_s4_006_testlike_gate_check/summary.csv
outputs/analysis/exp_s4_006_testlike_gate_check/new_accepts.csv
outputs/analysis/exp_s4_006_testlike_gate_check/accepted_new_errors.csv
outputs/analysis/exp_s4_006_testlike_gate_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_gate_check/metadata.json
outputs/analysis/exp_s4_006_testlike_gate_check/exports/
outputs/analysis/exp_s4_006_testlike_gate_check/samples/
```

该复核不重训模型，只加载 `outputs/EXP-S4-006/checkpoints/best.pt`，在新导出的 `sample_000256.png` 到 `sample_000319.png` 上重新生成 refined、top-1 final 和 candidate final。该 split 没有参与 `EXP-S4-006` 的 refiner 训练、验证或此前 gate sweep；但仍属于同一个 COCO val subset 和同一个 pseudo-label 评价口径，因此只能作为 test-like 派生风险复核，不是最终 test 结论。

全局关键结果：

| Metric | Value |
|---|---:|
| Num images | 320 |
| Candidate accept rate | 0.7063 |
| Newly accepted by candidate | 26 |
| Candidate final failure | 0.4313 |
| Baseline top-1 final failure | 0.4719 |
| Candidate minus baseline failure | -0.0406 |
| Candidate final PSNR | 32.2374 dB |
| Candidate delta PSNR vs top-1 | +0.0814 dB |
| Candidate delta PSNR vs M0 | +0.4927 dB |
| Accepted repairs | 17 |
| Accepted new errors | 4 |

分 SNR 关键结果：

| SNR(dB) | M0 Failure | Refined Failure | Top-1 Failure | Candidate Failure | New Accept | Repair | New Error | Delta PSNR vs top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.6562 | 0.5000 | 0.6562 | 0.5938 | 11 | 6 | 2 | +0.2120 |
| 4 | 0.5469 | 0.4375 | 0.5469 | 0.4375 | 11 | 8 | 1 | +0.1443 |
| 7 | 0.4688 | 0.4062 | 0.4688 | 0.4375 | 2 | 2 | 0 | +0.0238 |
| 13 | 0.3281 | 0.2500 | 0.3281 | 0.3281 | 2 | 1 | 1 | +0.0269 |
| 19 | 0.3594 | 0.3125 | 0.3594 | 0.3594 | 0 | 0 | 0 | +0.0000 |

解释：test-like split 上方向仍复现，candidate final failure 比 top-1 gate 低 `0.0406`，PSNR 高 `+0.0814` dB，并额外接受 17 个 pseudo-label repair；但 accepted new error 增至 4 个。`samples/accepted_new_error_review.png` 显示其中既有真实语义漂移风险，也有 AlexNet pseudo-label 本身较吵的样本。结论仍应保守：raw confidence-gain gate 是有收益但不安全的候选，不能写成最终 M3。

#### 派生 confidence-gain CLIP veto sweep

已运行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_conf_gain_clip_veto.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/per_sample_with_clip.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/joint_policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/galleries/
```

该派生分析读取 validation 的 `per_sample_audit.csv` 和 held-out 的 `per_sample.csv`，用本地 OpenCLIP ViT-B/32 只计算 receiver-side `CLIP(M0, refined)`。Original pseudo-label 仍只用于离线评价 final failure，不参与 veto 决策。

全局关键结果：

| Policy | Validation failure | Held-out failure | Validation repair | Held-out repair | Validation new error | Held-out new error | Sum delta PSNR vs top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `top1_equal` | 0.3750 | 0.3250 | 0 | 0 | 0 | 0 | +0.0000 |
| `top1_equal_or_refined_conf_gain_ge_0p05` | 0.3188 | 0.2812 | 21 | 9 | 3 | 2 | +0.2159 |
| `top1_equal_or_conf_gain_0p05_clip_m0_refined_ge_0p98` | 0.3719 | 0.3187 | 1 | 1 | 0 | 0 | +0.0073 |

解释：`CLIP(M0, refined) >= 0.98` 是当前扫描中能在 validation 和 held-out 同时清零 accepted new error、且不完全退回 top-1 的最保守阈值。它挡掉了 raw confidence-gain 的 5 个 accepted new error，但也挡掉了 28/30 个 repair，因此收益几乎被压平。这个结果说明单一 CLIP image-image veto 可作安全参考，但不够作为最终 M3；下一步需要 SNR-calibrated threshold、classifier ensemble 或 receiver-side risk predictor。

#### 派生 SNR-calibrated confidence-gain CLIP veto

已运行：

```bash
python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_calibrate_conf_gain_clip_veto_by_snr.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/calibrated_schedules.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/independent_threshold_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/monotonic_schedule_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_clip_veto_snr_calibration/galleries/
```

该派生分析只读取 `outputs/analysis/exp_s4_006_conf_gain_clip_veto_sweep/per_sample_with_clip.csv`，不训练模型、不联网、不重算 CLIP。阈值只在 validation split 上选择，再到 held-out split 上做风险复核。扫描网格包含 `no_veto`、`0.90/0.92/0.94/0.96/0.97/0.98/0.985/0.99/0.995` 和 `top1_only`；monotonic schedule 额外约束 `threshold(1 dB) >= threshold(4 dB) >= threshold(7 dB) >= threshold(13 dB) >= threshold(19 dB)`，对应低 SNR 语义控制不弱于高 SNR。

校准得到的 schedule：

| Policy | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB |
|---|---:|---:|---:|---:|---:|
| `fixed_clip_ge_0p98` | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| `snr_independent_calibrated` | 0.96 | no_veto | 0.98 | no_veto | no_veto |
| `snr_monotonic_calibrated` | 0.98 | 0.98 | 0.98 | no_veto | no_veto |

全局关键结果：

| Split | Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | `top1_equal` | 0.3750 | +0.0000 | 31.9814 | +0.0000 | 0 | 0 |
| validation | `raw_conf_gain` | 0.3187 | -0.0563 | 32.0966 | +0.1153 | 21 | 3 |
| validation | `fixed_clip_ge_0p98` | 0.3719 | -0.0031 | 31.9852 | +0.0038 | 1 | 0 |
| validation | `snr_independent_calibrated` | 0.3438 | -0.0312 | 32.0346 | +0.0533 | 10 | 0 |
| validation | `snr_monotonic_calibrated` | 0.3719 | -0.0031 | 31.9873 | +0.0059 | 1 | 0 |
| heldout | `top1_equal` | 0.3250 | +0.0000 | 31.7602 | +0.0000 | 0 | 0 |
| heldout | `raw_conf_gain` | 0.2812 | -0.0437 | 31.8609 | +0.1007 | 9 | 2 |
| heldout | `fixed_clip_ge_0p98` | 0.3187 | -0.0062 | 31.7637 | +0.0035 | 1 | 0 |
| heldout | `snr_independent_calibrated` | 0.3063 | -0.0187 | 31.7985 | +0.0383 | 4 | 1 |
| heldout | `snr_monotonic_calibrated` | 0.3187 | -0.0062 | 31.7637 | +0.0035 | 1 | 0 |

解释：independent per-SNR calibration 在 validation 上比全局 `0.98` 更有用，能在 0 accepted new error 条件下保留 10 个 repair；但它选择了 4 dB `no_veto`，既违反当前 SNR-aware semantic-control 的单调纪律，也在 held-out 上漏出 1 个 accepted new error。monotonic schedule 在 held-out 上安全，但只保留 1 个 repair，基本退回全局 `0.98` 的保守状态。因此，单一 `CLIP(M0, refined)` 标量阈值即使按 SNR 校准，也不足以作为最终 M3；后续应优先做 classifier ensemble 或 receiver-side risk predictor。

#### 派生 receiver-side confidence-gain risk rule sweep

已运行：

```bash
python3 scripts/s5_sweep_conf_gain_risk_rules.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_conf_gain_risk_rules.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/rule_candidates.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_by_snr.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/selected_rule.json
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/REPORT.md
outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/galleries/
```

该派生分析只读取已有 validation/held-out 的 `CLIP(M0, refined)` CSV 和 M0/refined top-k classifier CSV，不训练模型、不联网、不重算 CLIP。规则只使用 receiver-side 特征：`CLIP(M0, refined)`、M0/refined top-5 overlap、M0 top-1 在 refined top-5 中的 rank、M0 top-1 margin、refined top-1 margin。Original pseudo-label 仍只用于 validation 规则选择和离线 held-out 风险复核。

选中的规则：

```text
baseline top-1 agreement 仍直接接受 refined。
对 confidence-gain 新增接受样本：
  要求 clip_sim_m0_refined >= 0.90；
  无 top-5 overlap 最小要求；
  若 m0_top1_rank_in_refined_top5 <= 2
     且 m0_top1_margin <= 0.07
     且 refined_top1_margin >= 0.05，
     则触发 shadow veto，回退 M0。
```

直觉：当 M0 的 top-1 label 在 refined 中仍是非常靠前的候选，而 M0 自身 top-1 margin 很弱、refined top-1 margin 又明显变强时，这类“分类边界被推过头”的样本更容易是假修复或新错。该规则把这个 shadow pattern 作为风险信号。

全局关键结果：

| Split | Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | `top1_equal` | 0.3750 | +0.0000 | 31.9814 | +0.0000 | 0 | 0 |
| validation | `raw_conf_gain` | 0.3187 | -0.0563 | 32.0966 | +0.1153 | 21 | 3 |
| validation | `fixed_clip_ge_0p98` | 0.3719 | -0.0031 | 31.9852 | +0.0038 | 1 | 0 |
| validation | `selected_risk_rule` | 0.3156 | -0.0594 | 32.0767 | +0.0953 | 19 | 0 |
| heldout | `top1_equal` | 0.3250 | +0.0000 | 31.7602 | +0.0000 | 0 | 0 |
| heldout | `raw_conf_gain` | 0.2812 | -0.0437 | 31.8609 | +0.1007 | 9 | 2 |
| heldout | `fixed_clip_ge_0p98` | 0.3187 | -0.0062 | 31.7637 | +0.0035 | 1 | 0 |
| heldout | `selected_risk_rule` | 0.2812 | -0.0437 | 31.8350 | +0.0748 | 7 | 0 |

解释：这是目前最强的 gate 候选。相比 raw confidence-gain，它在 held-out 上保留同样的 final failure 改善，同时把 2 个 accepted new error 清零；相比全局 `CLIP >= 0.98`，它在 held-out 上从 1 个 repair 提升到 7 个 repair，并保留 `+0.0748` dB PSNR vs top-1 gate。`galleries/selected_risk_rule/heldout_vetoed_candidate_new_errors.png` 已固化被挡掉的两个 held-out 新错；`heldout_accepted_repairs.png` 固化 7 个被保留的 repair。

限制：该规则仍在 COCO pseudo-label validation 上选择，held-out 也只是同一 COCO val export 的未用样本段，不是最终 test split。它可以作为下一版 M3 gate 候选，但不能直接写成最终结论。

#### 派生 selected risk-rule final PNG materialization

已运行：

```bash
python3 scripts/s5_materialize_risk_rule_policy.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_materialize_risk_rule_policy.py
```

输出：

```text
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/per_sample.csv
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/summary.csv
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/REPORT.md
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/metadata.json
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/exports/{validation,heldout}/snr_XXdb/final/
outputs/analysis/exp_s4_006_risk_rule_candidate_outputs/samples/
```

该派生流程只读取 `outputs/analysis/exp_s4_006_conf_gain_risk_rule_sweep/policy_decisions.csv`，筛选 `policy == selected_risk_rule` 的 480 条决策，并按 `accept_refined` 从已有 M0/refined PNG 复制 final 输出；不训练、不联网、不重算 CLIP 或分类器。`summary.csv` 同时写入 top-1 gate 的 per-sample reference，方便核对 final failure 和 PSNR 增量。

关键结果：

| Split | Images | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error | Shadow Veto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | 320 | 0.3156 | -0.0594 | 32.0767 | +0.0953 | 19 | 0 | 5 |
| heldout | 160 | 0.2812 | -0.0437 | 31.8350 | +0.0748 | 7 | 0 | 5 |

说明：这是 risk-rule sweep 的 artifact 固化，不是新实验结论；作用是把当前最强 M3 gate 候选变成可复查的 final PNG/CSV/report，为后续正式 split 或更大 held-out 复核做准备。

#### 派生 selected risk-rule classifier ensemble audit

已检查代理变量，当前环境存在 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY=http://127.0.0.1:17890`。本次按项目流量规则清空代理变量，从 PyTorch/torchvision 官方 model zoo 直连下载缺失的 ResNet18 和 MobileNetV3-Small ImageNet 权重，规模约 `44.7MB + 9.83MB`；未下载数据集或 diffusion 模型。

已运行：

```bash
python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --device cuda:0 --allow-download
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --device cuda:0 --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/per_model_per_sample.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/model_summary.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/vote_summary.csv
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/REPORT.md
outputs/analysis/exp_s4_006_risk_rule_classifier_ensemble_audit/galleries/
```

该派生流程固定使用已经 materialize 的 `selected_risk_rule` 决策，不重新搜索 gate，也不把 ensemble 用作 receiver-side decision。它只用 AlexNet、ResNet18、MobileNetV3-Small 各自的 original top-1 pseudo-label 重新评价 M0/refined/selected-final 的 failure、repair 和 accepted-new-error 风险。

按分类器拆分：

| Classifier | Split | Selected Failure | Delta vs M0 | Repair | New Error |
|---|---|---:|---:|---:|---:|
| AlexNet | validation | 0.3156 | -0.0594 | 19 | 0 |
| AlexNet | heldout | 0.2812 | -0.0437 | 7 | 0 |
| ResNet18 | validation | 0.3688 | -0.0187 | 24 | 18 |
| ResNet18 | heldout | 0.4000 | -0.0062 | 8 | 7 |
| MobileNetV3-Small | validation | 0.4313 | -0.0563 | 28 | 10 |
| MobileNetV3-Small | heldout | 0.3562 | +0.0125 | 7 | 9 |

按样本投票：

| Split | Images | Any new-error vote | Majority new-error vote | Any repair vote | Majority repair vote |
|---|---:|---:|---:|---:|---:|
| validation | 320 | 26 | 2 | 64 | 7 |
| heldout | 160 | 15 | 1 | 17 | 4 |

解释：这个结果把边界说清楚了。`selected_risk_rule` 在 AlexNet pseudo-label 口径下确实是当前最强 gate 候选，但并非跨语义模型安全：ResNet18 和 MobileNetV3-Small 都能发现额外 accepted-new-error 风险，且有 3 个样本得到多数票 new-error。它仍可作为候选，但后续必须加入 ensemble-aware veto、辅助语义 veto 或更正式 split 复核，不能把它直接写成最终 M3。

#### 派生 ensemble-risk 二级 veto sweep

已运行：

```bash
python3 scripts/s5_sweep_ensemble_risk_veto.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_ensemble_risk_veto.py
```

输出：

```text
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/rule_candidates.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/selected_rule.json
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/metadata.json
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/REPORT.md
outputs/analysis/exp_s4_006_ensemble_risk_veto_sweep/galleries/
```

该派生流程固定使用 `selected_risk_rule` 的 480 条决策和 classifier ensemble audit 的离线投票标签，不重训、不联网、不下载、不重算分类器。搜索阶段只在 validation 上用 ensemble 多数票 new-error 作为风险约束；规则本身只使用接收端可得特征，包括 refined top-1 margin、refined 相对 M0 的 confidence gain、M0 top-1 margin 和 selected-risk-rule 的接受类型。

选中的二级 veto：

```text
在 selected_risk_rule 已接受 refined 的样本上：
  若它是 new_accept_vs_top1 且 refined_top1_margin <= 0.005，则额外 veto；
  若它是 top1-equal accept，且 refined_conf_gain_vs_m0 <= 0.05，
     且 m0_top1_margin >= 0.10，则额外 veto。
```

关键结果：

| Split | Extra Veto | Remaining Majority New Error | Remaining Any New Error | Remaining Majority Repair | Remaining Any Repair | Delta PSNR vs selected |
|---|---:|---:|---:|---:|---:|---:|
| validation | 96 | 0 | 16 | 5 | 40 | -0.1834 dB |
| heldout | 58 | 0 | 8 | 4 | 14 | -0.2538 dB |

按分类器复核最终 failure：

| Split | Classifier | Candidate Failure | Delta vs selected | Repair | New Error |
|---|---|---:|---:|---:|---:|
| validation | AlexNet | 0.3187 | +0.0031 | 18 | 0 |
| validation | ResNet18 | 0.3719 | +0.0031 | 14 | 9 |
| validation | MobileNetV3-Small | 0.4688 | +0.0375 | 13 | 7 |
| heldout | AlexNet | 0.2812 | +0.0000 | 7 | 0 |
| heldout | ResNet18 | 0.3938 | -0.0062 | 5 | 3 |
| heldout | MobileNetV3-Small | 0.3313 | -0.0250 | 7 | 5 |

解释：该规则能把 `selected_risk_rule` 暴露出的 validation/held-out 多数票 new-error 从 `2/1` 清到 `0/0`，说明 ensemble 暴露的高置信风险样本可以被简单 receiver-side 特征部分捕捉。但代价很明显：额外 veto 数达到 validation/held-out `96/58`，多数票 repair 只剩 `5/4`，且 any-new-error 仍有 `16/8`。因此它是收紧 gate 的风险分析结果，不是最终 M3；后续更合理的方向是把这个二级 veto 当作 conservative safety upper-bound，再训练/选择更细的 receiver-side risk predictor 或扩展正式 split。

#### 派生 receiver-side risk score sweep

已运行：

```bash
python3 scripts/s5_sweep_receiver_risk_score.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_sweep_receiver_risk_score.py --overwrite
```

输出：

```text
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/score_candidates.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/policy_decisions.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/policy_summary.csv
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/selected_score.json
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/metadata.json
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/REPORT.md
outputs/analysis/exp_s4_006_receiver_risk_score_sweep/galleries/
```

该派生流程固定使用 `selected_risk_rule` 的 480 条决策和 classifier ensemble audit 的离线投票标签，不重训、不联网、不下载、不重算分类器。它扫描 12 个透明 receiver-side risk score 模板和 1930 个 validation 阈值候选，目标是验证是否能用更少 extra veto 替代上一节的保守二级 veto。score 特征只来自接收端可得的 AlexNet/CLIP/top-k 派生量，如 `CLIP(M0, refined)`、top-5 overlap、refined top-1 是否偏离 M0 top-k、confidence gain 和 margin。

repair-pref validation 目标选中的分数：

```text
risk_score = low_top5_overlap + refined_top1_not_in_m0_safe_rank + low_clip
threshold = 0.444446
```

关键结果：

| Split | Extra Veto | Remaining Majority New Error | Remaining Any New Error | Remaining Majority Repair | Remaining Any Repair | Delta PSNR vs selected |
|---|---:|---:|---:|---:|---:|---:|
| validation | 48 | 0 | 17 | 4 | 36 | -0.1396 dB |
| heldout | 26 | 1 | 9 | 2 | 8 | -0.1581 dB |

按分类器复核最终 failure：

| Split | Classifier | Candidate Failure | Delta vs selected | Repair | New Error |
|---|---|---:|---:|---:|---:|
| validation | AlexNet | 0.3594 | +0.0437 | 5 | 0 |
| validation | ResNet18 | 0.3719 | +0.0031 | 17 | 12 |
| validation | MobileNetV3-Small | 0.4469 | +0.0156 | 18 | 5 |
| heldout | AlexNet | 0.3187 | +0.0375 | 1 | 0 |
| heldout | ResNet18 | 0.3938 | -0.0062 | 6 | 4 |
| heldout | MobileNetV3-Small | 0.3625 | +0.0063 | 3 | 6 |

解释：该 score 在 validation 上用更少 extra veto 清零多数票 new-error，但 held-out 漏掉 1 个多数票 new-error，即 19 dB `sample_000031.png`；该样本的 M0/refined AlexNet top-1 同为 `komondor`，且 top-k/CLIP 接收端分数很低风险，说明浅层 receiver-side score 很难覆盖所有跨模型语义风险。进一步查看 `score_candidates.csv`，若要求 validation 和 held-out 同时清零多数票 new-error，repair-pref 最好的 score 模板需要额外 veto validation/held-out `143/81` 张，PSNR held-out 相对 `selected_risk_rule` 回吐 `-0.3511` dB，比上一节的保守二级 veto 还重。因此这一步是负/部分结果：少 veto risk score 目前不够稳，不能作为最终 M3。

#### 派生 test-like frozen risk-rule check

已运行：

```bash
python3 scripts/s5_apply_testlike_risk_rules.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_apply_testlike_risk_rules.py --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_risk_rule_check/per_sample_with_clip.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_decisions.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_by_snr.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_check/REPORT.md
outputs/analysis/exp_s4_006_testlike_risk_rule_check/metadata.json
outputs/analysis/exp_s4_006_testlike_risk_rule_check/exports/
outputs/analysis/exp_s4_006_testlike_risk_rule_check/galleries/
```

该派生流程固定使用已经在 validation/held-out 阶段选出的 `selected_risk_rule` 和保守 ensemble-risk veto，不在 test-like split 上重新搜索阈值。它读取 `outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv`，重新计算本地 `CLIP(M0, refined)`，并把 final PNG materialize 到 `outputs/analysis/exp_s4_006_testlike_risk_rule_check/exports/`。本次不联网、不下载，CLIP 权重来自本地 cache。

全局关键结果：

| Policy | Final Failure | Delta Failure vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair | New Error | New Accept | Vetoed Raw New Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `top1_equal` | 0.4719 | +0.0000 | 32.1560 | +0.0000 | 0 | 0 | 0 | 4 |
| `raw_conf_gain` | 0.4313 | -0.0406 | 32.2374 | +0.0814 | 17 | 4 | 26 | 0 |
| `fixed_clip_ge_0p98` | 0.4688 | -0.0031 | 32.1636 | +0.0076 | 2 | 1 | 3 | 3 |
| `selected_risk_rule` | 0.4437 | -0.0281 | 32.1995 | +0.0434 | 10 | 1 | 15 | 3 |
| `selected_risk_rule_plus_ensemble_veto` | 0.4437 | -0.0281 | 32.0092 | -0.1468 | 10 | 1 | 14 | 3 |

解释：冻结的 `selected_risk_rule` 在 test-like split 上仍有迁移收益：相比 raw confidence-gain，它把 accepted new error 从 4 降到 1，同时保留 10 个 pseudo-label repair 和 `+0.0434` dB PSNR vs top-1 gate。但它没有清零风险。剩余 accepted new error 是 13 dB `sample_000312.png`：original/M0 AlexNet top-1 为 `ear`，refined 为 `seat belt`，`CLIP(M0, refined)=0.9950`，M0 top-1 在 refined top-k 中 rank=3，因此旧 shadow-margin 规则和保守 ensemble veto 都没有触发。视觉样例显示该 case 也包含明显 pseudo-label 噪声，因此它应被记录为辅助语义风险，而不是最终真值错误。

保守 ensemble-risk veto 在 test-like 上没有降低 accepted new error 或 final failure，却额外 veto 93 张、PSNR 相比 `selected_risk_rule` 回吐 `-0.1902` dB，说明它作为 safety upper-bound 太保守，不能直接作为第一版 M3。当前结论进一步支持：浅层 receiver-side 标量/规则已经接近瓶颈，下一步应转向更正式语义标签/ensemble test-like 审计，或在 residual CNN 训练/选择阶段加入 semantic-risk-aware 约束。

#### 派生 test-like classifier-ensemble audit

已运行：

```bash
python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_audit_risk_rule_classifier_ensemble.py --config configs/s5_testlike_risk_rule_classifier_ensemble_audit_exp_s4_006.yaml --device cuda:0
```

输出：

```text
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/per_model_per_sample.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/per_sample_votes.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/model_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/vote_summary.csv
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/REPORT.md
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/metadata.json
outputs/analysis/exp_s4_006_testlike_risk_rule_classifier_ensemble_audit/galleries/
```

该派生流程固定使用 test-like `selected_risk_rule` 决策，不重新搜索 gate，也不把 ensemble 用作 receiver-side decision。脚本从 `policy_decisions.csv` 中只抽取 `selected_risk_rule` 320 行，使用本地 AlexNet、ResNet18、MobileNetV3-Small 权重分别以各自 original top-1 pseudo-label 评价 M0/refined/selected-final 的 failure、repair 和 accepted-new-error 风险。本次不联网、不下载，正式运行前 dry-run 确认 3 个分类器权重均已在本地 cache。

按分类器拆分：

| Classifier | Split | Selected Failure | Delta vs M0 | Repair | New Error |
|---|---|---:|---:|---:|---:|
| AlexNet | testlike | 0.4437 | -0.0281 | 10 | 1 |
| ResNet18 | testlike | 0.4344 | -0.0563 | 31 | 13 |
| MobileNetV3-Small | testlike | 0.5406 | -0.0719 | 32 | 9 |

按样本投票：

| Split | Images | Any new-error vote | Majority new-error vote | Any repair vote | Majority repair vote |
|---|---:|---:|---:|---:|---:|
| testlike | 320 | 23 | 0 | 58 | 12 |

按 SNR 的 any-new-error vote 为 1/4/7/13/19 dB `3/4/6/6/4`，majority new-error vote 全部为 0。23 个 any-new-error 都只有单模型投票，其中 ResNet18 13 个、MobileNetV3-Small 9 个、AlexNet 1 个；AlexNet 的唯一风险仍是 13 dB `sample_000312.png`。

解释：test-like ensemble 审计比 validation/held-out 的跨模型结果更温和：没有 majority-vote accepted new error，说明 frozen `selected_risk_rule` 在 test-like 上没有暴露出明显多数票语义灾难；但 any-model new-error 仍有 23 张，且 ResNet18/MobileNetV3-Small 下 selected accepted new error 分别为 13/9 个。因此它只能说明当前 rule 有一定迁移性和辅助 repair 信号，不能说明跨模型安全。下一步更应该补带标签 clean-correct 评估或把 semantic-risk-aware 约束进入 residual CNN 训练/选择，而不是继续只在 AlexNet/CLIP/top-k 标量上调阈值。

#### 派生 test-like COCO object CLIP clean-correct eval

该派生流程不训练、不联网、不下载，读取 `outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_decisions.csv`、`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/source_manifest.json`、`data/coco/annotations/instances_val2017.json` 和本地 `outputs/cache/open_clip/ViT-B-32.pt`。它先用 COCO instance 面积得到 dominant object label，再用 OpenCLIP ViT-B/32 对 80 个 COCO object prompt 做 zero-shot 分类；只有 dominant label 面积占比满足阈值，且 original 的 CLIP top-1 与 dominant label 一致、prob/margin 过阈值的样本进入辅助 clean-correct 子集。

运行命令：

```bash
python3 scripts/s5_coco_object_clip_clean_eval.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_coco_object_clip_clean_eval.py --device cuda:0
```

输出路径：`outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/`

clean-correct 总表：

| Policy | Rows | Final Failure GT | Delta vs Top1 | Final PSNR | Delta PSNR vs Top1 | Repair GT | New Error GT |
|---|---:|---:|---:|---:|---:|---:|---:|
| top1_equal | 135 | 0.0815 | +0.0000 | 31.7925 | +0.0000 | 1 | 2 |
| raw_conf_gain | 135 | 0.0815 | +0.0000 | 31.8457 | +0.0533 | 1 | 2 |
| fixed_clip_ge_0p98 | 135 | 0.0815 | +0.0000 | 31.8042 | +0.0117 | 1 | 2 |
| selected_risk_rule | 135 | 0.0815 | +0.0000 | 31.8182 | +0.0257 | 1 | 2 |
| selected_risk_rule_plus_ensemble_veto | 135 | 0.0741 | -0.0074 | 31.6197 | -0.1727 | 0 | 0 |

解释：64 个 test-like 原图中有 55 个满足 dominant object 面积规则，其中 27 个 original 被 CLIP 判为 clean-correct，形成每个 policy 135 行统计。该辅助 GT-like 口径下，`selected_risk_rule` 没有比 top-1 gate 降低 final failure，也没有减少 GT-like accepted new error，只提供小幅 PSNR 增益；保守 ensemble veto 可把 GT-like new error 清零并稍降 final failure，但也清掉 repair 且 PSNR 低于 top-1。这个结果进一步确认当前浅层 gate 的语义保护和 restoration 收益存在硬 tradeoff。它比 ImageNet pseudo-label 更贴 COCO 物体标注，但仍依赖 CLIP zero-shot 和 dominant-object 假设，不能包装成最终监督真值指标。

#### 复现备注

`EXP-S4-006` 本体不联网、不下载模型或数据，只读取已有正式 M0 export 和本地 AlexNet/LPIPS 权重。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`。本体 `summary.csv` 有 5 个 SNR 汇总行，`per_sample.csv` 有 320 个 eval 样本行，`train_history.csv` 有 40 个 epoch 行。

#### 下一步

围绕 `EXP-S4-006` 继续收敛 detector：`selected_risk_rule` final PNG、classifier ensemble audit、ensemble-risk 二级 veto sweep、receiver-side risk score sweep、raw confidence-gain test-like 复核、frozen risk-rule test-like 复核、test-like classifier-ensemble audit 和 COCO object CLIP clean-correct 辅助诊断都已完成。test-like 证据说明 raw confidence-gain 有 PSNR/repair 收益但会引入 accepted new error；`selected_risk_rule` 可挡掉其中 3/4 个 AlexNet new error，且在 test-like ensemble 下没有 majority-vote new error，但仍有 23 个 any-model accepted new-error vote；在 COCO-object clean-correct 口径下它仍有 2 个 GT-like new error。下一步应停止只在同一套浅层 receiver-side 标量上拧阈值，优先做真正带监督标签的 clean-correct 评估，或在 residual CNN 训练阶段加入 semantic-risk-aware 约束。当前证据显示 raw confidence-gain、全局 CLIP veto、SNR-calibrated scalar CLIP veto、当前 AlexNet-tuned selected rule、保守 ensemble-risk veto 和少 veto risk score 都不能直接定为第一版 M3。

### EXP-S4-007：SNR-conditioned pixel residual diffusion pilot

- 日期：2026-07-06
- 项目版本：`4f4eefb5f08096e5efdd57d6019b97683ea7648b`
- 仓库地址：`https://github.com/daiqizai/Channel-Adaptive-Semantic-Drift-Controlled-Diffusion-JSCC.git`
- 第三方 commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 阶段：S5 Adaptive Control / latent-free residual diffusion design probe
- 方法：SNRConditionedPixelResidualDiffusionPilot
- 数据集：COCO2017 `val2017` subset export
- 数据 split / 样本 ID：
  - 输入 export：`outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
  - train：`sample_000032.png` 到 `sample_000111.png`，每个 SNR 80 张，共 400 对 M0/original
  - eval：`sample_000192.png` 到 `sample_000207.png`，每个 SNR 16 张，共 80 对 M0/original
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；sampling seed 1234
- checkpoint：`outputs/train/s2_deepjscc_coco256_awgn_snr7_cbr017/checkpoints/best.pt`
- diffusion checkpoint：`outputs/EXP-S4-007/checkpoints/best.pt`
- config：`outputs/EXP-S4-007/config.yaml`
- 运行命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_diffusion_pilot.py --device cuda:0`
- 关键源码：`scripts/s5_residual_diffusion_pilot.py`, `scripts/s5_residual_refiner_pilot.py`, `src/cadsd_jscc/metrics.py`
- 输出路径：`outputs/EXP-S4-007/`
- 状态：完成；负结果；不是最终 M2/M3/Ours

#### 方法设置

- 模型：小型 SNR-conditioned pixel residual DDPM，参数量 77,187
- 输入：noisy normalized residual + M0 reconstruction + SNR map + timestep map，共 8 通道
- 目标：学习 `(original - M0) / residual_gate` 后 clamp 到 `[-1, 1]` 的 residual
- diffusion：20 timesteps，linear beta `0.0001 -> 0.02`，epsilon prediction，deterministic DDIM sampling 20 steps
- residual gate：1/4/7/13/19 dB 使用 `0.12/0.10/0.08/0.05/0.04`
- 训练：20 epoch，batch size 16，128x128 random crop，epsilon loss + 0.1 x0 loss
- semantic failure handling：若 `c(refined) == c(M0)` 则接受，否则 fallback 到 M0；detector 不看原图

#### 指标

All-subset，使用原图 ImageNet top-1 作为离线 pseudo-label 评价。`Refined Failure` 是 refined 相对原图 pseudo top-1 的 failure；`M3 Failure` 是 top-1 agreement fallback 后最终输出的 failure。

| SNR(dB) | Gate | M0 PSNR | Refined PSNR | Refined Delta | M3 PSNR | M3 Delta | M0 LPIPS | Refined LPIPS | M3 LPIPS | M0 Failure | Refined Failure | M3 Failure | Accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.12 | 28.6189 | 21.4555 | -7.1634 | 27.2033 | -1.4156 | 0.1893 | 0.5139 | 0.2472 | 0.6250 | 0.8750 | 0.6250 | 0.2500 |
| 4 | 0.10 | 30.6216 | 23.1373 | -7.4843 | 28.9598 | -1.6618 | 0.1138 | 0.4236 | 0.1785 | 0.5625 | 0.8125 | 0.5625 | 0.2500 |
| 7 | 0.08 | 32.0814 | 24.9932 | -7.0882 | 29.4795 | -2.6019 | 0.0673 | 0.3338 | 0.1641 | 0.4375 | 0.7500 | 0.4375 | 0.3750 |
| 13 | 0.05 | 33.6698 | 28.2494 | -5.4204 | 31.5131 | -2.1567 | 0.0335 | 0.1999 | 0.1142 | 0.2500 | 0.6250 | 0.2500 | 0.4375 |
| 19 | 0.04 | 34.1760 | 29.7543 | -4.4217 | 32.0758 | -2.1002 | 0.0284 | 0.1489 | 0.0969 | 0.1875 | 0.5000 | 0.1875 | 0.5625 |

#### 结果总结

该实验回答了“是否只要把 diffusion 挪到像素 residual 域就会变好”：当前朴素设计不成立。训练 loss 确实下降，eval epsilon loss 最低出现在 epoch 14，但最终 DDIM sampling 得到的 residual 噪声化很强，refined PSNR 在所有 SNR 上显著低于 M0，下降范围为 `-4.4217` 到 `-7.4843` dB；LPIPS 也全部变差。

top-1 agreement fallback 仍有语义保护作用：同一冻结 AlexNet 口径下，M3 final failure 在每个 SNR 上都回到 M0 failure。但这不是有效增强，因为 M3 final PSNR 仍比 M0 低 `-1.4156/-1.6618/-2.6019/-2.1567/-2.1002` dB，M3 LPIPS 也全部高于 M0。

#### Semantic drift 观察

Pure refined 的 pseudo failure 明显高于 M0：1/4/7/13/19 dB 分别为 `0.8750/0.8125/0.7500/0.6250/0.5000`。`refined_refinement_drift` 也很高，分别为 `0.7500/0.7500/0.6250/0.5625/0.4375`，说明随机残差采样即使有 M0/SNR/timestep conditioning，也容易改变冻结分类器 top-1。当前 gate 把这些变化大多回退，但代价是 final 图像仍被 accepted refined 样本拖低。

#### 失败案例和样例

样例拼图位于：

- `outputs/EXP-S4-007/samples/snr_01db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_04db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_07db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_13db_original_m0_refined_m3final.png`
- `outputs/EXP-S4-007/samples/snr_19db_original_m0_refined_m3final.png`

逐样本 detector 决策、pseudo-label 和 false accept/reject 标记见 `outputs/EXP-S4-007/per_sample.csv`。

#### 复现备注

本实验不联网、不下载模型或数据，只读取已有正式 M0 export 和本地 AlexNet/LPIPS 权重。运行命令显式清空代理变量，`metrics.json` 中记录 `proxy_environment_present: []`、`download_note: No model or data download is required`、`git_dirty_state: clean`。`summary.csv` 有 5 个 SNR 汇总行，`per_sample.csv` 有 80 个 eval 样本行，`train_history.csv` 有 20 个 epoch 行。

#### 下一步

不要把该 naive residual DDPM 作为正向 M2/M3 路线。若继续研究 diffusion，应改成 restoration-aware 的条件短链：从 M0 或 residual CNN 输出附近初始化，只做小幅 residual correction；或以 `EXP-S4-006` residual CNN 作为 mean / teacher，再训练低噪声 conditional diffusion。第一版论文闭环仍应优先收敛 `EXP-S4-006` 的 residual CNN + semantic gate。

### ANALYSIS-S6-004：Minimal Closure Report with Shrink M3

- 日期：2026-07-07
- 项目版本：`371833e` + uncommitted report script/config at run time
- 阶段：S6 minimal closure derived analysis
- 方法：MinimalClosureReportWithHeldoutShrinkM3
- 数据集：COCO2017 `val2017` subset outputs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB；M1 negative reference 仅覆盖 `[1, 7, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 输入：
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export/metrics.json`
  - `outputs/EXP-S2-002/metrics.json`
  - `outputs/EXP-S4-006/summary.csv`
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_risk_rule_check/policy_summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_coco_object_clip_clean_eval/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；派生汇总，不训练、不推理、不下载

#### 核心表

| Method | Role | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | Status |
|---|---|---|---:|---:|---:|---|
| M0-DeepJSCC-HR | baseline | formal_coco512 | 0.0000 | N/A | N/A | usable M0 baseline |
| M1-BlindDiffusion-SDImg2Img | negative reference | exp_s2_002_16img_per_snr | -14.7485 | +0.3877 | N/A | failed due quality and semantic drift |
| M2-SNRConditionedPixelResidualRestoration | positive restoration anchor | exp_s4_006_eval | +0.7235 | -0.0274 | 0.3344 | positive quality, needs semantic handling |
| M3-ResidualRestorationTop1Fallback | conservative first M3 | exp_s4_006_eval | +0.4011 | -0.0104 | 0.3750 | safe conservative closure on pseudo-label metric |
| M3-ResidualRestorationTop1ShrinkFallback | stronger conservative M3 candidate | validation selected / frozen held-out/test-like | +0.4584 | -0.0153 | 0.3750 | best conservative M3 candidate so far; held-out/test-like PSNR delta +0.4689/+0.4552 and new error 0/0 |
| M3-SelectedRiskRuleCandidate | test-like candidate gate | testlike_policy | N/A | N/A | 0.4437 | not final; leaves AlexNet/GT-like risk |

#### 结果总结

该汇总把当前第一版闭环口径固定下来：M1 使用 SD img2img 空 prompt 是明确负结果，只作为 blind diffusion reference；M2 应写成 SNR-conditioned pixel residual restoration，是当前正向质量提升来源；M3 的保守第一版采用 top-1 semantic fallback，可以在 `EXP-S4-006` pseudo-label 口径下保证 final failure 不高于 M0，同时保留平均 `+0.4011` dB PSNR 和 `-0.0104` LPIPS 收益。

刷新后的报告新增 `M3-ResidualRestorationTop1ShrinkFallback`：validation-only schedule 选择 `1 dB alpha=0.5`、其余 SNR `alpha=0.75`，validation 平均 PSNR delta 为 `+0.4584` dB，LPIPS delta 为 `-0.0153`；冻结到 held-out 后，平均 PSNR delta 为 `+0.4689` dB，比 full-strength top-1 fallback 高 `+0.0236` dB，accepted new error 为 0；冻结到 test-like 后，平均 PSNR delta 为 `+0.4552` dB，比 full-strength top-1 fallback 高 `+0.0439` dB，accepted new error 为 0。因此它是当前最强保守 M3 候选，但仍是 pseudo-label/held-out/test-like 证据，不是监督标签安全证明。

`selected_risk_rule` 继续作为候选/消融：test-like AlexNet 口径下有 1 个 accepted new error，COCO-object clean-correct 口径下仍有 2 个 GT-like new error；保守 ensemble veto 可清 COCO-object new error，但 PSNR 相比 top-1 为 `-0.1727` dB，过于保守。

#### 复现备注

该流程只读已有本地 outputs，不重新运行模型或分类器。正式运行时清空代理变量，metadata 中记录 `proxy_environment_present: []`。生成文件包括 `REPORT.md`、6 个 CSV 和 4 张 figure：

- `outputs/analysis/minimal_closure_report/REPORT.md`
- `outputs/analysis/minimal_closure_report/method_closure_summary.csv`
- `outputs/analysis/minimal_closure_report/residual_per_snr_quality_semantics.csv`
- `outputs/analysis/minimal_closure_report/blind_diffusion_negative_reference.csv`
- `outputs/analysis/minimal_closure_report/residual_shrink_policy_tradeoff.csv`
- `outputs/analysis/minimal_closure_report/testlike_policy_tradeoff.csv`
- `outputs/analysis/minimal_closure_report/coco_object_clean_correct_tradeoff.csv`
- `outputs/analysis/minimal_closure_report/figures/`

#### 下一步

围绕这个闭环继续推进：优先把 residual strength / alpha 选择前移到 semantic-risk-aware residual training 或 validation model selection；若继续研究 diffusion，只做以 M2/refined/M0 附近初始化的短链 conditional residual correction。

### ANALYSIS-S6-005：EXP-S4-006 Held-Out Frozen Residual Shrink Schedule Check

- 日期：2026-07-07
- 项目版本：`371833e` + uncommitted generic split script/config at run time
- 阶段：S6 held-out derived analysis
- 方法：FrozenHeldoutResidualShrinkScheduleCheck
- 数据集：COCO2017 `val2017` held-out `sample_000000`-`sample_000031`
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_residual_shrink_schedule.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_heldout_residual_shrink_schedule_check_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_apply_residual_shrink_schedule.py`, `scripts/s6_residual_shrink_selection.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json`
  - `outputs/analysis/exp_s4_006_heldout_gate_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_heldout_gate_check/exports/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/`
- 输出路径：`outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/`
- 状态：完成；frozen schedule held-out 复核，不训练、不下载、不调参

#### 指标

| Policy | Delta PSNR vs M0 | Delta LPIPS vs M0 | Final Failure | Accept | Repair | Accepted New Error |
|---|---:|---:|---:|---:|---:|---:|
| top1_full_strength | +0.4454 | -0.0113 | 0.3250 | 0.6687 | 0 | 0 |
| validation_top1_shrink_schedule | +0.4689 | -0.0150 | 0.3250 | 0.7625 | 0 | 0 |
| always_full_strength | +0.6853 | -0.0223 | 0.2250 | 1.0000 | 26 | 10 |
| validation_always_m0_failure_constrained_schedule | +0.5292 | -0.0217 | 0.2375 | 0.8000 | 17 | 3 |

#### 结果总结

Validation 选出的 top-1 shrink schedule 在 held-out split 继续成立：平均 PSNR delta 从 full-strength top-1 fallback 的 `+0.4454` dB 提升到 `+0.4689` dB，LPIPS delta 从 `-0.0113` 改到 `-0.0150`，pseudo final failure 仍等于 M0，accepted new error 为 0。always-accept 两条路线仍有 10/3 个 accepted new error，不能作为最终 M3。

#### 复现备注

该流程只读取已有 held-out refined PNG 和 frozen validation schedule，不重新训练 residual refiner，不运行 diffusion，不下载模型或数据。metadata 中记录 `proxy_environment_present: []` 和 `split_name: held-out`。

#### 下一步

把 validation、held-out、test-like 三段 shrink 证据并入 minimal closure report；后续把 alpha/残差幅度约束前移到 residual CNN 训练或 validation model selection。

### ANALYSIS-S6-006：Residual Shrink M3 Artifact Gallery

- 日期：2026-07-07
- 项目版本：`c19cc0f` + uncommitted artifact-gallery script/config at run time
- 阶段：S6 derived artifact / failure-case organization
- 方法：ResidualShrinkM3ArtifactGallery
- 数据集：COCO2017 `val2017` validation、held-out、test-like residual shrink outputs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_residual_shrink_artifact_gallery_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_residual_shrink_gallery.py
python3 scripts/s6_make_residual_shrink_gallery.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_residual_shrink_gallery.py --overwrite
```

- 关键源码：`scripts/s6_make_residual_shrink_gallery.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv`
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/summary.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv`
- 输出路径：`outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/`
- 状态：完成；只整理已有 CSV/PNG，不训练、不运行 diffusion、不重算分类器、不下载、不调参

#### 指标

| Split | M3 Delta PSNR | M3 Delta LPIPS | M3 New Error | Safe Accept | Protective Reject | Rejected Good | Always Full New Error | Always Constrained New Error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | +0.4584 | -0.0153 | 0 | 183 | 17 | 34 | 28 | 19 |
| held-out | +0.4689 | -0.0150 | 0 | 102 | 6 | 19 | 10 | 3 |
| test-like | +0.4552 | -0.0152 | 0 | 156 | 13 | 44 | 25 | 12 |

#### 结果总结

该派生 artifact 把 validation、held-out、test-like 三段 residual shrink 证据合并到一个可引用目录。`M3-ResidualRestorationTop1ShrinkFallback` 在三段上 accepted new error 均为 0，同时提供 safe accept、protective reject、rejected good candidate 和 unsafe always-accept new-error 的样例 sheet。它进一步明确了当前 M3 的性质：保守质量增强，而不是冒险追求 repair 数。

Always-accept 仍作为负对照：full strength 在 validation/held-out/test-like 上分别有 28/10/25 个 accepted new error；validation-constrained always-accept 仍有 19/3/12 个 accepted new error，不能写成最终 M3。

#### 复现备注

正式运行时清空代理变量，metadata 中记录 `proxy_environment_present: []`。输出包括：

- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/REPORT.md`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/policy_summary.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/case_counts.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/case_index.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_artifact_gallery/samples/`

#### 下一步

把这些样例用于第一版 failure-case / reliability 小节；方法侧继续把 residual alpha/幅度控制前移到 residual CNN 训练、validation model selection 或短链 conditional residual diffusion。

### ANALYSIS-S6-007：Adaptive Residual Alpha Policy

- 日期：2026-07-07
- 项目版本：`fbcfe72` + uncommitted adaptive-alpha script/config at run time
- 阶段：S6 derived policy / residual strength control
- 方法：AdaptiveResidualAlphaPolicy
- 数据集：COCO2017 `val2017` validation、held-out、test-like residual alpha candidates
- 数据 split / 样本 ID：
  - validation：`sample_000192`-`sample_000255`，64 images/SNR
  - held-out：`sample_000000`-`sample_000031`，32 images/SNR
  - test-like：`sample_000256`-`sample_000319`，64 images/SNR
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_adaptive_residual_alpha_policy_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_adaptive_residual_alpha_policy.py
python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_adaptive_residual_alpha_policy.py --device cuda:0
```

- 关键源码：`scripts/s6_apply_adaptive_residual_alpha_policy.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv`
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/candidates/`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_heldout_residual_shrink_schedule_check/candidates/`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/candidates/`
- 输出路径：`outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/`
- 状态：完成；只读取已有 alpha candidate PNG、本地 AlexNet 和 LPIPS 权重，不训练、不运行 diffusion、不重新生成 residual、不下载、不在 held-out/test-like 上调参

#### 指标

| Split | Policy | Delta PSNR | Delta LPIPS | Failure Delta | Accept Rate | Mean Alpha | Repair | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | top1_full_strength | +0.4011 | -0.0104 | +0.0000 | 0.6406 | 1.0000 | 0 | 0 | 45 |
| validation | fixed_validation_top1_shrink_schedule | +0.4584 | -0.0153 | +0.0000 | 0.7438 | 0.7111 | 0 | 0 | 45 |
| validation | adaptive_max_top1_consistent_alpha | +0.5584 | -0.0189 | +0.0000 | 0.9062 | 0.8457 | 0 | 0 | 45 |
| validation | always_full_strength | +0.7235 | -0.0274 | -0.0406 | 1.0000 | 1.0000 | 41 | 28 | 4 |
| held-out | top1_full_strength | +0.4454 | -0.0113 | +0.0000 | 0.6687 | 1.0000 | 0 | 0 | 31 |
| held-out | fixed_validation_top1_shrink_schedule | +0.4689 | -0.0150 | +0.0000 | 0.7625 | 0.7131 | 0 | 0 | 31 |
| held-out | adaptive_max_top1_consistent_alpha | +0.5664 | -0.0174 | +0.0000 | 0.9187 | 0.8605 | 0 | 0 | 31 |
| held-out | always_full_strength | +0.6853 | -0.0223 | -0.1000 | 1.0000 | 1.0000 | 26 | 10 | 5 |
| test-like | top1_full_strength | +0.4113 | -0.0116 | +0.0000 | 0.6250 | 1.0000 | 0 | 0 | 70 |
| test-like | fixed_validation_top1_shrink_schedule | +0.4552 | -0.0152 | +0.0000 | 0.7063 | 0.7102 | 0 | 0 | 70 |
| test-like | adaptive_max_top1_consistent_alpha | +0.5691 | -0.0201 | +0.0000 | 0.8906 | 0.8482 | 0 | 0 | 70 |
| test-like | always_full_strength | +0.7180 | -0.0270 | -0.0906 | 1.0000 | 1.0000 | 54 | 25 | 16 |

Adaptive policy 的 per-SNR PSNR delta：

| Split | 1 dB | 4 dB | 7 dB | 13 dB | 19 dB |
|---|---:|---:|---:|---:|---:|
| validation | +0.6850 | +0.5843 | +0.4802 | +0.5129 | +0.5294 |
| held-out | +0.6843 | +0.6055 | +0.4704 | +0.5143 | +0.5573 |
| test-like | +0.7739 | +0.6078 | +0.4638 | +0.4754 | +0.5246 |

#### 结果总结

`adaptive_max_top1_consistent_alpha` 在每个样本上从 `alpha=1.0/0.75/0.5/0.25` 中选择最大且 candidate top-1 与 M0 top-1 一致的 residual 强度，否则回退 M0。该规则不使用原图，只使用接收端已有 M0、alpha candidates 和冻结 AlexNet 的 top-1 一致性。

它在 validation/held-out/test-like 上把 PSNR delta 提升到 `+0.5584/+0.5664/+0.5691` dB，明显强于固定 per-SNR shrink schedule 的 `+0.4584/+0.4689/+0.4552` dB，并且在同一 AlexNet pseudo-label 口径下 accepted new error 保持 `0/0/0`。always-accept 仍然质量更高但有 `28/10/25` 个 new error，继续作为负对照。

需要特别记录的是：adaptive policy 没有产生 repair，且 missed repair 为 `45/31/70`。因此它是当前最强的保守质量增强候选，不是语义修复方法。下一步应把这种 per-sample alpha 选择前移到 residual CNN 的训练目标、validation model selection 或短链 conditional residual diffusion 的幅度控制里，而不是继续只做离线后验选择。

#### 复现备注

正式运行时清空代理变量，未下载模型或数据。metadata 记录 `proxy_environment_present: []`；由于脚本/config 在运行时尚未提交，`git_dirty_state` 为 `dirty`。输出包括：

- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/REPORT.md`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/metadata.json`
- `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/samples/`

#### 下一步

把 adaptive alpha policy 写入 M3 方法候选：短期可作为 `M3-AdaptiveResidualAlphaTop1Fallback` 的派生方案；中期应训练一个 receiver-side alpha/risk predictor 或在 residual CNN 中加入 semantic-risk-aware amplitude loss，使方法不依赖离线枚举 alpha candidates。

### ANALYSIS-S6-008：Minimal Closure Report With Adaptive Alpha M3

- 日期：2026-07-07
- 项目版本：`bcfc1f1` + uncommitted closure-report script/config at run time
- 阶段：S6 derived closure report
- 方法：MinimalClosureReportWithAdaptiveAlphaM3
- 数据集：COCO2017 `val2017` existing outputs and analysis CSVs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_minimal_closure_report.py
python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 新增输入：`outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；只读取已有 metrics/CSV，不训练、不运行 diffusion、不重算分类器、不下载

#### 指标

| Method | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | New Error | Status |
|---|---|---:|---:|---:|---:|---|
| M3-ResidualRestorationTop1Fallback | validation | +0.4011 | -0.0104 | 0.3750 | 0 | conservative first closure |
| M3-ResidualRestorationTop1ShrinkFallback | validation / held-out / test-like | +0.4584 / +0.4689 / +0.4552 | -0.0153 / -0.0150 / -0.0152 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | fixed schedule candidate |
| M3-AdaptiveResidualAlphaTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5664 / +0.5691 | -0.0189 / -0.0174 / -0.0201 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | strongest conservative candidate |

#### 结果总结

本轮刷新把 `ANALYSIS-S6-007` 的 adaptive alpha policy 并入最小闭环报告。`outputs/analysis/minimal_closure_report/REPORT.md` 现在明确区分：

- `M3-ResidualRestorationTop1Fallback`：保守第一版闭环；
- `M3-ResidualRestorationTop1ShrinkFallback`：固定 per-SNR schedule 消融/备选；
- `M3-AdaptiveResidualAlphaTop1Fallback`：当前最强保守质量增强候选；
- `M3-SelectedRiskRuleCandidate`：有 repair 但仍有 new-error 风险，不能作为最终安全方法。

新增输出包括 `adaptive_residual_alpha_policy_tradeoff.csv` 和 `figures/adaptive_residual_alpha_policy_tradeoff.png`。报告仍保留 caveat：adaptive alpha 不使用原图，但还是后验枚举 alpha candidates 的 receiver-side policy，还不是带 learned amplitude/risk control 的 residual CNN。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。由于脚本/config 在运行时尚未提交，metadata 中 `git_dirty_state` 为 `dirty`。

#### 下一步

论文口径上可把 adaptive alpha 作为当前 M3 主候选；方法侧下一步应把该 per-sample alpha 选择前移到训练/模型选择流程，例如训练 receiver-side alpha predictor、把 residual amplitude loss 加入 residual CNN，或设计从 M0/refined 附近初始化的短链 conditional residual diffusion。

### ANALYSIS-S6-009：Two-Stage Residual Alpha Policy

- 日期：2026-07-07
- 项目版本：`9cacff5` + local script/config at run time
- 阶段：S6 deployability ablation
- 方法：TwoStageResidualAlphaPolicy
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha decisions
- 数据 split：validation `320` 行、held-out `160` 行、test-like `320` 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_two_stage_residual_alpha_policy_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_two_stage_residual_alpha_policy.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_two_stage_residual_alpha_policy.py --device cuda:0
```

- 关键源码：`scripts/s6_apply_two_stage_residual_alpha_policy.py`
- 输入：
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
- 输出路径：`outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/`
- 状态：完成；只读取已有 adaptive alpha 决策表和 final 图，不训练、不运行 diffusion、不重算分类器、不下载；LPIPS 省略以避免外部权重加载

#### 策略

```text
full_then_fixed_schedule:
  first try top1_full_strength
  if alpha=1.0 candidate top-1 equals M0 top-1, accept full strength
  otherwise use fixed_validation_top1_shrink_schedule with the same top-1 gate
  otherwise fallback to M0
```

#### 指标

| Split | Delta PSNR | Final Failure Delta | Accept | Full Accept | Fallback Stage | Fallback Accept When Used | New Error | Missed Repair |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | +0.4831 | +0.0000 | 0.7469 | 0.6406 | 0.3594 | 0.2957 | 0 | 45 |
| held-out | +0.5009 | +0.0000 | 0.7625 | 0.6687 | 0.3312 | 0.2830 | 0 | 31 |
| test-like | +0.4875 | +0.0000 | 0.7250 | 0.6250 | 0.3750 | 0.2667 | 0 | 70 |

对比：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| `top1_full_strength` | +0.4011 | +0.4454 | +0.4113 | 0/0/0 |
| `fixed_validation_top1_shrink_schedule` | +0.4584 | +0.4689 | +0.4552 | 0/0/0 |
| `full_then_fixed_schedule` | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Two-stage policy 用最多两次 candidate 检查，质量上稳定优于 fixed schedule，但没有追上 exhaustive adaptive alpha。它的价值是证明可以把“残差强度控制”向更少候选、更接近接收端部署的策略压缩，同时保持同一 AlexNet pseudo-label 口径下 accepted new error 为 0。

该策略仍没有 repair，missed repair 仍为 `45/31/70`，因此仍是保守质量增强，不是语义修复。下一步如果继续这条线，应训练 receiver-side alpha predictor 或把 alpha/risk 控制并入 residual CNN，而不是继续增加后验枚举规则。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。本轮曾有一次 ad hoc 指标探针误触发 LPIPS/AlexNet 临时权重下载；下载进程已停止，`/tmp/alpha_twostage_cache` 已删除，未使用任何该探针结果。正式脚本默认不加载 LPIPS，避免再次触发外部权重加载。

### ANALYSIS-S6-010：Minimal Closure Report With Two-Stage Alpha Ablation

- 日期：2026-07-07
- 项目版本：`9cacff5` + local script/config at run time
- 阶段：S6 derived closure report
- 方法：MinimalClosureReportWithTwoStageAlphaAblation
- 数据集：COCO2017 `val2017` existing outputs and analysis CSVs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_minimal_closure_report.py scripts/s6_apply_two_stage_residual_alpha_policy.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 新增输入：`outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；只读取已有 metrics/CSV，不训练、不运行 diffusion、不重算分类器、不下载

#### 指标

| Method | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | New Error | Status |
|---|---|---:|---:|---:|---:|---|
| M3-ResidualRestorationTop1Fallback | validation | +0.4011 | -0.0104 | 0.3750 | 0 | conservative first closure |
| M3-ResidualRestorationTop1ShrinkFallback | validation / held-out / test-like | +0.4584 / +0.4689 / +0.4552 | -0.0153 / -0.0150 / -0.0152 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | fixed schedule candidate |
| M3-TwoStageResidualAlphaTop1Fallback | validation / held-out / test-like | +0.4831 / +0.5009 / +0.4875 | N/A | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | deployability ablation |
| M3-AdaptiveResidualAlphaTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5664 / +0.5691 | -0.0189 / -0.0174 / -0.0201 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | strongest conservative candidate |

#### 结果总结

本轮刷新把 `ANALYSIS-S6-009` 的 two-stage alpha 消融并入最小闭环报告。`outputs/analysis/minimal_closure_report/REPORT.md` 现在明确区分：

- `M3-AdaptiveResidualAlphaTop1Fallback`：当前最强保守质量增强候选；
- `M3-TwoStageResidualAlphaTop1Fallback`：少候选检查的部署折中，质量高于 fixed schedule 但低于 exhaustive adaptive alpha；
- `M3-ResidualRestorationTop1ShrinkFallback`：固定 per-SNR schedule 消融/备选；
- `M3-SelectedRiskRuleCandidate`：有 repair 但仍有 new-error 风险，不能作为最终安全方法。

新增输出包括 `two_stage_residual_alpha_policy_tradeoff.csv`。报告仍保留 caveat：two-stage alpha 的 LPIPS 被刻意省略，不能把空 LPIPS 项与其他策略的 LPIPS 数值横向比较。

#### 下一步

方法侧下一步不应继续堆后验策略，而应把 alpha 选择变成可学习或训练期约束：训练 receiver-side alpha/risk predictor，或在 residual CNN/短链 conditional residual diffusion 中加入 semantic-risk-aware residual amplitude 控制。

### ANALYSIS-S6-011：Receiver Alpha Predictor

- 日期：2026-07-09
- 项目版本：`4a466e8` + local script/config at run time
- 阶段：S6 learned deployability pilot
- 方法：ReceiverAlphaPredictor
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha decisions and candidate PNGs
- 数据 split：validation `320` 行用于训练，held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_receiver_alpha_predictor_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_receiver_alpha_predictor.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --device cuda:0
```

- 关键源码：`scripts/s6_train_receiver_alpha_predictor.py`
- 输入：
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - validation/held-out/test-like residual alpha candidate PNG roots
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_receiver_alpha_predictor/`
- 状态：完成；validation-only 训练小型 tabular predictor，不训练图像模型、不运行 diffusion、不重新生成 residual、不下载；LPIPS 省略以避免外部权重加载

#### 方法

Predictor 使用接收端可见特征：

- SNR 数值和 SNR one-hot；
- M0 top-1 confidence；
- full-strength candidate top-1 confidence、confidence delta/ratio、是否与 M0 top-1 一致；
- M0 图像均值/方差/edge proxy；
- full-strength residual 的 MAE/RMSE/P95/max/signed mean。

训练目标是 validation 上 `adaptive_max_top1_consistent_alpha` 的 `selected_alpha` pseudo target。评估时只对预测 alpha 的候选图运行冻结 AlexNet；若 candidate top-1 与 M0 top-1 不一致，则回退 M0。

#### 指标

| Split | Delta PSNR | Failure Delta | Accept | Target Alpha Acc | Pred Alpha <= Oracle | New Error | Missed Repair |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | +0.5584 | +0.0000 | 0.9062 | 1.0000 | 1.0000 | 0 | 45 |
| held-out | +0.5099 | +0.0000 | 0.8375 | 0.7500 | 0.8813 | 0 | 31 |
| test-like | +0.4871 | +0.0000 | 0.7906 | 0.7000 | 0.8563 | 0 | 70 |

对比：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| `fixed_validation_top1_shrink_schedule` | +0.4584 | +0.4689 | +0.4552 | 0/0/0 |
| `full_then_fixed_schedule` | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| `receiver_alpha_predictor_top1_fallback` | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Receiver alpha predictor 在 validation 上完全拟合 adaptive alpha pseudo target，并在 held-out 上略高于 two-stage；但 test-like 只与 two-stage 基本持平，仍明显低于 exhaustive adaptive alpha。它说明“学 alpha”方向有价值，但当前 tabular 特征不足以稳定复制 oracle adaptive alpha。由于最终仍用 top-1 consistency gate，accepted new error 维持 `0/0/0`，但 repair 仍为 0，missed repair 仍为 `45/31/70`。

结论：该结果应写成 learned deployability pilot，而不是当前最强 M3。下一步应把 alpha/risk 预测并入 residual CNN 训练或使用更强的 receiver-side 特征，而不是继续堆浅层后验规则。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。

### ANALYSIS-S6-012：Minimal Closure Report With Receiver Alpha Predictor

- 日期：2026-07-09
- 项目版本：`4a466e8` + local script/config at run time
- 阶段：S6 derived closure report
- 方法：MinimalClosureReportWithReceiverAlphaPredictor
- 数据集：COCO2017 `val2017` existing outputs and analysis CSVs
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_minimal_closure_report.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_make_minimal_closure_report.py scripts/s6_train_receiver_alpha_predictor.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_make_minimal_closure_report.py --overwrite
```

- 关键源码：`scripts/s6_make_minimal_closure_report.py`
- 新增输入：`outputs/analysis/exp_s4_006_receiver_alpha_predictor/summary.csv`
- 输出路径：`outputs/analysis/minimal_closure_report/`
- 状态：完成；只读取已有 metrics/CSV，不训练、不运行 diffusion、不重算分类器、不下载

#### 指标

| Method | Split | Mean Delta PSNR | Mean Delta LPIPS | Mean Failure | New Error | Status |
|---|---|---:|---:|---:|---:|---|
| M3-TwoStageResidualAlphaTop1Fallback | validation / held-out / test-like | +0.4831 / +0.5009 / +0.4875 | N/A | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | deployability ablation |
| M3-ReceiverAlphaPredictorTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5099 / +0.4871 | N/A | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | learned deployability pilot |
| M3-AdaptiveResidualAlphaTop1Fallback | validation / held-out / test-like | +0.5584 / +0.5664 / +0.5691 | -0.0189 / -0.0174 / -0.0201 | 0.3750 / 0.3250 / 0.4719 | 0 / 0 / 0 | strongest conservative candidate |

#### 结果总结

本轮刷新把 `ANALYSIS-S6-011` 的 receiver alpha predictor 并入最小闭环报告。`outputs/analysis/minimal_closure_report/REPORT.md` 现在把 alpha-control 线拆成：

- `M3-AdaptiveResidualAlphaTop1Fallback`：当前最强保守质量增强候选；
- `M3-ReceiverAlphaPredictorTop1Fallback`：learned 部署 pilot，held-out 略优于 two-stage，但 test-like 未超过 two-stage；
- `M3-TwoStageResidualAlphaTop1Fallback`：少候选检查的非学习部署消融；
- `M3-ResidualRestorationTop1ShrinkFallback`：固定 schedule 消融/备选。

新增输出包括 `receiver_alpha_predictor_tradeoff.csv`。报告仍保留 caveat：receiver predictor 是 validation pseudo-target 训练结果，LPIPS 被省略，不能作为 supervised semantic proof。

#### 下一步

停止在浅层后验 alpha 规则上继续细调；下一步应进入训练侧：在 residual CNN 中加入 alpha/risk head，或设计短链 conditional residual diffusion，从模型内部学习何时放大/收缩 residual。

### ANALYSIS-S6-013：Alpha-Head Residual Refiner Pilot

- 日期：2026-07-09
- 项目版本：`a7076eb` + local script/config at run time
- 阶段：S6 training-side alpha-control exploration
- 方法：AlphaHeadResidualRefinerPilot
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha pseudo targets
- 数据 split：validation `320` 行用于训练 alpha head，held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_pilot_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_pilot/`
- 状态：完成；加载 `EXP-S4-006` residual CNN，默认冻结 refiner，仅训练 alpha head；不运行 diffusion、不下载、不加载 LPIPS

#### 方法

Alpha head 附着在 `EXP-S4-006` residual refiner 的 feature map 上。正式运行中：

- residual CNN 从 `outputs/EXP-S4-006/checkpoints/best.pt` 加载；
- `head/body/tail` 冻结，只训练 `alpha_head`；
- 训练目标是 validation 上 `adaptive_max_top1_consistent_alpha` 的 `selected_alpha` pseudo target；
- 评估时预测一个 alpha，生成 `M0 + alpha * (full_refined - M0)`，再用冻结 AlexNet top-1 consistency gate 决定接受或回退 M0。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Alpha Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4011 | +0.0000 | 0.6406 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.3846 | +0.0000 | 0.7312 | 0.6687 | 0 | 37 |
| held-out | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6687 |  | 0 | 26 |
| held-out | alpha_head_predicted_top1_fallback | +0.3808 | +0.0000 | 0.7438 | 0.6500 | 0 | 21 |
| test-like | full_strength_top1_fallback | +0.4113 | +0.0000 | 0.6250 |  | 0 | 54 |
| test-like | alpha_head_predicted_top1_fallback | +0.3623 | +0.0000 | 0.7094 | 0.5844 | 0 | 44 |

对比当前 alpha-control 线：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| `alpha_head_predicted_top1_fallback` | +0.3846 | +0.3808 | +0.3623 | 0/0/0 |
| `full_strength_top1_fallback` | +0.4011 | +0.4454 | +0.4113 | 0/0/0 |
| `full_then_fixed_schedule` | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| `receiver_alpha_predictor_top1_fallback` | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Alpha-head pilot 没有超过 full-strength top-1 fallback，也明显低于 two-stage、receiver predictor 和 exhaustive adaptive alpha。它的价值是把 alpha 控制第一次接进 residual refiner 模型内部，并暴露了当前训练设计的瓶颈：validation target 中 `alpha=1.0` 占 `205/320`，而 alpha head 预测 `alpha=1.0` 达到 `280/320`，说明普通 CE 在类别不平衡下偏向 majority alpha。held-out/test-like 也有同样倾向。

结论：这是训练侧方向的部分负结果，不进入 minimal closure 主表，也不能作为新 M3。下一步应尝试 inverse-frequency alpha loss、unfreeze/refiner joint fine-tune，或者直接设计 semantic-risk-aware residual amplitude loss，而不是只在冻结 feature 上训普通分类头。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `a7076eb + local script/config`。

### ANALYSIS-S6-014：Weighted Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`594db31` + local script/config at run time
- 阶段：S6 training-side alpha-control exploration
- 方法：WeightedAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like adaptive-alpha pseudo targets
- 数据 split：validation `320` 行用于训练 alpha head，held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_weighted_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_weighted/`
- 状态：完成；加载 `EXP-S4-006` residual CNN，冻结 refiner，仅训练 alpha head；启用 tempered inverse-frequency CE weights；不运行 diffusion、不下载、不加载 LPIPS

#### 方法

本实验直接验证上一版 alpha-head 的主要怀疑点：pseudo target 类别不均衡。validation 训练目标分布为 `alpha=0.0/0.25/0.5/0.75/1.0 = 30/34/26/25/205`，因此训练中使用：

- `class_weighting: inverse_frequency`
- `class_weight_power: 0.5`
- `class_weight_normalize_mean: true`

得到 class weights `[1.1132, 1.0457, 1.1958, 1.2195, 0.4259]`。其余结构与 `ANALYSIS-S6-013` 一致：冻结 residual CNN，只训练 alpha head；评估时预测一个 alpha 候选，再用 AlexNet top-1 consistency gate 保护输出。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Alpha Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4011 | +0.0000 | 0.6406 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.3851 | +0.0000 | 0.8094 | 0.6375 | 0 | 25 |
| held-out | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6687 |  | 0 | 26 |
| held-out | alpha_head_predicted_top1_fallback | +0.3506 | +0.0000 | 0.7875 | 0.5750 | 0 | 16 |
| test-like | full_strength_top1_fallback | +0.4113 | +0.0000 | 0.6250 |  | 0 | 54 |
| test-like | alpha_head_predicted_top1_fallback | +0.3166 | +0.0000 | 0.7562 | 0.4969 | 0 | 36 |

对比上一版普通 CE：

| Policy | validation | held-out | test-like | Target Alpha Acc | New Error |
|---|---:|---:|---:|---:|---:|
| unweighted alpha head | +0.3846 | +0.3808 | +0.3623 | 0.6687 / 0.6500 / 0.5844 | 0/0/0 |
| weighted alpha head | +0.3851 | +0.3506 | +0.3166 | 0.6375 / 0.5750 / 0.4969 | 0/0/0 |
| full-strength top-1 fallback | +0.4011 | +0.4454 | +0.4113 | N/A | 0/0/0 |
| `adaptive_max_top1_consistent_alpha` | +0.5584 | +0.5664 | +0.5691 | N/A | 0/0/0 |

#### 结果总结

Weighted CE 把普通 CE 的 majority collapse 缓和了，但没有变成更好的 alpha policy。validation 上 unweighted 预测 `alpha=1.0` 为 `280/320`，weighted 降到 `223/320`，少数 alpha 预测明显增加；test-like 上 weighted 预测分布为 `0.0/0.25/0.5/0.75/1.0 = 64/21/5/21/209`，也比 unweighted 的 `39/5/0/9/267` 更分散。

问题是更分散不等于更优。weighted 版 accept rate 更高，accepted new error 仍为 0，但 held-out/test-like PSNR 明显低于 unweighted 和 full-strength top-1 fallback。这说明 alpha head 当前学到的是“少数类覆盖”，不是“何时某个 alpha 能带来最大质量收益且不引发 semantic drift”。冻结 residual feature 本身也可能没有足够信息区分 `0.25/0.5/0.75` 的边界。

结论：类别不均衡是症状之一，但不是主因。下一步不宜继续只调 CE 权重；应转向 benefit/risk-aware alpha 目标、联合微调 residual CNN，或在短链 conditional residual diffusion 中用 M0/refined 附近初始化并加入 identity/semantic-risk 约束。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `594db31 + local script/config`。

### ANALYSIS-S6-015：Benefit-Aware Alpha Predictor

- 日期：2026-07-09
- 项目版本：`050b0c2` + local script/config at run time
- 阶段：S6 receiver-side learned alpha-control exploration
- 方法：BenefitAwareAlphaPredictor
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练小型 predictor；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_benefit_alpha_predictor_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_receiver_alpha_predictor.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --config configs/s6_benefit_alpha_predictor_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_receiver_alpha_predictor.py --config configs/s6_benefit_alpha_predictor_exp_s4_006.yaml --device cuda:0 --overwrite
```

- 关键源码：`scripts/s6_train_receiver_alpha_predictor.py`
- 输入：
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/summary.csv`
  - `outputs/analysis/exp_s4_006_two_stage_residual_alpha_policy/summary.csv`
  - validation/held-out/test-like residual alpha candidate PNG roots
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_benefit_alpha_predictor/`
- 状态：完成；只训练小型 tabular predictor，不训练图像模型、不运行 diffusion、不下载、不加载 LPIPS

#### 方法

上一版 `ReceiverAlphaPredictor` 直接把 `adaptive_max_top1_consistent_alpha` 当 hard pseudo-label 分类。这个 follow-up 改为 utility soft labels：

- 对每个样本枚举 `alpha in [0.0, 0.25, 0.5, 0.75, 1.0]`；
- 若候选 alpha 的 AlexNet top-1 与 M0 top-1 一致，则 utility 为该候选相对 M0 的 PSNR delta；
- 若候选不满足 top-1 安全，则 utility 设为 `-2.0`；
- `alpha=0.0` 表示 fallback M0，utility 为 `0.0`；
- 用 temperature `0.20` 把 utility 转成 soft label 训练 predictor。

训练标签可用 validation 原图计算 PSNR，但 predictor 输入仍只包含接收端可见特征：SNR、M0/full candidate 的分类器置信度、full candidate 是否与 M0 top-1 一致，以及 M0 到 full candidate 的 residual 图像统计。评估时仍对预测 alpha 候选执行 top-1 fallback。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | benefit_alpha_predictor_top1_fallback | +0.5538 | +0.0000 | 0.8906 | 0.7188 | 0 | 45 |
| held-out | benefit_alpha_predictor_top1_fallback | +0.4474 | +0.0000 | 0.7562 | 0.3812 | 0 | 31 |
| test-like | benefit_alpha_predictor_top1_fallback | +0.4627 | +0.0000 | 0.7469 | 0.4250 | 0 | 70 |

对比当前 alpha-control 线：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| benefit-aware predictor | +0.5538 | +0.4474 | +0.4627 | 0/0/0 |
| receiver alpha predictor | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| full_then_fixed_schedule | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| adaptive_max_top1_consistent_alpha | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Benefit-aware 目标在 validation 上有效：PSNR delta `+0.5538` dB，几乎追上 exhaustive adaptive alpha 的 `+0.5584` dB。但它没有在 held-out/test-like 上迁移，分别只有 `+0.4474/+0.4627` dB，低于 two-stage 和上一版 receiver predictor。utility target 分布比原 adaptive pseudo target 更均衡，validation target 为 `0.0/0.25/0.5/0.75/1.0 = 30/34/35/115/106`，但 held-out/test-like target accuracy 只有 `0.3812/0.4250`。

结论：把 alpha 目标改成“安全前提下的质量收益”是更贴近问题的方向，但当前 tabular feature + 小 MLP 泛化不足。下一步不宜继续只换浅层 predictor loss；更合理的是把 benefit/risk 约束前移到 residual CNN joint fine-tune，或让模型内部特征直接预测 residual amplitude/risk。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 candidate PNG 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `050b0c2 + local script/config`。

### ANALYSIS-S6-016：Benefit-Aware Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`53b71b3` + local script/config at run time
- 阶段：S6 training-side residual alpha-control exploration
- 方法：BenefitAwareAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练 alpha head；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_benefit/`
- 状态：完成；冻结 residual CNN，仅训练 alpha head，不运行 diffusion、不下载、不加载 LPIPS

#### 方法

该实验复用 alpha-head residual refiner 流程，但把训练标签从 `adaptive_max_top1_consistent_alpha` hard pseudo target 换成上一轮 benefit predictor feature table 中的 `utility_target_alpha`。这些 utility target 使用 validation 原图构造：候选 alpha 必须满足 AlexNet top-1 与 M0 top-1 一致，安全候选按 PSNR delta 选最大收益，否则回退 M0。

模型输入和推理仍只使用接收端可见的 M0/SNR/refiner feature。评估阶段仍对 predicted-alpha candidate 使用冻结 AlexNet top-1 fallback，因此该实验是训练侧 alpha 控制探索，不是新的 M3 闭环。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4011 | +0.0000 | 0.6406 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.4251 | +0.0000 | 0.7812 | 0.5406 | 0 | 28 |
| held-out | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6687 |  | 0 | 26 |
| held-out | alpha_head_predicted_top1_fallback | +0.4192 | +0.0000 | 0.8000 | 0.4313 | 0 | 16 |
| test-like | full_strength_top1_fallback | +0.4113 | +0.0000 | 0.6250 |  | 0 | 54 |
| test-like | alpha_head_predicted_top1_fallback | +0.3530 | +0.0000 | 0.7406 | 0.4062 | 0 | 40 |

对比 alpha-control 线：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| benefit-aware alpha head | +0.4251 | +0.4192 | +0.3530 | 0/0/0 |
| unweighted alpha head | +0.3846 | +0.3808 | +0.3623 | 0/0/0 |
| weighted alpha head | +0.3851 | +0.3506 | +0.3166 | 0/0/0 |
| benefit-aware predictor | +0.5538 | +0.4474 | +0.4627 | 0/0/0 |
| receiver alpha predictor | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| full_then_fixed_schedule | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| adaptive_max_top1_consistent_alpha | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

#### 结果总结

Benefit-aware alpha head 比普通/weighted alpha-head 有部分进展。validation 从 `+0.3846/+0.3851` 提到 `+0.4251` dB，held-out 从 `+0.3808/+0.3506` 提到 `+0.4192` dB，accepted new error 仍为 `0/0/0`。但是它没有超过 receiver predictor、two-stage policy 或 exhaustive adaptive alpha，test-like 也低于普通 alpha-head。

预测分布显示模型仍没有学到细粒度 alpha 边界：validation predicted alpha 为 `0.0/0.25/0.5/0.75/1.0 = 35/0/10/154/121`，而 target 为 `30/34/35/115/106`；test-like predicted 为 `50/1/12/113/144`，target 为 `35/35/30/123/97`。模型几乎不预测 `alpha=0.25`，说明冻结 residual feature + alpha classifier 仍主要学到粗粒度 fallback/strong-refine，而不是 utility target 中的收益/风险排序。

结论：benefit/risk 目标本身有价值，但只把标签换到冻结 alpha head 上还不够。下一步应优先 joint fine-tune residual CNN，或把 semantic-risk-aware residual amplitude loss 直接放进 residual restoration 训练，而不是继续只换 alpha 分类标签。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `53b71b3 + local script/config`。

### ANALYSIS-S6-017：Benefit-Aware Joint Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`901420f` + local script/config at run time
- 阶段：S6 training-side residual alpha-control exploration
- 方法：BenefitAwareJointAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于 joint fine-tune；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_joint_benefit/`
- 状态：完成；解冻 residual CNN joint fine-tune，不运行 diffusion、不下载、不加载 LPIPS；负/诊断结果

#### 方法

该实验在 `ANALYSIS-S6-016` 的 benefit-aware alpha-head 基础上解冻 residual CNN，并新增训练损失：

- `soft_refiner_detach: false`：predicted soft-alpha reconstruction loss 反传到 residual CNN；
- `target_alpha_mse_weight: 100.0`：utility target alpha 对应的 refined 图像对 original 做 MSE；
- `full_mse_weight: 10.0`：保留一个弱 full-strength restoration anchor；
- `ce_weight: 0.30`：继续训练 alpha head 预测 utility alpha；
- `refiner_lr: 0.00005`，alpha head `lr: 0.001`。

评估阶段仍对 predicted-alpha candidate 使用冻结 AlexNet top-1 fallback。该实验测试的是“全量 unfreeze + benefit/risk alpha loss”是否能让 residual CNN 内部学到更好的 amplitude control。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.2530 | +0.0000 | 0.6656 |  | 0 | 45 |
| validation | alpha_head_predicted_top1_fallback | +0.3294 | +0.0000 | 0.8688 | 0.7719 | 0 | 17 |
| held-out | full_strength_top1_fallback | +0.2236 | +0.0000 | 0.7188 |  | 0 | 19 |
| held-out | alpha_head_predicted_top1_fallback | +0.2303 | +0.0000 | 0.8562 | 0.3875 | 0 | 8 |
| test-like | full_strength_top1_fallback | +0.1855 | +0.0000 | 0.6687 |  | 0 | 39 |
| test-like | alpha_head_predicted_top1_fallback | +0.1869 | +0.0000 | 0.8219 | 0.3719 | 0 | 21 |

对比 alpha-head 训练侧路线：

| Policy | validation | held-out | test-like | Target Acc | New Error |
|---|---:|---:|---:|---:|---:|
| frozen benefit alpha head | +0.4251 | +0.4192 | +0.3530 | 0.5406 / 0.4313 / 0.4062 | 0/0/0 |
| joint benefit alpha head | +0.3294 | +0.2303 | +0.1869 | 0.7719 / 0.3875 / 0.3719 | 0/0/0 |
| full-strength top-1 fallback before joint | +0.4011 | +0.4454 | +0.4113 | N/A | 0/0/0 |
| full-strength top-1 fallback after joint | +0.2530 | +0.2236 | +0.1855 | N/A | 0/0/0 |

#### 结果总结

Joint fine-tune 成功改善了 validation alpha 分类：target accuracy 从 frozen benefit alpha-head 的 `0.5406` 提升到 `0.7719`，预测分布也从几乎不用 `alpha=0.25` 变成 `0.0/0.25/0.5/0.75/1.0 = 28/24/23/127/118`。这说明解冻 shared feature 后，模型确实能更好地读出 utility alpha。

但图像 restoration anchor 被明显损伤。full-strength top-1 fallback 从原始 `+0.4011/+0.4454/+0.4113` dB 掉到 `+0.2530/+0.2236/+0.1855` dB；predicted-alpha final 也只有 `+0.3294/+0.2303/+0.1869` dB，低于 frozen benefit alpha-head。训练日志中 full MSE 从约 `0.000816` 升到约 `0.000872`，与最终 PSNR 下滑一致。

结论：benefit/risk 目标可以改善 alpha 分类，但全量 unfreeze 且 CE 仍占主导会破坏 residual restoration 表征。下一步应避免让分类目标直接改写 shared residual feature；更合理的是 partial fine-tune（只调 tail/amplitude/head）、更强 reconstruction-dominant objective、或在固定 residual feature 上学习单独的 amplitude/risk head。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `901420f + local script/config`。

### ANALYSIS-S6-018：Benefit-Aware Tail-Only Alpha-Head Residual Refiner

- 日期：2026-07-09
- 项目版本：`c69743a` + local script/config at run time
- 阶段：S6 training-side residual alpha-control exploration
- 方法：BenefitAwareTailAlphaHeadResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_joint_benefit_exp_s4_006.yaml --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_benefit/`
- 状态：完成；只训练 residual tail 与 alpha head，不运行 diffusion、不下载、不加载 LPIPS

#### 方法

该实验直接接续 `ANALYSIS-S6-017` 的负结果。全量 joint fine-tune 会破坏 shared residual feature，因此本轮只允许：

- `trainable_refiner_parts: [tail]`：冻结 head/body，只更新 residual tail；
- `alpha_head` 正常训练；
- `ce_weight: 0.05`：降低 alpha 分类目标对训练的主导性；
- `full_mse_weight: 100.0`：用 reconstruction-dominant loss 保护 full-strength restoration anchor；
- `soft_refiner_detach: false`、`soft_mse_weight: 25.0`、`target_alpha_mse_weight: 25.0`：让 predicted/target alpha reconstruction 对 tail 做温和幅度校准。

metadata 记录的可训练参数为：head `0/1776`、body `0/207840`、tail `1299/1299`、alpha head `3461/3461`。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4454 | +0.0000 | 0.6281 |  | 0 | 41 |
| validation | alpha_head_predicted_top1_fallback | +0.4749 | +0.0000 | 0.7531 | 0.5437 | 0 | 32 |
| held-out | full_strength_top1_fallback | +0.4820 | +0.0000 | 0.6687 |  | 0 | 27 |
| held-out | alpha_head_predicted_top1_fallback | +0.4552 | +0.0000 | 0.7937 | 0.4313 | 0 | 19 |
| test-like | full_strength_top1_fallback | +0.4259 | +0.0000 | 0.5938 |  | 0 | 53 |
| test-like | alpha_head_predicted_top1_fallback | +0.4061 | +0.0000 | 0.7312 | 0.4250 | 0 | 38 |

对比 alpha-head 训练侧路线：

| Policy | validation | held-out | test-like | Target Acc | New Error |
|---|---:|---:|---:|---:|---:|
| frozen benefit alpha head | +0.4251 | +0.4192 | +0.3530 | 0.5406 / 0.4313 / 0.4062 | 0/0/0 |
| full joint benefit alpha head | +0.3294 | +0.2303 | +0.1869 | 0.7719 / 0.3875 / 0.3719 | 0/0/0 |
| tail-only benefit alpha head | +0.4749 | +0.4552 | +0.4061 | 0.5437 / 0.4313 / 0.4250 | 0/0/0 |

#### 结果总结

Tail-only partial fine-tune 是训练侧正向阶段结果：它明显恢复了全量 joint 损伤的 restoration anchor，full-strength top-1 fallback 达到 `+0.4454/+0.4820/+0.4259` dB；predicted-alpha final 也达到 `+0.4749/+0.4552/+0.4061` dB，三段 accepted new error 均为 0。

该结果说明：上一轮失败不是 benefit/risk 目标本身无效，而是全量解冻让分类/target-alpha loss 改写了 shared residual feature。把可训练范围限制在 tail，并用 reconstruction-dominant loss 后，可以获得比冻结 benefit alpha-head 更好的泛化质量。

限制也明确：tail-only 仍低于 receiver predictor、two-stage policy 和后验 adaptive alpha，因此不能作为最终 M3。预测分布仍不使用 `alpha=0.25`（validation/held-out/test-like predicted counts for `0/0.25/0.5/0.75/1.0` 为 `27/0/9/159/125`、`20/0/0/66/74`、`42/0/8/123/147`），说明细粒度 alpha 边界还没有被学好。下一步若继续训练侧，应考虑显式 amplitude head、连续 alpha regression 或 validation model-selection loss，而不是单纯扩大解冻范围。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `c69743a + local script/config`。

### ANALYSIS-S6-019：Benefit-Aware Tail-Only Continuous-Alpha Residual Refiner

- 日期：2026-07-09
- 项目版本：`9b6f74a` + local script/config at run time
- 阶段：S6 training-side residual amplitude-control exploration
- 方法：BenefitAwareTailContinuousAlphaResidualRefiner
- 数据集：COCO2017 `val2017` validation / held-out / test-like alpha candidates
- 数据 split：validation `320` 行用于训练；held-out `160` 行和 test-like `320` 行只评估
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_train_alpha_head_residual_refiner.py
python3 scripts/s6_train_alpha_head_residual_refiner.py --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_benefit_exp_s4_006.yaml --dry-run
python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_train_alpha_head_residual_refiner.py --config configs/s6_alpha_head_residual_refiner_tail_regression_benefit_exp_s4_006.yaml --device cuda:0
```

- 关键源码：`scripts/s6_train_alpha_head_residual_refiner.py`
- 输入：
  - `outputs/EXP-S4-006/checkpoints/best.pt`
  - `outputs/analysis/exp_s4_006_adaptive_residual_alpha_policy/per_sample.csv`
  - `outputs/analysis/exp_s4_006_benefit_alpha_predictor/features.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/`
- 状态：完成；只训练 residual tail 与 continuous alpha head，不运行 diffusion、不下载、不加载 LPIPS

#### 方法

该实验接续 `ANALYSIS-S6-018` 的限制：离散 alpha 分类仍几乎不预测 `alpha=0.25`。本轮保持 head/body 冻结、只训练 residual tail 和 alpha head，但把 alpha head 输出从 5 类 logits 改为单个连续 alpha：

- `model.alpha_mode: regression`：alpha head 输出经 `sigmoid` 映射到 `[0, 1]`；
- `trainable_refiner_parts: [tail]`：冻结 residual head/body，只微调 residual tail；
- `alpha_loss_weight: 0.20`，`regression_loss: smooth_l1`，`regression_beta: 0.10`；
- `full_mse_weight: 100.0`、`soft_mse_weight: 25.0`、`target_alpha_mse_weight: 25.0`：继续用 reconstruction-dominant loss 保护 restoration anchor；
- 评估时使用连续 predicted alpha 生成 candidate，再用同一冻结 AlexNet top-1 fallback 做 final decision。

metadata 记录的可训练参数为：head `0/1776`、body `0/207840`、tail `1299/1299`、alpha head `3201/3201`。

#### 指标

| Split | Policy | Delta PSNR | Failure Delta | Accept | Target Acc | Mean Alpha | New Error | Missed Repair |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | full_strength_top1_fallback | +0.4463 | +0.0000 | 0.6281 |  |  | 0 | 41 |
| validation | continuous_alpha_top1_fallback | +0.5010 | +0.0000 | 0.7125 | 0.4188 | 0.7270 | 0 | 41 |
| held-out | full_strength_top1_fallback | +0.4824 | +0.0000 | 0.6687 |  |  | 0 | 27 |
| held-out | continuous_alpha_top1_fallback | +0.5049 | +0.0000 | 0.7438 | 0.3625 | 0.7381 | 0 | 23 |
| test-like | full_strength_top1_fallback | +0.4298 | +0.0000 | 0.5969 |  |  | 0 | 52 |
| test-like | continuous_alpha_top1_fallback | +0.5012 | +0.0000 | 0.7250 | 0.3469 | 0.7123 | 0 | 40 |

对比 learned / deployable alpha control：

| Policy | validation | held-out | test-like | New Error |
|---|---:|---:|---:|---:|
| tail-only discrete alpha head | +0.4749 | +0.4552 | +0.4061 | 0/0/0 |
| two-stage alpha policy | +0.4831 | +0.5009 | +0.4875 | 0/0/0 |
| receiver alpha predictor | +0.5584 | +0.5099 | +0.4871 | 0/0/0 |
| tail-only continuous alpha head | +0.5010 | +0.5049 | +0.5012 | 0/0/0 |
| posterior adaptive alpha upper bound | +0.5584 | +0.5664 | +0.5691 | 0/0/0 |

连续 alpha 分布：

| Split | Mean | Min | Q1 | Median | Q3 | Max | Nearest Alpha Counts `0/0.25/0.5/0.75/1.0` |
|---|---:|---:|---:|---:|---:|---:|---|
| validation | 0.7270 | 0.2574 | 0.6490 | 0.7437 | 0.8281 | 0.9955 | `0/6/66/206/42` |
| held-out | 0.7381 | 0.3104 | 0.6218 | 0.7654 | 0.8650 | 0.9938 | `0/2/38/87/33` |
| test-like | 0.7123 | 0.1219 | 0.6072 | 0.7407 | 0.8379 | 0.9815 | `1/11/78/176/54` |

#### 结果总结

这是当前训练侧 amplitude-control 最明确的正向结果。连续 alpha head 在三段 split 上都保持 accepted new error 为 0，同时 PSNR delta 达到 `+0.5010/+0.5049/+0.5012` dB，明显超过离散 tail-only alpha head，并在 held-out/test-like 上达到或超过 two-stage policy 和 receiver predictor。

该结果说明，上一轮离散 alpha-head 的瓶颈很可能来自分类目标和离散候选表达，而不是 tail-only 微调本身。连续 alpha 的 nearest-class target accuracy 较低并不是直接负面信号，因为它没有强行复刻离散 utility label，而是在 `[0,1]` 上学到更平滑的幅度折中；test-like 最近 alpha 分布覆盖 `0.5/0.75/1.0`，不再完全跳过中间强度。

限制：该方法仍低于 posterior adaptive alpha upper bound，且本轮训练实验本身省略 LPIPS、classifier ensemble audit 和 COCO-object/CLIP 辅助诊断。后续 `ANALYSIS-S6-020` 已补 LPIPS 和三分类器 ensemble 审计，结论是它可以作为 learned deployable amplitude-control 的强候选，但仍不能直接写成最终 M3。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`。正式脚本只使用本地 `EXP-S4-006` checkpoint、本地 benefit feature table 和本地 AlexNet 权重，不加载 LPIPS，不下载任何模型或数据。运行时 `git_dirty_state=dirty` 是因为脚本和配置为本轮新增本地文件，结果记录为 `9b6f74a + local script/config`。

### ANALYSIS-S6-020：Continuous-Alpha Tail Refiner LPIPS / Classifier-Ensemble Audit

- 日期：2026-07-09
- 项目版本：`3c8a0bd` + local script/config at run time
- 阶段：S6 derived perceptual and semantic robustness audit
- 方法：ContinuousAlphaTailRefinerPerceptualEnsembleAudit
- 数据集：COCO2017 `val2017` validation / held-out / test-like continuous-alpha outputs
- 数据 split：validation `320` 行，held-out `160` 行，test-like `320` 行；审计 continuous-alpha 与 full-strength top-1 fallback 两个 policy，共 `1600` 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- config：`configs/s6_continuous_alpha_tail_refiner_audit_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_audit_continuous_alpha_tail_refiner.py
python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_audit_continuous_alpha_tail_refiner.py --device cuda:0
```

- 关键源码：`scripts/s6_audit_continuous_alpha_tail_refiner.py`
- 输入：
  - `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/per_sample.csv`
  - `outputs/analysis/exp_s4_006_alpha_head_residual_refiner_tail_regression_benefit/summary.csv`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
  - `outputs/cache/torch/hub/checkpoints/resnet18-f37072fd.pth`
  - `outputs/cache/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth`
- 输出路径：`outputs/analysis/exp_s4_006_continuous_alpha_tail_refiner_audit/`
- 状态：完成；只读取已有 PNG/CSV，不训练、不运行 diffusion、不调参

#### 指标

| Policy | Split | Delta PSNR | Delta LPIPS | AlexNet New Error | Any-Classifier New Error | Majority New Error | Any Repair |
|---|---|---:|---:|---:|---:|---:|---:|
| continuous_alpha_top1_fallback | validation | +0.5010 | -0.0149 | 0 | 17 | 1 | 41 |
| continuous_alpha_top1_fallback | held-out | +0.5049 | -0.0149 | 0 | 9 | 0 | 11 |
| continuous_alpha_top1_fallback | test-like | +0.5012 | -0.0162 | 0 | 14 | 0 | 47 |
| full_strength_top1_fallback | validation | +0.4463 | -0.0097 | 0 | 19 | 1 | 38 |
| full_strength_top1_fallback | held-out | +0.4824 | -0.0106 | 0 | 12 | 1 | 7 |
| full_strength_top1_fallback | test-like | +0.4298 | -0.0098 | 0 | 20 | 0 | 51 |

#### 结果总结

LPIPS 证据支持 continuous-alpha：三段 split 上 final LPIPS delta 为 `-0.0149/-0.0149/-0.0162`，明显优于同 checkpoint full-strength top-1 fallback 的 `-0.0097/-0.0106/-0.0098`。这说明连续 alpha 的 PSNR 提升不是单纯牺牲感知质量换来的。

跨分类器审计给出更谨慎的边界。AlexNet source gate 下 continuous-alpha 仍保持 accepted new error `0/0/0`，但 ResNet18/MobileNetV3-Small 作为离线 pseudo reference 时，any-classifier new error 为 `17/9/14`，majority-vote new error 为 `1/0/0`。唯一 majority case 是 validation 4 dB `sample_000248.png`，由 MobileNetV3-Small 与 ResNet18 同时标为 accepted new error。相比 full-strength fallback，continuous-alpha 在 LPIPS、PSNR 和多数 split 的 ensemble 风险上更好，但仍不能声称跨模型完全安全。

结论：continuous-alpha tail refiner 是当前最强 learned training-side amplitude-control 候选；它可以进入下一轮方法设计依据，但不能直接升级为最终 M3。下一步应加入 semantic-risk-aware / ensemble-aware 训练或选择约束，或先做 labeled clean-correct subset 复核。

#### 复现备注

正式运行时清空代理变量，metadata 记录 `proxy_environment_present: []`，`lpips_error: null`。首次正式运行前脚本曾把 LPIPS `TORCH_HOME` 指向输出目录，触发临时 AlexNet 权重下载；该运行被中断、输出目录删除，脚本修正为使用项目本地 `outputs/cache/torch` 后重新正式运行，未使用中断结果。

### ANALYSIS-S6-002：EXP-S4-006 Residual Shrink Selection

- 日期：2026-07-07
- 项目版本：运行时基于 `20f9cc3d6d0444b3eee2a2ccab76bb04b9a18369` 之后的本轮新增脚本
- 阶段：S6 validation-only model-selection analysis
- 方法：ResidualShrinkSelection
- 数据集：COCO2017 `val2017` subset outputs from `EXP-S4-006`
- 数据 split / 样本 ID：`sample_000192`-`sample_000255`，5 个 SNR，共 320 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；本分析本身不使用随机采样
- checkpoint：不重新加载 JSCC/refiner checkpoint；读取 `EXP-S4-006` 已有 PNG
- config：`configs/s6_residual_shrink_selection_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_residual_shrink_selection.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --device cuda:0 --overwrite
```

- 关键源码：`scripts/s6_residual_shrink_selection.py`
- 输入：
  - `outputs/EXP-S4-006/per_sample.csv`
  - `outputs/EXP-S4-006/exports/snr_XXdb/refined/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/exports/original/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_256/exports/snr_XXdb/reconstruction/`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_residual_shrink_selection/`
- 状态：完成；派生分析，不训练、不运行 diffusion、不下载

#### 指标

| Policy | Alpha/Schedule | Mean Delta PSNR | Mean Delta LPIPS | Final Failure | Delta Failure | Accept | New Error |
|---|---|---:|---:|---:|---:|---:|---:|
| `top1_fallback_alpha` | 1.0 | +0.4011 | -0.0104 | 0.3750 | +0.0000 | 0.6406 | 0 |
| `top1_fallback_alpha` | validation-selected `[0.5,0.75,0.75,0.75,0.75]` | +0.4584 | -0.0153 | 0.3750 | +0.0000 | 0.7438 | 0 |
| `always_alpha` | 1.0 | +0.7235 | -0.0274 | 0.3344 | -0.0406 | 1.0000 | 28 |
| `selected_always_m0_failure_constrained_schedule` | validation-selected | +0.5505 | -0.0253 | 0.3281 | -0.0469 | 0.8000 | 19 |

#### 结果总结

缩放残差强度能提升保守 top-1 fallback 的质量/语义 tradeoff：validation-only per-SNR schedule 在不提高 pseudo final failure 的前提下，比 full-strength top-1 fallback 多 `+0.0573` dB PSNR，并进一步改善 LPIPS。选出的 schedule 为：1 dB 用 `alpha=0.5`，4/7/13/19 dB 用 `alpha=0.75`。

但是 always-accept 不能作为最终 M3。它的平均 final failure 低于 M0，是因为 repair 数量多于 new error；逐样本看仍有 19-28 个 accepted new error。该结果说明下一步应把 residual strength / alpha 控制放入训练或 validation model selection，而不是把 always-accept 包装成安全方法。

#### 复现备注

正式运行时清空代理变量，dry-run 记录 `proxy_environment_present: []`。输出包括：

- `outputs/analysis/exp_s4_006_residual_shrink_selection/REPORT.md`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/summary.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/per_sample.csv`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/alpha_tradeoff.png`
- `outputs/analysis/exp_s4_006_residual_shrink_selection/samples/`

#### 下一步

把该分析作为 M3 训练/选择设计依据：优先做 semantic-risk-aware residual CNN model selection 或在训练中加入残差幅度/语义风险约束；若继续 diffusion，只做从 M0/M2 附近初始化的短链 residual correction。

### ANALYSIS-S6-003：Frozen Residual Shrink Schedule Test-Like Check

- 日期：2026-07-07
- 项目版本：运行时基于 `7ef1753d` 之后的本轮新增脚本
- 阶段：S6 frozen schedule test-like analysis
- 方法：FrozenResidualShrinkScheduleCheck
- 数据集：COCO2017 `val2017` test-like outputs from `EXP-S4-006`
- 数据 split / 样本 ID：`sample_000256`-`sample_000319`，5 个 SNR，共 320 行
- 信道：AWGN
- SNR：`[1, 4, 7, 13, 19]` dB
- CBR：0.17
- 随机种子：42；本分析本身不使用随机采样
- checkpoint：不重新加载 JSCC/refiner checkpoint；读取 test-like gate check 已有 PNG
- config：`configs/s6_testlike_residual_shrink_schedule_check_exp_s4_006.yaml`
- 运行命令：

```bash
python3 -m py_compile scripts/s6_apply_residual_shrink_schedule.py
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --device cuda:0
```

- 关键源码：`scripts/s6_apply_residual_shrink_schedule.py`, `scripts/s6_residual_shrink_selection.py`
- 输入：
  - `outputs/analysis/exp_s4_006_residual_shrink_selection/selected_schedule.json`
  - `outputs/analysis/exp_s4_006_testlike_gate_check/per_sample.csv`
  - `outputs/analysis/exp_s4_006_testlike_gate_check/exports/snr_XXdb/refined/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/exports/original/`
  - `outputs/eval/s2_deepjscc_coco256_awgn_best_m0_export_384/exports/snr_XXdb/reconstruction/`
  - `outputs/cache/torch/hub/checkpoints/alexnet-owt-7be5be79.pth`
- 输出路径：`outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/`
- 状态：完成；frozen schedule 复核，不训练、不调参、不运行 diffusion、不下载

#### 指标

| Policy | Mean Delta PSNR | Mean Delta LPIPS | Final Failure | Delta Failure | Accept | Repair | New Error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `m0` | 0.0000 | 0.0000 | 0.4719 | 0.0000 | 0.0000 | 0 | 0 |
| `top1_full_strength` | +0.4113 | -0.0116 | 0.4719 | +0.0000 | 0.6250 | 0 | 0 |
| `validation_top1_shrink_schedule` | +0.4552 | -0.0152 | 0.4719 | +0.0000 | 0.7063 | 0 | 0 |
| `always_full_strength` | +0.7180 | -0.0270 | 0.3812 | -0.0906 | 1.0000 | 54 | 25 |
| `validation_always_m0_failure_constrained_schedule` | +0.5555 | -0.0257 | 0.4031 | -0.0688 | 0.8000 | 34 | 12 |

#### 结果总结

Validation 上选出的 top-1 shrink schedule 在 test-like 上迁移成功：相对 full-strength top-1 fallback，PSNR 额外提升 `+0.0439` dB，LPIPS 进一步改善，同时 pseudo final failure 仍不高于 M0，accepted new error 为 0。分 SNR 看，固定 schedule 在 1/4/7/13/19 dB 的 PSNR delta 分别为 `+0.5087/+0.4268/+0.3769/+0.4499/+0.5137` dB。

Always-accept 路线继续不安全：full-strength always-accept 有 25 个 accepted new error，validation 的 always-constrained schedule 仍有 12 个 accepted new error。因此可写成证据的是“残差强度控制 + top-1 semantic fallback”提升了保守 M3 的质量，而不是 always-accept。

#### 复现备注

正式运行时清空代理变量，dry-run 记录 `proxy_environment_present: []`。输出包括：

- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/REPORT.md`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/summary.csv`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/per_sample.csv`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/metadata.json`
- `outputs/analysis/exp_s4_006_testlike_residual_shrink_schedule_check/samples/`

#### 下一步

把 `validation_top1_shrink_schedule` 作为当前更强的 conservative M3 候选，后续需要在带标签 clean-correct subset 或更正式 test split 上复核；训练侧则应考虑直接学习 SNR-aware residual amplitude / alpha，而不是只在输出后缩放。

### EXP-S4-008 / EXP-S4-009 / EXP-S4-010：Edge × Capacity/Training-Budget 受控消融

- 日期：2026-07-10
- 项目版本：`abf117b` + dirty worktree；精确 config/checkpoint/PNG SHA256 见 `ANALYSIS-S6-026` metadata
- 阶段：S5 residual restoration controlled ablation
- 数据集：COCO2017 `val2017`
- split：train `sample_000032`-`sample_000191`；validation `sample_000192`-`sample_000255`
- 信道：AWGN；SNR `[1,4,7,13,19]` dB；CBR `0.17`
- 随机种子：42
- 共同输入：正式 DeepJSCC `best.pt` 对应的 256-image-per-SNR M0 export

#### 2×2 设计

| Arm | Experiment | Edge | Channels × Blocks | Epochs | Parameters | Best Epoch |
|---|---|---|---:|---:|---:|---:|
| small no-edge | `EXP-S4-006` | no | `48 × 5` | 40 | 210,915 | 39 |
| small edge | `EXP-S4-010` | Sobel + Laplacian | `48 × 5` | 40 | 211,779 | 39 |
| large no-edge | `EXP-S4-009` | no | `64 × 6` | 60 | 447,235 | 59 |
| large edge | `EXP-S4-008` | Sobel + Laplacian | `64 × 6` | 60 | 448,387 | 54 |

两个 matched pair 除 `model.input_channels`、`condition_features`、模型名和相应第一层参数外，split、seed、loss、residual gates、训练轮数与容量完全一致。结构条件只从 receiver-visible M0 计算，不读取原图 edge。

#### 配置与命令

- `EXP-S4-008`：`configs/s5_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml`
- `EXP-S4-009`：`configs/s5_capacity_matched_no_edge_residual_refiner_validation_coco256_awgn.yaml`
- `EXP-S4-010`：`configs/s5_small_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml`

```bash
python3 -m py_compile scripts/s5_residual_refiner_pilot.py
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_capacity_matched_no_edge_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_capacity_matched_no_edge_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips
python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_small_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s5_residual_refiner_pilot.py --config configs/s5_small_edge_conditioned_residual_refiner_validation_coco256_awgn.yaml --device cuda:0 --skip-lpips
```

#### Validation 结果

| Arm | Raw ΔPSNR | Top-1 Fallback ΔPSNR | Raw Failure | Raw New Error | Raw Repair |
|---|---:|---:|---:|---:|---:|
| small no-edge | +0.7235 | +0.4011 | 0.3344 | 28 | 41 |
| small edge | +0.7736 | +0.4466 | 0.3406 | 28 | 39 |
| large no-edge | +0.8009 | +0.4738 | 0.3187 | 26 | 44 |
| large edge | +0.9398 | +0.5356 | 0.3625 | 34 | 38 |

结论：edge 对质量的独立贡献成立，但不是 semantic improvement。特别是 large edge 虽多 `+0.1389` dB raw PSNR，却增加 raw pseudo failure、新错并减少 repair；后续必须依赖独立语义审计，而不能把 source-AlexNet fallback 的零 new-error 当作泛化安全证据。

### ANALYSIS-S6-026：2×2 Paired Bootstrap 与精确复现审计

- config：`configs/s6_edge_capacity_ablation_exp_s4_006_008_009_010.yaml`
- 脚本：`scripts/s6_compare_edge_capacity_ablation.py`
- 输出：`outputs/analysis/exp_s4_006_008_009_010_edge_capacity_ablation/`
- 方法：从保存的 PNG 重算逐样本 raw/M3 PSNR；以 sample ID 为 cluster，每个 bootstrap draw 保留同一样本的五个 SNR；10,000 次、seed 42、percentile 95% CI

```bash
python3 -m py_compile scripts/s6_compare_edge_capacity_ablation.py
python3 scripts/s6_compare_edge_capacity_ablation.py --config configs/s6_edge_capacity_ablation_exp_s4_006_008_009_010.yaml --dry-run
python3 scripts/s6_compare_edge_capacity_ablation.py --config configs/s6_edge_capacity_ablation_exp_s4_006_008_009_010.yaml
```

| Contrast | Outcome | Estimate | 95% CI | CI Excludes 0 |
|---|---|---:|---|---|
| small edge − no-edge | raw PSNR | +0.0501 dB | `[+0.0249,+0.0696]` | yes |
| small edge − no-edge | M3 PSNR | +0.0455 dB | `[+0.0167,+0.0702]` | yes |
| large edge − no-edge | raw PSNR | +0.1389 dB | `[+0.1031,+0.1805]` | yes |
| large edge − no-edge | M3 PSNR | +0.0617 dB | `[+0.0236,+0.0997]` | yes |
| edge × capacity interaction | raw PSNR | +0.0888 dB | `[+0.0461,+0.1369]` | yes |
| edge × capacity interaction | M3 PSNR | +0.0163 dB | `[-0.0264,+0.0610]` | no |

验证：四 arm 共 1,280 行；PNG 重算与 source summary 最大差 `1.16e-6` dB；metadata 保存所有主输入 SHA256、2,944 个唯一 PNG 的 manifest SHA256、参数量、best epoch、脚本/config SHA256。覆盖复跑的四个核心 artifact SHA 完全一致。

### ANALYSIS-S6-027：Matched Large Edge 跨 Split 与 Fresh-Holdout 审计

- config：`configs/s6_matched_edge_holdout_audit_exp_s4_008_009.yaml`
- 脚本：`scripts/s6_compare_matched_edge_holdouts.py`
- 输出：`outputs/analysis/exp_s4_008_009_matched_edge_holdout_audit/`
- fresh-holdout：`sample_000320`-`sample_000383`，运行前未被任何 downstream residual 实验引用；结果查看后不再调整 alpha/threshold

```bash
python3 -m py_compile scripts/s6_compare_matched_edge_holdouts.py
python3 scripts/s6_compare_matched_edge_holdouts.py --config configs/s6_matched_edge_holdout_audit_exp_s4_008_009.yaml --dry-run
python3 scripts/s6_compare_matched_edge_holdouts.py --config configs/s6_matched_edge_holdout_audit_exp_s4_008_009.yaml
```

| Split | Edge − No-Edge Raw PSNR | 95% CI | Positive SNR | Pseudo Failure Δ |
|---|---:|---|---:|---:|
| validation | +0.1389 dB | `[+0.1031,+0.1805]` | 5/5 | +0.0438 |
| held-out | +0.1565 dB | `[+0.1221,+0.1975]` | 5/5 | -0.0312 |
| test-like | +0.1585 dB | `[+0.1337,+0.1854]` | 5/5 | +0.0531 |
| fresh-holdout | +0.1411 dB | `[+0.1201,+0.1634]` | 5/5 | -0.0344 |

结论：edge 的 matched 质量收益跨样本段稳定且统计 CI 明确排除 0；pseudo semantic 变化则跨 split 改变方向，不能声称 edge 本身稳定提升语义。

### ANALYSIS-S6-028：单调 Effective-Strength Schedule 与 LPIPS 复核

原 `EXP-S4-008` 独立 per-SNR schedule 的 `gate×alpha={0.06,0.05,0.06,0.05,0.03}` 在 4→7 dB 上升，违反 `MILESTONES.md` 的强度单调约束。本轮把 selection 改为全局枚举，在逐 SNR final failure 不高于 M0 的前提下最大化 mean PSNR，并强制 `gate(SNR)×alpha(SNR)` 随 SNR 非增。

- validation config：`configs/s6_edge_monotonic_residual_shrink_selection_exp_s4_008.yaml`
- frozen configs：`configs/s6_edge_monotonic_{heldout,testlike,fresh_holdout}_residual_shrink_schedule_check_exp_s4_008.yaml`
- 选中 alpha：`{1:0.75,4:0.75,7:0.75,13:1.0,19:0.75}`
- 有效强度：`{1:0.09,4:0.075,7:0.06,13:0.05,19:0.03}`

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_residual_shrink_selection.py --config configs/s6_edge_monotonic_residual_shrink_selection_exp_s4_008.yaml --device cuda:0
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_edge_monotonic_heldout_residual_shrink_schedule_check_exp_s4_008.yaml --device cuda:0
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_edge_monotonic_testlike_residual_shrink_schedule_check_exp_s4_008.yaml --device cuda:0
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_apply_residual_shrink_schedule.py --config configs/s6_edge_monotonic_fresh_holdout_residual_shrink_schedule_check_exp_s4_008.yaml --device cuda:0
```

| Split | ΔPSNR | ΔLPIPS | Failure Δ | Source-AlexNet New Error |
|---|---:|---:|---:|---:|
| validation | +0.5734 | -0.0145 | +0.0000 | 0 |
| held-out | +0.6128 | -0.0148 | +0.0000 | 0 |
| test-like | +0.5700 | -0.0163 | +0.0000 | 0 |
| fresh-holdout | +0.5668 | -0.0162 | +0.0000 | 0 |

四段每个 SNR 的 LPIPS 都小于 M0。source-AlexNet new error 为 0 是 top-1 fallback 定义保证，只能说明该决策模型内自洽，不能作为独立安全结论。

### ANALYSIS-S6-029：Monotonic Edge Policy 三分类器审计

- config：`configs/s6_edge_monotonic_policy_ensemble_audit_exp_s4_008.yaml`
- 入口：`scripts/s6_audit_residual_policy.py`
- 实现：`scripts/s6_audit_continuous_alpha_tail_refiner.py` 的通用 multi-source policy audit
- 输出：`outputs/analysis/exp_s4_008_edge_monotonic_policy_ensemble_audit/`
- 分类器：AlexNet（source decision）、ResNet18、MobileNetV3-Small；全部使用本地缓存，无联网/下载

```bash
python3 scripts/s6_audit_residual_policy.py --config configs/s6_edge_monotonic_policy_ensemble_audit_exp_s4_008.yaml --device cuda:0 --skip-lpips --skip-quality-metrics --dry-run
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy -u NO_PROXY -u no_proxy python3 scripts/s6_audit_residual_policy.py --config configs/s6_edge_monotonic_policy_ensemble_audit_exp_s4_008.yaml --device cuda:0 --skip-lpips --skip-quality-metrics
```

| Split | Images | Any-Classifier New Error | Majority New Error | Any Repair | Majority Repair |
|---|---:|---:|---:|---:|---:|
| validation | 320 | 20 | 1 | 47 | 6 |
| held-out | 160 | 7 | 1 | 8 | 1 |
| test-like | 320 | 17 | 0 | 50 | 6 |
| fresh-holdout | 320 | 17 | 3 | 55 | 7 |

结论：monotonic edge policy 的质量和感知收益已跨 split 成立，但跨分类器安全未完全过关。它不能直接升级为最终 M3；下一步优先补 supervised clean-correct 或在训练/model selection 中加入 independent semantic risk，而不是继续把 AlexNet 内生零错误包装成安全性。

### ANALYSIS-S6-030：Imagenette 严格监督 Clean-Correct Policy-Dev 审计

- 配置：`configs/s6_imagenette_supervised_clean_eval.yaml`
- 训练脚本：`scripts/s6_train_imagenette_scratch_classifiers.py`
- 评估脚本：`scripts/s6_imagenette_supervised_clean_eval.py`
- 预注册：`reports/imagenette_supervised_preregistration_2026-07-10.md`
- 输出：`outputs/analysis/imagenette_supervised_policy_dev/`

官方 Imagenette2-320 归档通过精确大小 `341663724`、MD5 `3df6f0d01a2c9592104656642f5e78a3`、gzip/tar、训练/验证成员字节校验、跨 split SHA-256 与感知哈希近重复审计。分类器使用随机初始化、不同架构、严格 `cls_train/cls_cal/policy_dev` 切分；`G_gate` cal macro top-1 `0.8961`，`T_cls` `0.9046`，两者 checkpoint 均通过质量门槛。

policy-dev 固定使用 COCO 训练的 DeepJSCC、EXP-S4-008 edge refiner、EXP-S4-009 no-edge control 与 frozen monotonic alpha。结果如下：

| Endpoint | Result | 95% CI / bound | Preregistered status |
|---|---:|---|---|
| M2 edge failure vs M0 | `-0.0202` | — | positive diagnostic |
| M3 fallback failure vs M2 | `+0.0079` | `[+0.0041,+0.0116]` | FAIL |
| M3 failure vs M0 | `-0.0124` | `[-0.0169,-0.0079]` | PASS |
| M3 PSNR vs M0 | `+0.7434 dB` | `[+0.7296,+0.7571]` | PASS |
| M3 LPIPS vs M0 | `-0.0307` | `[-0.0317,-0.0297]` | PASS |
| accepted-new-error conservative upper | `0.0108` | 11/1683 image clusters | FAIL (`<=0.005`) |

该实验的结论是“质量迁移成立、当前 M3 安全泛化失败”，不是 M3 正结果。由于 policy-dev 未通过，未填充 final lock，未访问 official val；这条 negative boundary 必须保留。

### ANALYSIS-S6-031：Imagenette Sender Coarse-Semantic Description Nested Audit

- 配置：`configs/s6_imagenette_source_semantic_description_eval.yaml`
- 脚本：`scripts/s6_imagenette_source_semantic_description_eval.py`
- 预注册：`reports/imagenette_source_semantic_description_preregistration_2026-07-10.md`
- 输出：`outputs/analysis/imagenette_source_semantic_description_policy_dev/`
- 输入：`ANALYSIS-S6-030` 的 1894-image/9470-row frozen policy-dev；official val 未访问
- 侧信息：scratch `G_gate(original)` 的 4-bit top-1 或 10×uint8 probability vector（80 raw bits）；无噪声诊断，不换算 CBR

policy-dev 在每个 WNID 内按 seed `57721` 的 SHA-256 rank 固定拆为 `semantic_select=945` 与 `semantic_audit=949`。连续 source/M0/candidate probability 在拆分与阈值网格冻结后才提取；脚本逐行重建并核对原审计 G_gate top-1/置信度。四类预注册距离中没有规则同时满足 select 的零 accepted-new-error image cluster 与至少 50% M2 PSNR 保留，因此按预注册 fallback 选中 `fullprob_cross_entropy_risk <= -1`。

| Scope | Failure | Δ vs M2 | New-error image clusters | Repair | ΔPSNR | M2 gain retained |
|---|---:|---:|---:|---:|---:|---:|
| semantic-select | `0.021251` | `+0.008264` | 0 | 24 | `+0.0312 dB` | `0.0385` |
| semantic-audit | `0.028627` | `+0.016078` | 1 | 19 | `+0.0258 dB` | `0.0326` |

audit failure delta vs M2 的 paired cluster-bootstrap 95% CI 为 `[+0.008627,+0.023922]`；accepted-new-error 保守上界 `0.005622`（1/842 image clusters），也略高于 0.5% 门槛。结论：模仿 SGD-JSCC 的 coarse description 有概念价值，但只把它接到末端 accept/reject router 不足以同时保留 repair 与控制 new error；该路线记录为负结果。

### EXP-S4-011 / ANALYSIS-S6-032：Sender Source-Edge Oracle Feasibility

- 训练配置：`configs/s5_source_edge_oracle_residual_refiner_validation_coco256_awgn.yaml`
- 训练入口：`scripts/s5_residual_refiner_pilot.py`
- 比较配置：`configs/s6_source_edge_oracle_comparison_exp_s4_008_011.yaml`
- 比较脚本：`scripts/s6_compare_source_edge_oracle.py`
- 预注册：`reports/source_edge_oracle_preregistration_2026-07-10.md`
- 训练输出：`outputs/EXP-S4-011/`
- paired 输出：`outputs/analysis/exp_s4_011_source_edge_oracle_vs_receiver_edge/`

`EXP-S4-011` 与 `EXP-S4-008` 匹配 COCO split、seed、五个 AWGN SNR、现有主图 CBR、`64×6` 容量、两个结构输入通道、residual gates、optimizer/loss/crop、60 epochs。唯一方法差异是 Sobel/Laplacian 条件来自 sender original，而不是 receiver M0。source maps 在本实验中 perfect/unaccounted，因此总 CBR 未定义。

| SNR | Receiver-edge ΔPSNR | Source-edge ΔPSNR | Source − receiver | Receiver/Source raw pseudo failure |
|---:|---:|---:|---:|---:|
| 1 | `+1.3121` | `+5.1594` | `+3.8473` | `0.5156/0.2500` |
| 4 | `+0.9537` | `+4.7981` | `+3.8443` | `0.4062/0.2500` |
| 7 | `+0.7831` | `+4.5086` | `+3.7255` | `0.3281/0.1875` |
| 13 | `+0.7965` | `+4.0409` | `+3.2444` | `0.2969/0.1719` |
| 19 | `+0.8536` | `+3.7664` | `+2.9128` | `0.2656/0.1719` |

同一 64 张图的五 SNR 值按 sample cluster 做 10,000 次 paired bootstrap，source-edge 相对 receiver-edge 的 raw PSNR 增益为 `+3.5149 dB`，95% CI `[+3.2602,+3.7652]`；五个 SNR 的 point estimate 均为正，预注册 feasibility gate 通过。raw pseudo failure 均值从 `0.3625` 降到 `0.2062`，但只作辅助诊断。

结论是“fine source structure 对当前 restoration architecture 有巨大可达收益”，不是“系统在相同通信预算下提升 3.51 dB”。下一步必须训练/接入较低 CBR 主图路径和独立 lossy edge-JSCC，使 main≈`1/8` + edge≈`1/24`、total≈`1/6`，并重新做监督 semantic audit。

### EXP-S7-001：Matched-Rate `c=6` Main / `c=2` Structure DeepJSCC Pilot

- 配置：`configs/s7_matched_rate_jscc_pilot_coco256_awgn.yaml`
- 入口：`scripts/s7_train_matched_rate_jscc.py`
- 预注册：`reports/matched_rate_jscc_pilot_preregistration_2026-07-11.md`
- 输出：`outputs/train/s7_matched_rate_main_cbr0125_pilot20k/`、`outputs/train/s7_matched_rate_structure_cbr004167_pilot20k/`

从稳定 `c=8` epoch-73 checkpoint 按 encoder/decoder 联合 L2 importance 选择 latent channels，分别训练 `c=6` RGB main 和 `c=2` packed Sobel/Laplacian structure。两个 arm 均使用 COCO train 前 20,000 图、固定 512 图 validation、AWGN 7 dB，并保留 epoch `-1` initialization 作为可选 best。

| Arm | CBR | Best epoch | Validation PSNR | Validation SSIM | Checkpoint SHA-256 |
|---|---:|---:|---:|---:|---|
| main RGB | `6/48=1/8` | 7 | `30.7497` | `0.8880` | `40f36f131b83ec1b3154402525904972d023b4211cb2e53ccb4b8d4e80385b6d` |
| structure RGB | `2/48=1/24` | 11 | `30.4991` | `0.8268` | `4ff825130987d6faa201fb25dcbfc4976fda2aa0e5fba0da56ef3816e3e4734e` |

训练全程有限，无 NaN；`6+2=8` 与 reference `c=8` 精确匹配。

### EXP-S7-002 / ANALYSIS-S7-001：Decoded-Structure Refiner 与 COCO Cross-Split Equal-Rate 比较

- export：`configs/s7_matched_rate_jscc_export_coco256_awgn.yaml` / `scripts/s7_export_matched_rate_jscc.py`
- refiner：`configs/s7_matched_rate_decoded_structure_refiner_validation.yaml` / `outputs/EXP-S7-002/`
- downstream configs：`configs/s7_matched_rate_refiner_{heldout,testlike,fresh_holdout}.yaml`
- comparison：`configs/s7_matched_rate_system_cross_split_comparison.yaml` / `scripts/s7_compare_matched_rate_system.py`
- 输出：`outputs/analysis/s7_matched_rate_system_cross_split_comparison/`

Refiner 只读取 receiver-visible `c=6` 重建、decoded `c=2` structure 的前两个通道和 SNR。与 reference `c=8` 比较：validation/held-out/test-like/fresh-holdout raw PSNR 分别提升 `+0.3974/+0.3261/+0.4198/+0.3600 dB`，所有 paired image-cluster bootstrap 95% CI 下界大于 0；三个冻结 downstream splits 合并为 `+0.3772 dB`，95% CI `[+0.3274,+0.4253]`。四段×五 SNR 的 20 个点估计均为正。

### ANALYSIS-S7-002：Imagenette Matched-Total-Rate Supervised Policy-Dev Audit

- 配置：`configs/s7_imagenette_matched_rate_supervised_eval.yaml`
- 脚本：`scripts/s7_imagenette_matched_rate_eval.py`
- 预注册：`reports/imagenette_matched_rate_preregistration_2026-07-11.md`
- 输出：`outputs/analysis/imagenette_matched_rate_policy_dev/`
- 范围：1,894 policy-dev images × 5 SNR；scratch `T_cls` independent evaluator；official val sealed

| Endpoint | Result | 95% CI / bound | Status |
|---|---:|---:|---|
| matched raw failure − reference | `-0.021410` | `[-0.028678,-0.013946]` | PASS |
| matched raw PSNR − reference | `+1.8341 dB` | `[+1.7742,+1.8949]` | PASS |
| matched raw LPIPS − reference | `-0.0305` | `[-0.0318,-0.0293]` | PASS |
| conditional new-error upper | `0.024764` | 31/1684 image clusters | **FAIL** (`<=0.005`) |

主 SNR supervised failure 从 reference `3.3785%` 降到 matched raw `1.2375%`，说明净语义效果显著改善；但 50 个 new-error rows/31 image clusters 使严格安全门槛失败。`G_gate` top-1 fallback 的 failure 为 `3.0053%`，也不能替代 raw。该结果允许写作“等码率质量提升 + 净监督失败下降”，禁止写作“语义无损/安全通过”；official val 未解锁。

### EXPORT-S8-001/002：Rate-Accounted Hybrid Structure + Semantic Sketch

- 实现：`src/cadsd_jscc/semantic_sketch.py`、`scripts/s8_export_hybrid_semantic_structure.py`
- 配置：`configs/s8_hybrid_structure_semantic_export_{coco256_awgn,r4_coco256_awgn}.yaml`
- 输出：`outputs/eval/s8_hybrid_structure_semantic_export_384/`、`outputs/eval/s8_hybrid_structure_semantic_r4_export_384/`

32-D frozen-AlexNet probability sketch 使用固定随机投影并占用已有 `c=2` latent。R16 的 512 real-symbol payload 因结构损伤超过 10% 而 FAIL；R4 只占 128/16384 symbols，五 SNR sketch cosine 均 `>=0.9552`，结构 MSE 增加均 `<=5.84%`，通过 stage gate。

### EXP-S8-001/002/003 与 ANALYSIS-S8-001--005：Semantic FiLM 反事实闭环

- 训练入口：`scripts/s5_residual_refiner_pilot.py`
- 因果分析：`scripts/s8_semantic_sketch_ablation.py`
- 最终配置：`configs/s8_per_sample_counterfactual_semantic_refiner_validation.yaml`
- 最终输出：`outputs/EXP-S8-003/`
- validation ablation：`outputs/analysis/s8_per_sample_semantic_sketch_validation_ablation/`
- downstream ablation：`outputs/analysis/s8_per_sample_semantic_sketch_downstream_ablation/`

`EXP-S8-001` 的 ordinary semantic loss 在 received-zero 因果消融中为负；`EXP-S8-002` batch-mean ranking 使 received-zero 显著为正，但 shuffled CI 仍跨 0；`EXP-S8-003` 将 hinge 改为逐样本后，frozen 160-image downstream 得到：

| Endpoint | Estimate | 95% CI | Status |
|---|---:|---:|---|
| S8 raw − reference c=8 PSNR | `+0.4691 dB` | `[+0.4231,+0.5159]` | quality positive |
| S8 raw − S7 raw PSNR | `+0.0919 dB` | `[+0.0746,+0.1147]` | quality positive |
| received − zero sketch PSNR | `+0.0849 dB` | `[+0.0728,+0.0982]` | PASS |
| received − shuffled sketch PSNR | `+0.0072 dB` | `[-0.0023,+0.0170]` | **FAIL** |

因此 S8 是“有用 side signal + 更强等码率质量”的部分正结果，不是“样本特异语义描述已成立”。按预注册纪律不把 S8 独立送入 Imagenette；后续只在主线 M3 semantic controller 整体协议中审计，official val 未访问。

### ANALYSIS-S9-001：Mainline M3 Hybrid Semantic-Controller Policy-Dev Audit

- 配置：`configs/s9_imagenette_hybrid_semantic_controller_eval.yaml`
- 入口：`scripts/s7_imagenette_matched_rate_eval.py` 的 `hybrid_semantic_controller` mode
- 预注册：`reports/imagenette_hybrid_semantic_controller_preregistration_2026-07-11.md`
- 输出：`outputs/analysis/imagenette_hybrid_semantic_controller_policy_dev/`

该实验把 S8 sketch 正式合并为 M3 semantic consistency control。内部 `matched_raw` 字段表示 M3 alpha controller，`matched_top1_fallback` 表示 M2 hybrid raw。

| Endpoint | Result | 95% CI / bound | Status |
|---|---:|---:|---|
| M3 failure − c8 | `-0.023964` | `[-0.030642,-0.017482]` | PASS |
| hybrid raw failure − M3 | `+0.000393` | `[-0.001768,+0.002554]` | FAIL |
| M3 new-error upper | `0.015875` | 18/1677 clusters | FAIL |
| M3 PSNR − c8 | `+1.4234 dB` | `[+1.3693,+1.4799]` | PASS |
| M3 LPIPS − c8 | `-0.0265` | `[-0.0276,-0.0255]` | PASS |
| raw PSNR gain retained | `74.8%` | — | PASS |

M3 将 hybrid-raw new-error rows/clusters 从 `41/23` 降至 `29/18`，但同时少保留 10 个 repair rows，净 failure 改善不显著。结论是“已合并主线、风险下降、质量收益保留”的部分正结果，不是 supervised-safe 最终 M3；official val 未访问。

### EXP-S10-001：Matched-Rate Short-Chain Residual-Shift Diffusion Pilot

- 日期：2026-07-12
- 预注册：`reports/short_chain_residual_shift_diffusion_preregistration_2026-07-12.md`
- 配置：`configs/s10_short_chain_residual_shift_diffusion_pilot.yaml`
- 入口：`scripts/s10_short_chain_residual_shift_diffusion.py`
- 输出：`outputs/EXP-S10-001/`
- 数据：冻结 S7 split，160 train / 64 eval images per SNR；不访问 Imagenette
- 通信契约：`c=6` main + `c=2` decoded structure，总 CBR `1/6`
- anchor：冻结 `EXP-S7-002` decoded-structure residual CNN
- diffusion：pixel-domain residual-shift bridge，6 个 deterministic steps；从 anchor 精确起步，不使用 Stable Diffusion、VAE、prompt 或 pure-Gaussian full residual

桥接过程定义为：

`x_tau = (1-tau) * x + tau * anchor + sigma * sqrt(tau*(1-tau)) * epsilon`

best checkpoint 为 epoch 2。相对冻结 anchor 的五 SNR 汇总如下：

| SNR | raw diffusion ΔPSNR (dB) | raw diffusion ΔLPIPS |
|---:|---:|---:|
| 1 | -0.1894 | -0.000014 |
| 4 | -0.1683 | -0.000302 |
| 7 | -0.1707 | -0.000292 |
| 13 | -0.1668 | -0.000158 |
| 19 | -0.0787 | -0.000207 |
| mean | **-0.1548** | **-0.000195** |

预注册质量条件通过：LPIPS 在 5/5 SNR 改善，mean ΔPSNR 大于 `-0.20 dB`，采样步数为 6。语义条件失败：冻结 AlexNet pseudo 诊断中，raw diffusion 相对 anchor 新增错误 12 rows、修复 7 rows，不满足 `new_error <= repair`。top-1 fallback 后 mean ΔPSNR 为 `-0.1395 dB`、mean ΔLPIPS 为 `-0.000153`，但该 fallback 仅继承 anchor prediction，不消除 raw candidate 的风险证据。

正式决策为 **NEGATIVE / no-go for this exact variant**。这不是对 diffusion 总方向的否定：相比 `EXP-S4-007` 的 `-4` 至 `-7 dB`，anchor-near short-chain bridge 已把质量损失压到 `-0.155 dB` 并取得微弱感知收益，证明架构选择明显更合理；但在扩大数据、加入感知/语义风险目标并补齐公平对照前，不允许仅靠调采样步数或随机 seed 把它包装成正结果。

### EXP-S11-001 / ANALYSIS-S11-001：P0 `c8 + Same Refiner` Fairness Audit

- 预注册：`reports/p0_c8_same_refiner_preregistration_2026-07-12.md`
- B1 配置：`configs/s11_p0_c8_same_refiner_validation.yaml`
- 配对分析：`configs/s11_p0_b1_b3_paired_comparison.yaml`、`scripts/s11_compare_p0_b1_b3.py`
- B1 输出：`outputs/EXP-S11-001/`
- 分析输出：`outputs/analysis/s11_p0_b1_b3_paired_comparison/`

B1 为裸 `c=8` reference 增加只读取 receiver-visible reconstruction/SNR/Sobel/Laplacian 的 residual refiner。它与 B3 `EXP-S7-002` 严格匹配 seed、160/64 split、`64×6` 容量、60 epochs、loss、crop、batch、residual gates，并且两者均为 448,387 refiner 参数、约 2.5 ms/image。

| Comparison | Estimate | 95% image-cluster CI |
|---|---:|---:|
| B1 raw − bare B0 | `+1.0192 dB` | — |
| B3 raw − bare B0 | `+0.3974 dB` | — |
| B3 raw − B1 raw | `-0.6217 dB` | `[-0.6654,-0.5839]` |

B3 在 5/5 SNR 均低于 B1，LPIPS 也更差 `+0.00664`；B1/B3 raw pseudo new-error/repair 为 `31/45` 与 `37/57`。预注册三个 gate 全部失败，故不能继续把当前 decoded-structure side path 写成主要增益来源。该结果不否定 diffusion；它将 B1 确立为后续 short-chain diffusion 的更强、更公平 deterministic anchor。完整解释见 `reports/p0_c8_same_refiner_result_2026-07-12.md`。

### EXP-S12-001：B1-Anchored Semantic-Preserving Short-Chain Diffusion

- 预注册：`reports/b1_anchored_semantic_preserving_diffusion_preregistration_2026-07-12.md`
- 配置：`configs/s12_b1_anchored_semantic_preserving_diffusion.yaml`
- 入口：`scripts/s10_short_chain_residual_shift_diffusion.py`
- 输出：`outputs/EXP-S12-001/`
- anchor：冻结 B1 `EXP-S11-001`，即 `c8 + receiver-structure residual CNN`
- condition：从 B1 anchor 计算 receiver-visible Sobel/Laplacian，不使用 `c2` side path
- training preservation：edge L1 + 本地冻结 ResNet18 target KL；最终 pseudo 审计仍使用 AlexNet

正式结果 mean raw ΔPSNR `-0.0775 dB`、mean raw ΔLPIPS `-0.000652`，5/5 SNR 的 LPIPS 均改善。相对 S10，PSNR 回吐减半且 LPIPS 改善更大；但 raw new-error/repair 为 `8/4`，预注册 semantic-risk gate 仍失败，因此总判定 NEGATIVE。top-1 fallback 的 mean ΔPSNR/ΔLPIPS 为 `-0.0747 dB/-0.000613`。6-step sampling latency `14.97 ms/image`，约为 B1 anchor 的 6 倍。

训练 best 为 epoch 2，后续 train loss 下降但 eval PSNR 明显恶化，说明 160-image bridge 强过拟合。按预注册停止该小数据 bridge 家族的继续调权重；若再做 diffusion，必须转为 COCO train2017-scale、独立 validation 和直接 risk calibration。完整结果见 `reports/b1_anchored_diffusion_result_2026-07-12.md`。

### EXPORT-S13-001 / EXP-S13-001：COCO Train2017 Scale-Up Cache 与 B1 Anchor

- cache 预注册：`reports/coco_train2017_scaleup_protocol_preregistration_2026-07-12.md`
- cache 配置/入口：`configs/s13_coco_train2017_c8_scaleup_export.yaml`、`scripts/s13_export_coco_train2017_c8_scaleup.py`
- anchor 预注册/配置：`reports/scaleup_b1_anchor_preregistration_2026-07-12.md`、`configs/s13_scaleup_b1_anchor_train.yaml`
- cache：`outputs/eval/s13_coco_train2017_c8_scaleup_10k_1k/`
- anchor：`outputs/EXP-S13-001/`

使用 SHA-256(seed:path) 排序从本地 train2017 固定 10,000 train + 1,000 validation，逐文件 SHA 排除 local val2017 重复。manifest hash 为 `93ae3f3b...de9`；11k original 和五个 11k reconstruction 目录均完整，共 55k rows/6.9GB。

scale-up B1 在 10 epochs 后 best 为 epoch 9；五 SNR raw PSNR 全改善，mean `+1.3632 dB`，mean LPIPS `-0.03272`。raw pseudo new-error/repair 为 `339/951`，通过 `new<=repair` gate；top-1 fallback mean PSNR/LPIPS 为 `+0.8384 dB/-0.01529`。全部预注册 anchor gate 通过，冻结 best SHA-256 `80133f9d...65562` 作为下一阶段 scale-up diffusion anchor。完整报告：`reports/scaleup_b1_anchor_result_2026-07-13.md`。

### EXP-S14-001：Train2017-Scale B1-Anchored Diffusion

使用冻结 S13 B1、10k/1k split、S12 原样 loss/gates/steps 训练 3 epochs。mean raw ΔPSNR `-0.0736 dB`、ΔLPIPS `+0.000081`，LPIPS 仅 2/5 SNR 改善；new-error/repair `63/76`，首次通过净 incremental-risk gate。总判定 NEGATIVE：扩大数据解决净语义伤害但没有产生额外感知收益，且 6-step `14.97 ms/image`。停止该 residual-shift bridge 家族的 validation tuning。详见 `reports/scaleup_b1_anchored_diffusion_result_2026-07-13.md`。

### ANALYSIS-PC-001：Received-Latent Posterior Correction Pilot

- 日期：2026-07-13
- 方法：冻结 S13 B1 与 S14 diffusion，从 S14 raw 出发，按实际 received latent 做 3 次 normalized-gradient proximal correction
- 数据：未被 S13/S14 使用的 train2017 SHA-rank `11000--11063`，64 图 × 5 SNR
- SNR / CBR：`[1,4,7,13,19] dB` / `1/6 (c=8)`
- 随机种子：`20260715`
- config：`configs/pc001_posterior_consistency_pilot.yaml`
- 入口：`scripts/pc_posterior_consistency_pilot.py`
- 输出：`outputs/analysis/s15_received_latent_posterior_pilot/`
- 状态：**POSITIVE pilot；全部预注册 gate 通过**

5/5 SNR 的 normalized received-latent loss 均下降，mean `-0.020876`，总体从 `0.10363` 降至 `0.08275`。相对未约束 S14 raw，mean PSNR `+0.2124 dB`、LPIPS `-0.00991`，两个指标均在 5/5 SNR 改善。B1-anchor-relative pseudo new error `5→5`，repair `2→17`；直接 raw→posterior classifier flip 为 new/repair `4/19`。

该结果证明 S14 的负结果可以被真实信道观测约束显著纠正，支持把 diffusion 转为 posterior/data-consistency sampler；它只授权新方法开发，不构成最终 semantic-safety 结论，也不允许在该 64 图 split 上继续调三步协议。详见 `reports/posterior_consistency_pilot_result_2026-07-13.md`。

### ANALYSIS-PC-002 / ANALYSIS-PC-003：独立复现与 Failure Handling

PC-002 在新 SHA-rank `11064--11319`、256 图×5 SNR 上冻结复现 PC-001：latent loss `0.10460→0.08328`，相对 S14 raw PSNR `+0.2125 dB`、LPIPS `-0.01078`，三项均 5/5 SNR 同向。质量/一致性结论强正向；但 ensemble-majority new error `0→2`，AlexNet/ResNet18/MobileNetV3-Small new error 分别 `15→20/11→23/15→53`，故完整判定 NEGATIVE。

PC-003 换用再次全新的 SHA-rank `11320--11575`，冻结三步 correction，并加入 receiver-only AlexNet posterior-anchor top-1 agreement fallback。accept rate `87.66%`，final 相对 S14 raw 仍有 PSNR `+0.2062 dB`、LPIPS `-0.00910`；ensemble-majority new error 从 uncontrolled posterior `4` 降到 `1`，但 raw 为 `0`，且 ResNet/MobileNet new error 仍增加，故完整判定仍为 NEGATIVE。

阶段结论：received-latent posterior correction 已成为可复现的 diffusion restoration mechanism，但单模型 semantic fallback 不具跨模型可靠性，不能晋级最终 M3。详见 `reports/posterior_consistency_independent_replication_result_2026-07-13.md` 与 `reports/posterior_consistency_failure_handling_result_2026-07-13.md`。

### ANALYSIS-PC-CTRL-001：Consensus Controller Holdout Audit

冻结 AlexNet+ResNet18 作为 receiver-side consensus controller，把 MobileNetV3-Small 完全留作 holdout audit，并在新 SHA-rank `11576--11831` 上一次性运行。posterior candidate 再次得到 PSNR `+0.2119 dB`、LPIPS `-0.01061`；controlled final coverage `78.05%`，仍保留 `+0.1927 dB/-0.00791`。

controller models 的 final new error 均被规则机械清零，三模型 majority new error 也为 0；但 holdout MobileNet new error 从 raw `12` 增至 final `34`，严格 gate 失败。该结果明确停止继续堆 classifier-consensus routing；后续 semantic control 必须使用与审计分离的训练/校准信号。详见 `reports/posterior_consensus_controller_holdout_result_2026-07-13.md`。

### ANALYSIS-PC-GT-001 / ANALYSIS-PC-SUP-001：独立标注与真实类别监督

PC-GT 在新 512 图×5 SNR 上使用 COCO instance dominant-object 标签和独立 OpenCLIP clean-correct 口径。195 张 clean 图中，raw/posterior/final failure 为 `36/32/32`，new error 为 `2/5/4`，repair 为 `1/8/7`。净 failure 改善但 new error 增加，完整 gate 失败；说明跨模型风险不是纯 pseudo-label disagreement。

PC-SUP 仅使用已有 Imagenette policy-dev 和 scratch `T_cls`，官方 validation 未访问。1697 张 clean-correct 图上，primary `[1,4,7] dB` raw/posterior/final failure 为 `69/56/62`，new error 均为 `4`；final 保留 mean `+0.2543 dB/-0.00531 LPIPS`。但 7 dB final/raw new error 为 `1/0`，违反逐 SNR gate，故仍判 NEGATIVE。该结果首次在真实 WNID 监督下确认 posterior diffusion 有净语义收益，同时明确其尚不具逐信道可靠性。

### ANALYSIS-PC-RISK-001：Frozen Scratch-Gate Follow-Up

- 日期：2026-07-13
- config：`configs/pc_imagenette_scratch_gate_audit.yaml`
- 入口：`scripts/pc_imagenette_supervised_audit.py`
- 输出：`outputs/analysis/pc_imagenette_scratch_gate_policy_dev/`
- 控制器：冻结 scratch MobileNetV3-Small `G_gate`；只检查 `G_gate(posterior).top1 == G_gate(anchor).top1`
- 审计器：独立 scratch ResNet18 `T_cls`，不参与决策
- 状态：**NEGATIVE strict；controller 明显改善但 7 dB tail gate 仍失败**

本轮与 PC-SUP 复用 seed 和 batch，9470 行 raw/posterior 的 sample、语义结果、latent consistency、PSNR/LPIPS 逐值核对为 0 mismatch，仅替换 final controller。1697 张 clean 图上，scratch gate 五 SNR 合计接受 `8428/8485` 行（`99.33%`），primary failure 从 raw `69` 降至 `57`，优于旧 consensus final 的 `62`；primary new error 从 raw `4` 降至 `3`，也优于旧 controller 的 `4`。final mean PSNR/LPIPS 相对 raw 为 `+0.26394 dB/-0.005966`。

严格 gate 仍因 7 dB new error `1 > 0` 失败；该事件中 `G_gate` 对 anchor/posterior 都预测 class index 2，无法发现独立 `T_cls` 的 correct→incorrect flip。不得在已查看的 policy-dev 上追加 threshold 或逐样本例外。该结果保留 frozen posterior+scratch gate 为 supervised development candidate，但不解锁 official validation。详见 `reports/posterior_imagenette_scratch_gate_result_2026-07-13.md`。

### ANALYSIS-PC-RISK-REP-001：Scratch-Gate Multi-Seed Replication

- 日期：2026-07-13
- config：`configs/pc_imagenette_scratch_gate_multiseed_replication.yaml`
- 入口：`scripts/pc_imagenette_supervised_audit.py`
- 输出：`outputs/analysis/pc_imagenette_scratch_gate_policy_dev_multiseed/`
- channel seeds：`[20260722,20260723,20260724]`
- 数据量：1894 images × 3 seeds × 5 SNR = `28,410` rows
- 状态：**NEGATIVE tail control；POSITIVE robust restoration**

三个新 seed 上 primary failure 分别从 raw `63/67/66` 降为 final `56/53/54`，且 15/15 seed×SNR received-latent consistency 均下降。mean final-minus-raw PSNR/LPIPS 为 `+0.26334 dB/-0.005937`，每个 seed 均同向，说明 posterior restoration 的正向效果不依赖单一 AWGN realization。

严格风险结论相反：primary raw/posterior/final new-error rows 为 `13/15/14`，image clusters 为 raw/final `10/11`。final `11/1691` cluster rate `0.6505%` 的单侧 95% Clopper-Pearson upper 为 `1.0744%`，超过冻结 `0.5%`。1 dB `8→10`、seed 20260722 `5→7`，因此 total、per-SNR、per-seed 与 cluster-UCB 四个 gate 均失败。旧 7 dB event `n03425413_3069` 在新 seed 再现，确认尾部是 image susceptibility × channel noise，而不是可替换的坏 seed。official validation 继续封存；详见 `reports/posterior_imagenette_scratch_gate_multiseed_result_2026-07-13.md`。

### TRAIN-PC-AUX-001：Independent Scratch `G_aux`

- 日期：2026-07-14
- config：`configs/pc_imagenette_scratch_aux_classifier.yaml`
- 入口：`scripts/s6_train_imagenette_scratch_classifiers.py --roles G_aux`
- 输出：`outputs/analysis/imagenette_scratch_risk_classifier/G_aux/`
- 状态：**PASS as auxiliary feature extractor only**

EfficientNet-B0 从零初始化，只用 6629 个 `cls_train` 样本训练、946 个 `cls_cal` 样本选点/温度校准。80 epoch 最优为 epoch 64，cal macro top-1 `0.90269984`、top-1 `0.90380550`，temperature `0.80484366`，ECE15 `0.0739071→0.0427925`。checkpoint SHA-256 为 `8e074be6ec854edbc144d95d9fe5cd7d098c61bca853915108952acfa094b455`；policy-dev training/selection/calibration=false，official-val-accessed=false。该结果不等价于 controller 或 semantic safety 通过。

### ANALYSIS-PC-RISK-FEAT-001：Receiver-Risk Feature Table

- config：`configs/pc_imagenette_receiver_risk_features_multiseed.yaml`
- 入口：`scripts/pc_imagenette_supervised_audit.py`
- 输出：`outputs/analysis/pc_imagenette_receiver_risk_features_multiseed/`
- 状态：**INTEGRITY PASS；DEVELOPMENT DATA ONLY**

`receiver_risk_v1` 保存 43 个 receiver-visible feature，`T_cls` teacher target 单独前缀隔离。28,410 行键唯一、全部特征 finite、三 classifier checkpoint 分离且 hash 固定；普通审计所有共同字段与 PC-RISK-REP 在 `1e-9` 内一致。`risk_features.csv` SHA-256 `7b81120f8a2a23140800845257d19126800fae1e5f2cb78a4c6398266917233d`。这些 seed/outcome 已暴露，只能用于 controller development。

### ANALYSIS-PC-RISK-CTRL-DEV-001：Transparent Percentile Controller

- config：`configs/pc_imagenette_receiver_risk_controller_dev.yaml`
- 入口：`scripts/pc_fit_receiver_risk_controller.py`
- 输出：`outputs/analysis/pc_imagenette_receiver_risk_controller_dev/`
- 状态：**DEVELOPMENT PASS；NOT HOLDOUT EVIDENCE**

冻结 score 为 `G_gate/G_aux` 四个 JS-change empirical percentile 与两个 sign-reversed posterior-confidence percentile 的算术均值。固定候选网格中首次通过全部门槛的是 target reject rate 10%，threshold `0.8537265316368728`。开发 primary raw/posterior/final failure `196/164/180`，new error `13/15/3`；final 3 个 new-error clusters 的 upper95 `0.4579%`；mean final-minus-raw PSNR/LPIPS `+0.23834/-0.004799`，保留 posterior PSNR gain `89.83%`。controller JSON SHA-256 `3ff792a366074202d1727042c40c0cbc777843a5c65d48276e3dfd9be6199f6f`，CDF NPZ SHA-256 `2a7f062ab53da9309f6ccad8b9c2a8977b9b79e5d95c760559d674d2503e6956`。

### ANALYSIS-PC-RISK-SEED-AUDIT-001：Frozen New-Seed Audit

- feature config：`configs/pc_imagenette_receiver_risk_seed_20260725_features.yaml`
- audit config：`configs/pc_imagenette_receiver_risk_seed_20260725_audit.yaml`
- 入口：`scripts/pc_imagenette_supervised_audit.py`，随后 `scripts/pc_apply_receiver_risk_controller.py`
- 输出：`outputs/analysis/pc_imagenette_receiver_risk_seed_20260725_{features,audit}/`
- 状态：**NEGATIVE**

seed `20260725` 在生成前确认未出现于 source/config/report ledgers，并冻结 extraction-config/controller/CDF SHA 与阈值。9470 行完整性通过。posterior mean PSNR/LPIPS 为 `+0.26535/-0.006064`，primary failure `50→45`，继续证明 diffusion posterior restoration 稳定。冻结 risk final reference reject rate `9.861%`、PSNR/LPIPS `+0.23827/-0.004800`，但 final failure `56>raw 50`、new error `2>raw 0`；两个 1 dB new error 都未拒绝，同时拒绝 11 个 posterior repair，故总/per-SNR failure 与 new-error gate 失败。

其中 `n03425413/n03425413_24914.JPEG` 的 `G_gate/G_aux` posterior confidence 为 `0.973/0.963`，四个 JS 特征接近 0，属于高置信共享语义盲点；事后同分数要覆盖两例需拒绝约 `38.3%` 参考行。禁止据此调 threshold 或替换 seed。保留 posterior diffusion，淘汰本 receiver-only controller 晋级资格；完整报告见 `reports/posterior_receiver_risk_controller_stage_result_2026-07-14.md`。

### ANALYSIS-PC-SENDER-DUAL-EVIDENCE-DEV-001B：独立 receiver guard 未解决 shared blind spot

- 配置：`configs/pc_imagenette_sender_dual_evidence_seed20260726_dev.yaml`
- 输出：`outputs/analysis/pc_imagenette_sender_dual_evidence_seed20260726_dev/`
- 状态：**NEGATIVE development**

在严格等总码率 `UInt4+BPSK×4` sender-JS zero veto 上加冻结 `G_gate(posterior).top1 == G_gate(anchor).top1`，不增加 bit 或阈值。seed `20260726` 的 receiver guard 额外 veto `0.6230%` 行，但 primary final failure/new-error 仍为 `55/5`，与单 `G_aux` veto 相同；mean final-minus-M2 PSNR/LPIPS 仅为 `+0.02523/-0.003094`。5 个 new-error 均被 `G_aux` 与 `G_gate` 同时接受，说明该简单 receiver-side 交集不足。该失败已记录，未在该 seed 上补阈值。

### ANALYSIS-PC-SENDER-CROSSMODEL-DEV-001A/B 与 SEED-AUDIT-001：固定码率 cross-model triplet controller

- development configs：`configs/pc_imagenette_sender_crossmodel_triplet_seed20260725_dev.yaml`、`configs/pc_imagenette_sender_crossmodel_triplet_seed20260726_dev.yaml`
- frozen audit reference/config：`configs/pc_imagenette_sender_crossmodel_seed20260727_reference.yaml`、`configs/pc_imagenette_sender_crossmodel_triplet_seed20260727_audit.yaml`
- 入口：`scripts/pc_imagenette_sender_inbudget_awgn_audit.py`
- audit 输出：`outputs/analysis/pc_imagenette_sender_crossmodel_triplet_seed20260727_audit/`
- 状态：**NEGATIVE after system-endpoint correction；原 POSITIVE 已作废**

controller 只接受同时满足 source `G_aux` JS 非恶化、recovered `G_aux(source)` top-1 等于独立 `G_gate(anchor)` top-1、且 `G_gate(anchor/posterior)` top-1 一致的 posterior。它保持 40-bit payload、160 保留符号、65536 总实符号、总 CBR `1/6` 和共同 AWGN。

development seed `20260725/20260726` 分别得到 primary M2 failure/final `50→48`、`58→55` 和 in-budget raw/final new-error `4→1`、`3→0`。seed `20260727` 原记录报告 in-budget raw-relative `2→0`、0 个 anchor-relative new-error clusters（upper95 `0.1771%`）和 mean final-minus-M2 `+0.01158 dB/-0.002566 LPIPS`。后续统计审计确认该 endpoint 不足：相对 paired unpunctured M2 的 system new/repair clusters 为 `7/8`、upper95 `0.7766%`，且 1 dB failure `32→34`。因此原 POSITIVE 作废、正式改记 NEGATIVE；详情见带更正声明的 `reports/posterior_sender_crossmodel_triplet_stage_result_2026-07-14.md`。
### EXP-S15-001（历史标签）：UInt2 Reservation-Aware B1 Fine-Tune

- 日期：2026-07-14；实际阶段归属为 S5 validation，`S15` 仅为已落盘历史标签。
- 预注册：`reports/reservation_aware_diffusion_jscc_preregistration_2026-07-14.md`
- cache/config：`outputs/eval/s15_coco_train2017_c8_uint2_reserved_2k_200/`、`configs/s15_coco_uint2_reserved_c8_export_pilot.yaml`
- fine-tune config/output：`configs/s15_uint2_reservation_aware_b1_finetune_pilot.yaml`、`outputs/EXP-S15-001/`
- checkpoint：`outputs/EXP-S15-001/checkpoints/best.pt`，SHA-256 `57aa528345b90b06a3daadd1069b27d320534a0124769d46118b760fbbc85495`
- paired comparison：`scripts/s15_compare_reservation_aware_b1.py` → `outputs/analysis/s15_reservation_aware_b1_paired_comparison/`
- 状态：**POSITIVE internal paired validation；不等于 semantic-safe M3**

在相同 200 images×5 SNR reserved inputs 上，新 B1 相对旧 S13 B1 的 aggregate PSNR `+0.102782 dB`，image-cluster bootstrap 95% CI `[+0.093375,+0.114000]`；逐 SNR增益 `[+0.11514,+0.10223,+0.09737,+0.09747,+0.10170] dB`，全部 CI 下界大于 0。aggregate LPIPS `-0.001682`，95% CI `[-0.002022,-0.001333]`。official Imagenette 未访问。

### ANALYSIS-PC-UINT2-RESAWARE-001：Reservation-Aware B1 Full Policy-Dev Replay

- config：`configs/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_full_dev.yaml`
- output：`outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_full_dev/`
- channel seed：`20260727`；1894 images×5 SNR = 9470 rows
- 状态：**NEGATIVE strict；POSITIVE quality**

UInt2 总码率保持 CBR `1/6`；final−paired-M2 PSNR/LPIPS `+0.073967/-0.002633`，质量 CI 与五 SNR point gate 全过。paired M2/final primary failure `59→60`，system new/repair `7/6`、new cluster upper95 `0.7761%`，故不能晋级。7 个新增中 6 个是 rejected→wrong-anchor，而 raw/posterior 都正确，促成三路 fallback 但不能把本 run 改写成正结果。

### ANALYSIS-PC-UINT2-ROUTING-SEL-001：Seed20260727 Offline Three-Way Routing

- 入口：`scripts/pc_analyze_mismatch_raw_routing.py`
- source：上述 `ANALYSIS-PC-UINT2-RESAWARE-001/per_sample.csv`
- output：`outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_offline_seed20260727/`
- 状态：**NEGATIVE strict development selection**

固定规则 accept→posterior、reject+source/anchor mismatch→raw、其余→anchor。point 结果为 M2/final `59→56`、system new/repair `2/5`、upper95 `0.3718%`；PSNR/LPIPS `+0.065360/-0.002595`，5/5 SNR PSNR 正。但 failure bootstrap 95% CI `[-0.001571,+0.000393]` 未严格低于 0，因此只用于在新 seed 前冻结规则。

### ANALYSIS-PC-UINT2-ROUTING-SEED20260728-001：Frozen Independent-Channel Replication

- 预注册：`reports/reservation_aware_fallback_routing_seed20260728_preregistration_2026-07-14.md`
- config：`configs/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_seed20260728.yaml`
- output：`outputs/analysis/pc_imagenette_sender_crossmodel_triplet_uint2_r4_resaware_b1_routing_seed20260728/`
- channel seed：`20260728`；1894 images×5 SNR = 9470 rows
- 状态：**NEGATIVE preregistered replication；POSITIVE robust restoration**

final−paired-M2 PSNR `+0.065798 dB`，image-cluster 95% CI `[+0.060055,+0.071703]`；LPIPS `-0.002540`，95% CI `[-0.002805,-0.002286]`；五 SNR PSNR 全正。primary M2/final failure `62→62`，system new/repair rows `4/4`，new error 4 clusters/1690 eligible、upper95 `0.5408%`，超过 0.5%；1 dB failure `34→36`。按预注册保留负结果，不在该 seed 调路由。完整解释见 `reports/uint2_reservation_aware_diffusion_stage_result_2026-07-14.md`。

### SMOKE-EXT-SGDJSCC-001：SGD-JSCC Author-Native One-Image Integration Smoke

- 日期：2026-07-14
- config：`configs/external_sgdjscc_native_smoke.yaml`
- 入口：`scripts/external_sgdjscc_native_smoke.py --run`
- 作者源码：`third_party/SGDJSCC` commit `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`，tracked files clean
- 输入：`outputs/eval/s13_coco_train2017_c8_scaleup_10k_1k/exports/original/sample_010000.png`，integration-smoke only
- channel：AWGN `1 dB`，seed `2025`，`use_gt_csi=false`
- method：BLIP2 caption + main JSCC + MuGE soft edge + rate-adaptive edge JSCC + CLIP ViT-L/14 + ControlNet + 50-step continuous diffusion
- 输出：`outputs/smoke/external_sgdjscc_native_snr1_seed2025_20260714/`
- 状态：**PASS integration smoke；NOT an outcome/comparison result**

第一次真实 run 直接完成，生成 finite RGB `[1,3,128,128]` reconstruction，无 failure artifact。caption 为 `a blurry picture of a baseball field with a few players on it`；smoke-only PSNR `25.055894 dB`，模型加载加单图前向耗时 `12.6837 s`，peak allocated GPU memory `7234.28 MiB`。单图不做 semantic-safety 或方法优劣推断。

运行时 hook 观测 main `4096` real symbols；edge dense tensor `16384` elements、nonzero-active `832` elements。main+edge-active CBR 为 `0.1002604`，main+edge-dense CBR 为 `0.4166667`；caption UTF-8 成本为 `488 bits`，但作者没有 text channel/coding/symbol 定义。因此 `common_contract_direct_ranking_allowed=false`，本结果只能进入 author-native 表。

所有作者 checkpoint、BLIP2 两 safetensors 分片和 CLIP 均在运行前按 frozen config 做精确 hash 校验；official Imagenette validation 未访问。完整解释见 `reports/sgdjscc_author_native_smoke_stage_result_2026-07-14.md`。

### SMOKE-EXT-SGDJSCC-COMMON-001：SGD-JSCC Common-Contract 256×256 One-Image Smoke

- 日期：2026-07-15
- config：`configs/external_sgdjscc_common_smoke.yaml`
- 入口：`scripts/external_sgdjscc_common_smoke.py --run`
- 输入：frozen S13 eval manifest 的 `sample_010000.png`，256×256，integration-smoke only
- channel：AWGN `1 dB`，canonical channel seed `20260729`
- 方法标签：**SGD-JSCC common-contract adapter；不是 author-native**
- 输出：`outputs/smoke/external_sgdjscc_common_snr1_seed20260729_20260715/`
- 状态：**PASS protocol/rate integration smoke；NOT an outcome/comparison result**

作者四 patch 完整生成链实际运行，caption 显式通过 fixed 67-byte UTF-8/CRC16 packet、BPSK×21 和同一 AWGN；edge 仅调度确定性 active mask。runtime 观测 main `16,384`、active edge `3,328`、text `45,024`、padding `800` 个实坐标，总计 `65,536`=`32,768` complex uses，CBR `1/6`，rate gate PASS。

1 dB 下四个文本 packet 的 raw hard-symbol errors 为 `5,981/45,024`，R21 后 packet bit errors 为 `0/2,144`，CRC `4/4` 通过。输出 finite `[1,3,256,256]`，smoke-only PSNR `24.785109 dB`，耗时 `13.1588 s`，peak allocated VRAM `7364.35 MiB`。肉眼可见 patch seams，右边缘存在疑似 text-driven enlarged-player hallucination；单图不作 semantic new-error 统计结论。完整报告见 `reports/sgdjscc_common_contract_smoke_stage_result_2026-07-15.md`。

### SMOKE-EXT-SGDJSCC-COMMON-CXAWGN-001：复信道 AWGN 口径更正 smoke

- 日期：2026-07-15
- config：`configs/external_sgdjscc_common_complex_awgn_smoke.yaml`
- 输出：`outputs/smoke/external_sgdjscc_common_complex_awgn_snr1_seed20260729_20260715/`
- 状态：**PASS corrected integration smoke；NOT an outcome result**

旧 common smoke 继承作者实值 AWGN 方差 `P/SNR`；本轮显式改为项目复信道每实坐标方差 `P/(2×SNR)`，其余输入、seed、rate layout、caption 和模型均不变。结果 PSNR 从旧口径 `24.785109` 变为 `26.128782 dB`，四 caption CRC 仍 `4/4`，总码率仍为 65,536 real / 32,768 complex uses / CBR `1/6`。旧输出不覆盖，只降格为更严苛噪声下的接入证据。

### ANALYSIS-EXT-COMMON-PILOT-001：外部共同协议 8 图×5 SNR pilot

- 日期：2026-07-15
- prereg config：`configs/external_common_comparison_pilot.yaml`
- runners：`scripts/external_common_project_pilot.py`、`scripts/external_sgdjscc_common_pilot.py`
- aggregate validator：`scripts/external_common_aggregate.py`
- population：Imagenette train `policy_dev` 中按预注册 SHA 规则选出的 8 张 frozen clean-correct 图；official val 未访问
- channel：复 AWGN `[1,4,7,13,19] dB`，base seed `20260729`，每 sample/SNR 一个 65,536-D canonical noise vector
- rate：所有方法每行 65,536 real = 32,768 complex uses = CBR `1/6`
- 输出根：`outputs/external_baselines/ANALYSIS-EXT-COMMON-PILOT-001/`
- 状态：**PASS integration/directional pilot；NO superiority claim**

三方法各完成 40 行，aggregate gate 对全部 120 行验证相同 key/noise SHA/DeepJSCC reference/rate/AWGN convention。总结果：

| 方法 | PSNR | MS-SSIM | LPIPS | failure | new/repair vs DeepJSCC |
|---|---:|---:|---:|---:|---:|
| DeepJSCC reference | `31.743847` | `0.972978` | `0.078612` | `0` | — |
| ours M3 | `33.059380` | `0.982034` | `0.035320` | `0` | `0/0` |
| SGD-JSCC common adapter | `26.888224` | `0.948619` | `0.077630` | `0` | `0/0` |
| SING-Zero-style final-only | `24.659262` | `0.961176` | `0.317252` | `1` | `1/0` |

ours−SGD 的 40-row 配对均值为 PSNR `+6.171157 dB`、MS-SSIM `+0.033415`、LPIPS `-0.042310`；PSNR/MS-SSIM wins `40/40`，LPIPS wins `39/40`。SGD 的 160/160 captions CRC 通过，但 sender captions 已含 dog→cat、chainsaw scene→driver/computer/camera 等 patch-level semantic error。SING-style projection 把 mean-pool measurement MSE `2.0819e-2→5.7928e-6`，仍在 1 dB 引入一个 hard new error，表明 final-only range/null projection 不足。

SGD 完整批跑前有两次 scheduler-cache 初始化失败，均发生在第一条 reconstruction 前、完成行数 0；失败目录与 `failure.json` 保留。根因是 pinned hub cache constants 在 transitive import 时提前初始化，最终把 `HF_HOME` 与 offline flags 移到所有相关 import 之前后通过。无下载、无代理、未访问 official val。

该 pilot 只有 8 张已暴露 development 图，且 SGD/SING 分别是 common adapter 与 mechanism-level approximation；结果不得表述为强于两篇论文。完整中文解释见 `reports/external_common_comparison_pilot_stage_result_2026-07-15.md`。

### EXP-EXT-AUTHOR-RATE-DJSCC-001/002/003：精确 19,712-real DeepJSCC

- 日期：2026-07-15
- configs：`configs/external_author_rate_deepjscc_train.yaml`、`external_author_rate_deepjscc_train_stable.yaml`、`external_author_rate_deepjscc_fullcoco_continue.yaml`
- 方法：c3 dense latent 24,576 中固定发送 19,712 个均匀实坐标；活动功率归一化；复 AWGN `P/(2×SNR)`；c8 importance-pruned warm start
- rate：9,856 complex uses，精确 CBR `0.0501302083`
- official val：未访问

001 在 epoch 1 batch 213 出现 non-finite loss，状态 **FAILED**，目录和 `failure.json` 保留。002 改用 FP32/`lr=2e-5` 后完成 20k×12，COCO-512 PSNR/SSIM `25.8609/0.74830`。检查到 c8 使用完整 COCO 后，003 继承 002、保持码率/掩码/信道不变，以 FP32/`lr=1e-5` 完整 COCO 118,287 张×12 继续训练；最终 `26.6981 dB/0.77855`，best checkpoint SHA-256 `bca5b67a...bb606`。状态：**PASS exact-rate training-budget follow-up**。

### ANALYSIS-EXT-AUTHOR-RATE-PILOT-001/002：SGD 作者工作点对齐

- prereg：`reports/external_two_working_point_alignment_preregistration_2026-07-15.md`
- final config：`configs/external_author_rate_alignment_fullcoco_followup.yaml`
- population/channel：冻结 8 图×`[1,4,7,13,19]`，base seed `20260730`，相同 19,712-D canonical noise
- 输出：`outputs/external_baselines/ANALYSIS-EXT-AUTHOR-RATE-PILOT-002/`
- 状态：**PASS directional pilot；SGD is paper-protocol upper bound**

full-COCO low-rate DeepJSCC 为 PSNR/MS-SSIM/LPIPS `25.9260/0.92189/0.28716`、failure `3/40`；SGD 免费/无误 caption 论文协议为 `26.8389/0.94861/0.07856`、failure `0/40`。SGD−DeepJSCC 配对 `+0.91283 dB/+0.02672/-0.20860 LPIPS`，wins `31/40、33/40、40/40`，并修复 3 个 hard failure。因 SGD caption 不计码率，该结果禁止称为严格端到端物理 rate match。

### ANALYSIS-EXT-SGD-REALLOC-PILOT-001：SGD CBR=1/6 预算重分配

- config：`configs/external_project_rate_sgd_reallocation_pilot.yaml`
- allocation：main R2 `32,768` + active edge R1 `3,328` + text R13 `27,872` + padding `1,568` = `65,536` real
- 输出：`outputs/external_baselines/ANALYSIS-EXT-SGD-REALLOC-PILOT-001/`
- 状态：**PASS released-weight allocation sensitivity；NOT increased capacity**

新分配 PSNR/MS-SSIM/LPIPS `27.3933/0.95534/0.07246`，相对旧 R1/R21 为 `+0.50510 dB/+0.006721/-0.005167`，PSNR/MS-SSIM `40/40` 行改善；160/160 caption CRC 通过，failure/new error 为 0。当前 M3 相对新 SGD 仍为 `+5.66606 dB/+0.026694/-0.037143 LPIPS`，PSNR/MS-SSIM `40/40`、LPIPS `38/40` 更优。完整解释见 `reports/external_two_working_point_alignment_stage_result_2026-07-15.md`。
### EXPORT-S16-LOWRATE-M3-001：精确 19,712-real 预留感知缓存

- 日期：2026-07-15
- config：`configs/lowrate_m3_exact19712_cache_export.yaml`
- runner：`scripts/s13_export_coco_train2017_c8_scaleup.py`
- 输入：COCO train2017 冻结 10000/1000 split；排除 val2017
- rate：19712 total real = 80 payload-reserved + 19632 image-active，9856 complex uses，CBR `0.0501302083`
- channel：复 AWGN `[1,4,7,13,19] dB`；载荷代理为固定均衡 ±1，接收端擦除载荷坐标后图像解码
- 输出：`outputs/eval/s16_lowrate_m3_exact19712_uint2r4_coco10k_1k/`
- 结果：55000/55000 行完整；11000 图 PSNR 为 `24.18879/25.58552/26.47436/27.27843/27.50629`
- 状态：**PASS cache/rate closure**

### EXP-S16-B1-001：低码率 receiver-structure B1

- config：`configs/lowrate_m3_b1_anchor_train.yaml`
- checkpoint：`outputs/EXP-S16-B1-001/checkpoints/best.pt`
- checkpoint SHA-256：`7a295976105a9c43c25604c9070e676d25512c7a09b5c50655b6671477b7615a`
- budget：10000 train/1000 eval、10 epochs、冻结五档 residual gate
- 结果：五档 ΔPSNR `+1.3817/+1.1053/+0.9863/+0.8864/+0.8311 dB`；平均 `+1.03815 dB`。五档 LPIPS 全改善，平均 `-0.11412`。
- 状态：**PASS；低码率新主 anchor**

### EXP-S16-DIFF-001：低码率 B1-anchored 短链 diffusion

- config：`configs/lowrate_m3_b1_anchored_diffusion.yaml`
- checkpoint SHA-256：`44915d7e116cc9a46cb590501f075196afdc4827cb658d170795540de448bc8a`
- budget：20 train timesteps、6 sampling steps、3 epochs；低码率 B1 55k anchor 独立物化
- 结果：五档 ΔPSNR `-1.0120/-0.8639/-0.6039/-0.4659/-0.2499 dB`；LPIPS 五档全恶化；raw new/repair `318/139`。
- 判定：五项检查仅 sampling-step 通过。
- 状态：**NEGATIVE；完整保留，不作为主要提升**

### ANALYSIS-S16-LOWRATE-M3-STAGE-001：严格低码率 8×5 闭环

- config：`configs/lowrate_m3_stage_pilot.yaml`
- runner：`scripts/lowrate_m3_stage_pilot.py --run`
- population：外部共同协议第一组 frozen 8 图；base seed20260730；official val 未访问
- 输出：`outputs/analysis/ANALYSIS-S16-LOWRATE-M3-STAGE-001/`
- payload：BER 0，40/40 vector exact
- 均值：reference exact B0 `25.9260/0.28716 LPIPS/3 failures`；strict payload B0 `25.8765/0.28872/3`；B1 `26.8461/0.17140/0`；raw `26.1961/0.18882/0`；posterior `26.3329/0.17636/0`；final `26.6434/0.17390/0`。
- consistency：`0.13253→0.11138`，40/40 行不增。
- 状态：**NEGATIVE overall；B1 positive，posterior restores raw，full-SNR routing harms B1**

### ANALYSIS-S16-LOWRATE-M3-TAIL-HOLDOUT-001：独立高 SNR 尾部验证

- configs：`configs/lowrate_m3_tail_holdout_population.yaml`、`configs/lowrate_m3_tail_holdout_pilot.yaml`
- population：冻结 clean membership 的 SHA rank 9–16，独立 8 图；新 base seed20260731
- frozen policy：`SNR<19→B1`；`SNR=19→仅接受三重语义门通过的 posterior`；拒绝回 B1
- 输出：`outputs/analysis/ANALYSIS-S16-LOWRATE-M3-TAIL-HOLDOUT-001/`
- 结果：payload BER 0；B1−B0 `+1.02732 dB/-0.10969 LPIPS`、failure `3→0`。final−B1 全五档 `-0.009895 dB/-0.000389 LPIPS`、failure/new/repair `0/0/0`；19 dB `-0.04947 dB/-0.001945 LPIPS`。
- consistency：`0.13114→0.10951`，40/40 行不增；7/7 预注册 checks PASS。
- 状态：**PASS independent directional holdout；只授权高 SNR diffusion tail，不授权全 SNR/final-safe claim**

### EXP-S17-LATDIFF-001：Channel-Matched Latent Diffusion AMP 失败运行

- 日期：2026-07-15
- 状态：失败，完整保留
- 配置：失败输出内的原始 `amp: true` 配置副本
- 命令：`python3 scripts/s17_channel_matched_latent_diffusion.py --mode train --device cuda:0`
- 失败点：epoch 0 / batch 13，`RuntimeError: non-finite training loss`
- 输出：`outputs/failed/EXP-S17-LATDIFF-001_amp_nonfinite_batch13_20260715/`
- 启动器诊断：更早一次 PTY 会话在首批结果前异常退出，只有 config/run_plan/script，保留于 `outputs/failed/EXP-S17-LATDIFF-001_pre_epoch_session_exit_20260715/`；独立 max-batch debug 随后证明 runner 可执行，正式非有限失败以上述非 PTY run 的 batch13 traceback 为准。
- 结论：发生在任何 epoch selection 输出之前，不产生方法结论；后继只允许 FP32 数值稳定化，不改变数据、模型或成功判据。

### EXP-S17-LATDIFF-002：Exact-Rate Channel-State-Matched Latent Diffusion

- 日期：2026-07-15
- 项目 commit（注册时）：`abf117bc9aa194cd0a4fa80bd63df6baae1f6d29`
- 第三方 DeepJSCC commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 配置：`configs/s17_channel_matched_latent_diffusion.yaml`
- 预注册：`reports/channel_matched_latent_diffusion_preregistration_2026-07-15.md`
- 命令：`env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 scripts/s17_channel_matched_latent_diffusion.py --mode train --device cuda:0`
- 数据：现有 source manifest 的 10,000 train；selection 为 validation role 前 256 图；official Imagenette validation 未访问。
- 信道/码率：AWGN `[1,4,7,13,19]`，`P/(2×SNR)`，19,712 active real，其中 19,632 image + 80 payload，CBR `0.050130208333333336`。
- 模型：323,574 参数 masked latent epsilon predictor；6-step deterministic DDIM；FP32。
- DeepJSCC checkpoint：`bca5b67a...bb606`
- 最佳 checkpoint：epoch 5，`outputs/EXP-S17-LATDIFF-002/checkpoints/best.pt`，SHA-256 `cfc5271660103e29567e05f5d0242dcb02edc5e625054e115490fbed4b3cb4e1`
- selection：mean matched PSNR `26.318562 dB`，同噪声 B0 `26.167079 dB`，delta `+0.151483 dB`。
- 输出：`outputs/EXP-S17-LATDIFF-002/`

### ANALYSIS-S17-LATDIFF-HOLDOUT-002 / BOOTSTRAP-001：一次性 256×5 机制审计

- 日期：2026-07-15
- 命令：`TORCH_HOME=.../outputs/cache/torch python3 scripts/s17_channel_matched_latent_diffusion.py --mode holdout --device cuda:0`
- holdout：validation role 第 256--511 图，base channel seed `20260733`，1,280 rows；与 selection 不重叠。
- frozen per-sample SHA-256：`a13bfe7ffa827d421c8f64c28226546ca79d98d2cfbf7f783672cba2236e1363`
- matched DDIM−B0：PSNR `+0.148715 dB`，image-cluster 95% CI `[+0.129607,+0.168857]`；LPIPS `-0.035305`，CI `[-0.038907,-0.031814]`。
- raw→matched latent MSE：`0.145516→0.060453`；五档全部下降。
- matched−fixed-7dB-step：`+0.233455 dB`，CI `[+0.220234,+0.246661]`。
- 分 SNR PSNR delta：`+0.614949/+0.199503/+0.010328/-0.054083/-0.027123 dB`。
- B1 对照：B1 `27.016222 dB/0.190905 LPIPS`；matched DDIM `26.131949/0.271167`；naive matched→B1 比 B1 `-0.231266 dB`，CI `[-0.241379,-0.221355]`。
- pseudo semantic：AlexNet eligible new/repair `26/83`；三分类器 majority new/repair `7/38`。只作 COCO pseudo 漂移诊断。
- 预注册检查：6/8 通过；未通过 4/5 SNR PSNR 正收益和 matched→B1 不劣于 B1。verdict `NEGATIVE_OR_PARTIAL`。
- 输出：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-002/`、`outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-001/`
- 报告：`reports/channel_matched_latent_diffusion_stage_result_2026-07-15.md`

### ANALYSIS-S17-LATDIFF-LOSS-DIAGNOSTIC-001：decoder image loss 尺度冻结

- 日期：2026-07-15
- 状态：完成；只使用 train population，不访问 selection/holdout
- parent：`EXP-S17-LATDIFF-002` best，SHA `cfc52716...b4e1`
- 规则：前 16 train batches；目标 weighted decoder/base loss ratio `0.075`；候选 `[5,10,20,40]` 按对数最近选择。
- 结果：base loss `0.67200185`，decoder image MSE `0.002449218`，连续权重 `20.5781`，冻结 `lambda_img=20`，实际 ratio `0.0728932`。
- 输出：`outputs/analysis/ANALYSIS-S17-LATDIFF-LOSS-DIAGNOSTIC-001/`

### EXP-S17-LATDIFF-003-CONTROL / 004-DECODER：同预算 decoder-aware 因果对照

- 日期：2026-07-15
- 项目 commit（注册时）：`abf117bc9aa194cd0a4fa80bd63df6baae1f6d29`
- 预注册：`reports/decoder_aware_latent_diffusion_preregistration_2026-07-15.md`
- 配置：`configs/s17_decoder_aware_latent_diffusion_control.yaml`、`configs/s17_decoder_aware_latent_diffusion.yaml`
- 命令：`python3 scripts/s17_channel_matched_latent_diffusion.py --config <config> --mode train --device cuda:0`
- 共同合同：同 `EXP-S17-LATDIFF-002` parent warm-start、同 10k train、同 validation 512--767 selection、同 seed、同三轮 FP32/AdamW/LR `1e-4`、同 6-step matched DDIM。
- 唯一方法差异：control `decoder_image_mse_weight=0`；decoder-aware `decoder_image_mse_weight=20`，frozen decoder 只对输入回传。
- control best：epoch 1，selection matched PSNR `26.356077 dB`，SHA `edbcbdbd7f78384decab40572728fab384e06bef854445af9452ee555aab2b1f`。
- decoder best：epoch 2，selection matched PSNR `26.377630 dB`，SHA `5b708117a5d25cad0a5909a24f85bb32d1b5dc11c83146ba8c98fad5ee35d98f`。
- 输出：`outputs/EXP-S17-LATDIFF-003-CONTROL/`、`outputs/EXP-S17-LATDIFF-004-DECODER/`。

### ANALYSIS-S17-LATDIFF-HOLDOUT-003 / BOOTSTRAP-002：fresh 232×5 审计

- 日期：2026-07-15
- holdout：validation role 第 768--999 图，base channel seed `20260736`，1,160 rows；此前未暴露。
- 命令：`TORCH_HOME=outputs/cache/torch python3 scripts/s17_channel_matched_latent_diffusion.py --config configs/s17_decoder_aware_latent_diffusion.yaml --mode holdout --device cuda:0`
- frozen per-sample SHA-256：`9a42ce71c05036f6401a0509b4aa6cde200b660a5acad603b0ce0293926baf92`。
- decoder−control：PSNR `+0.021605 dB`，95% CI `[+0.018883,+0.024640]`；LPIPS `-0.002502`，CI `[-0.002824,-0.002203]`；latent MSE `-0.002324`。
- decoder−parent：PSNR `+0.021006 dB`，CI `[+0.019132,+0.022939]`；decoder−B0：`+0.174221 dB/-0.038540 LPIPS`。
- 分 SNR decoder−B0 PSNR：`+0.668454/+0.240784/+0.037157/-0.047710/-0.027578 dB`；仍只有 3/5 为正。
- pseudo semantic：AlexNet decoder new/repair `29/72`，majority `5/31`；只作 COCO pseudo audit。
- decoder-aware DDIM→旧 B1 相对 B1：`-0.198803 dB/+0.033593 LPIPS`，确认 naive fusion 失败。
- 预注册 checks：7/8；verdict `NEGATIVE_OR_PARTIAL`。报告：`reports/decoder_aware_latent_diffusion_stage_result_2026-07-15.md`。
- 输出：`outputs/analysis/ANALYSIS-S17-LATDIFF-HOLDOUT-003/`、`outputs/analysis/ANALYSIS-S17-LATDIFF-BOOTSTRAP-002/`。

### EXPORT-S18-FRESH-POPULATION-001：未使用 COCO 512 图冻结 population

- 日期：2026-07-15
- 配置/预注册：`configs/s18_snr_identity_envelope.yaml`、`reports/snr_identity_envelope_preregistration_2026-07-15.md`
- 命令：`python3 scripts/s18_prepare_fresh_coco_population.py`
- 规则：旧 11,000 source path/SHA 双重排除，val2017 同名排除，`SHA256(20260738:relative_path)` 排序；前 256 selection、后 256 holdout。
- 结果：剩余候选 107,287；path/SHA overlap=`0/0`；manifest SHA `c467d2ccadd94242f51ff683f09d7b43d91a07d5edc538d06c06f2b6d93a8bed`。
- 输出：`outputs/eval/s18_fresh_coco_identity_envelope_population/`。

### ANALYSIS-S18-IDENTITY-SELECTION-001：SNR envelope policy 冻结

- 日期：2026-07-15
- frozen diffusion：`EXP-S17-LATDIFF-004-DECODER`，SHA `5b708117...5d98f`；selection seed `20260739`，256图×5SNR=1,280 rows。
- 候选：smooth power `p=0.25/0.5/1/2` 与 `hard_identity_7db`；网络/码率/sampler 全冻结。
- selection：smooth `p=0.5` mean ΔPSNR `+0.199099 dB` 但仅 3/5 SNR 非负；hard policy `+0.194020 dB/-0.036302 LPIPS` 且 5/5 非负，按预注册字典序入选。
- 冻结策略：1/4/7 dB `g=1`，13/19 dB `g=0`；policy SHA `c31d68533bf4e470d585ff5d279e948a71dca3ead1964902647393b2f37d05eb`。
- 输出：`outputs/analysis/ANALYSIS-S18-IDENTITY-SELECTION-001/`。

### ANALYSIS-S18-IDENTITY-HOLDOUT-001 / BOOTSTRAP-001：一次性 256×5 正式审计

- 日期：2026-07-15
- 命令：`TORCH_HOME=outputs/cache/torch python3 scripts/s18_snr_identity_envelope.py --mode holdout --device cuda:0`；bootstrap：`python3 scripts/s18_snr_identity_bootstrap.py`。
- holdout：新 population 后 256 图，channel seed `20260740`；1,280 rows。CSV SHA `b488fc0205fe3535ea5d128a99f7808ce0c701d379ed242c08e458e1502f95ea`。
- selected−B0：PSNR `+0.189717 dB`，95% CI `[+0.170601,+0.210902]`；LPIPS `-0.036284`，CI `[-0.039420,-0.033268]`。
- 分 SNR PSNR：`+0.677172/+0.240940/+0.030472/0/0 dB`；低中 SNR full-gain retention `99.999998%`。
- selected−full PSNR：`+0.015642 dB`，CI `[+0.014230,+0.016915]`；selected−full LPIPS `+0.001005`，记录为可靠性/感知 tradeoff。
- pseudo semantic：AlexNet full/selected new-repair=`17/81` 与 `16/71`；majority=`7/32` 与 `5/31`。
- B1−selected PSNR=`+0.830617 dB`，CI `[+0.791172,+0.869723]`；不得声称整体超过 B1。
- 预注册 10/10 checks PASS；报告：`reports/snr_identity_envelope_stage_result_2026-07-15.md`。
- 输出：`outputs/analysis/ANALYSIS-S18-IDENTITY-HOLDOUT-001/`、`outputs/analysis/ANALYSIS-S18-IDENTITY-BOOTSTRAP-001/`。

### EXPORT-S19-FUSION-POPULATION/CACHE-001：全新融合训练与审计 population

- 日期：2026-07-16
- 配置/预注册：`configs/s19_diffusion_fusion_ablation.yaml`、`reports/diffusion_fusion_ablation_preregistration_2026-07-16.md`
- population：5000 train / 256 selection / 256 holdout；排除旧 11,000、S18 512 与 val2017 同名；path/SHA overlap=`0/0`
- source manifest SHA：`b73c05656865eb6023c40dd57dfde176d05141eaf9b996feff92a41894522fe9`
- cache：27,560 行；27,560 B0 PNG + 16,536 低 SNR identity-diffusion PNG；最终 manifest SHA `8d88daf70a5ad07c213674f883c02d0a5f9b84ca082ff26727a5baead60775e3`
- 中断记录：第一次 cache 在 19 dB `sample_001161.png` 写入时被中断，留下唯一损坏 PNG；第一次训练读图失败、无 epoch outcome。旧 cache/失败训练/selection 均改名保留。runner 改成原子写入后只重生成损坏项，全量 49,608 PNG 验证 bad=0。
- 状态：**PASS population/cache closure；一次基础设施失败完整记录**

### EXP-S19-FUSION-001：同容量 B0-control vs diffusion-fusion

- 模型：control/fusion 各 450,115 参数，均由 S16-B1 checkpoint 展开；初始 state 相同，新增 auxiliary head slice 为零，实际 batch 上两分支与 B1 输出最大差 0。
- 唯一输入差异：control=`[B0,B0,SNR,Sobel,Laplacian]`；fusion=`[B0,D_identity,SNR,Sobel,Laplacian]`
- 训练：25,000 行/epoch×10；同 batch/crop/flip；epoch0 纳入候选，各分支独立按 selection mean PSNR 选 checkpoint。
- control best：epoch7，selection `27.44533 dB`，SHA `c9eab7648bd0120e5db2820b7a94edd13251ed8e1050cec6edb6f6187fbdbcf6`
- fusion best：epoch9，selection `27.50549 dB`，SHA `e7577d59e9c2362e40feb72bb030c1c7b9302b707115d3e000c9c55bbb8942c5`
- 输出：`outputs/EXP-S19-FUSION-001/`、`outputs/analysis/ANALYSIS-S19-FUSION-SELECTION-001/`
- 状态：**PASS checkpoint freeze；holdout 读取前 SHA 已写入配置**

### ANALYSIS-S19-FUSION-HOLDOUT/BOOTSTRAP-001：Diffusion 互补信息因果审计

- holdout：全新 256 图×5 SNR=1,280 行；per-sample SHA `cd5ba7899b52790cd242a447b22a455395be32939bc180f16e8ddd2ffa260f54`
- 平均 PSNR：B0 `26.24075`、identity diffusion `26.44454`、B1 `27.30482`、control `27.34803`、fusion `27.40649 dB`
- fusion−control：PSNR `+0.05846 dB`，cluster-bootstrap 95% CI `[+0.05198,+0.06423]`；LPIPS `-0.001493`，CI `[-0.002162,-0.000824]`
- fusion−B1：PSNR `+0.10168 dB`，CI `[+0.09431,+0.10915]`；LPIPS `-0.006394`
- 分 SNR fusion−control：`+0.09186/+0.11031/+0.13407/-0.02492/-0.01903 dB`
- pseudo semantic majority：fusion new/repair/failure `54/280/727`；control `60/276/737`
- checks：6/7；主 CI、LPIPS、B1、三项 pseudo semantic 均过；仅 4/5 nonnegative-SNR gate 未过（实际 3/5）。
- 报告：`reports/diffusion_fusion_ablation_stage_result_2026-07-16.md`
- 输出：`outputs/analysis/ANALYSIS-S19-FUSION-HOLDOUT-001/`、`outputs/analysis/ANALYSIS-S19-FUSION-BOOTSTRAP-001/`
- 状态：**PRIMARY PASS；严格证明 diffusion 对 B0-only/B1 路径有互补信息，但高 SNR 共享权重负迁移未解决**

### ANALYSIS-S20-SGD-B1-DECISION-001：SGD 全程替代 B1 的扩展判定

- 日期：2026-07-17
- 配置/预注册：`configs/s20_sgd_b1_decision.yaml`、`reports/sgd_b1_decision_preregistration_2026-07-17.md`
- population：64 张 Imagenette `policy_dev`、`T_cls` clean-correct 图；排除旧 8 图 author-rate pilot，10 类分层为 `7/7/7/7/6/6/6/6/6/6`；official validation 未访问。
- population reference SHA：`a08b0d3f3dead68919bea42a0a28c7854e998aea6173fe62d4669bd537ab393f`。
- channel：AWGN，SNR `[1,4,7,13,19] dB`，base seeds `[20260748,20260749,20260750]`；每方法 `64×5×3=960` rows；同 `(seed,sample,SNR)` canonical 19,712-D noise SHA 必须一致。
- 方法：B0-full；严格 `19,632 image + 80 UInt2-R4 payload` 的 B0-strict/B1；作者发布 SGD-JSCC main+active-edge `19,712 real`，caption 按论文免费、完美传输，明确标为 paper upper bound。
- B0-full/B0-strict/B1 总体：PSNR `27.10576/27.05489/28.12459 dB`，LPIPS `0.255417/0.257027/0.159398`，failure `111/115/35`。
- SGD-paper-upper 总体：PSNR `27.74037 dB`、MS-SSIM `0.952973`、LPIPS `0.072101`、failure `25`；三种子输出均 `320/320` rows，每 SNR 64 张。
- SGD−B0-full：PSNR `+0.634607 dB`，95% CI `[+0.380138,+0.890939]`；LPIPS `-0.183316`，CI `[-0.198140,-0.168312]`；failure `111→25`。确认论文方法在其有利协议下明显强于普通 JSCC。
- SGD−B1：PSNR `-0.384224 dB`，95% CI `[-0.615292,-0.160262]`；MS-SSIM `+0.006276`，CI `[+0.004162,+0.008386]`；LPIPS `-0.087297`，CI `[-0.100439,-0.075641]`；failure `35→25`，但 failure-rate CI `[-0.042708,+0.014583]` 跨零，new/repair=`11/21`。
- 分 SNR SGD−B1 PSNR 为 `-0.10747/-0.55384/-0.42660/-0.62374/-0.20946 dB`，LPIPS 为 `-0.11858/-0.09200/-0.08599/-0.06834/-0.07157`；五档均体现 fidelity/perception tradeoff，而不是全面支配。
- rate audit：公开 SGD main+active-edge 已为 `19,712 real`；四个 536-bit caption 的最低未保护 BPSK 成本为 `2,144 real`，最低总量 `21,856`，超预算 `2,144 real=10.8766%`。严格 full-SGD route 不可执行，除非重新分配并训练。
- 计算：SGD `2064.738 ms/图`、7458.9 MiB；B1 `2.642 ms/图`、5551.5 MiB；本机实测平均 runtime ratio `781.4×`。
- aggregate SHA：`3023ac917f4705fd6a705ea08b7ebe99b5dec4e529c0b77dcc1d414a7dd364d5`。
- 输出：`outputs/external_baselines/ANALYSIS-S20-SGD-B1-DECISION-001/`
- 报告：`reports/sgd_b1_decision_stage_result_2026-07-17.md`
- 状态：**PASS protocol / NEGATIVE full-SGD dominance；保留严格保真路径与受控 diffusion，不支持无条件全程 SGD**

### EXP-S21/S22：B1 与 matched diffusion 合并 development 闭环

- 日期：2026-07-20
- 配置：`configs/s21_b1_anchored_gated_fusion.yaml`、`configs/s21_b1_anchored_gated_fusion_no_gate_penalty.yaml`、`configs/s21_b1_anchored_auxiliary_residual.yaml`、`configs/s21_b1_diffusion_convex_envelope.yaml`、`configs/s22_b1_feature_injection.yaml`
- population/cache：5000 train / 256 selection / 256 sealed holdout；source manifest SHA `4f31ecb6...c7db5`；cache 27,560 rows，SHA `dd79fe2f...84b87`；与 S16/S18/S19 path/SHA overlap=`0/0`。
- S21 learned gate：带 penalty 第1轮塌零；无 penalty 第4轮塌零，best eligible 均为 exact B1。
- S21 fixed gate：第3轮 mean abs injection 到达 envelope ceiling `0.06`，selection PSNR=`22.73128 dB`，按失败处理停止。
- S21 convex：120 个单调 alpha 组合仅全零可行；policy SHA `a6fe207b...80c7`，未访问 holdout。
- S22 方法：冻结 B1，仅训练 zero-init `Conv3x3(3→64,bias=False)` 的 1,728 参数，输入为 `D-B0`；13/19 dB envelope=0，control `B0-B0=0`。
- smoke：初始 fusion−B1 最大逐像素差=`0`；projection gradient L1=`0.03053785`，有限且非零。
- S22 selection：epoch1/6/10 的 PSNR delta=`-0.01887/-0.01789/-0.02031 dB`；LPIPS delta=`-0.01096/-0.01487/-0.01580`。10 个训练 epoch 均未同时超过 B1 PSNR。
- 决策：按冻结规则选择 epoch0；checkpoint SHA `b7eac7ece17cef6d4c478f69a1d6f623a24e545a7eda3bd8d1681250bbda0d79`；`holdout_accessed=false`，不运行 holdout/bootstrap。
- 运行命令：`PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --mode smoke --device cuda:0`；`PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --mode train --device cuda:0`。
- 关键源码：`src/cadsd_jscc/b1_feature_injection.py`、`scripts/s21_b1_anchored_gated_fusion.py`、`scripts/s21_b1_diffusion_convex_envelope.py`、`scripts/s22_b1_feature_injection.py`。
- 报告：`reports/b1_merge_stage_result_2026-07-20.md`。
- 状态：**NEGATIVE current parameterization / POSITIVE perceptual-direction diagnostic；holdout sealed**

### EXP/ANALYSIS-S23-B1FS-001：冻结 B1 的非零特征 Shrink 闭环

- 日期：2026-07-20
- 事后性：注册时已知 S22 selection 轨迹；明确作为 development follow-up，不伪装为独立事前方法发现。S22/S23 official Imagenette validation 均未访问。
- 配置/预注册：`configs/s23_b1_feature_shrink.yaml`、`reports/b1_feature_shrink_preregistration_2026-07-20.md`。
- 冻结设计：复现 S22 最早的 epoch1 projection endpoint；全局 alpha 网格 `[0,.01,.025,.05,.075,.1,.15,.2,.35,.5,.75,1]` 在结果前冻结；不扫描 epoch、per-SNR alpha、LR 或 loss。
- selection：选中 `alpha=0.15`；fusion−B1 PSNR=`+0.000536 dB`、LPIPS=`-0.001681`；1/4/7 dB PSNR=`+0.000566/+0.001114/+0.001000 dB`；13/19 dB exact B1。
- checkpoint SHA：`53692278be57a918a70146a6c72bb5a44fe76e871d590a82f2f6fd5c7a21abbf`；policy SHA：`54c2639f569df12b06f65d3356b1d64d14184a32e2603b69c2115d3c7bc8c68f`。二者冻结后才首次访问 holdout。
- holdout：256 images×5 SNR=1,280 rows；per-sample SHA `9f4dd87d0923121da89e7e4c968d63d15164e4d72707665b6a08ebc58e771fa3`。
- fusion−B1 PSNR=`+0.000568 dB`，source-cluster 95% CI `[+0.000378,+0.000771]`；LPIPS=`-0.001731`，CI `[-0.001849,-0.001622]`。
- 分 SNR PSNR=`+0.000701/+0.001158/+0.000979/0/0 dB`；LPIPS=`-0.003789/-0.003000/-0.001864/0/0`；13/19 dB 最大逐像素差 0。
- pseudo semantic：AlexNet new/repair=`1/2`；三分类器 majority new/repair=`3/7`。不是零 new error，不声明绝对语义安全。
- bootstrap：10,000 source-image cluster replicates，seed `20260764`；5/5 checks PASS；summary SHA `14ce060d7481300de6f510b7475a13b5febd593255b308c5cff717b681e10ca7`。
- 运行命令：`PYTHONPATH=src python3 scripts/s23_b1_feature_shrink.py --device cuda:0`；冻结 SHA 后运行 `PYTHONPATH=src python3 scripts/s22_b1_feature_injection.py --config configs/s23_b1_feature_shrink.yaml --mode holdout --device cuda:0` 与 `--mode bootstrap`。
- 报告：`reports/b1_merge_stage_result_2026-07-20.md`。
- 状态：**5/5 PASS / POSITIVE mechanism closure；效应量太小，尚非强主方法**

### ANALYSIS-S24-RECENT-PROGRESS-SUMMARY-001：近期进度与指标统一复核

- 日期：2026-07-20。
- 性质：对已冻结 S19/S20/S23 产物的派生汇总；不训练、不重新选模型、不用 holdout 调参、不访问 official Imagenette validation。
- 配置/脚本：`configs/s24_recent_progress_metrics.yaml`、`scripts/s24_recent_progress_metrics.py`。运行前验证 S19/S20/S23 输入 SHA，按 source image cluster 跨 5 个 SNR 做 10,000 次 bootstrap，seed=`20260766`。
- S23 同一 COCO holdout：B0/diffusion/B1/fusion 的 PSNR 为 `26.46104/26.67858/27.56772/27.56829 dB`，MS-SSIM 为 `0.919915/0.932561/0.943591/0.943613`，LPIPS 为 `0.300710/0.261052/0.183951/0.182220`。
- S23−B1：PSNR `+0.000567`，95% CI `[+0.000376,+0.000762]`；MS-SSIM `+0.0000224`，CI `[+0.0000116,+0.0000338]`；LPIPS `-0.001731`，CI `[-0.001844,-0.001619]`。majority `3 new / 7 repair`，failure difference CI `[-0.0078125,+0.00078125]` 跨 0，因此只声明质量增益，不声明显著语义改善。
- S19 复核：fusion−control PSNR `+0.05843 dB`、LPIPS `-0.001487`；fusion−B1 PSNR `+0.10173 dB`、LPIPS `-0.006397`。两项质量 CI 均有利，但 majority failure CI 跨 0，且 S19 仍有高 SNR 负迁移。
- S20 外部定位沿用冻结结果：SGD 免费/完美文本上界相对 B0 明显有利；相对 B1 为 PSNR `-0.38422 dB`、LPIPS `-0.08730`，同时有 `11 new / 21 repair`。caption 最低增加 `2,144 real`，严格预算至少超出 `10.88%`，不能与 S23 宣布端到端公平胜负。
- receiver postprocessor microbenchmark：RTX 4090 D、batch16、warmup50、timed200；给定已缓存 B0/diffusion，B1=`2.49145 ms/图`，S23=`2.60177 ms/图`，增量 `0.11032 ms/图`（`4.43%`）。该范围不含 DeepJSCC、6-step diffusion、分类器和指标，不是端到端延迟。
- 参数/码率：B1=`448,387` 参数，S23 新增 `1,728`、总计 `450,115`；S19/S23 均仍为 `19,712 real`，无额外 side-information symbols。
- 输出哈希：`summary.json` SHA `72b607fa...4f327`；same-population CSV SHA `75c14d37...d4b2`。
- 中文报告：`reports/recent_progress_metrics_and_data_flow_2026-07-20.md`。
- 状态：**COMPLETE DERIVED ANALYSIS；S19 为质量幅度上限，S23 为 exact-fallback 机制基线，尚无统一强终点**

### ANALYSIS-S25-B1FA-HEADROOM-001：S23 逐图幅度上限诊断

- 日期：2026-07-20。
- 性质：已知 S23 outcome 后的 selection-only feasibility diagnostic；12 个 alpha、继续门槛和 oracle 定义均在输出前冻结，不训练、不访问 holdout。
- fixed `0.15`：PSNR `27.099880`、LPIPS `0.185948`，majority `4 new / 5 repair`。
- semantic-safe PSNR oracle：PSNR `27.101245`、LPIPS `0.184132`，相对 fixed 为 `+0.001365 dB/-0.001817 LPIPS`，majority `0 new / 10 repair`。
- source-image cluster bootstrap：PSNR headroom 95% CI `[+0.001186,+0.001562]`；统计上非零，但效应量远低于预注册 `+0.02 dB`。
- 纯 LPIPS oracle：相对 fixed LPIPS `-0.01030`，但 PSNR `-0.01329 dB` 且 majority new=`18`，显示大幅注入主要换取感知纹理并增加 drift 风险。
- 判定：4 项 gate 通过 3 项，`continue_to_receiver_visible_controller=false`；不再在该 feature direction 上训练/扫描 controller。
- 报告：`reports/b1_feature_amplitude_headroom_stage_result_2026-07-20.md`。
- 状态：**CLEAR NEGATIVE FOR S23 AMPLITUDE-CONTROLLER ROUTE；holdout sealed**

### ANALYSIS-S26-S19-XF-REPLICATION-001：S19 强融合的 exact-B1 跨总体复现

- 日期：2026-07-20。
- 事后性：S19/S23/S25 outcome 已知，目标 population 的 B1/S23 outcome 已知；但 frozen S19 checkpoint 的目标输出未知，route/checkpoint/门槛在运行前冻结，不访问目标 selection。
- 方法：1/4/7 dB 使用 frozen S19 fusion 及 paired frozen control；13/19 dB 两者均结构性返回 frozen B1。新增参数=0，side-information symbols=0。
- routed fusion−B1：PSNR `+0.093267 dB`，95% CI `[+0.087945,+0.098806]`；MS-SSIM `+0.002188`，CI `[+0.001972,+0.002412]`；LPIPS `-0.007661`，CI `[-0.008438,-0.006915]`。
- diffusion causality：routed fusion−control PSNR `+0.065486 dB`，CI `[+0.061088,+0.069994]`；LPIPS `-0.003100`，CI `[-0.003670,-0.002528]`。
- semantic diagnostic：B1/routed-control/routed-fusion majority failure=`744/734/720`；routed fusion 相对 B1 `27 new / 51 repair`，failure difference CI `[-0.03203,-0.00547]`。
- per-SNR fusion−B1 PSNR=`+0.141105/+0.154616/+0.170612/0/0 dB`；13/19 dB exact max difference=0。
- 判定：9/9 checks PASS；当前最强安全融合阶段性结果。目标图片总体此前已用于别的方法，仍需 fresh population final replication。
- 报告：`reports/s19_exact_fallback_replication_stage_result_2026-07-20.md`。
- 状态：**POSITIVE FROZEN CROSS-POPULATION REPLICATION / CURRENT BEST METHOD**

### ANALYSIS-S27-S19-XF-FRESH-001：完全新图总体的主方法复现

- 日期：2026-07-21。
- population：从 COCO train2017 排除 S16/S18/S19/S21 22,536 个唯一 source path/SHA 后冻结 512 张；旧总体 path/SHA overlap=`0/0`，内部重复 SHA=0，无 selection。
- cache：512×5=2,560 个 canonical-noise B0；1/4/7 dB 共 1,536 个 matched-diffusion；cache manifest SHA `2f5b6ec3...556fa`。
- routed fusion−B1：PSNR `+0.092662 dB`，95% CI `[+0.089147,+0.096313]`；MS-SSIM `+0.002310`，CI `[+0.002149,+0.002482]`；LPIPS `-0.007922`，CI `[-0.008465,-0.007398]`。
- diffusion causality：fusion−control PSNR `+0.065799 dB`，CI `[+0.062673,+0.068775]`；LPIPS `-0.003494`，CI `[-0.003884,-0.003110]`。
- semantic：B1/control/fusion majority failure=`1561/1537/1517`；fusion 相对 B1 `60 new / 104 repair`，failure difference CI `[-0.02813,-0.00664]`。
- per-SNR fusion−B1 PSNR=`+0.135781/+0.153492/+0.174039/0/0 dB`；13/19 dB exact max difference=0。
- 判定：9/9 PASS；S26 数字在 pristine population 高精度复现，内部方法冻结。
- 报告：`reports/s19_exact_fallback_fresh_replication_stage_result_2026-07-21.md`。
- 状态：**POSITIVE PRISTINE-POPULATION REPLICATION / INTERNAL METHOD FROZEN**

### ANALYSIS-S28-CURRENT-VS-SGD-001：当前方法与 SGD-JSCC 同总体外部定位

- 日期：2026-07-21。
- population/channel：冻结 S20 Imagenette policy-dev 64 图，3 个 canonical AWGN seed×5 SNR=960 行/方法；无训练、调参或 official validation 访问。
- current−B1：PSNR `+0.099085 dB`，95% CI `[+0.088053,+0.111284]`；MS-SSIM `+0.002360`；LPIPS `-0.007314`；failure `35→29`，`6 new / 12 repair`。
- current−matched-control：PSNR `+0.059681 dB`，CI `[+0.050030,+0.069327]`；LPIPS `-0.002990`，CI `[-0.004083,-0.001982]`；failure `36→29`。
- current−SGD paper upper：PSNR `+0.483309 dB`，CI `[+0.258829,+0.711956]`；MS-SSIM `-0.003916`；LPIPS `+0.079983`；failure `29 vs 25`，为 fidelity/perception Pareto。
- rate：current=19,712 real、auxiliary side information=0；SGD main+edge=19,712 real 且至少 2,144 caption real 未计费，严格计费最低超预算 `10.88%`。
- 技术判定：B1 改善 5/5 checks 均通过；但 batch=16 重算 B1 的最大 PSNR 浮点差 `0.0004768 dB` 超过预注册 `0.0001 dB`，故 S28 原 verdict 不事后修改，保留 `NEGATIVE`。
- 报告：`reports/current_method_external_positioning_stage_result_2026-07-21.md`。
- 状态：**CLEAR EXTERNAL PARETO POSITIONING / TECHNICAL VERDICT NEGATIVE PENDING DIAGNOSTIC**

### ANALYSIS-S29-S28-B1-EXACT-BATCH-001：S28 合同浮点差诊断

- 日期：2026-07-21。
- 事后性：S20/S28 outcome 已知；仅诊断，不是独立复现。
- 方法：完全恢复 S20 原 batch=64，重放同 64 图、3 seed、5 SNR 的 B1；不改模型、人口、载荷、噪声或阈值。
- 结果：960 行 noise SHA、T_cls prediction、failure、PSNR、MS-SSIM、LPIPS 与冻结 S20 B1 全部逐项一致，所有最大绝对差为 `0`。
- 判定：6/6 PASS；S28 的形式失败已由 batch-dependent 浮点次序完全解释，不是人口/信道合同错位。S28 原 verdict 仍保留，不回写历史。
- 状态：**CONTRACT DIAGNOSTIC PASS / S28 COMPARISON VALID WITH DOCUMENTED NUMERICAL CAVEAT**

### ANALYSIS-S30-DIFFJSCC-CHECKPOINT-AUDIT-001：官方 DiffJSCC 检查点结构审计

- 日期：2026-07-21。
- 性质：任何 DiffJSCC 重建前的官方资产完整性审计；不访问图像总体、不运行信道或 diffusion，不产生方法效果结论。
- 固定资产：`Mingyuyang/DiffJSCC-OpenImage-CBR-1-96/model.ckpt`，实际/预期尺寸均为 `9,859,655,693` bytes，SHA-256=`ae1e6df0b706d09857cfa02d399f94cc171d8d0ce44f851d96cb032bd7dec579`。
- state dict：`1,896` 个 tensor key、`1,734,266,992` 个元素、tensor bytes=`6,937,068,068`；`1,871` 个 float32 tensor、`25` 个 int64 tensor。
- 核心前缀均非空：OpenCLIP text=`294` keys / `354,032,641` numel；DeepJSCC=`219` / `31,289,260`；ControlNet=`328` / `365,203,840`；diffusion UNet=`686` / `865,910,724`；VAE=`248` / `83,653,863`；spatial condition encoder=`108` / `34,163,664`。
- `blip_model.*` 为 `0` key，与作者 `on_save_checkpoint` 主动排除逻辑一致；因此必须另载固定 base `Salesforce/blip2-opt-2.7b`，禁止用 SGD-JSCC COCO-BLIP、随机权重或其他 caption 模型替代。
- 输出 summary SHA-256=`1c3a0ef130f348a7377db1b0270e9cdaa9ace2ccaa38422a3f76fb15273c4eae`。
- 当时状态：**PASS / OFFICIAL DIFFJSCC GENERATIVE CHECKPOINT COMPLETE；等待精确 BLIP2 后做端到端 preload/smoke**。后续 BLIP2、preload、smoke 和 full 已完成，见下一节；保留本句作为阶段时序记录。

### ANALYSIS-S30-DIFFJSCC-PREFLIGHT/SMOKE/FIRST-SEED-001：精确资产到阶段推理闭环

- 日期：2026-07-21。
- preflight：作者源码 commit、checkpoint、BLIP2 两个分片、OpenCLIP/Transformers runtime、64 图总体、960 个 canonical noise key 和 symbol ledger 全部 PASS。BLIP2 分片实际总量 `14,979,207,136` bytes，两个 SHA-256 与固定 revision 精确匹配。
- preload：checkpoint 只缺作者设计上排除的 `1,248` 个 `blip_model.*` key；其他 missing/unexpected=`0/0`，由精确 base BLIP2 补齐。
- smoke：第一图、seed `20260748`、1 dB、作者 100-step 链 PASS；DiffJSCC `29.15878 dB/0.099317 LPIPS`，author-JSCC `31.21237/0.098876`，current `29.68875/0.105058`；全部 T_cls correct。总耗时 `5.347 s`、peak allocated VRAM `14,924.41 MiB`。只验证可执行性，不作胜负结论。
- first-seed：320/320 行完成。current−DiffJSCC PSNR `+0.629471 dB`，CI `[+0.412868,+0.839893]`；LPIPS `+0.052415`，CI `[+0.041445,+0.064039]`；failure current/Diff=`8/7`。阶段 verdict=`PARETO_OR_INCONCLUSIVE`。
- 状态：**PASS STAGED EXECUTION；冻结设置不变进入 full**。

### ANALYSIS-S30-DIFFJSCC-COMPARISON/POST-003：官方 DiffJSCC 完整外部对比

- 日期：2026-07-21。
- population/channel：冻结 S20/S28 Imagenette policy-dev 64 图×3 canonical AWGN seed×5 SNR=`960` 行；official validation 未访问。完整 CSV 为 960 个唯一键、960 个噪声 SHA、960 张三联图，全部 finite；归一化复功率 `[0.999999821,1.000000238]`。
- 码率：DiffJSCC latent=`16×32×32=16,384 real`、8,192 complex uses，相对原始 256 源 CBR=`1/24`；使用项目 19,712-real ceiling 的 `83.1169%`。receiver BLIP2 caption 传输符号=`0`。19 dB 超作者 `[0,14]` dB 训练范围，单列为外推。
- 总体方法：author-JSCC/DiffJSCC/current/B1 的 PSNR=`29.986135/27.598398/28.223678/28.124602`；MS-SSIM=`0.963092/0.940799/0.949057/0.946698`；LPIPS=`0.128342/0.100223/0.152084/0.159396`；failure=`22/23/29/35`。
- current−DiffJSCC：PSNR `+0.625280 dB`，source-image cluster 95% CI `[+0.423123,+0.824753]`；MS-SSIM `+0.008258`；LPIPS `+0.051861`（更差），CI `[+0.041360,+0.063002]`；failure-rate delta `+0.00625`，CI `[-0.017708,+0.034375]`。
- current−author-JSCC：PSNR `-1.762457 dB`，CI `[-1.938592,-1.601835]`；MS-SSIM `-0.014035`；LPIPS `+0.023742`，CI `[+0.019926,+0.027543]`；failure `29 vs 22`，差值 CI 跨零。强 backbone 质量三轴显著领先 current。
- DiffJSCC−author-JSCC：PSNR `-2.387737 dB`、MS-SSIM `-0.022293`、LPIPS `-0.028119`；三个 CI 均不跨零。failure `22→23`，failure-rate CI 跨零；new/repair=`10/9`，source clusters=`4/3`。
- 分 SNR DiffJSCC new/repair=`1/3、2/4、3/2、3/0、1/0`。1/4 dB 为净修复，7 dB 转为净风险，13 dB 在 3/3 seed 都为 `1 new/0 repair`；19 dB 为外推。
- 系统：author-JSCC/caption/diffusion/total 平均 `5.787/113.600/5115.503/5237.712 ms/图`；peak allocated VRAM `14,927.42 MiB`。S28 current 的毫秒字段不含端到端链，禁止直接算倍数。
- 输出哈希：`per_sample.csv=549720b804df6cfd87a7035ba37be096b0cf8e683634ebb7f93feff47c49b6e2`；runner summary=`6547e3b35480a8e1132f9b943fcca9b8889bc72e8bab9901fee5d72894cb137d`；final post analysis=`87c2ffcb2a699c4f39c1ab92ae88c8d515ed406ad5d808427626e8829fc2aa1f`。
- 验证：960 行独立 CSV/噪声/symbol/image/snapshot 审计和 `git diff --check` 通过。首次 `unittest` 调用漏 `PYTHONPATH=src`，产生 3 个 import error；标准入口 `PYTHONPATH=src python3 -m unittest discover -s tests` 重跑 `122/122` PASS。
- 报告：`reports/diffjscc_external_comparison_stage_result_2026-07-21.md`。
- 状态：**PASS PROTOCOL / PARETO_OR_INCONCLUSIVE；强 backbone + 风险受控 diffusion 成为下一建议，不声明全面战胜外部方法**。

### EXP-S33/S33B 与 ANALYSIS-S33：16,384-real Strong JSCC 严格等码率 Gate

- 日期：2026-07-21。
- 预注册：`reports/s33_strong_jscc_16384_preregistration_2026-07-21.md`、`reports/s33b_strong_jscc_16384_fp32_continuation_preregistration_2026-07-21.md`、`reports/s33_strong_jscc_16384_external_comparison_preregistration_2026-07-21.md`。
- 模型/码率：clean-room 四级 SNR-conditioned JSCC，`64x16x16=16,384 real`、8,192 complex uses、31,028,163 参数；mask/padding/side information=0。随机初始化，FP32 4+8 epochs，训练 SNR 为 `[1,4,7,13,19]` 离散逐图均匀采样，不是连续 SNR。
- 主训练：4-epoch PSNR=`26.062294/27.459733/28.280950/28.587876 dB`。epoch2 落盘后外部进程终止，config/snapshot/checkpoint SHA 一致，按同配置从 epoch3 恢复；无重复 history。主 best SHA=`b698797f...26c4`。
- 续训：首点 `28.510495 dB` 低于初始化，负点保留；随后升至 epoch7 `29.415098 dB/0.966782 MS-SSIM`。最终 SHA=`2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`；COCO per-SNR PSNR=`27.311886/28.571358/29.544483/30.634386/31.013377 dB`。
- 外部合同：冻结 S20/S32 policy-dev 64 图×3 seed×5 SNR；每 key 先验证完整 19,712-real canonical noise SHA，再取前 16,384 坐标，strong 与 author 严格同 noise prefix、同总实符号。official Imagenette validation 未访问。
- 聚合：strong/author PSNR=`30.466064/29.986135`、MS-SSIM=`0.969708/0.963092`、LPIPS=`0.119985/0.128342`、failure=`9/22`。strong−author PSNR=`+0.479929 dB`，95% CI=`[+0.370006,+0.598197]`；MS-SSIM=`+0.006616`、LPIPS=`-0.008357`、failure-rate=`-0.013542`，CI 均显著有利。
- 分档 PSNR delta=`+0.966257/+0.771715/+0.526119/+0.131229/+0.004323 dB`。13 dB LPIPS显著较差；19 dB PSNR CI=`[-0.161540,+0.202654]`，按 0.10 dB margin 未过非劣，且 LPIPS显著较差。
- artifact audit：960 rows/unique keys、960 full/prefix noise、960 author rows 全部 PASS；`per_sample.csv` SHA=`5e585ca5...7dbe1`，`post_analysis.json` SHA=`d7a89ef2...bdfc`。
- 报告：`reports/strong_jscc_16384_equal_rate_stage_result_2026-07-21.md`。
- 状态：**SIGNIFICANTLY SUPERIOR AT EXACT 16,384-REAL AGGREGATE / HIGHEST-PRIORITY PAPER GATE PASS；KNOWN POLICY-DEV, NOT FINAL TEST**。

### SMOKE-S34A-SWINJSCC-CALIBRATION-001：SwinJSCC 双臂 exact-rate GPU 校时

- 日期：2026-07-22。
- 范围：只做 official Base-SA 和 capacity-matched CM-SA 各一个真实 COCO FP32 microbatch；不训练 epoch、不做质量排名、不判断收敛、不访问 official Imagenette validation。
- 源码：`semcomm/SwinJSCC@a6d0e6da53548976acbe9317839a077ef31f190f`；直连 codeload tarball=`17,887 bytes`，SHA-256=`3f837eef...21688`；本地官方文件未改，项目侧 adapter 单独保存。
- Base：参数=`28,182,512`，latent=`[8,256,64]`，输出=`[8,3,256,256]`，loss=`0.331543`，gradient norm=`1.70264`，peak allocated/reserved=`8.988/9.748 GiB`，单 microbatch=`0.5366s`（含进程首臂 CUDA cold-start）。
- CM：参数=`31,348,752`，latent/output 与 Base 相同，loss=`0.318521`，gradient norm=`2.62209`，peak allocated/reserved=`9.563/10.396 GiB`，单 microbatch=`0.1788s`（warm-cache 参考）。
- 物理/可恢复性：两臂均为每图 `16,384 real`；单位功率最大误差=`1.19209e-7`；finite backward、AdamW step、strict checkpoint round-trip 均 PASS。scalar-SNR 下 adapter 与官方原始 encoder/decoder forward 最大绝对差=`0/0`。
- 正式 microbatch：冻结为 8，gradient accumulation=4，保持 effective batch=32。warm 单步线性估算约 `0.715s/optimizer-step`，单个 12-epoch 臂纯训练约 9 小时；最终以正式日志为准。
- 结果：`outputs/smoke/EXP-S34A-SWINJSCC-CALIBRATION-001/smoke_result.json`，SHA-256=`010d9befe939f3c9288755888c34857e68bcfcbeb7135fd0f9f685ef944cacb3`。
- 状态：**SYSTEMS PASS / CONVERGENCE UNKNOWN / FORMAL TRAINING NOT STARTED**。
