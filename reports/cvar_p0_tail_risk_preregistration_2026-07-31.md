# CVaR 候选方向二 P0：条件信道尾部风险诊断预注册

- 日期：2026-07-31
- analysis_id：`ANALYSIS-CVAR-P0-TAIL-RISK-001`
- 配置：`configs/cvar_p0_tail_risk_diagnostic.yaml`
- 入口：`scripts/cvar_p0_diagnose_tail_risk.py`
- 任务书来源：`候选二_CVaR尾部风险JSCC_Codex实验任务书.md`
- 状态：**在任何正式诊断统计产生前预注册**。仅完成 4 图 × 4 realization 的 smoke dry-run。

本阶段**不训练、不做 checkpoint 选择、不下载、不访问 official Imagenette validation**。

---

## 1. 与项目主线的关系（必读）

`PROJECT.md` 与 `MILESTONES.md` 明确把 Rayleigh 列为 COCO-256 AWGN 最小闭环**之后**的扩展项，且当前主线是 2026-07-23 修订的 S35R（P1 轻量接收端 refiner，smoke/训练均未获授权）。

因此本阶段的定位被严格限制为：

- 这是一次**只读诊断**，用于判断"候选方向二"是否值得投入，不是新主线。
- 任务书中的 P4/P5（repeated-mean 与 CVaR 训练）**不在本次授权范围内**，必须在诊断出 GO 且用户另行放行后才可启动。
- 本阶段不修改任何既有训练/评测/复现流程，不覆盖任何既有实验目录。

## 2. 仓库审计结果（P0）

### 2.1 模型

| 项 | 值 |
|---|---|
| 类名 | `StrongJSCC` |
| 实现 | `src/cadsd_jscc/strong_jscc.py` |
| 构造入口 | `scripts/s32_strong_jscc_external_comparison.py::build_model` |
| encoder / decoder | `model.encode(image, snr_db)` / `model.decode(received, snr_db)` |
| 信道输入归一化 | `model.normalize_channel_input`（逐样本全实坐标功率归一） |
| checkpoint | `outputs/train/EXP-S33B-STRONG-JSCC-16384-FP32-001/checkpoints/best.pt` |
| checkpoint SHA256 | `2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`（与 `MILESTONES.md` 冻结值一致，已在脚本内强制校验） |
| best epoch | 7 |
| latent | `64 × 16 × 16 = 16,384` 实符号，`8,192` 复信道使用 |
| exact CBR | `0.0416667 = 1/24` |
| 训练损失 | 图像 MSE（均值口径） |
| 训练数据 | COCO2017 `train2017`，`256×256` |
| 训练 SNR | 离散均匀 `[1, 4, 7, 13, 19]` dB |
| CSI 假设 | 发端与收端均已知**标称 SNR**（`transmitter_csi/receiver_csi: perfect_snr`） |

注意：这是一个 **SNR-conditioned（信道自适应）** 模型，encoder 与 decoder 都吃 `snr_db` 条件嵌入（`strong_jscc.py:44`）。这一点决定了本诊断的核心设计（见 §3）。

### 2.2 信道

- 现有实现只有 **AWGN**：`strong_jscc.py::transmit` 与 `src/cadsd_jscc/external_common.py::complex_awgn_from_standard_normal`。
- 全仓（排除 `third_party/`）**没有任何 Rayleigh 实现**；`rayleigh` 仅出现在 `.md` 文档中。
- SNR 定义：每**复**信道使用的 Es/N0。
- 功率归一化：逐样本在全部实坐标上测功率 `P`。
- 噪声：每个**实**坐标方差 `P / (2 · SNR_linear)`。
- 无均衡（AWGN 不需要）。

因此本阶段必须新增 Rayleigh，按任务书 §4.3 实现最小可复现版本。

### 2.3 数据与指标

- 数据：COCO2017 `val2017`（5,000 图，本地已存在）。
- 预处理：`Resize(256) → CenterCrop(256) → ToTensor()`（与 `s31_train_strong_jscc.py::make_transform(train=False)` 一致）。
- 指标：`src/cadsd_jscc/metrics.py` 提供 `psnr_per_sample` / `ms_ssim_per_sample`；LPIPS 经 `scripts/s5_residual_refiner_pilot.py::try_load_lpips`（AlexNet）。
- 输出量化：`floor_uint8`（与全仓 `primary_quantization` 一致）。

### 2.4 新增与修改文件

```text
新增:
- src/cadsd_jscc/tail_risk.py                  CVaR 工具 + block-fading Rayleigh 信道
- tests/test_tail_risk.py                      18 项单元测试
- scripts/cvar_p0_diagnose_tail_risk.py        诊断入口
- scripts/cvar_p0_analyze_tail_risk.py         尾部统计、图表、GO/NO-GO
- configs/cvar_p0_tail_risk_diagnostic.yaml    配置

修改:
- 无。既有信道、训练、评测代码零改动。
```

`external_common.py` **未被修改**：realization 级噪声通过 `canonical_standard_normal(base_seed, f"{image_id}|r{k}", snr, 16384)` 复用既有冻结函数。

## 3. 关键方法设计：为什么必须有四个 arm

任务书 §4.3 要求"接收端已知 CSI + ZF 均衡"。在 block fading 下，ZF 均衡后

```text
ỹ = x + conj(h)/|h|² · n
```

每实坐标噪声方差为 `P / (2 γ |h|²)`，即**等价于有效 SNR 为 `γ|h|²` 的 AWGN 信道**。
（`tests/test_tail_risk.py::test_equalized_noise_variance_matches_effective_snr` 与
`test_unit_gain_reduces_to_the_existing_awgn_path` 已固定这两条性质。）

由于本模型是 SNR-conditioned 的，"喂给 decoder 什么 SNR"就变成一个**独立的、会完全改变结论的自由变量**。任务书完全没有规定这一点。若随手选一种，得到的 GO 无法归因。故本阶段固定四个 arm，**共享同一次 encoder 前向与同一条标准正态噪声**，因此逐 realization 严格配对：

| arm | fading | decoder 条件 SNR | 作用 |
|---|---|---|---|
| `awgn_control` | 否（`h=1`） | 标称 | 隔离"无衰落时是否本来就有尾部" |
| `rayleigh_nominal_csi` | 是 | 标称 | 朴素部署：decoder 不知道衰落 |
| `rayleigh_effective_csi` | 是 | `标称 + 10log10|h|²` | 接收端用真实有效 SNR |
| `rayleigh_effective_csi_clamped` | 是 | 上者 clamp 到 `[1, 19]` dB | 有效 SNR 会跌出训练范围，此臂把条件保持在分布内 |

encoder 在所有 arm 中一律使用**标称 SNR**：block fading 无反馈链路，发端不可能知道 `h`。这是因果上诚实的设定。

`awgn_control` 由 `h=1` 实现，与既有 AWGN 路径按单元测试**逐元素相等**，因此它是同一 realization 的 `h=1` 反事实，而不是另一次独立采样。

### 3.1 主 arm 选择规则（tail-blind，防止 outcome-based selection）

三个 Rayleigh arm 只差接收端条件，所以"有没有尾部"在固定接收机之前无意义。规则：

> 逐 SNR 选择**平均 PSNR 最高**的 Rayleigh arm 作为该 SNR 的主 arm。

该规则**对一切尾部统计量盲**（只用均值），因此"用均值选 arm、再检验尾部"不构成循环论证。规则在 4×4 smoke dry-run 之后、任何尾部统计量计算之前冻结。四个 arm 的尾部表**全部**报告，不只报告主 arm。

## 4. 与任务书的偏离及理由

| 任务书 | 本阶段 | 理由 |
|---|---|---|
| `snr_db_list: [-2,0,2,4,6,8,10]` | `[1, 4, 7, 13, 19]` | 该 backbone 只在这五档训练过。用 `-2/0/2` 会把"训练范围外"混进"尾部风险"，无法归因。五档满足"至少两个有意义 SNR 点"。 |
| `num_images: 200` | 200 | 不变。 |
| `num_channel_realizations: 64` | 64 | 不变。M=64 时 `ceil(64×0.1)=7`、`ceil(64×0.2)=13`，CVaR-10 与 CVaR-20 可区分。 |
| 单一 checkpoint、单一 arm | 四 arm | 见 §3。这是本阶段最重要的新增。 |
| `outputs/cvar_tail_risk/` | `outputs/analysis/ANALYSIS-CVAR-P0-TAIL-RISK-001/` | 遵循仓库既有输出约定。 |
| 未要求预注册 | 本文件 | `AGENTS.md` 要求。 |
| 未要求更新项目记录 | 完成后更新 `PROGRESS.md`/`EXPERIMENTS.md` | `AGENTS.md` 强制。 |
| 图像集未指定 | COCO val2017 中 SHA 排序前 200、且与 S33 的 512 图 checkpoint-selection 子集**不相交** | 避免复用已用于选 checkpoint 的图。 |
| `B=4, M=4, α∈{0.1,0.2}`（§7.2） | 训练阶段未授权；若放行须 M≥8 | M=4 时 `ceil(4×0.1)=ceil(4×0.2)=1`，CVaR-10/CVaR-20/worst-one 三者退化为同一目标，任务书 §7.2 与 §7.1-B4 自相矛盾。 |

## 5. 判定规则（在结果前冻结）

主 arm 上，**GO 需同时满足**：

1. 至少两个 SNN 点满足 `median PSNR − p10 PSNR ≥ 2 dB`；
2. `mean PSNR − worst10-mean PSNR ≥ 1 dB`；
3. 至少一个阈值下 outage probability 不可忽略；
4. 尾部可归因于 `|h|²` 而非图像难度（见下）。

**NO-GO 若出现任一**：

1. 所有 SNR 下 `median − p10 < 1 dB`；
2. 尾部在换用更好的 SNR 条件 arm 后消失；
3. 尾部只出现在无实际意义的 SNR 档。

归因证据（必须报告，不作为 gate）：

- 四个 arm 的完整尾部表；
- 图像内（信道）方差 vs 图像间（内容）方差分解；
- `PSNR` 与 `|h|²` 的 Spearman 秩相关。

outage 阈值：绝对 `20 / 22 / 24` dB，另加相对阈值 `该 SNR 下 AWGN 中位数 − 3 dB`。绝对阈值参照 S33B 在 AWGN 下约 `26–28.6 dB` 的工作区间。

## 6. 重要的已知局限（必须写进结果报告）

1. **S33B 从未见过衰落。** 它是纯 AWGN 训练的。诊断出的尾部同时包含"风险不敏感"与"train/test 信道错配"两个来源。`awgn_control` 只能隔离前者的一部分，不能完全分离。这意味着：即使本阶段 GO，也**不能**直接把 S33B 当作 CVaR 训练的公平 B0 对照——B0 必须在 Rayleigh 上重训。任务书 §10 已要求主对照是 `Repeated-fading mean control` 而非 B0，本局限进一步强化了这一点。
2. **有效 SNR 会跌出训练范围。** `|h|² ~ Exp(1)`，深衰落时有效 SNR 可低至 `−20` dB 量级。`rayleigh_effective_csi` 因此把条件嵌入推向分布外；`clamped` arm 缓解但不消除。
3. 本阶段无语义指标。`PROJECT.md` 要求任何生成/感知模块都要报 semantic failure；本阶段只做信道尾部诊断，不产出方法 claim，故只报 PSNR/MS-SSIM/LPIPS。若进入 P4/P5，semantic failure 必须补齐。
4. 单一 backbone、单一码率（`1/24`）、单一 block-fading 假设。结论不外推到 fast fading、per-channel fading 或其他 CBR。

## 7. 运行命令

```bash
# 环境
python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"

# 单元测试
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_tail_risk.py' -v

# dry-run（4 图 × 4 realization × 2 SNR × 4 arm = 128 行）
python3 scripts/cvar_p0_diagnose_tail_risk.py --dry-run

# 正式诊断（200 图 × 64 realization × 5 SNR × 4 arm = 256,000 行）
python3 scripts/cvar_p0_diagnose_tail_risk.py

# 尾部统计、图表与 GO/NO-GO
python3 scripts/cvar_p0_analyze_tail_risk.py
```

## 8. 验收

- [x] `REPO_AUDIT`（本文件 §2）
- [x] 可复现 Rayleigh realization 采样（`fading_seed` / `block_fading_coefficient`）
- [x] 同一图像可重复采样 M 个独立信道实现
- [x] tail metric 单元测试（18/18）
- [x] dry-run
- [ ] 逐样本 CSV
- [ ] 按图像尾部统计
- [ ] 5 类诊断图
- [ ] 最差重建案例
- [ ] `GO` / `NO-GO`
- [x] 未在诊断前启动 CVaR 训练
