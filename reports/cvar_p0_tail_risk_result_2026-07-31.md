# CVaR 候选方向二 P0/P2/P3：条件信道尾部风险诊断结果

- 日期：2026-07-31
- analysis_id：`ANALYSIS-CVAR-P0-TAIL-RISK-001`
- 预注册：`reports/cvar_p0_tail_risk_preregistration_2026-07-31.md`（在本结果前冻结）
- 输出：`outputs/analysis/ANALYSIS-CVAR-P0-TAIL-RISK-001/`
- checkpoint：`EXP-S33B-STRONG-JSCC-16384-FP32-001/checkpoints/best.pt`，SHA `2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`（脚本内强制校验通过）
- 规模：200 图 × 64 realization × 5 SNR × 4 arm = **256,000 行**，用时 `576.1 s`，单卡 RTX 4090 D
- 本阶段无训练、无 checkpoint 选择、无下载，official Imagenette validation 未访问

---

## 1. 最终判定

```text
Decision: NO-GO
```

按预注册判定规则机械执行的结果为 `NO-GO`：四项 GO 条件通过 3 项，第 4 项（尾部归因）未通过。

**但本报告必须同时披露一处我自己造成的预注册缺陷，并给出两种读法，不得只取有利的一种。**见 §4。

比机械判定更重要的是两项实质发现（§3），它们指向同一个结论：**当前证据不支持把 CVaR 作为论文贡献投入训练。**

---

## 2. 主结果

### 2.1 四 arm 条件尾部表（逐图 M=64 realization 内统计，再跨 200 图平均）

| arm | SNR | mean | median | p10 | worst10-mean | median−p10 | outage(<24dB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `awgn_control` | 1 | 27.58 | 27.58 | 27.47 | 27.42 | **0.11** | 0.075 |
| `rayleigh_nominal_csi` | 1 | 24.56 | 26.66 | 16.60 | 13.42 | **10.06** | 0.353 |
| `rayleigh_effective_csi` | 1 | 21.48 | 22.44 | 14.47 | 12.04 | 7.97 | 0.631 |
| `rayleigh_effective_csi_clamped` | 1 | 23.66 | 25.38 | 16.60 | 13.42 | 8.79 | 0.430 |
| `awgn_control` | 4 | 28.86 | 28.86 | 28.77 | 28.74 | **0.09** | 0.040 |
| `rayleigh_nominal_csi` | 4 | 26.47 | 28.17 | 20.12 | 16.03 | **8.05** | 0.227 |
| `awgn_control` | 7 | 29.85 | 29.85 | 29.78 | 29.76 | **0.07** | 0.020 |
| `rayleigh_nominal_csi` | 7 | 27.91 | 29.31 | 23.55 | 18.94 | **5.76** | 0.149 |
| `awgn_control` | 13 | 30.96 | 30.96 | 30.93 | 30.91 | **0.04** | 0.015 |
| `rayleigh_nominal_csi` | 13 | 29.94 | 30.74 | 27.94 | 24.79 | **2.79** | 0.055 |
| `awgn_control` | 19 | 31.35 | 31.35 | 31.33 | 31.33 | **0.02** | 0.015 |
| `rayleigh_nominal_csi` | 19 | 30.92 | 31.28 | 30.28 | 28.42 | **1.01** | 0.026 |

完整 20 行见 `diagnostic_summary.csv`。

### 2.2 预注册要求的两项差值（主 arm = `rayleigh_nominal_csi`，逐 SNR 按平均 PSNR 最高选出）

| SNR | Δ_tail = median − p10 | Δ_worst10 = mean − worst10-mean |
|---:|---:|---:|
| 1 | 10.06 dB | 11.14 dB |
| 4 | 8.05 dB | 10.44 dB |
| 7 | 5.76 dB | 8.97 dB |
| 13 | 2.79 dB | 5.15 dB |
| 19 | 1.01 dB | 2.50 dB |

### 2.3 CVaR MSE 相对均值 MSE 的放大倍数

| SNR | AWGN `cvar10/mean` | Rayleigh `cvar10/mean` | Rayleigh `cvar20/mean` |
|---:|---:|---:|---:|
| 1 | **1.03×** | **5.67×** | 3.80× |
| 4 | 1.03× | 6.02× | 3.76× |
| 7 | 1.02× | 5.85× | 3.56× |
| 13 | 1.01× | 4.34× | 2.72× |
| 19 | 1.01× | 2.65× | 1.84× |

### 2.4 方差分解与秩相关

| arm | SNR | 图像内（信道）方差 | 图像间（内容）方差 | 信道占比 | Spearman(PSNR, \|h\|²) |
|---|---:|---:|---:|---:|---:|
| `awgn_control` | 1 | 0.008 | 7.553 | **0.001** | — |
| `awgn_control` | 19 | 0.000 | 8.664 | **0.000** | — |
| `rayleigh_nominal_csi` | 1 | 25.621 | 6.370 | 0.801 | 0.748 |
| `rayleigh_nominal_csi` | 4 | 20.103 | 6.818 | 0.747 | 0.685 |
| `rayleigh_nominal_csi` | 7 | 14.826 | 7.292 | 0.670 | 0.620 |
| `rayleigh_nominal_csi` | 13 | 5.920 | 7.647 | 0.436 | 0.385 |
| `rayleigh_nominal_csi` | 19 | 1.927 | 8.433 | 0.186 | 0.187 |

### 2.5 最差样本可视化

`worst_examples/` 共 40 组，每组为 `原图 | median realization | worst-10% | worst`，文件名含 `image_id / snr / realization_id / |h|² / psnr`。全部 40 组均经**重放校验**：重建 PSNR 与 CSV 记录值误差 `< 0.01 dB`。

典型案例 `snr1dB_000000013004_worst_r54_h0.0003_psnr8.40.png`：median realization（`|h|²=0.627`）为 `31.08 dB` 视觉良好，worst-10%（`|h|²=0.0385`）为 `16.25 dB` 已完全语义崩塌，worst（`|h|²=0.0003`）为 `8.40 dB` 纯噪声。同图 median−worst 跨度 `24.71 dB`。

---

## 3. 两项实质发现

### 3.1 AWGN 下**不存在**条件尾部风险

这是本次诊断最干净、也最重要的结果。在 `awgn_control` arm（同一 checkpoint、同一批图、同一 M=64 次独立噪声重采样、仅令 `h=1`）：

- `median − p10 ≤ 0.11 dB`，五档分别为 `0.11 / 0.09 / 0.07 / 0.04 / 0.02 dB`；
- 信道引起的方差占总方差 `≤ 0.001`；
- `CVaR-10% MSE / mean MSE ≤ 1.03×`。

也就是说：**在项目当前主线信道（AWGN）上，均值训练的模型对信道随机性已经极其稳定，CVaR 没有任何可优化的对象。** 该 arm 与既有 AWGN 实现按单元测试逐元素相等，所以这不是新实现的伪结果。

尾部**完全**是引入 Rayleigh 之后才出现的。而 Rayleigh 按 `PROJECT.md` 与 `MILESTONES.md` 属于 AWGN 最小闭环之后的扩展项。

### 3.2 接收端「忽略」真实有效 SNR 反而更好 → 尾部主因是 OOD，不是风险不敏感

主 arm 选择规则（tail-blind，只看平均 PSNR）在**五档全部**选出了 `rayleigh_nominal_csi`，即 decoder 被告知**标称** SNR 的那一臂：

| SNR | nominal | effective | effective_clamped |
|---:|---:|---:|---:|
| 1 | **24.56** | 21.48 | 23.66 |
| 4 | **26.47** | 23.38 | 24.74 |
| 7 | **27.91** | 24.54 | 25.27 |
| 13 | **29.94** | 25.08 | 25.32 |
| 19 | **30.92** | 25.31 | 27.38 |

接收端明明有完美 CSI、可以算出真实有效 SNR `γ|h|²`，但**喂进去反而全面变差**（1 dB 处差 `3.08 dB`，19 dB 处差 `5.61 dB`）。clamp 到训练范围 `[1,19]` 能挽回一部分，仍不如干脆不告诉它。

这说明该 backbone 的 SNR 条件嵌入**无法表示深衰落产生的有效 SNR**（`|h|²~Exp(1)`，深衰落时有效 SNR 可低至 `−20 dB` 量级，而它只在 `[1,19] dB` 训练过）。因此观测到的尾部主要来自两重分布外：

1. 模型从未见过衰落（纯 AWGN 训练）；
2. 有效 SNR 跌出条件嵌入的训练范围。

**这两者都不是「均值目标掩盖了尾部风险」，而是 train/test 信道错配。** 对应的正确修法是把衰落放进训练、把有效 SNR 正确地喂给条件嵌入，而不是换成 CVaR 目标。

---

## 4. 预注册缺陷披露（必须读）

预注册 §5 把第 4 项 GO 条件写为定性表述「尾部可归因于 `|h|²` 而非图像难度」，同时把用于归因的三项具体统计量明确标为「**必须报告，不作为 gate**」。而我在 `scripts/cvar_p0_analyze_tail_risk.py` 中把这些统计量实现成了 gate，并自行选定了两个未在预注册中数值化的阈值：

```python
any(channel_share >= 0.5)  and  all(spearman >= 0.5)
```

这带来两个问题：

1. **阈值未预先数值化。** `0.5` 是我在实现时选的，不是预注册值。
2. **口径与第 1 项 GO 条件不一致。** 第 1 项只要求「至少两个 SNR 点」，而我对归因用了 `all(...)` 跨全部五档。

两种读法及其结果：

| 读法 | 归因判据 | 结果 |
|---|---|---|
| **A（脚本字面执行，即上文 §1 的判定）** | `all(spearman ≥ 0.5)` 跨五档 | 13 dB 为 `0.385`、19 dB 为 `0.187`，**FAIL** → `NO-GO` |
| **B（与第 1 项 GO 条件同口径）** | 仅在满足第 1 项的 SNR 点（1/4/7/13 dB）上评估 | 1/4/7 dB 的占比 `0.80/0.75/0.67`、Spearman `0.75/0.69/0.62` 明确通过；13 dB 边缘 → 四项全过 → `GO` |

我**不**把这两种读法之一说成唯一正确答案，也不修改脚本去获得有利结果（`AGENTS.md` 明确禁止「看到结果不理想后修改统计口径」）。`verdict.json` 保留读法 A 的原始输出。

**但两种读法都不改变 §3 的实质结论**：读法 B 只能说明「Rayleigh 下确实存在信道驱动的尾部」，而 §3.1 与 §3.2 说明该尾部主要由信道错配与条件 OOD 造成，因此**都不足以支持「CVaR 是有效贡献」**。判定的分歧不影响下一步建议。

---

## 5. 结论与建议的下一步

### 5.1 结论

1. AWGN 下无条件尾部风险（`median−p10 ≤ 0.11 dB`，信道方差占比 `≤ 0.001`）。项目当前主线上 CVaR 无优化对象。
2. Rayleigh block fading 下存在很大的条件尾部（1 dB 处 `median−p10 = 10.06 dB`、`CVaR-10 MSE` 为均值的 `5.67×`、`outage(<24dB) = 35.3%`），且在低中 SNR 明确由 `|h|²` 驱动（信道方差占比 `0.80/0.75/0.67`）。
3. 但该尾部主要归因于「纯 AWGN 训练的模型被放到从未见过的衰落信道上」+「有效 SNR 跌出条件嵌入训练范围」，而非均值目标本身掩盖风险。最直接的证据是接收端喂入真实有效 SNR 反而五档全面变差。
4. 因此**不建议**按任务书 P4/P5 直接启动 CVaR 训练。

### 5.2 建议的下一步（若仍要推进衰落方向）

任务书 §10 已经要求 CVaR 必须打败 `Repeated-fading mean control`，否则不算贡献。该对照**目前根本不存在**，而按 §3 的证据，它很可能自己就吸收掉本次测得的大部分尾部。所以正确的下一个实验是**更便宜的那个**：

> 在 Rayleigh block fading 上、用正确的有效 SNR 条件，训练**均值**基线（即任务书的 B0'/B1），然后用本阶段完全相同的诊断脚本重测它的条件尾部。

- 若该均值基线仍有 `median − p10 ≥ 2 dB` 的尾部 → CVaR 才值得测，且此时你已经顺带得到任务书要求的公平 B1 对照。
- 若尾部大幅消失 → 该方向结束，省下一整轮 CVaR 训练。

这条路径同时满足任务书 §17 的研究纪律（不在没有诊断证据时包装论文方向），也不与 `MILESTONES.md` 冲突——它仍是 AWGN 最小闭环之后的扩展项，需要用户单独放行。

### 5.3 与主线的关系

本阶段是只读诊断，未改动任何既有训练/评测/复现流程，未覆盖任何既有实验目录，S35R 主线未受影响。P4/P5 未启动，等待用户决定。

---

## 6. 已知局限

1. **S33B 从未在衰落上训练。** 这是 §3.2 的核心，也是本诊断最大的局限：无法把「风险不敏感」与「信道错配」完全分离。`awgn_control` 只能证明 AWGN 下无尾部，不能反推 Rayleigh 尾部中风险不敏感成分的大小。要分离必须有 §5.2 的衰落训练均值基线。
2. **无语义指标。** `PROJECT.md` 要求任何生成/感知模块报 semantic failure。本阶段只做信道尾部诊断、不产出方法 claim，故只报 PSNR/MS-SSIM/LPIPS。若进入 P4/P5 必须补齐 semantic failure。
3. **单一设定。** 单 backbone、单码率 `1/24`、单 block-fading 假设（逐图一个复系数）、ZF 均衡且 `ε=0`、发端无 CSI。结论不外推到 fast fading、per-channel fading、MMSE 均衡、其他 CBR 或其他 backbone。
4. **深衰落事件稀有但被采到。** `|h|² ≈ 1e-4` 量级事件在 `Exp(1)` 下概率约 `1e-4`，每档 12,800 个 realization 期望约 1.3 次，实际在 19 dB 采到多例（`worst_examples/` 中 `h0.0001`）。这类点主导 worst-case 但对 p10 影响有限；p10 与 worst10-mean 应分开读。
5. **13 dB 为归因边缘点。** 信道方差占比 `0.436`、Spearman `0.385`，介于低 SNR 的信道主导与 19 dB 的内容主导之间，两种读法在此点分歧最大。
6. **未做 bootstrap 置信区间。** 本阶段为 go/no-go 诊断，报告点估计。若进入 P4/P5，方法对比必须按仓库惯例做 10,000 次 image-cluster bootstrap。
7. 诊断用图为 COCO val2017 中与 S33 的 512 图 checkpoint-selection 子集不相交的 SHA 排序前 200 图；这些图此前未被用于选 checkpoint，但也不是封存的 official validation。

---

## 7. 复现命令

```bash
python3 -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_tail_risk.py' -v   # 18/18
PYTHONPATH=src python3 -m unittest discover -s tests                            # 140/140
python3 scripts/cvar_p0_diagnose_tail_risk.py --dry-run                         # 128 行
python3 scripts/cvar_p0_diagnose_tail_risk.py                                   # 256,000 行
python3 scripts/cvar_p0_analyze_tail_risk.py                                    # 统计 + 图 + verdict
python3 scripts/cvar_p0_export_worst_cases.py                                   # 40 组最差案例
```

## 8. 验收清单

- [x] `REPO_AUDIT`（预注册 §2）
- [x] 可复现 Rayleigh realization 采样
- [x] 同一图像可重复采样 M=64 个独立信道实现
- [x] 逐样本 CSV（256,000 行）
- [x] 按图像尾部统计（`per_image_tail_stats.csv`，4,000 行）
- [x] 诊断图（7 张，含任务书要求的 5 类 + 2 张归因图）
- [x] 最差重建案例（40 组，全部通过重放校验）
- [x] tail metric 单元测试（18/18，全仓 140/140）
- [x] 明确 `GO` / `NO-GO`
- [x] 未在诊断前启动 CVaR 训练
- [ ] P4/P5（repeated-mean 与 CVaR 训练）——**未启动，按 §5.2 建议不以当前形式启动**
