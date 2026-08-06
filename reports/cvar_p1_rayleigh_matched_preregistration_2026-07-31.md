# CVaR 候选方向二 P1：Rayleigh matched mean-training 归因闭环预注册

- 日期：2026-07-31
- experiment_id：`EXP-CVAR-P1-RAYLEIGH-MATCHED-MEAN-001`
- 训练配置：`configs/cvar_p1_rayleigh_matched_mean_training.yaml`
- 训练入口：`scripts/cvar_p1_train_rayleigh_matched.py`
- 上游：`reports/cvar_p0_tail_risk_result_2026-07-31.md`（判定 `NO-GO`）
- 状态：**在任何正式训练输出前冻结**。仅完成 FakeData smoke 与 20-batch 吞吐测量。

## 0. 授权范围

用户 2026-07-31 授权：候选二原始版本正式 `NO-GO`；允许进行**一次独立的** Rayleigh matched mean-training 归因闭环。该闭环若消除尾部则彻底结束 CVaR；只有匹配均值训练后仍残留显著条件尾部，才进入 CVaR。

本预注册把「消除」「显著残留」「以及一个用户未指定但必须存在的第三种结局」全部数值化并提前冻结。

## 1. 要回答的问题

P0 证明：AWGN 下无条件尾部（`median−p10 ≤ 0.11 dB`）；Rayleigh 下尾部很大（1 dB 处 `10.06 dB`），但 S33B 存在两重分布外——从未见过衰落，且有效 SNR 跌出条件嵌入训练范围（接收端喂入真实有效 SNR 反而五档全面变差）。

因此 P0 无法区分：

> Rayleigh 尾部是「均值目标掩盖了风险」，还是「训练/测试信道错配」？

本闭环把两重错配都去掉，只保留均值目标，再测残余尾部。

## 2. 训练设定

| 项 | 值 |
|---|---|
| 初始化 | 冻结 S33B，SHA `2daad9e7…dd5bfb`（脚本强制校验；原文件不修改） |
| 信道 | block fading `y=hx+n`，`h~CN(0,1)` 逐图，ZF 均衡 `ε=0`，与 P0 诊断**同一函数** `apply_block_fading_channel` |
| encoder 条件 | **标称** SNR（无反馈，发端不可能知道 `h`） |
| decoder 条件 | **真实有效** SNR `nominal + 10log10|h|²`，**不 clamp**（让条件嵌入覆盖衰落真实产生的全范围） |
| 损失 | 纯 MSE（这是均值基线，不是 CVaR） |
| 训练 SNR | 逐图离散均匀 `[1,4,7,13,19]` |
| 数据 | COCO2017 train2017 全量，`256×256`，与 S33B 同 crop/flip |
| 预算 | 6 epoch = `22,182` optimizer steps；实测 `246 ms/batch`、`15.2 min/epoch`、峰值 `12.34 GiB` |
| 优化器 | AdamW fresh state，lr `5e-5` cosine→`1e-6`，wd `1e-4`，grad clip `1.0`，FP32（S33 的 AMP 曾非有限失败，沿用 FP32） |
| checkpoint 选择 | 匹配信道上 512 图 × 4 realization × 5 SNR 的**聚合验证 PSNR 最高**（与 S31/S33 同规则，只换信道） |
| seed | `20260731` |

预算说明：6 epoch 是**刻意给足**的。本控制组的作用是让匹配均值训练尽全力消除尾部，因此残余尾部不能被解释为欠训练。收敛证据（最后两 epoch 验证 PSNR 增量）将在结果报告中报告；若仍在明显上升，必须记为局限。

## 3. 评测

训练结束后，用**与 P0 完全相同**的脚本 `scripts/cvar_p0_diagnose_tail_risk.py` 与 `scripts/cvar_p0_analyze_tail_risk.py` 评测新 checkpoint，仅替换 checkpoint 路径与 SHA：

- 同 200 图（COCO val2017，SHA 排序，与 S33 的 512 图 selection 子集不相交）
- 同 64 realization、同 5 SNR、同 `base_seed=20260731`
- 同四 arm（`awgn_control` / `nominal` / `effective` / `effective_clamped`），共享 encoder 前向与噪声，逐 realization 严格配对
- 同 `floor_uint8` 量化与同指标

**主 arm 事前固定为 `rayleigh_effective_csi`**，即与训练匹配的部署方式。这里不用 P0 的 tail-blind「最高平均 PSNR」规则，因为匹配训练之后主 arm 有了原理性答案，事前指定比数据依赖的规则更强。四 arm 全部报告；若另有 arm 平均 PSNR 更高，记为异常并**在两个 arm 上都评估尾部门槛**。

## 4. 三种结局（全部事前冻结）

用户只描述了两种结局。必须补第三种，否则一个退化模型（输出糊平均图）会因为尾部天然小而被误读成「尾部已消除」。

### 4.1 能力门槛（先判，防退化）

匹配模型在 Rayleigh 上的平均 PSNR 必须达到：

- **聚合**（五档平均）`≥ 27.958479 dB`，即 P0 中 S33B 最强 Rayleigh arm（`rayleigh_nominal_csi`）的聚合值；
- **且**任一档相对该 arm 的退化不超过 `0.5 dB`。参照值逐档为 `24.5588 / 26.4680 / 27.9057 / 29.9419 / 30.9180 dB`。

能力门槛在主 arm 上评估；若主 arm 未过但另一 arm 过，如实报告并以过门槛的 arm 继续。

### 4.2 残余尾部门槛

「显著残留条件尾部」定义为**同时**满足：

1. **幅度**：`median PSNR − p10 PSNR ≥ 2.0 dB` 在五档中**至少 2 档**成立；
2. **归因**：在**上述触发档上**，信道方差占总方差 `≥ 0.5` **且** Spearman(PSNR, `|h|²`) `≥ 0.5`。

第 2 条明确修正 P0 结果报告 §4 披露的缺陷：阈值 `0.5` 在此**提前数值化**，且归因只在触发档评估（不再用跨五档 `all(...)`），与第 1 条同口径。

### 4.3 判定表

| 能力门槛 | 残余尾部门槛 | 结论 |
|---|---|---|
| 未过 | — | **`INCONCLUSIVE`**：匹配均值训练在本 recipe 下未产出合格模型。既不结束也不进入 CVaR，如实报告并停止，等待用户决定是否改 recipe 重试。 |
| 过 | 未过 | **`END-CVAR`**：尾部由信道错配造成，匹配均值训练即可消除。候选方向二彻底结束。 |
| 过 | 过 | **`ENTER-CVAR`**：匹配均值训练后仍有信道驱动的显著尾部，CVaR 值得测试，且本模型即为任务书 §10 要求的公平 `Repeated-fading mean control`。 |

**不允许在看到结果后修改上述任何阈值**（`AGENTS.md` 明令禁止）。若出现门槛边缘或规则未覆盖的情形，如实披露并同时报告两种读法，如 P0 所做。

## 5. 事前记录的预期与风险

事前写下预期，避免事后合理化：

1. **均值 MSE 在重尾衰落下本身被尾部主导。** `|h|²~Exp(1)`，深衰落样本的 MSE 极大，因此「均值」训练的梯度实际上大量来自尾部样本。这意味着匹配均值训练**可能自己就吸收掉大部分尾部**——这正是 P0 结果报告 §5.2 的预判，也是 `END-CVAR` 的主要可能路径。
2. **反向风险**：同一机制也可能让训练不稳定或把模型推向对深衰落过度保守（整体变糊），从而**触发能力门槛失败**。故 §4.1 必须存在。
3. 深衰落是不可恢复的：`|h|²≈1e-4` 时有效 SNR ≈ `nominal−40 dB`，信息论上无法重建。因此**任何**模型都会有一个由物理决定的残余尾部下界。这意味着 `ENTER-CVAR` 即便触发，也仍需在 CVaR 阶段证明 CVaR 能改善**可恢复**区间，而不是把不可恢复的样本算作收益。此点必须写进结果报告。

## 6. 局限（事前声明）

1. 单一 recipe：continuation 而非 from-scratch，单一预算、单一 lr。若 `INCONCLUSIVE`，不能推广为「匹配均值训练不可行」。
2. 单 backbone、单码率 `1/24`、block fading 逐图一系数、ZF 且 `ε=0`、发端无 CSI。不外推到 fast fading、per-channel fading、MMSE 均衡。
3. 无语义指标。本闭环只回答尾部归因问题，不产出方法 claim。若 `ENTER-CVAR`，semantic failure 必须在 CVaR 阶段补齐（`PROJECT.md` 要求）。
4. 未做 bootstrap CI；go/no-go 阶段只报点估计。若 `ENTER-CVAR`，正式方法对比必须按仓库惯例做 10,000 次 image-cluster bootstrap。
5. 与 `MILESTONES.md` 的关系不变：Rayleigh 仍属 AWGN 最小闭环之后的扩展项，本闭环是用户单独授权的一次性归因实验，不改主线，S35R 不受影响。

## 7. 运行命令

```bash
# smoke（FakeData，仅验证管线）
python3 scripts/cvar_p1_train_rayleigh_matched.py --dry-run

# 正式训练（约 1.6 小时）
python3 scripts/cvar_p1_train_rayleigh_matched.py

# 训练后：把 best.pt 的 SHA 填入诊断配置，再复用 P0 同一套脚本
python3 scripts/cvar_p0_diagnose_tail_risk.py --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml
python3 scripts/cvar_p0_analyze_tail_risk.py  --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml
```

## 8. 验收

- [x] 训练脚本与配置
- [x] per-sample SNR 信道支持 + 单测（20/20）
- [x] FakeData smoke 通过、吞吐与显存实测
- [x] 三种结局与全部阈值事前冻结
- [ ] 正式训练完成
- [ ] 匹配 checkpoint 上重跑同一诊断
- [ ] `INCONCLUSIVE` / `END-CVAR` / `ENTER-CVAR` 明确判定
