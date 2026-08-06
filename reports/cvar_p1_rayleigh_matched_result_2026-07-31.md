# CVaR 候选方向二 P1：Rayleigh matched mean-training 归因闭环结果

- 日期：2026-07-31 / 2026-08-01
- experiment_id：`EXP-CVAR-P1-RAYLEIGH-MATCHED-MEAN-001`
- 诊断：`ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001`（seed `20260731`）与 `-002-SEED20260802`（独立复现）
- 预注册：`reports/cvar_p1_rayleigh_matched_preregistration_2026-07-31.md`（训练输出前冻结）
- 上游：`reports/cvar_p0_tail_risk_result_2026-07-31.md`（`NO-GO`）
- git commit：`c435f89e`

---

## 1. 最终判定

```text
Decision: END-CVAR
```

两个独立 seed 均判定 `END-CVAR`。按预注册 §4.3 判定表：能力门槛**通过**，残余尾部门槛**未通过**（幅度通过、归因未通过）。

**候选方向二（CVaR 尾部风险 JSCC）到此彻底结束，不进入 CVaR 训练。**

---

## 2. 训练结果

匹配训练：block fading + ZF 均衡，encoder 条件为标称 SNR，decoder 条件为**真实有效 SNR（不 clamp）**，纯 MSE 损失，6 epoch = `22,182` steps，`92.5 min`，峰值 `12.34 GiB`。

| epoch | train_mse | val aggregate PSNR |
|---:|---:|---:|
| 0 | 0.003087 | 27.6263 |
| 1 | 0.002822 | 27.5082 |
| 2 | 0.002754 | 27.8484 |
| 3 | 0.002725 | 27.9127 |
| 4 | 0.002645 | 28.0181 |
| 5 | 0.002594 | **28.1409** |

最后一 epoch 增量 `+0.12 dB`，已基本收敛。best checkpoint SHA `4a52028480c7317c7084c7922af7d22e216b3798613036b278131122e44dbc20`。

### 2.1 能力门槛（预注册 §4.1）：**PASS**

主 arm `rayleigh_effective_csi` 上，200 图 × 64 realization：

| SNR | P0 最强 arm 参照 | P1 匹配模型 | 差 |
|---:|---:|---:|---:|
| 1 | 24.5588 | 25.7742 | **+1.2154** |
| 4 | 26.4680 | 27.2261 | **+0.7581** |
| 7 | 27.9057 | 28.3770 | **+0.4713** |
| 13 | 29.9419 | 30.0583 | **+0.1164** |
| 19 | 30.9180 | 30.9177 | −0.0003 |
| **聚合** | **27.9585** | **28.4707** | **+0.5122** |

要求聚合 `≥ 27.9585` 且逐档退化 `≤ 0.5 dB`：聚合超出 `+0.51 dB`，最差逐档退化 `+0.0003 dB`（即无退化）。**未出现预注册 §5.2 担心的退化风险**——匹配训练没有把模型推向过度保守。

---

## 3. 核心结果：尾部大幅收缩但未消失

同一 arm（`rayleigh_effective_csi`）、同一批图、同一 realization 编号，P0 冻结 S33B vs P1 匹配模型：

| SNR | mean PSNR | | median−p10 | | worst10-mean | | outage(<24dB) | | CVaR-10 MSE | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | P0 | P1 | P0 | P1 | P0 | P1 | P0 | P1 | P0 | P1 |
| 1 | 21.48 | **25.77** | 7.98 | **4.18** | 12.04 | **19.82** | 0.631 | **0.292** | 0.0748 | **0.0147** |
| 4 | 23.38 | **27.23** | 7.33 | **3.99** | 14.21 | **21.60** | 0.488 | **0.173** | 0.0507 | **0.0105** |
| 7 | 24.54 | **28.38** | 5.14 | **3.66** | 16.68 | **23.11** | 0.401 | **0.110** | 0.0330 | **0.0073** |
| 13 | 25.08 | **30.06** | 2.72 | **2.34** | 20.73 | **26.30** | 0.368 | **0.041** | 0.0131 | **0.0037** |
| 19 | 25.31 | **30.92** | 2.73 | **0.95** | 21.69 | **28.89** | 0.343 | **0.021** | 0.0086 | **0.0020** |

匹配训练带来 `+3.8 ~ +5.6 dB` 的平均提升，尾部差 `median−p10` 缩小 `0.38 ~ 3.80 dB`，outage(<24dB) 下降 `2.4 ~ 16×`，`CVaR-10 MSE` 下降 `4.4 ~ 5.1×`。

失效模式也发生质变。同一图、同一 realization（`000000013004`, `r54`, `|h|²=0.0003`）：

- P0：worst-10%（`|h|²=0.0385`）为 `16.25 dB` 彩色噪声，完全语义崩塌；worst 为 `8.40 dB` 纯噪声。
- P1：worst-10% 为 `23.95 dB`，**内容清晰可辨、仅模糊**；worst 为 `14.07 dB`。

**但尾部没有被消除**：1 dB 处仍有 `4.18 dB` 的 `median−p10`，五档中 4 档（1/4/7/13 dB）超过 `2.0 dB` 门槛。所以幅度条款**通过**。

### 3.1 归因条款：**FAIL**（这是判定的决定性一环）

预注册要求在触发档上，信道方差占比 `≥ 0.5` **且** Spearman(PSNR, `|h|²`) `≥ 0.5`：

| SNR | 信道方差占比 | Spearman | 占比通过？ |
|---:|---:|---:|:--:|
| 1 | 0.54612 | 0.6379 | 是 |
| 4 | **0.49995** | 0.5966 | **否（差 5.5e-5）** |
| 7 | 0.44453 | 0.5539 | 否 |
| 13 | 0.28600 | 0.3523 | 否 |

**残余尾部里，图像内容难度的贡献已经追平甚至超过信道随机性。** P0 时低中三档占比为 `0.80/0.75/0.67`（信道主导），匹配训练后降到 `0.55/0.50/0.44`。也就是说，匹配训练把**信道造成**的那部分方差压掉了大半，剩下的越来越是「这张图本身难」。

而 CVaR 的全部立论基础是「同一图像在不同信道实现下的尾部」。当残余尾部主要来自图像内容而非信道时，逐图 CVaR 无从发力——它对图像间难度差异是不敏感的（这正是任务书 §4.1 与 §7.2 反复强调、且我们在 `test_per_image_cvar_is_not_a_global_pool` 中固定下来的性质）。

---

## 4. 边缘性与稳健性（必须披露）

4 dB 档的占比为 `0.499945`，**距 0.5 门槛仅差 `5.5e-5`**。这是一个统计学上的掷硬币，不能当作干净的判定依据。已做两项检验：

**（1）2,000 次 image-cluster bootstrap**（seed `20260731`）：

| SNR | 点估计 | 95% CI | P(占比 ≥ 0.5) |
|---:|---:|---|---:|
| 1 | 0.5461 | [0.5000, 0.5970] | 0.975 |
| 4 | 0.4999 | [0.4525, 0.5547] | **0.542** |
| 7 | 0.4445 | [0.3983, 0.4999] | 0.025 |

4 dB 确认为掷硬币（`P=0.542`，CI 跨越门槛）。

**（2）独立 seed 复现**（`base_seed=20260802`，全新 64×5×200 realization）：

| SNR | seed 20260731 占比 | seed 20260802 占比 |
|---:|---:|---:|
| 1 | 0.54612 | 0.54048 |
| 4 | 0.49995 | 0.49331 |
| 7 | 0.44453 | 0.42965 |
| 13 | 0.28600 | 0.26738 |
| 判定 | `END-CVAR` | `END-CVAR` |

**判定不依赖那个边缘点。** 即使 4 dB 判为通过，7 dB（`0.44`）与 13 dB（`0.29`）仍明确失败，归因条款（要求触发档**全部**满足）依然 FAIL，结论不变。两个 seed 的能力门槛也都通过（聚合 `28.4707` / `28.4760 dB`）。

---

## 5. 结论

1. **匹配均值训练确实是 P0 尾部的主因解释。** 去掉两重分布外错配后，平均性能提升 `+3.8~5.6 dB`，尾部差缩小最多 `3.80 dB`，outage 降低最多 `16×`，最差重建从纯噪声变为可辨模糊。P0 结果报告 §5.2 的预判得到证实。
2. **残余尾部存在，但已不再由信道主导。** 触发档的信道方差占比从 `0.80/0.75/0.67` 降到 `0.55/0.50/0.44`，内容难度追平或超过信道随机性。逐图 CVaR 对这部分无从发力。
3. **按预注册判定表，结论为 `END-CVAR`**，两个独立 seed 一致，且不依赖 4 dB 的边缘点。
4. 该结论同时符合任务书 §17 的研究纪律：本任务的目的是尽快可靠地判断 CVaR 是否值得投入，`NO-GO`/结束是可接受的结果。

### 5.1 顺带得到的正面产物

`EXP-CVAR-P1-RAYLEIGH-MATCHED-MEAN-001` 本身是一个**合格的 Rayleigh block-fading channel-adaptive JSCC 基线**：聚合 `28.47 dB`，五档全部不劣于冻结 S33B 在其最强 Rayleigh arm 上的表现，且在低 SNR 明显更好（1 dB `+1.22 dB`）。它就是任务书 §7.1 要求的 `Repeated-fading mean control`。

但按 `MILESTONES.md`，Rayleigh 仍属 AWGN 最小闭环之后的扩展项，**本模型不自动进入主线**，也未做语义评估。是否保留为将来 Rayleigh 扩展的起点，由用户决定。

### 5.2 明确不做的事

- 不训练 CVaR-10 / CVaR-20 / worst-one。
- 不把「匹配训练带来的 `+3.8~5.6 dB`」包装成 CVaR 或本项目的方法贡献——它只是修正了 train/test 信道错配，是应有的对照，不是新方法。
- 不把残余的 `4.18 dB` 尾部写成「CVaR 仍有机会」，因为归因已表明它主要不是信道尾部。

---

## 6. 已知问题与局限

1. **单一 recipe。** continuation 而非 from-scratch，单预算（6 epoch）、单 lr。`END-CVAR` 的强度限于「在这一 recipe 下，匹配均值训练足以把信道尾部压到不再主导」。
2. **深衰落物理下界。** `|h|²≈1e-4` 时有效 SNR ≈ `nominal−40 dB`，信息论上不可恢复。任何模型都有由物理决定的残余尾部下界，本次 worst-case（`14.07 dB`）已接近该性质，这也是残余尾部无法归零的原因之一。
3. **4 dB 归因为掷硬币**（`P=0.542`）。已用 bootstrap 与独立 seed 双重检验，判定不依赖该点，但该点本身不可作为任何单独结论的依据。
4. **预注册阈值 `0.5` 的合理性未独立论证。** 它在 P1 预注册中被提前数值化（修正了 P0 的缺陷），但阈值本身是判断值而非理论推导值。占比 `0.44`（7 dB）与 `0.29`（13 dB）距门槛足够远，故结论不敏感；`0.4999`（4 dB）敏感。
5. **无语义指标。** 本闭环只回答尾部归因，未评估 semantic drift / semantic failure。P1 模型若将来进入主线，必须按 `PROJECT.md` 补齐。
6. **单 backbone、单码率 `1/24`、block fading 逐图一系数、ZF 且 `ε=0`、发端无 CSI。** 不外推到 fast fading、per-channel fading、MMSE 均衡、其他 CBR。
7. **一次 CUDA OOM 失败已保留**：外部 VLLM 进程占用 `20.8/24 GiB` 导致首次诊断在 32 行处 OOM，失败目录保留为 `ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-001_failed_oom_20260801`，重跑时 `realization_chunk` 由 32 降为 8。
8. **chunk 不是逐比特不变的。** 实测（8 vs 2，按 `(arm,image,snr,realization)` 键比较）：`|h|²` 与 decoder SNR 逐比特相同，但 `max|ΔPSNR| = 8.3e-4 dB`、`max|ΔMSE| = 4.4e-7`、`max|ΔLPIPS| = 1.7e-4`，来自 GPU kernel 在不同 batch 尺寸下的非确定性，比 `2.0 dB` 门槛低四个数量级。**行顺序会随 chunk 改变**，任何跨运行比较必须按键而非按位置。P0 与 P1 使用了不同 chunk（32 vs 8），本报告所有跨阶段对比均按键进行。

---

## 7. 复现命令

```bash
# 单元测试（新增 20 项 tail_risk / 全仓 142 项）
PYTHONPATH=src python3 -m unittest discover -s tests

# 匹配训练（约 92 分钟）
python3 scripts/cvar_p1_train_rayleigh_matched.py --dry-run   # FakeData smoke
python3 scripts/cvar_p1_train_rayleigh_matched.py

# 匹配模型上重跑同一诊断（约 10 分钟）
python3 scripts/cvar_p0_diagnose_tail_risk.py --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml
python3 scripts/cvar_p0_analyze_tail_risk.py  --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml
python3 scripts/cvar_p0_export_worst_cases.py --config configs/cvar_p1_matched_tail_risk_diagnostic.yaml

# 判定表
python3 scripts/cvar_p1_attribution_verdict.py

# 独立 seed 复现
python3 scripts/cvar_p0_diagnose_tail_risk.py --config configs/cvar_p1_matched_tail_risk_seed_replication.yaml
python3 scripts/cvar_p0_analyze_tail_risk.py  --config configs/cvar_p1_matched_tail_risk_seed_replication.yaml
python3 scripts/cvar_p1_attribution_verdict.py \
  --matched-directory outputs/analysis/ANALYSIS-CVAR-P1-MATCHED-TAIL-RISK-002-SEED20260802
```

## 8. 验收

- [x] 匹配训练完成并收敛（6 epoch，最后增量 `+0.12 dB`）
- [x] 能力门槛通过（聚合 `+0.51 dB`，无逐档退化）
- [x] 匹配 checkpoint 上重跑**完全相同**的诊断（同图、同 realization、同 seed、同四 arm）
- [x] 残余尾部门槛按预注册机械评估
- [x] 边缘点做 bootstrap + 独立 seed 双重检验
- [x] 最差案例导出并通过重放校验（40 组）
- [x] 明确判定 `END-CVAR`
- [x] 未训练任何 CVaR 模型
