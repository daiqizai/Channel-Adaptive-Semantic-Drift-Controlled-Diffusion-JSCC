# 固定码率发送端语义载荷阶段结果

日期：2026-07-14。

## 一句话结论

项目没有放弃 diffusion。S14 六步 diffusion 加三步 received-latent posterior correction 继续是稳定的 restoration 主体；本阶段首次把发送端语义描述真正放进 `c=8` 固定总符号预算并通过同一 AWGN 传输。`UInt4+BPSK×4` 解决了模拟概率载荷的决策不稳定问题，在开发 seed 上通过全部门槛，但冻结到新 seed 后仍因单一 `G_aux`/JS 判据漏过语义新错而失败。因此当前成果是：**码率公平、可纠错的 sender semantic channel 已做通；最终 semantic failure handler 尚未跨 seed 可靠。**

## 1. 本阶段实现了什么

### 1.1 严格码率合同

- DeepJSCC 仍为 `c=8`，256×256 图像对应 65,536 个实信道符号，CBR 仍为 `1/6`；
- 固定保留 160 个实符号承载发送端描述，占总预算 `0.244140625%`；
- 图像主载荷剩 65,376 个实符号，没有增加第二条信道、额外 bit budget 或无噪声旁路；
- 语义载荷覆盖到原 latent 后，整段 latent 重新归一化，并在一次共同 AWGN channel call 中传输；
- receiver 恢复描述后擦除 160 个位置，再送入冻结 DeepJSCC decoder；
- posterior data-consistency 只在 65,376 个图像位置上计算，避免错误地要求候选图像编码复现 sender payload。

### 1.2 控制链

冻结 scratch EfficientNet-B0 `G_aux` 在 sender 计算 10 类校准概率。receiver 用恢复概率 `q` 比较 B1 anchor 和 posterior：

```text
r_JS = JS(q, G_aux(posterior)) - JS(q, G_aux(anchor))
r_JS <= 0: 接受 posterior
r_JS > 0 : 回退 in-budget B1 anchor
```

S13 B1、S14 六步 diffusion、三步 posterior correction、零阈值、`G_aux`、独立 `T_cls` 和 clean threshold 均保持冻结。官方 Imagenette validation 未访问。

## 2. 模拟概率载荷：质量可行，但决策不稳定

第一种严格率实现直接发送 L2 单位功率的 10 维概率，每维重复 16 次，共 160 个模拟符号。

开发 seed `20260725` 的完整 1894×5 结果：

- source/recovered top-1：`99.8416%`；
- mean cosine：`0.997844`；
- primary reference-raw / in-budget-raw / posterior / final failure：`50 / 54 / 45 / 49`；
- primary in-budget raw / posterior / final new-error：`4 / 4 / 3`；
- final new-error cluster upper95：`0.45761%`；
- mean final-minus-reference-raw：`+0.038905 dB / -0.003242 LPIPS`。

总体数字看似接近成功，但 1 dB 的 final new-error 为 `2`，高于 in-budget raw 的 `1`，违反冻结的逐 SNR 门槛，正式 verdict 为 `NEGATIVE`。

完美载荷反事实进一步定位原因：

- 实际模拟载荷与完美载荷的接受/拒绝决策有 `40.4963%` 不同；
- 完美载荷可把 primary failure `49→44`、new-error `3→2`；
- 3 个实际 final new-error 中，`n03425413_24914` 会被完美载荷保护，属于连续概率噪声翻转；
- `n03888257_20624` 和 `n02979186_3088` 即使使用完美 source 概率仍被接受，属于 `G_aux`/JS 的语义盲区。

关键认识是：**top-1 恢复率接近 100% 不代表零阈值差分分数稳定。** 当两个 JS 值都很小时，小的连续概率扰动足以翻转差值符号。

## 3. UInt4+BPSK×4：固定码率编码开发通过

第二种实现不扫描模拟 repetitions，而是固定为：

- 10 类概率逐维 UInt4，共 40 raw bits；
- BPSK `{-1,+1}`；
- 每 bit 重复 4 次；
- 总占用仍为 160 个实符号。

开发 seed `20260725` 的完整结果：

| 指标 | 结果 |
|---|---:|
| bit error rate | `0.014520%` |
| 40-bit 整向量无误率 | `99.4298%` |
| source/recovered top-1 | `99.6304%` |
| mean source/recovered cosine | `0.999356` |
| 载荷噪声决策翻转率 | `0.07392%` |
| primary reference raw failure | `50` |
| primary in-budget raw / posterior / final failure | `54 / 45 / 45` |
| primary in-budget raw / posterior / final new-error | `4 / 4 / 2` |
| final new-error cluster upper95 | `0.37162%` |
| mean final-minus-reference-raw PSNR | `+0.026653 dB` |
| mean final-minus-reference-raw LPIPS | `-0.003165` |

所有预注册门槛通过，开发 verdict 为 `POSITIVE`。这证明数字硬判决能在相同 160 符号下把模拟方案的分数抖动基本消除，同时保留 diffusion/posterior 的感知收益。

逐 SNR 结果：

| SNR | final-reference raw PSNR | final-reference raw LPIPS | reference raw failure | in-budget raw failure | final failure | raw→final new-error | 整向量无误率 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `-0.00459` | `-0.00420` | 25 | 28 | 25 | `1→1` | `97.1489%` |
| 4 | `-0.00910` | `-0.00400` | 16 | 16 | 10 | `2→1` | `100%` |
| 7 | `-0.01453` | `-0.00314` | 9 | 10 | 10 | `1→0` | `100%` |
| 13 | `+0.04708` | `-0.00225` | 4 | 2 | 4 | `0→1` | `100%` |
| 19 | `+0.11441` | `-0.00224` | 2 | 1 | 1 | `0→0` | `100%` |

注意：低 SNR 的 PSNR 相对 unpunctured raw 略负，说明擦除 0.244% latent 的代价真实存在；五 SNR 平均为正主要由高 SNR posterior gain覆盖。LPIPS 则五个 SNR 全部改善。

## 4. 新 seed 审计：编码迁移，单模型语义控制未迁移

方法冻结后只更换此前未使用的 channel seed `20260726`。先生成同 seed unpunctured reference，再以其逐样本表和 SHA256 为公平对照。

审计结果：

| 指标 | seed 20260725 开发 | seed 20260726 审计 |
|---|---:|---:|
| bit error rate | `0.014520%` | `0.017159%` |
| 整向量无误率 | `99.4298%` | `99.3242%` |
| 载荷噪声决策翻转率 | `0.07392%` | `0.23231%` |
| mean final-reference raw PSNR | `+0.026653 dB` | `+0.026429 dB` |
| mean final-reference raw LPIPS | `-0.003165` | `-0.003182` |
| primary reference raw failure | 50 | 58 |
| primary in-budget raw failure | 54 | 55 |
| primary final failure | 45 | 55 |
| primary in-budget raw new-error | 4 | 3 |
| primary final new-error | 2 | 5 |
| final cluster upper95 | `0.37162%` | `0.45896%` |
| verdict | `POSITIVE (development)` | `NEGATIVE (frozen audit)` |

编码和质量几乎原样迁移，primary failure 也仍优于 reference raw 的 `58`。失败集中在 semantic tail：final new-error `5 > in-budget raw 3`，总量和逐 SNR gate 均失败。

5 个审计 new-error 行的 payload vector 全部正确，完美载荷反事实仍为 5 个，因此这次失败不归因于信道编码。它们集中为三个 image cluster：

- 1 dB：`n03394916/n03394916_40715.JPEG`；
- 4 dB：`n02979186/n02979186_3088.JPEG`、`n03394916/n03394916_33456.JPEG`、`n03394916/n03394916_40715.JPEG`；
- 7 dB：`n03394916/n03394916_40715.JPEG`。

这些行的 `r_JS` 均为明显负值，约 `-0.0021` 到 `-0.0100`，不是零附近的信道噪声翻转。结论是单一 `G_aux` 与独立 `T_cls` 存在稳定语义盲区。

## 5. 视觉检查

三个开发 failure batch 已按原始 batch start 精确重放。每张 sheet 从上到下为 source、in-budget B0、B1 anchor、S14 raw、posterior、final：

- `outputs/analysis/pc_imagenette_sender_aux_inbudget_awgn_failure_case_replay/samples/seed_20260725_snr_04_batch_0512_source_b0_anchor_raw_post_final.png`
- `outputs/analysis/pc_imagenette_sender_aux_inbudget_awgn_failure_case_replay/samples/seed_20260725_snr_01_batch_1456_source_b0_anchor_raw_post_final.png`
- `outputs/analysis/pc_imagenette_sender_aux_inbudget_awgn_failure_case_replay/samples/seed_20260725_snr_01_batch_1760_source_b0_anchor_raw_post_final.png`

人工观察下，CD player、gas pump、parachute 三类的错误变化都很细微，并非显著的对象替换。这再次说明不能只看“图像更干净、更自然”，必须保留冻结 `T_cls` semantic drift 统计。

## 6. 研究方向判断

### 已经做通的部分

1. diffusion/posterior restoration 继续稳定改善 LPIPS；
2. sender semantic description 已从无噪声额外 80 bit，推进到真正 matched-rate、同 AWGN 的 160-symbol payload；
3. UInt4+BPSK×4 显著优于模拟概率重复，编码层在新 seed 上迁移；
4. masked posterior consistency 与 payload erasure 的工程闭环已具备复用价值。

### 仍未做通的部分

1. 单一 `G_aux` 的 source-grounded JS veto 不是跨 seed 可靠的最终 M3；
2. failure cluster 会跨 SNR 重复出现，说明是 image susceptibility 与模型盲区的组合；
3. 不能再通过在 seed `20260726` 上调阈值、位数或 repetitions 来补救。

### 下一步建议

保留 `UInt4+BPSK×4` 作为通信层，不再重开码率设计；下一方法只改 semantic decision layer。优先考虑自然的双证据交集：source-grounded JS 接受后，再要求独立 `G_gate` 的 anchor/posterior top-1 一致；它不增加 payload 和 channel use，也不需要新阈值。必须把 seed `20260726` 降格为 development evidence，并在方法冻结后用再新的 channel seed 审计。若双证据仍失败，停止堆手工 classifier gate，转向在独立 labeled development population 上训练 source-conditioned candidate risk model，并保留新的图像 population 作最终审计。

## 7. 复现与哈希

- 项目版本：工作树包含连续协作中的未提交改动，按项目规则记为 `N/A (not a clean project commit)`；
- 无联网、无下载、未访问 official Imagenette validation；
- 主入口：`scripts/pc_imagenette_sender_inbudget_awgn_audit.py`；
- 当前脚本 SHA256：`98ffc689a18d2e8415c2c2fb0c5971d4c95a88e8739fc2d58254fdce974a48e4`；
- `G_aux` SHA256：`8e074be6ec854edbc144d95d9fe5cd7d098c61bca853915108952acfa094b455`；
- `T_cls` SHA256：`b846c8d81dc3dd604f916c82cb2ca5584ecaf88191e1a3009da1b81e3f00924d`；
- S13 B1 SHA256：`80133f9d9649c1a5d9514cf2b4f0d04802b6ebe03cc970bfcec86eddfd165562`；
- S14 diffusion SHA256：`e10c7da6e7e7cd155114e7d1b94bc7f524320c91025ba18b1783f8934ba5a4b1`；
- DeepJSCC SHA256：`5943bb96d4522fb9f707bfd5cb6691a1663e30c2c2d47ebd35e9814f97627d7d`；
- 模拟严格率正式 metrics/per-sample/summary：`2d6a3b... / a5d227... / 348fd3...`；
- 完美载荷诊断 metrics/per-sample/summary：`e3e3a2... / 22f171... / f09dc1...`；
- UInt4 开发 metrics/per-sample/summary：`97925a... / a909eb... / 865401...`；
- seed-20260726 reference per-sample SHA256：`0906f4ebcdee78bd066da6e40416f1b2acd167bd0e2e11855d2e3d96fd04abff`；
- seed-20260726 审计 metrics/per-sample/summary：`681523... / 27fc54... / 9005c8...`。

完整输出目录：

- `outputs/analysis/pc_imagenette_sender_aux_inbudget_awgn_zero_veto_dev/`
- `outputs/analysis/pc_imagenette_sender_aux_inbudget_awgn_perfect_payload_diagnostic/`
- `outputs/analysis/pc_imagenette_sender_aux_uint4_bpsk_inbudget_awgn_zero_veto_dev/`
- `outputs/analysis/pc_imagenette_sender_aux_seed20260726_unpunctured_reference/`
- `outputs/analysis/pc_imagenette_sender_aux_uint4_bpsk_seed20260726_audit/`

验证：标准库 `unittest` 58 项全部通过；`py_compile`、dry-run、reference hash、9470 行键完整性和 `git diff --check` 均通过。
