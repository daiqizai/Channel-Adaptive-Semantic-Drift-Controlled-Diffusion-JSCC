# Cross-model triplet sender checksum 阶段结果（旧端点，已更正）

日期：2026-07-14。正式审计：`ANALYSIS-PC-SENDER-CROSSMODEL-SEED-AUDIT-001`。

> **统计更正（2026-07-14）：本报告原 POSITIVE 结论已作废。** 后续 sender official-statistics 审计发现，原报告的 `0/1690` 与 `0.1771%` 使用的是 anchor/in-budget-raw-relative 事件，不能代替相对论文 M2（paired unpunctured `reference_raw`）的系统 new-error endpoint。按正确端点重算，seed20260727 是 7 个 `reference_raw_correct→final_wrong` image clusters、8 个 repair，eligible 1690，单侧 upper95 `0.7766%>0.5%`；1 dB failure 为 M2/final `32→34`。因此当前严格 verdict 是 **NEGATIVE**，不得据此解封 official val。下面保留的是更正前的历史叙述，仅用于追溯，所有 `POSITIVE` 字样均被本段取代。

## 原结论（已被上述统计更正取代）

在全新 AWGN channel seed `20260727` 上，固定码率的 cross-model triplet controller 通过全部预注册 strict-rate gate，结论为 **POSITIVE**。它首次在独立新 channel realization 下同时做到：

- 总 CBR 不变（`1/6`），payload 与图像同一次 AWGN；
- primary final new-error 相对 in-budget raw `2→0`，每个 primary SNR 均不增加；
- primary final failure 相对同 seed unpunctured M2 `61→60`；
- 五 SNR 平均 final-minus-M2 PSNR `+0.01158 dB`、LPIPS `-0.002566`；
- 0 个 primary new-error image cluster，单侧 95% Clopper-Pearson upper 为 `0.1771%`，低于冻结 `0.5%`。

这不是“全面强于 M2”的结论：1/4/7 dB 相对 M2 的 final PSNR 分别为 `-0.00346/-0.01042/-0.02136 dB`，整体小幅正增益来自 13/19 dB；posterior 接受率仅 `46.01%`。更准确的表述是：**在固定总码率和新 channel seed 下，方法以约一半 posterior coverage 换取了可审计的 semantic-tail control，同时保留小幅 aggregate quality/perceptual gain。**

## 方法与隔离

总 `c=8` latent 为 65,536 个实符号。保持 40 bit `G_aux(source)` UInt4 payload，BPSK ×4，占 160 符号（`0.24414%`）；余下 65,376 符号承载图像。两者共同经过一次 AWGN，receiver 擦除 payload 位置，三步 received-latent posterior consistency 也排除相同位置。

posterior 只有同时满足下列三项时才被采用：

```text
JS(q_recovered, G_aux(posterior)) - JS(q_recovered, G_aux(anchor)) <= 0
argmax(q_recovered) == argmax(G_gate(anchor))
argmax(G_gate(anchor)) == argmax(G_gate(posterior))
```

`q_recovered` 是经信道恢复的 sender `G_aux(source)` probability；`G_gate` 是独立 scratch classifier；`T_cls` 只作为真实 WNID outcome evaluator，未参与 controller。没有新符号、margin/confidence threshold、逐 SNR 例外、标签或 source `G_gate` prediction。

规则来自此前已暴露的 `20260725/20260726` development seeds，因此它们不作为独立成功证据。reference M2 表先在从未使用过的 `20260727` 生成，随后只将 CSV SHA-256 `b58a1795183d998a663c836bec813b37fe20815b06098c74e9f3842c4e901567` 填入审计 config；reference outcome 没有用于调整方法。

## 三个 seed 的结果

| 角色 | seed | verdict | M2 failure → final | in-budget raw new → final | ΔPSNR / ΔLPIPS vs M2 | coverage |
|---|---:|---|---:|---:|---:|---:|
| development | 20260725 | POSITIVE | `50→48` | `4→1` | `+0.01091 / -0.002529` | 45.46% |
| development | 20260726 | POSITIVE | `58→55` | `3→0` | `+0.01108 / -0.002562` | 46.02% |
| **frozen audit** | **20260727** | **POSITIVE** | **`61→60`** | **`2→0`** | **`+0.01158 / -0.002566`** | **46.01%** |

所有三次均为 1894 张 policy-dev 图像 × 5 SNR = 9470 行；clean-correct 图像为 1697 张。它们验证了 channel-seed 稳定性，但不是独立 image-population 泛化，official Imagenette validation 仍未访问。

## 新 seed 逐 SNR 审计

| SNR (dB) | M2 failure | in-budget raw failure | final failure | raw new → final new | final ΔPSNR vs M2 | final coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 34 | 34 | `1→0` | -0.00346 | 49.38% |
| 4 | 18 | 18 | 18 | `1→0` | -0.01042 | 47.38% |
| 7 | 11 | 12 | 8 | `0→0` | -0.02136 | 47.61% |
| 13 | 2 | 2 | 2 | `0→0` | +0.01982 | 46.55% |
| 19 | 2 | 1 | 1 | `0→0` | +0.07333 | 45.96% |

载荷层也通过：mean BER `0.01610%`、40-bit 整向量无误率 `99.377%`、source/recovered top-1 agreement `99.609%`；五个 SNR 的 masked data-consistency 都下降。perfect-payload 反事实与实际决策只在 `0.1267%` 行不同，说明这次 tail gate 不依赖把 payload 噪声藏起来。

## 与失败的双证据规则的关系

此前的 `source-JS ∩ G_gate(anchor/posterior)` 在 seed `20260726` 额外 veto `0.623%` 行，却保留了全部 5 个 new-error，正式为 NEGATIVE。failure replay 显示这些行的 `G_aux` source top-1 与独立 `G_gate(anchor)` 不一致；把已有 payload 的 recovered source top-1 纳入三方自然一致性后，无需额外 bit 即可拦住该类 shared blind spot。该发现只在 development data 上形成，随后由 `20260727` 的冻结审计验证。

## 可复现性与下一步

```bash
# 新 seed reference（已完成）
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python3 scripts/pc_imagenette_supervised_audit.py \
  --config configs/pc_imagenette_sender_crossmodel_seed20260727_reference.yaml --device cuda:0

# frozen strict-rate audit（已完成；output_dir 不可覆盖）
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python3 scripts/pc_imagenette_sender_inbudget_awgn_audit.py \
  --config configs/pc_imagenette_sender_crossmodel_triplet_seed20260727_audit.yaml --device cuda:0
```

下一步不应继续在这些 seed 上调 coverage 或规则。优先级是：以完全冻结的同一 config 在未暴露的 image population 做一次 image-holdout audit；随后才启动已规划的单个外部 mechanism baseline。论文中必须保留此次成功的条件、低/中 SNR PSNR 回吐、约 46% coverage 和所有先前负结果。
