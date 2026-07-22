# 外部方法共同协议首轮对比阶段报告（2026-07-15）

## 结论先行

`ANALYSIS-EXT-COMMON-PILOT-001` 已完成第一轮真实外部对比：当前冻结方法 M3、SGD-JSCC common-contract adapter 和 SING-Zero-style 机制对照分别产生 `8 images × 5 SNR = 40` 条结果，聚合器对全部 `120` 条记录逐行验证了相同 sample/SNR key、相同 canonical channel-noise SHA-256、相同 DeepJSCC reference、相同复 AWGN 方差约定和精确 CBR `1/6`。

在这组小规模 policy-development pilot 上：

- 当前 M3 的均值为 `33.0594 dB / 0.98203 MS-SSIM / 0.03532 LPIPS`，`40/40` 条均保持 `T_cls` 正确，相对同噪声 DeepJSCC 为 `0` 个 new error；
- SGD-JSCC common adapter 为 `26.8882 dB / 0.94862 / 0.07763`，同样 `0` 个 new error；
- SING-Zero-style final-only 近似为 `24.6593 dB / 0.96118 / 0.31725`，在 1 dB 出现 `1` 个相对 DeepJSCC new error；
- 当前 M3 相对 SGD adapter 的逐行配对均值为 `+6.1712 dB`、`+0.03341 MS-SSIM`、`-0.04231 LPIPS`，PSNR/MS-SSIM 为 `40/40` 行更优，LPIPS 为 `39/40` 行更优。

这些结果是明确的阶段性正信号，但 **不能写成“已强于 SGD-JSCC/SING 论文”**。样本只有 8 张且来自已暴露的 policy-development population；SGD 是为公平计码率而改造的 project-side adapter，不是作者原生协议；SING 是 final-only 机制近似，不是论文的逐步 DDNM 实现。

## 公平协议

### Population 与冻结顺序

- 数据：Imagenette2-320 train 的 `policy_dev`，official validation 继续封存；
- 从已有 frozen clean-correct membership 中按 `sha256("external-stage-20260715:" + sample_id)` 取前 8 张；
- SNR：`[1, 4, 7, 13, 19] dB`；
- base seed：`20260729`；
- 每个 sample/SNR 由 SHA-256 派生一个 CPU `torch.float32` seed，生成固定 `65,536` 维标准高斯向量；
- 三个方法对同一个 sample/SNR 必须消费相同向量，聚合器已逐行比较 noise SHA。

### 信道与码率

本轮统一把相邻两个实坐标视为一个复信道使用。SNR 是每个 complex channel use 的 `Es/N0`，每个实坐标噪声方差为：

`signal_power / (2 × 10^(SNR_dB/10))`。

每行固定：

- `65,536` real channel coordinates；
- `32,768` complex channel uses；
- source dimensions `3×256×256=196,608`；
- CBR `32,768/196,608=1/6`。

SGD adapter 的账本为 main `16,384` + active edge `3,328` + caption `45,024` + no-information padding `800`。M3 在 c=8 latent 内用 `80` 个实坐标传 UInt2+BPSK×4 payload，总数不增加。SING-style 与 DeepJSCC reference 使用完整 c=8 latent。

## 三个方法的准确标签

### 当前 M3

冻结链路为 `UInt2-R4 → reservation-aware B1 → S14 6-step residual-shift diffusion → 3-step received-latent posterior correction → cross-model three-way route`。这是项目当前真实方法，不是为本 pilot 重新调参的简化版。

### SGD-JSCC common-contract adapter

保留作者发布的 BLIP2、main JSCC、MuGE/edge-JSCC、CLIP、ControlNet 和 50-step diffusion，以及四个无重叠 `128×128` patch。项目侧只改变通信协议：caption 经固定 UTF-8/CRC16+BPSK×21 过同一 AWGN，edge 只发送确定性 active coordinates，并用 padding 闭合总预算。因此它允许进入 common table，但必须保留 `adapter / not author-native` 标签。

### SING-Zero-style

以同一个 frozen DeepJSCC reconstruction 为输入，用本地 Stable Diffusion v1.5 做 25-step、strength `0.25` 的 img2img，再执行：

`x_final = A† y + (I - A†A) x_diffusion`，

其中 `A` 是非重叠 2×2 mean pooling，`A†` 是 2× nearest repeat。本轮只在最终图像做一次 range/null projection；论文 SING-Zero 的 DDNM 在 reverse diffusion 的每一步投影。因此当前实现只能回答“简单 final-only null-space 近似是否足够”，不能代表 SING 论文的真实上限。

## 结果

### 总表

| 方法 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | failure | new / repair vs DeepJSCC | runtime ms/图 | peak VRAM MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepJSCC reference | `31.7438` | `0.97298` | `0.07861` | `0` | — | — | — |
| 当前 M3 | **`33.0594`** | **`0.98203`** | **`0.03532`** | `0` | `0 / 0` | `24.10` | `697.49` |
| SGD-JSCC common adapter | `26.8882` | `0.94862` | `0.07763` | `0` | `0 / 0` | `2034.72` | `7374.85` |
| SING-Zero-style final-only | `24.6593` | `0.96118` | `0.31725` | `1` | `1 / 0` | `44.94` | `3184.97` |

SGD runtime 包含每次传输重新生成 BLIP2 patch captions 的实测开销估计；M3 与 SING-style 不需要该文本前端，因此 runtime 只作系统成本量级参考，不作为论文级严格 latency benchmark。

### 分 SNR PSNR

| SNR | DeepJSCC | 当前 M3 | SGD adapter | SING-style |
|---:|---:|---:|---:|---:|
| 1 | `28.0255` | **`29.7430`** | `25.1436` | `23.4678` |
| 4 | `30.1641` | **`31.4833`** | `26.1790` | `24.3418` |
| 7 | `31.8934` | **`32.9748`** | `27.0397` | `24.7558` |
| 13 | `33.9689` | **`35.0613`** | `27.7888` | `25.3872` |
| 19 | `34.6673` | **`36.0346`** | `28.2900` | `25.3437` |

当前 M3 相对 paired DeepJSCC 的五个 SNR PSNR 增益为 `[+1.7175,+1.3191,+1.0813,+1.0924,+1.3673] dB`，LPIPS 五点全部下降。这比此前 full-policy development 中相对强 M2 只有约 `+0.066 dB` 的增益大得多，原因是这里的共同 reference 是 bare DeepJSCC，不是 S14 raw M2；两组效果量不能混写。

### 分 SNR LPIPS

| SNR | DeepJSCC | 当前 M3 | SGD adapter | SING-style |
|---:|---:|---:|---:|---:|
| 1 | `0.17750` | **`0.06964`** | `0.09762` | `0.40175` |
| 4 | `0.10278` | **`0.04553`** | `0.08676` | `0.34944` |
| 7 | `0.05948` | **`0.03185`** | `0.07404` | `0.31903` |
| 13 | `0.02874` | **`0.01783`** | `0.06905` | `0.26672` |
| 19 | `0.02456` | **`0.01175`** | `0.06067` | `0.24932` |

## 语义与视觉诊断

### M3

8 张图在五个 SNR 下均保持 `T_cls` 正确，UInt2 payload `40/40` 整向量无误。pilot 说明固定 semantic payload 在这批 canonical noise 上没有成为失败源，但样本太小，不能替代 full-population 中已经观察到的 tail new error。

### SGD-JSCC

`160/160` caption packets 均 CRC 正确，说明本轮传输链不是免费文字。与此同时，BLIP2 在 sender 端已产生明显 patch-level 描述错误，例如：

- dog 图的一个 patch 被描述为 cat；
- chainsaw 图的其他 patch 被描述为 computer screen、snowy-road driver、camera；
- cassette-player 图的下半 patch 被描述为 apples/microwave。

R21 能可靠传输这些文字，却无法修复 sender caption 本身的语义错误。这是本项目一个很有价值的启发：**语义 side channel 的 packet reliability 与 semantic reliability 必须分开测量**。本 pilot 的 whole-image `T_cls` 没有因此出错，但后续应增加 object-level新增/遗漏审计，不能因为 CRC=100% 就把文本支路视为语义可靠。

视觉上 SGD 输出整体自然、主体大多正确，但多张图在 `x=128/y=128` 附近可见 patch seam。其平均 PSNR 甚至低于相同 DeepJSCC reference `4.8556 dB`，说明当前 common rate allocation 把 `68.7%` 预算给 R21 caption 后，并未在像素失真上得到回报；这不是对作者原协议的否定，而是 fixed total-rate 适配的真实代价。

### SING-Zero-style

final projection 把 mean-pool measurement MSE 从平均 `2.0819e-2` 压到 `5.7928e-6`，说明投影本身按设计工作；但高频/null-space 部分仍出现强过锐化、纹理重写和色彩改变。1 dB 的 `n02979186/n02979186_9758.JPEG` 从正确类别 2 被改判为类别 3，构成 1 个 refinement-induced hard new error。

这说明“最终一步投影满足低频 measurement”不足以控制语义漂移；若继续 SING 路线，应该实现 reverse-step DDNM projection 或论文的 degradation estimator，而不是继续调 final-only strength 后冒充 SING。

## 两次失败启动

SGD 批跑在第一条 reconstruction 前有两次 scheduler cache 定位失败。原因是 pinned diffusers/huggingface_hub 在 transitive import 时固定了默认 cache，而 `HF_HOME` 当时在函数内稍后才赋值。两个失败目录及 `failure.json` 均保留：

- `sgd_jscc_failed_scheduler_cache_resolution_20260715_0054/`；
- `sgd_jscc_failed_scheduler_cache_resolution_retry2_20260715_0056/`。

最终把 `HF_HOME` 和 offline flags 移到任何 project evaluation helper import 之前，独立 scheduler load 通过，第三次产生完整 40 行结果。全程没有联网下载。

## 当前能回答与不能回答的问题

可以回答：

1. 三种实现已经在同图、同复 AWGN realization、同 CBR 和同 evaluator 下实际运行；
2. 当前 M3 在这个小 pilot 的 distortion/perception/semantic 三轴都优于两个已接入 adapter；
3. SGD 的 caption 可靠传输不等于 caption 语义正确；
4. 简单 final-only null-space projection 不是足够强的 SING 对照。

不能回答：

1. 不能说当前方法强于 SGD-JSCC 或 SING 论文；
2. 不能把 8 张已暴露 development 图上的 `0 new error` 当成 tail safety；
3. 不能把 SING-style 的负结果归因于 SING 论文；
4. 不能据此解封 official validation 或改变已冻结的 M3 semantic decision layer。

## 下一步冻结建议

1. 先把当前 common pilot 扩成 `64 images × 3 channel seeds × 5 SNR`，样本从未用于本次 8 图选择的 frozen development block 中一次性 SHA 选取；继续保持 official validation 封存。
2. SGD 主表保留当前 R21 账本，另预注册一个更高效 caption FEC sensitivity；不得用看过结果后更省码率的协议替换主表。
3. SING 路线若推进，优先实现逐 reverse-step DDNM projection；当前 final-only 版本只保留作负对照。
4. 增加 object-level hallucination/omission 端点，并把 sender caption accuracy、packet CRC 和 final-image semantic error 三层分列。
5. 64×3 stage 通过后再接 DiffJSCC；现在不需要放弃 diffusion。当前结果反而说明受 measurement/posterior consistency 与语义路由约束的短链 diffusion，比自由生成或弱投影 prior 更适合本项目的 reliability 目标。

## 复现资产与验证

- config：`configs/external_common_comparison_pilot.yaml`
- M3/SING runner：`scripts/external_common_project_pilot.py`
- SGD runner：`scripts/external_sgdjscc_common_pilot.py`
- aggregate validator：`scripts/external_common_aggregate.py`
- aggregate：`outputs/external_baselines/ANALYSIS-EXT-COMMON-PILOT-001/aggregate/summary.json`
- 全仓标准库测试：`99/99 PASS`
- 三个方法：各 `40/40` rows；aggregate validation `PASS`
- official Imagenette validation：未访问
- 网络：全程 offline；无新增下载
