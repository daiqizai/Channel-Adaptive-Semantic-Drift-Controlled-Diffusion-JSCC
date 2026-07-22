# SGD-JSCC 共同协议单图闭环阶段报告（2026-07-15）

> **2026-07-15 信道口径更正：**本报告原始 `24.785109 dB` smoke 使用作者实值 AWGN 的每实坐标方差 `P/SNR`，比本项目复信道口径 `P/(2×SNR)` 严苛 3 dB。该 run 仍可证明 adapter、caption codec 和总码率闭环，但不得进入直接公平排名。新增 `configs/external_sgdjscc_common_complex_awgn_smoke.yaml` 后，同图/同 seed/1 dB 的复信道 smoke 为 `26.128782 dB`，输出在 `outputs/smoke/external_sgdjscc_common_complex_awgn_snr1_seed20260729_20260715/`。随后完成的 8 图×5 SNR 公平 pilot 见 `reports/external_common_comparison_pilot_stage_result_2026-07-15.md`。下文保留原始历史结果，不覆盖。

## 阶段结论

`SMOKE-EXT-SGDJSCC-COMMON-001` 已一次性跑通一个可计总码率的 SGD-JSCC 共同协议适配器：同一张 COCO-256 图像按作者 `split_image_v2` 切成四个 `128×128` patch，作者主 JSCC、MuGE edge、edge-JSCC、ControlNet 和 50-step diffusion 均使用发布权重；四段 BLIP2 caption 不再免费提供给接收端，而是经过固定 UTF-8 packet、CRC16、BPSK×21 和同一 AWGN；edge 只发送收发双方都能由 `cr=13` 确定的 active coordinates。

运行时账本精确闭合为 `65,536` 个实信道坐标，即 `32,768` 个复信道使用；以 `3×256×256=196,608` 个源实维度为分母，CBR 正好为 `1/6`。因此 **SGD-JSCC common-adapter 的 rate gate 已通过**，不再被“免费 caption / edge 码率未知”阻塞。

这仍只是单图 integration smoke，不是效果比较。它不授权“SGD-JSCC 强于或弱于本项目”的结论，也不能替代 8 图预检、64 图多 SNR 质量/语义统计和 fresh population 审计。

## 为什么它不是作者原生结果

作者原生链假设 caption 完美可用且忽略传输成本，并把带大量零值的 dense edge tensor 直接交给 AWGN module。共同协议适配器为了满足可比较性，做了三项明确、可审计的协议层改动：

1. 每个 patch 的 caption 被编码为 `1-byte length + 64-byte UTF-8 payload + CRC16`，共 `536 bits`；BPSK×21 后占 `11,256` 个实坐标，CRC 或 UTF-8 失败时接收端把 caption 擦除为空字符串。
2. edge encoder 的 mask 固定保留前 `cr=round(0.2×64)=13` 个通道；每块 `64×13=832` 个 active coordinates。接收端按同一确定性 mask scatter，未发送坐标置零。
3. 每张图、每个 SNR、每个 channel seed 生成一条固定的 `65,536` 维标准高斯向量，按 `main → edge → text → padding` 的顺序切片；剩余 `800` 坐标记为无信息 padding，不向方法额外泄露信息。

因此正式标签必须写成 **“SGD-JSCC common-contract adapter”**；作者原生表和共同协议表继续分开。

## 冻结协议与输出

- 配置：`configs/external_sgdjscc_common_smoke.yaml`
- 入口：`scripts/external_sgdjscc_common_smoke.py`
- 作者源码：`third_party/SGDJSCC` commit `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`，tracked files clean
- 输入：`sample_010000.png`，SHA-256 `42ebfa92489dac0ad4044b4b5edcdd785ca96453528e2c58a7197f9c5f78af75`
- channel：AWGN `1 dB`，channel seed `20260729`
- canonical noise SHA-256：`f8edbfe05eb1fb2ce9606bcdf0be8bd790fe267fbf46a6f2eacf32ec7d01e416`
- 输出：`outputs/smoke/external_sgdjscc_common_snr1_seed20260729_20260715/`
- official Imagenette validation 未访问。

## 实测码率账本

| 分支 | 实测实坐标 | 总预算占比 |
|---|---:|---:|
| 四块 main latent | `16,384` | `25.0000%` |
| 四块 active edge | `3,328` | `5.0781%` |
| 四段 caption，536 bits×R21 | `45,024` | `68.7012%` |
| 无信息 padding | `800` | `1.2207%` |
| **总计** | **`65,536`** | **`100%`** |

诊断用 dense edge tensor 共 `65,536` elements，但其中每块只有 `832/16,384` active。mask 是固定的通道前缀，不需要额外发送索引。共同协议只计实际调度的 active coordinates；dense 数字仍保留在 `rate_accounting.json`，避免丢失实现事实。

这里也修正了前一份作者原生报告中的术语歧义：一个复信道使用由两个实坐标组成。旧 JSON 的 `main_real_cbr=4096/49152=0.08333` 实际是 **real-coordinate/source-dimension ratio**；对应 complex-use CBR 是 `4096/2/49152=1/24`。本轮已在适配器和测试中把两种口径分字段报告。

## 单图运行结果

| 项目 | 结果 |
|---|---:|
| 状态 | `PASS` |
| patch 数 | `4` |
| caption packet CRC 成功 | `4/4` |
| BPSK hard-symbol errors | `5,981/45,024 = 13.2840%` |
| repetition 后 packet bit errors | `0/2,144` |
| 输出 shape | `[1,3,256,256]` |
| full-image smoke PSNR | `24.785109 dB` |
| patch PSNR | `[24.5430, 24.1065, 24.4626, 26.3563] dB` |
| 模型加载加单图前向耗时 | `13.1588 s` |
| peak allocated GPU memory | `7364.35 MiB` |

四个 sender caption 均小于 64 bytes，没有发生截断；1 dB 下 raw BPSK 硬判错误率约 `13.28%`，R21 多数表决后四个 packet 均零 bit error，接收端实际使用的文字与 sender 文字一致。这证明编码与失败处理链真的参与了运行，而不是只在报告中估算成本。

## 语义风险观察

肉眼 sanity check 仍暴露了不能忽略的风险：

- `x=128` 和 `y=128` 附近可见 patch 拼接边界，说明 patch-wise 生成不能视为无缝的 256 生成器；
- 右侧 patch 的 caption 是 `a man in a white suit standing on a baseball field`，重建右边缘出现一个尺寸明显放大的白衣人物，而原图相应位置没有同尺度目标。这是 **疑似 text-driven hallucination / semantic drift**，但单图目测不能把它统计认定为 new error；
- 球场、围栏和主要运动场景仍被保留，所以“整体场景看起来对”同样不能掩盖局部对象新增风险。

这项观察直接支持项目既定纪律：外部 diffusion 方法也必须报告 `T_cls new-error`、failure/repair 和 image-cluster tail upper bound，不能只报 PSNR 或视觉平滑度。

## 对后续对比的约束

1. author-native 结果继续禁止与本项目直接排名；只有明确标注的 common-adapter 可以在后续共同表中出现。
2. R21 是在查看本次结果前冻结的第一个可执行文本协议，不代表最优通信码。它占用约 `68.7%` 总预算，后续主表必须固定它；若做更高效 FEC，只能作为另一个预注册 sensitivity，不可事后替换主结果。
3. rate gate 通过不等于 semantic gate 通过。下一次 SGD-JSCC 扩展至少需要同一 frozen 8 图、五个 SNR，并保存 per-sample caption CRC、质量与语义失败记录。
4. 按冻结排期，下一项先实现 `SING-Zero-style` 共同协议机制对照；之后再统一进入 64 图×3 channel seeds 的外部方法 stage，避免只把算力集中在一个作者系统上。
5. 当前项目主线仍保留 diffusion + posterior/data consistency。外部对比的目的，是测清额外 text/edge 预算、patch-wise generation 和 inverse-restoration 各自贡献及语义代价，而不是退回纯非 diffusion 上限。

## 验证

- common adapter dry-run：`PASS`
- common contract checker：`PASS`，下一 milestone 为 `EXT3_SING_zero_style_common_contract`
- 新 common-adapter tests：8/8 通过
- native adapter tests：5/5 通过
- external contract tests：5/5 通过
- 全仓标准库测试：94/94 通过
- 第三方源码：commit 匹配、tracked files clean
- 全部 checkpoint、BLIP2 shard、CLIP：运行前按 frozen SHA-256 重新校验
- 网络：全程 offline，本轮无下载；运行命令清空全部代理变量
