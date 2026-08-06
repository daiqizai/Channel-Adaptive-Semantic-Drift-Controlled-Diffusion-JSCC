# S34D：生成式 JSCC 公平推理代价与少步质量曲线

日期：2026-07-23
分析 ID：`ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001`

## 一、最重要的结论

**“生成式 JSCC 比纯 JSCC 慢很多”在少步优化后仍然成立，但正确数字不是拿 100 步直接算出来的。**

- DiffJSCC 的 100 步官方工作点是 `5,089.67 ms/图`。
- 在不训练、不换 checkpoint、只减少 spaced-sampling 步数的条件下，预注册候选里**最低仍显著保持相对 S33 LPIPS 优势的是 25 步**：`1,458.47 ms/图`。
- 为排除 PyTorch 版本影响，S33 另在与 DiffJSCC/SGD 完全相同的 `PyTorch 2.1.0+cu121` 下重测为 `8.833 ms/图`。因此最保守的公平数字是：

> **DiffJSCC 在保持显著 LPIPS 优势时，最低实测仍约为 S33 的 165.1 倍。**

S33 在自己的当前原生 PyTorch 2.11 runtime 下为 `5.788 ms/图`，对应 25-step DiffJSCC 为 `252.0×`。论文主文建议使用更保守的 common-runtime `165×`，把 `252×` 作为 runtime sensitivity。

不过还有一个重要限定：25 步虽然通过 LPIPS gate，却有 `14/320` 个语义失败，S33 为 `4/320`；failure-rate 差 `+3.125 pp`，post-hoc source-cluster 95% CI `[+0.625,+6.563] pp`。50 步的失败为 `10/320`，差值 CI 跨零。因此：

- **只要求保持感知优势：最低点是 25 步、1.458 秒、165×；**
- **若还要求当前样本上不出现显著语义 failure 增加：25 步不合格，50 步是更稳妥的观测点，约 2.676 秒、303×；**但本轮没有预注册语义非劣 margin，不能把 50 步正式称为“已证明语义非劣”。

100 步相对 25 步慢 `3.49×`，所以把 100 步全算成 diffusion 的固有代价确实会夸大；但即使采用少步优化后的公平端点，生成链仍是百倍量级，而不是只慢两三倍。

## 二、测量合同

三种方法均在同一张 RTX 4090D（UUID=`GPU-e8c74486-e009-e254-2fb1-fbaea11d7f8a`）、batch size=1 上测量。

主计时从“已经从磁盘读入主存的 256×256 RGB 图”开始，到“256×256 RGB 重建已经回到主存”为止。

包含：

- 方法内部 resize、patch split/merge；
- host↔GPU transfer；
- JSCC 编码、信道、解码；
- BLIP2 caption；
- text/edge conditioning；
- 每一次真实执行的 diffusion denoiser evaluation；
- 每一次真实执行的 VAE encode/decode；
- wavelet/color fix 与输出回缩。

排除：

- 模型构造与 checkpoint 加载；
- 磁盘图像读取/结果写盘；
- LPIPS、分类器等指标计算。

因此这里是 persistent-resident receiver 的 steady-state latency。若每张图都需重新加载数十 GB 权重，实际延迟只会更高；本轮没有把 cold start 混入每图数字。

为了兼顾作者代码可运行性和公平性，模型精度保持各自冻结推理合同，不强行改成同一 dtype。S33 同时在项目原生 runtime 和生成式方法使用的共同 PyTorch 2.1 runtime 下测量；倍数主结论采用后者。

内部工作量保持方法原生：

- S33：256×256；
- DiffJSCC：内部 512×512；
- SGD：一张 256×256 拆成 4 个 128×128 patch，四 patch 是一张源图的内部 batch，不算 4 张源图。

## 三、同 runtime 的公平延迟

| 方法 | steps | mean ms/图 | median ms | 相对 S33 |
|---|---:|---:|---:|---:|
| S33 strong | 0 | **8.833** | 8.857 | 1.0× |
| DiffJSCC | 100 | 5,089.671 | 5,091.961 | 576.2× |
| DiffJSCC | 50 | 2,676.170 | 2,673.177 | 303.0× |
| DiffJSCC | **25** | **1,458.465** | 1,453.832 | **165.1×** |
| DiffJSCC | 10 | 726.259 | 722.168 | 82.2× |
| DiffJSCC | 4 | 433.605 | 429.472 | 49.1× |
| SGD paper upper | 50 | 2,044.701 | 2,046.847 | 231.5× |

这里的 SGD 只做系统代价测量，不做质量胜负：它仍是 `≥21,856 real`、完美 caption 的 non-ranking paper upper。

项目原生 PyTorch 2.11 下，S33 为 `5.788±0.177 ms`，主组件为 encoder `1.469 ms`、channel `0.153 ms`、decoder `2.831 ms`。共同 PyTorch 2.1 下，S33 为 `8.833±0.106 ms`，encoder/channel/decoder=`3.022/0.257/3.964 ms`。两个 runtime 的绝对值不同，但相对百倍结论不变。

## 四、DiffJSCC 延迟—质量曲线

质量总体为冻结 64 图×seed `20260748`×5 SNR=`320` 个相同键。100-step 重放与历史 S30 的 PSNR/LPIPS 最大绝对误差均为 0，确认少步实验没有偷换 100-step 算法。

| steps | mean ms | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | Diff−S33 LPIPS及95% CI | failures | LPIPS gate |
|---:|---:|---:|---:|---:|---:|---:|---|
| 100 | 5,089.7 | 27.5898 | 0.940673 | 0.099957 | `−0.019946 [−0.032494,−0.007571]` | 7/320 | PASS |
| 50 | 2,676.2 | 27.8712 | 0.943614 | **0.097870** | `−0.022032 [−0.034032,−0.011190]` | 10/320 | PASS |
| **25** | **1,458.5** | 28.1999 | 0.945952 | 0.101952 | `−0.017950 [−0.027709,−0.008826]` | 14/320 | **PASS，最低** |
| 10 | 726.3 | 28.5295 | 0.947427 | 0.117499 | `−0.002404 [−0.008177,+0.003302]` | 21/320 | FAIL，CI跨零 |
| 4 | 433.6 | 28.5422 | 0.946850 | 0.138976 | `+0.019073 [+0.015326,+0.022841]` | 24/320 | FAIL，显著更差 |

少步时 PSNR 上升而 LPIPS 先维持、后恶化，正是 fidelity–perception tradeoff。50 步的 LPIPS 观测值甚至略好于 100 步，说明作者固定 100 步并非该总体上的感知最优；但这不能事后把 50 步称为新 checkpoint，只能说同一权重下的 sampler working point 更好。

语义 failure 相对 S33 的补充审计：

| steps | Diff failures | S33 failures | new/repair | failure-rate差及95% CI |
|---:|---:|---:|---:|---:|
| 100 | 7 | 4 | 5/2 | `+0.938 pp [−0.938,+3.125]` |
| 50 | 10 | 4 | 8/2 | `+1.875 pp [−0.625,+5.000]` |
| 25 | 14 | 4 | 12/2 | `+3.125 pp [+0.625,+6.563]` |
| 10 | 21 | 4 | 17/0 | `+5.313 pp [+1.250,+10.313]` |
| 4 | 24 | 4 | 20/0 | `+6.250 pp [+1.875,+11.875]` |

该 failure CI 是结果出来后的安全解释，不改变预注册 LPIPS gate。它说明“少步更快”不能只看 LPIPS：25 步是感知最低点，却不是当前项目语义可靠性意义上的安全最低点。

## 五、DiffJSCC 25 步的组件分解

| 组件 | mean ms | 占总时间 |
|---|---:|---:|
| 256→512、H2D | 3.162 | 0.2% |
| JSCC frontend | 5.592 | 0.4% |
| BLIP2 caption | 104.239 | 7.1% |
| OpenCLIP text conditioning | 4.269 | 0.3% |
| sampler schedule | 0.223 | <0.1% |
| spatial condition encoder + latent init | 37.490 | 2.6% |
| **25 次 ControlNet/UNet denoising** | **1,226.110** | **84.1%** |
| VAE decode | 70.221 | 4.8% |
| wavelet color fix | 0.451 | <0.1% |
| D2H + 回缩到256 | 4.725 | 0.3% |
| **端到端** | **1,458.465** | 100% |

不同步数的非 denoiser 固定部分稳定在约 `232.4 ms/图`；单步 denoiser 约 `48.8–49.9 ms`。因此当前 checkpoint 的延迟基本可写成：

```text
DiffJSCC latency ≈ 232 ms + N × 49 ms
```

这是本卡、本实现的经验式，不是跨硬件定律。

## 六、SGD-JSCC 组件分解

SGD 50-step paper upper 的总时间为 `2,044.70 ms/图`：

| 组件 | mean ms | 占总时间 |
|---|---:|---:|
| 输入、H2D、四 patch | 1.318 | 0.1% |
| BLIP2 四 patch captions | 220.568 | 10.8% |
| **MuGE soft-edge extractor** | **849.366** | **41.5%** |
| **diffusion solver** | **908.018** | **44.4%** |
| main VAE encode | 7.683 | 0.4% |
| edge JSCC | 10.936 | 0.5% |
| edge VAE encode | 7.691 | 0.4% |
| CLIP text conditioning | 5.261 | 0.3% |
| pipeline 内部 VAE decode | 16.130 | 0.8% |
| final VAE decode | 16.078 | 0.8% |
| 其他 channel/mask/step-match | 1.426 | 0.1% |

源码与运行时共同确认：`pipeline.generate` 内部先把 latent VAE decode 成图，但调用方丢弃这张图，只取 latent，随后又做一次 final VAE decode。第一遍 `16.13 ms` 是**可无损删除的实现冗余**，但只占总时间 `0.79%`；删掉后 SGD 仍约 2.03 秒，不改变百倍结论。SGD 的大头不是这处小 bug，而是 MuGE edge extractor 和 50-step diffusion。

## 七、参数量与 FLOPs

参数按运行时实际驻留、parameter object 去重统计：

| 方法 | 去重参数量 | 相对 S33 |
|---|---:|---:|
| S33 | 31.03M | 1.0× |
| DiffJSCC | 5.479B | 176.6× |
| SGD | 4.597B | 148.2× |

DiffJSCC 的主要参数为 BLIP2 `3.745B`、diffusion UNet `865.9M`、ControlNet `365.2M`、OpenCLIP conditioner `354.0M`、VAE `83.7M`。SGD 也由 BLIP2 `3.745B` 主导。

FLOPs 使用 `torch.profiler(with_flops=True)`，只统计其支持的 conv/linear/mm/bmm executed FLOPs，因此是可复现的**下界**，不是理论完整 FLOPs：

| 方法/步数 | profiled FLOPs lower bound | 相对 S33 |
|---|---:|---:|
| S33 | 0.05693 TFLOPs | 1.0× |
| DiffJSCC 100 | 94.156 TFLOPs | 1,653.9× |
| DiffJSCC 50 | 49.303 TFLOPs | 866.0× |
| **DiffJSCC 25** | **26.877 TFLOPs** | **472.1×** |
| DiffJSCC 10 | 13.421 TFLOPs | 235.7× |
| DiffJSCC 4 | 8.038 TFLOPs | 141.2× |
| SGD 50 | 36.389 TFLOPs | 639.2× |

DiffJSCC 的 profiler 下界公式为：

```text
4.450 TFLOPs fixed + N × 0.897 TFLOPs/denoiser evaluation
```

固定部分中，VAE decode=`2.481T`、spatial condition encoder=`1.083T`、BLIP2 caption=`0.762T`。这些 FLOPs 与 wall time 不成严格比例，因为不同算子的并行度、memory-bound 程度和 xFormers 漏计不同；因此报告同时保留 ms 与 FLOPs，不用任一项替代另一项。

## 八、什么是“固有代价”，什么是“未优化代价”

### 当前 latent-diffusion 生成链的结构性成本

对当前 DiffJSCC，要生成 pixel output 至少需要：

- 若干次 conditional denoiser evaluation；
- latent-to-pixel VAE decode。

这两项是当前 latent diffusion 路径不可直接删除的。25 步是本轮候选网格中保持显著 LPIPS 优势的经验下限，不是数学上证明“24 步一定不行”；4 步只是最低已测点。

### 当前方法特有、但不是 diffusion 普遍固有的成本

- BLIP2 caption；
- OpenCLIP text conditioner；
- 512×512 内部工作分辨率；
- spatial condition encoder 的具体大小；
- wavelet color fix；
- 当前 SD2.1/ControlNet/UNet 的模型规模。

这些都可以通过无文本条件、更小模型、更低内部分辨率、latent consistency/distillation/新 solver 等路线降低，但会改变方法或权重，必须重新训练/蒸馏或至少重新验证质量与 semantic drift。不能仅凭愿景把这些成本从当前数字里扣掉。

### 明确属于未优化实现的成本

- 100 步相对本总体的 25/50 步不是必要工作点；
- SGD 有一次返回值被丢弃的重复 VAE decode；
- 模型量化、编译、TensorRT、蒸馏和更小 backbone 尚未做。

因此正确的立论不是“diffusion 天生必须慢 879×”，而是：

> **在同一 4090D、batch=1、同 runtime 的公平测量中，现有代表性生成式 JSCC 即使把 DiffJSCC 从100步压到仍保持显著感知优势的25步，仍需约165×延迟和至少472×已计 FLOPs；继续压到10/4步虽然更快，但感知优势不再显著或已经反转，语义失败也上升。**

这个结论足以支持“低复杂度/低延迟生成式 JSCC”作为研究方向，但不支持“所有未来生成式 JSCC 必然慢百倍”的绝对命题。

## 九、证据边界

- 质量曲线使用已知 policy-development 64图×1seed×5SNR，不是 official validation；
- official Imagenette validation 全程封存；
- 没有训练、下载或 checkpoint 选择；
- 100-step 质量对历史结果零误差重放；
- SGD 仍是非等码率、完美 caption paper upper，只测系统成本；
- FLOPs 是 supported-op lower bound；
- 不含冷启动/模型加载；
- 不包含尚不存在的蒸馏、小模型或 TensorRT 版本，不能对它们虚构速度。

## 十、产物

- 汇总：`outputs/analysis/ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001/aggregate_summary.json`
- common-runtime 复核：`common_runtime_post_analysis.json`
- 延迟/参数/FLOPs 表：`latency_parameter_flops_comparison.csv`
- common-runtime 延迟表：`latency_comparison_common_torch21.csv`
- DiffJSCC 曲线：`diffjscc/latency_quality_curve.csv`
- DiffJSCC 逐样本：`diffjscc/quality_rows.csv`
- 语义补充：`semantic_failure_post_analysis.json`
- S33/SGD/DiffJSCC 组件 summary：对应子目录 `summary.json`
