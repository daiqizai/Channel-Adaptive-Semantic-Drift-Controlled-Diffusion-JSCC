# S34C-Lite：生成式 JSCC 码率与先验透明度分析

日期：2026-07-23
分析 ID：`ANALYSIS-S34C-LITE-RATE-TRANSPARENCY-001`

## 一、结论先行

这份分析不能给三种方法排一个“总冠军”，但把当前最容易混淆的事实厘清了：

1. **S33 与 DiffJSCC 的信道码率是严格相同的。**两者都发送 `16,384 real`，都没有发送端文本或边缘 side information。DiffJSCC 的 caption 是接收端根据带噪初始重建本地生成的，因此不能把它的感知优势解释为“白嫖了发送端文本码率”。
2. **S33 与 DiffJSCC 是清楚的 fidelity–perception Pareto。**S33 的 PSNR 和 MS-SSIM 显著更高，DiffJSCC 的 LPIPS 显著更低；语义失败率的差异置信区间跨零。也就是说，当前证据支持“纯 JSCC 保真/结构端点”与“生成式感知端点”的取舍，而不是一方全面击败另一方。
3. **SGD-JSCC 当前结果只能作为 non-ranking paper upper。**其已执行的 main+edge 已占 `19,712 real`；再按最低、未保护的 caption 物理传输计费，总量至少 `21,856 real`，相对 S33 超出 `5,472 real / 33.40%`，而且当前 caption 是完美、无误传

输。它的 LPIPS 最好，但不能据此做严格等码率排名。

因此，轻量版最重要的判断是：**DiffJSCC 的等码率感知优势是真实需要正视的，而 SGD 的现有感知上界含有额外码率与完美文本条件。**当前项目可以把 S33 定位为低复杂度、强保真、较可靠的 exact-rate 端点；如果以后恢复 diffusion 方向，目标应是沿 Pareto 前沿改善感知，而不是把现有 DiffJSCC 说成不公平。

## 二、分析合同与完整性

本轮只读取既有冻结结果，没有训练、模型推理、下载或调参，也没有访问 official Imagenette validation。

- 总体：同一批已知 policy-development 64 张图；
- 信道：3 个相同 canonical AWGN seeds：`20260748/20260749/20260750`；
- SNR：`1/4/7/13/19 dB`；
- 每种方法：`64×3×5=960` 行；
- 置信区间：按 source image 聚类、10,000 次 bootstrap 的双侧 95% CI。

完整性审计通过：

- 三种方法的 960 个 `sample/seed/SNR` 键完全一致；
- canonical noise SHA 不一致数为 0；
- class/WNID 不一致数为 0；
- S33 文件内嵌的 DiffJSCC 指标与 S30 正式文件最大绝对误差为 0；
- official validation accessed=`false`。

需要注意：S33/DiffJSCC 的冻结主指标使用 uint8 截断图像，SGD 的冻结指标使用 float 重建张量。因此涉及 SGD 的跨合同数值只作描述，不能包装成精确排名。

## 三、严格码率账本

| 方法 | 主图像 | edge | caption 最低成本 | 最低物理总量 | 相对 16,384 | 发送端 side information | 接收端/外部先验 |
|---|---:|---:|---:|---:|---:|---|---|
| S33 strong | 16,384 | 0 | 0 | **16,384** | 0 | 无 | 无生成式大模型先验 |
| DiffJSCC | 16,384 | 0 | 0 | **16,384** | 0 | 无；caption 在接收端生成 | SD2.1 + BLIP2 + OpenCLIP |
| SGD paper upper | 16,384 | 3,328 | 至少 2,144 | **至少 21,856** | **+5,472 / +33.40%** | active edge + 4 个完美 captions | diffusion/ControlNet + BLIP2/CLIP |

SGD 的 `21,856 real` 仍是乐观下界：caption 按 4×536 raw bits、每 bit 一个未保护 BPSK 实坐标计费；若需要纠错保护，真实成本只会更高。当前实际运行经过信道的是 main+edge=`19,712 real`，caption 没有经过信道。

外部预训练参数或接收端计算不属于“信道码率”，所以不能把 DiffJSCC 的 SD2.1/BLIP2 参数量加进 real-symbol 账本；但必须在复杂度、训练数据和系统依赖中明确披露。

## 四、现有共同指标

以下是 960 行五档聚合均值；失败为冻结 `T_cls` 语义失败计数。

| 方法 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | 语义失败 | 失败率 | 平均推理时间/图 |
|---|---:|---:|---:|---:|---:|---:|
| S33 strong | **30.4661** | **0.969708** | 0.119985 | **9/960** | **0.9375%** | 2.66 ms |
| DiffJSCC | 27.5984 | 0.940799 | **0.100223** | 23/960 | 2.3958% | 5,237.71 ms |
| SGD paper upper | 27.7404 | 0.952973 | **0.072101** | 25/960 | 2.6042% | 2,064.74 ms |

这里的加粗只标每列观测值，不代表跨合同总排名。尤其 SGD 的 LPIPS 不能与两个 `16,384-real` 方法作公平胜负；S33 的峰值显存没有可比冻结记录，也不能补猜。

### S33 对 DiffJSCC：可比较的 exact-rate Pareto

`S33 − DiffJSCC` 的聚合差值为：

| 指标 | 差值 | source-cluster 95% CI | 解释 |
|---|---:|---:|---|
| PSNR ↑ | **+2.8677 dB** | `[+2.7473,+2.9884]` | S33 显著更高 |
| MS-SSIM ↑ | **+0.028909** | `[+0.025125,+0.032930]` | S33 显著更高 |
| LPIPS ↓ | **+0.019762** | `[+0.008057,+0.032420]` | DiffJSCC 显著更好 |
| failure ↓ | **−1.458 pp** | `[−3.854,+0.208] pp` | S33 观测更低，但 CI 跨零 |

结论是 fidelity–perception Pareto，而不是“DiffJSCC 被 S33 战胜”。此外，DiffJSCC 使用作者 OpenImage/连续 `[0,14] dB` 训练合同，S33 使用 COCO/离散五档任务训练；19 dB 还超出 DiffJSCC 的作者训练范围。这些差异不改变二者信道等码率事实，但限制了算法归因。

### 分 SNR 结果

| SNR | 方法 | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | failure |
|---:|---|---:|---:|---:|---:|
| 1 | S33 | 28.1617 | 0.946395 | 0.163678 | 6/192 |
| 1 | DiffJSCC | 25.5518 | 0.904052 | 0.149835 | 8/192 |
| 1 | SGD upper | 26.1182 | 0.932872 | 0.089137 | 5/192 |
| 4 | S33 | 29.5217 | 0.963134 | 0.131676 | 2/192 |
| 4 | DiffJSCC | 26.7736 | 0.929894 | 0.113992 | 6/192 |
| 4 | SGD upper | 26.8927 | 0.944893 | 0.084817 | 8/192 |
| 7 | S33 | 30.5933 | 0.972933 | 0.113743 | 1/192 |
| 7 | DiffJSCC | 27.8381 | 0.946594 | 0.091288 | 4/192 |
| 7 | SGD upper | 27.9112 | 0.956979 | 0.067310 | 5/192 |
| 13 | S33 | 31.8095 | 0.981705 | 0.097565 | 0/192 |
| 13 | DiffJSCC | 29.0896 | 0.962787 | 0.070053 | 4/192 |
| 13 | SGD upper | 28.5682 | 0.962917 | 0.063549 | 6/192 |
| 19 | S33 | 32.2441 | 0.984375 | 0.093265 | 0/192 |
| 19 | DiffJSCC | 28.7389 | 0.960668 | 0.075949 | 1/192 |
| 19 | SGD upper | 29.2115 | 0.967206 | 0.055694 | 1/192 |

S33 相对 DiffJSCC 在五档的 PSNR 优势分别为 `+2.6099/+2.7481/+2.7552/+2.7199/+3.5052 dB`，每档 CI 都全正；但 LPIPS 每档都由 DiffJSCC 更好，且每档 CI 也都全正。这个稳定模式进一步支持 Pareto 解释。

## 五、SGD paper upper 能说明什么、不能说明什么

描述性地看，SGD paper upper 相对 S33 的 LPIPS 低 `0.047884`，95% CI `[0.035195,0.061408]`；S33 的 PSNR 高 `2.725696 dB`，CI `[2.465822,3.009596]`。SGD 也比 DiffJSCC 有更低的 LPIPS，观测差 `0.028122`，CI `[0.024501,0.031841]`。

但这些差值都属于 `cross-contract_non-ranking`，原因同时包括：

- SGD 最低物理码率比 S33/DiffJSCC 高 33.40%；
- captions 完美且未经过信道；
- SGD JSCC backbone 按 ImageNet、固定 10 dB 训练，而非项目 COCO 五档合同；
- 外部预训练、模型容量和推理计算没有对齐；
- 冻结指标还存在 float 与 uint8 路径差异。

所以它只证明“在作者协议上，强生成式先验配合 edge/完美文本可以达到很低 LPIPS”，不能回答“压到严格 16,384 real 后还剩多少优势”。后一个问题仍需要未来的 rate-constrained retraining/adaptation，轻量分析无法替代。

## 六、证据缺口与投稿含义

当前共同结果没有 FID 或 KID，不能判断三种方法的分布级生成质量，也不能声称任一方法是感知质量总冠军。64 图总体还是已知 policy-development population，而不是最终独立测试。official validation 必须继续封存。

对近期论文最稳妥的写法是：

- 把 S33 写成严格 `16,384 real`、无 side information、低计算的强 fidelity/reliability backbone；
- 把 DiffJSCC 写成相同信道码率、依赖大型接收端生成先验的感知端点，并如实报告 Pareto；
- 把 SGD 放在独立的 protocol/rate-transparency 表中，明确 `≥21,856 real` 与 perfect-caption upper，不参与主排名；
- 不写“生成式方法优势来自码率作弊”，因为这一判断只适用于当前 SGD 协议，不适用于 DiffJSCC。

这张表足以支撑“码率与先验透明度”分析节，但不足以单独支撑“公平条件下生成式优势被高估”的强论文主张。是否投入 14–29 天公平重训，应该由投稿时程和是否必须回答感知主场问题决定；本轮不自动重启长版。

## 七、可复现文件

- 正式输出：`outputs/analysis/ANALYSIS-S34C-LITE-RATE-TRANSPARENCY-001/`
- 统一方法表：`unified_method_table.csv`
- 分 SNR 表：`per_snr_table.csv`
- 配对差值与 CI：`pairwise_descriptive_deltas_with_ci.csv`
- 码率/先验账本：`rate_and_prior_ledger.json`
- 输入一致性审计：`input_audit.json`
- 汇总：`summary.json`

历史执行命令（输出已存在，禁止覆盖）：

```bash
python3 scripts/s34c_lite_rate_transparency.py
```
