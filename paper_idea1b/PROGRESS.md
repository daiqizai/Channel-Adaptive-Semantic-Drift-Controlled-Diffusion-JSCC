# paper idea1b 进展

> **归档状态（2026-08-03 起）：** `reports/METHOD_TERMINATION_REPORT_2026-08-03.md` 已 supersede 本文件下方所有历史“已授权/未授权/下一步”状态；它们只说明当时时点，不再构成 A2、S35R-P1、S36、Swin extension、official validation 或任何训练/评测的执行许可。

## Gate A1 判别式评测（2026-07-23）

状态：完成；保守 verdict 为 S33 劣于 SwinJSCC。

已授权：

- S33 vs Swin official Base-SA / capacity-matched CM-SA；
- Kodak 24张×5 SNR×3 seeds；
- CLIC2020 test 428张×5 SNR×1 seed；
- 每阶段先做大图smoke并报告显存/时间。

冻结公平合同：

- 三臂共享A0 tile manifest、相同边界padding、相同逐图canonical noise；
- 每tile均发送`16,384 real`，逐图报告actual symbols/CBR；
- Kodak为`1/24`；CLIC因边界padding可高于`1/24`，但三臂逐图完全同码率；
- DiffJSCC、SGD、refiner与official Imagenette validation均不得由A1脚本加载。

预注册：`A1_DISCRIMINATIVE_PREREGISTRATION.md`。
DiffJSCC码率审计：`DIFFJSCC_RATE_ADJUSTABILITY.md`。

smoke `SMOKE-IDEA1B-A1-DISCRIMINATIVE-001`：

- preflight核对三臂参数=`31.03M/28.18M/31.35M`，均为`16,384 real/tile`；
- Kodak 768×512：6 tiles、actual CBR=`1/24`；S33/Base/CM完整图wall=`42.6/42.6/44.0 ms`；
- 最大CLIC 2048×2048：64 tiles、actual CBR=`1/24`；wall=`189.7/439.5/464.4 ms`；
- peak reserved VRAM=`1.21/2.20/2.21 GiB`；
- 三臂逐图actual rate和canonical-noise SHA完全一致，最大功率误差=`2.38e-7`；
- DiffJSCC/SGD未加载，official validation未访问。

正式结果：

- Kodak完成`1080/1080`行，CLIC完成`6420/6420`行；三臂逐图actual rate/noise均完全相同。
- Kodak aggregate：S33−Base PSNR=`+0.0477 dB`、CI=`[−0.0537,+0.1612]`，为追平/非劣；但LPIPS/DISTS显著更差。S33−CM PSNR=`−0.2003 dB`、CI=`[−0.3116,−0.0846]`，判为劣于。
- CLIC aggregate：S33−Base=`−0.2631 dB [−0.3211,−0.2074]`；S33−CM=`−0.4909 dB [−0.5513,−0.4352]`，两者均劣于。五个SNR档均未通过PSNR非劣门槛。
- CLIC上S33的LPIPS/DISTS/CLIP/FID/KID总体也弱于两条Swin臂；局部例外仅为1 dB DISTS优于Base，不能改变整体结论。
- CLIC逐图actual CBR范围=`0.041667–0.063210`、均值=`0.045472`；三臂在同图上严格相同。Kodak固定为`1/24`。
- 最大2048²图整图指标smoke通过：LPIPS约`9.7–17.8 ms`、peak reserved `1.29 GiB`；DISTS约`795 ms`、`6.37 GiB`；CLIP热身后约`2.1–9.3 ms`、`0.81 GiB`。
- 首次指标smoke因PyTorch 2.6/OpenCLIP TorchScript `weights_only`兼容性失败而保留；第二次以已核SHA的可信本地权重正确加载后通过。全量指标曾在6210/7500处被外部终止，断点续跑后越过该位置并闭合，没有缺失或覆盖。
- Kodak/CLIC无监督标签，本轮只报告原图—重建CLIP连续相似度，不事后构造“语义失败率”；official Imagenette validation仍封存。

结论：S33不能在本轮高分辨率主benchmark上写成“强于Swin的backbone”。它保留最大CLIC图上约`2.32–2.45×`延迟优势和约一半峰值显存，但对应明显质量损失，当前只能定位为低代价判别式端点。完整中文结果见`A1_DISCRIMINATIVE_RESULT.md`；机器可读结果见`outputs/ANALYSIS-IDEA1B-A1-DISCRIMINATIVE-001/summary.json`。

## Gate A0（2026-07-23）

状态：`complete`；A1 仍未授权。

已冻结边界：

- 训练集继续使用 COCO，S33 checkpoint 永久冻结，不启动 DIV2K 重训。
- 主画质 benchmark 改为 Kodak + CLIC2020 test。
- 训练仍使用 `256x256` crop，不追求4K/8K训练。
- 大图不强制共同 tile：DiffJSCC 优先官方整图入口；S33/Swin可使用冻结tile；SGD保留作者原生128 patch。
- 公平性统一在实际通信码率账本，而非统一预处理；sender-side side information、重叠、padding与重复发送全部计费。
- Kodak/CLIC 后续采用 CLIP/DreamSim 等原图-重建无标签语义相似度；本A0只接入DISTS/FID/KID并做identity sanity。
- Imagenette保留监督reliability，official validation封存。
- 根目录可复现资产只引用、不移动、不复制。

当前授权：

- 已下载并校验 Kodak 24 张和 CLIC2020 test 428 张；共452张、全部RGB、无内容重复。
- 已生成 source、method-native processing、tile/patch 和实际码率 manifests。
- 已接入 DISTS/FID/KID，并完成 identity sanity。

数据完整性：

- Kodak 官方逐图 CDN 对本服务器返回403，因此使用公开 GitHub mirror；24个tar成员逐个用官方页面公布的文件字节数核对。mirror archive SHA-256=`44e2569b71dd0b35950ca0b0ddc36cc974d307c6990066147893008940300223`。
- CLIC2020 Mobile 官方包：`721,789,976` bytes，SHA-256=`2025f07a6c652270e534640de4271feef3b3dd3260beed4ac4821064837aa732`。
- CLIC2020 Professional 官方包：`891,643,809` bytes，SHA-256=`857df244fc2bfa5da28d4c66bf0db16ee99bfc79eb807be8afa89cd507852884`。
- 所有大下载均显式清空 proxy 环境变量，使用服务器直连。

manifest 结果：

- source rows=`452`，method-rate rows=`2,260`，S33/Swin tile rows=`20,018`，SGD released `split_image_v2` patch rows=`882,675`。
- Kodak 上 S33/Swin 每图6 tile、`98,304 real`、actual CBR=`1/24`；DiffJSCC官方整图预处理每图 actual CBR公式值=`1/96`，是低于S33预算的原生臂，不得称exact-rate，且A1仍须runtime instrumentation复核。
- CLIC 上 S33/Swin因边界padding的actual CBR范围=`[1/24, 0.0632098765]`；DiffJSCC整图公式范围=`[1/96, 0.0113309556]`。SGD sender caption 未计费，只有main+edge下界，始终non-ranking。
- 冻结旧资产未移动/复制；S33 checkpoint、canonical noise、S34D aggregate 的既有SHA全部复核一致。

identity sanity：

- Kodak 4张恒等输入：PSNR=`120 dB`（共享实现的MSE clamp上限）、MS-SSIM=`1.0`、LPIPS=`0.0`、DISTS=`5.96e-8–1.19e-7`。
- CLIC完整428张目录自比：FID=`-4.5057e-5`，KID=`-0.00205318`；均在冻结浮点容差内。
- 首次sanity使用“PSNR必须为∞、|FID|≤1e-5”的过严判据而失败；失败summary/STATE已保留。判据按实现事实修正为PSNR≥119.999 dB、|FID|≤1e-4后重跑通过，没有改数据、指标实现或方法。
- 正式状态文件为 `outputs/GATE-A0-BENCHMARK-SETUP-001/STATE.json`，`status=complete`、`a1_authorized=false`。
- 完整中文结果与A1前决策点：`GATE_A0_RESULT.md`。

未授权：

- S33、Swin、author-JSCC、DiffJSCC、SGD 的A1正式benchmark推理；
- P1 smoke/训练；
- DIV2K下载或重训；
- official Imagenette validation。
