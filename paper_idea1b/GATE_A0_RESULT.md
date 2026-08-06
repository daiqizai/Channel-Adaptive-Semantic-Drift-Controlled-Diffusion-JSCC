# Gate A0 完成报告

日期：2026-07-23
实验ID：`GATE-A0-BENCHMARK-SETUP-001`
状态：`complete`；`a1_authorized=false`

## 结论

Kodak + CLIC2020 test 的数据、完整性校验、方法原生大图处理账本和DISTS/FID/KID指标链已经就绪。A0没有训练、没有运行任何方法推理、没有访问official Imagenette validation，也没有移动或复制既有S33/S34D资产。

A0同时发现一个必须在A1前正视的事实：**“各方法使用原生最优大图路径”不等于“它们自然拥有相同actual CBR”。** Kodak上S33/Swin是`1/24`，而当前DiffJSCC整图公式臂是`1/96`；CLIC边界padding还会让S33/Swin高于`1/24`。因此A1不能直接沿用256图上的“16,384 real exact-rate”标签，必须逐图报告actual symbols/CBR；只有actual CBR相同的臂才可作严格胜负，其余应作rate-quality Pareto或分层结果。

## 数据与来源

- Kodak：24张，全部RGB，尺寸为`768x512`或`512x768`。官方逐图CDN对本服务器返回403，因此使用公开GitHub mirror；tar中24个成员逐个通过官方页面字节数表。archive SHA-256=`44e2569b71dd0b35950ca0b0ddc36cc974d307c6990066147893008940300223`。
- CLIC2020 Mobile test：官方包`721,789,976` bytes，SHA-256=`2025f07a6c652270e534640de4271feef3b3dd3260beed4ac4821064837aa732`。
- CLIC2020 Professional test：官方包`891,643,809` bytes，SHA-256=`857df244fc2bfa5da28d4c66bf0db16ee99bfc79eb807be8afa89cd507852884`。
- 总计452张，全部RGB，逐文件SHA无重复。
- 全部大文件显式清空proxy环境变量，通过服务器直连下载。

## 原生处理与码率账本

| 方法 | Kodak原生处理 | Kodak actual CBR | CLIC预检 | A1排名资格 |
|---|---|---:|---|---|
| S33 strong | 6个非重叠256 tile，边界padding计费 | `1/24` | `0.041667–0.063210` | 逐图CBR相同才可strict ranking |
| Swin Base/CM | 同冻结256 tile adapter | `1/24` | `0.041667–0.063210` | 同上 |
| DiffJSCC | 官方短边至少512、pad到64倍数、整图 | `1/96`（公式预检） | `0.010417–0.011331` | under-budget；A1 runtime instrumentation前不排名 |
| DiffJSCC author-JSCC臂 | 跟随同一官方整图入口 | `1/96`（公式预检） | 同上 | 同上 |
| SGD released paper-upper | 作者`split_image_v2` 128 patch | caption未计费 | 直接套大图可产生80–24,560 patches/图 | 始终non-ranking |

注意：

- S33/Swin的padding tile按完整`16,384 real`计费，不能只按有效像素折算。
- DiffJSCC的receiver-side caption、VAE和diffusion tensor是0通信，但计算量单列；A1必须从真实channel latent shape采集actual symbols，A0公式只作fail-closed预检。
- SGD的main与active edge可计数，但sender caption成本未知；其released分块函数在部分大图尺寸会产生极高重叠，A1不得在未做小样本runtime/memory preflight时直接全量运行。

manifest规模：

- source images：452行；
- method-native rate ledger：2,260行；
- S33/Swin tile manifest：20,018行；
- SGD released patch manifest：882,675行。

## 指标identity sanity

Kodak前4张原图与自身比较：

- PSNR：全部`120 dB`。共享实现把MSE截断到`1e-12`，所以恒等上限不是正无穷；
- MS-SSIM：全部`1.0`；
- LPIPS：全部`0.0`；
- DISTS：`5.96e-8–1.19e-7`。

完整428张CLIC目录与自身比较：

- FID：`−4.5057309649e-5`；
- KID：`−0.00205317698`。

上述值通过冻结容差。首次attempt把PSNR误要求为正无穷，并把self-FID容差设为`1e-5`，因此fail-closed；失败summary和STATE均已保留。按共享实现和clean-fid浮点数值事实把判据修订为PSNR≥`119.999 dB`、`|FID|≤1e-4`后重跑通过，没有改数据或指标实现。

## 可复现资产

- 配置：`configs/gate_a0_benchmark_setup.yaml`，SHA-256=`37ffc8c76bf2622d0c3fc9fb8849e2f74f41790a40372d2a585c12f30802f6bd`。
- 正式summary SHA-256=`975c7d3ef8ab1fed4c019e4ec3da6006da0529f05ce524d8bd170a5cd588eff1`。
- S33 checkpoint、canonical noise实现、S34D aggregate仍在根目录原位；A0复核SHA分别为`2daad9e7...5bfb`、`01978a77...6d22`、`7fdeb1ff...f931`。
- 大数据、权重和输出由`paper_idea1b/.gitignore`排除，防止误提交；provenance由`data/README.md`、本报告和输出manifests记录。

## A1前的决策点

A0不自动建议或启动A1。若用户放行A1，建议先只做少量原生大图的shape/rate/runtime/VRAM smoke，重点确认：

1. DiffJSCC真实channel latent与A0公式完全一致；
2. CLIC上S33/Swin边界padding后的逐图actual CBR分布；
3. 主表采用“同actual CBR严格子表 + 不同CBR的rate-quality Pareto表”，而不是把所有臂硬称exact-rate；
4. SGD是否仅保留为non-ranking系统参考，避免released分块规则在CLIC上失控。
