# S19 强融合 + exact-B1 fallback 完全新总体复现预注册（S27）

日期：2026-07-21。分析 ID：`ANALYSIS-S27-S19-XF-FRESH-001`。

## 目的

S26 已在另一 population 上以 9/9 PASS 复现 `S19 low-SNR fusion + exact-B1 high-SNR fallback`，但该图片总体此前用于 S21/S23，因此还不是完全 pristine final test。S27 不再修改方法，只做一次更严格的冻结复现。

## 完全冻结的方法

- 1/4/7 dB：运行 frozen S19 fusion；paired control 使用相同参数量的 frozen S19 B0-only control。
- 13/19 dB：fusion/control 都结构性返回 frozen B1，必须逐像素精确一致。
- checkpoint、route、SNR、AWGN 方差、active coordinates、80-symbol reservation、三分类器、bootstrap seed 和全部成功门槛均在 population 产生前冻结。
- 不训练、不使用 selection、不根据新总体修改任何规则。

## 新总体

从本地 COCO train2017 以 seed `20260771` 做 SHA rank，抽取 512 张 holdout。必须同时排除：

- S16 11,000 张 source；
- S18 512 张 source；
- S19 5,512 张 source；
- S21/S23 5,512 张 source；
- COCO val2017 同名文件；
- 上述集合的路径与 SHA256，以及新总体内部重复 SHA。

正式 materialization 后 path overlap 和 SHA overlap 必须均为 0。每张图在 `[1,4,7,13,19]` dB 下用 role seed `20260772` 生成 canonical AWGN；总预算仍为 `19,712 real = 9,856 complex uses`，无额外 side information。

## 成功判据

沿用 S26，不因样本数增大而放宽：

1. 高 SNR exact-B1 最大逐像素差不超过 `1e-7`。
2. fusion−control PSNR CI 下界不小于 0，LPIPS CI 上界不大于 0。
3. fusion−B1 PSNR 均值至少 `+0.05 dB`、CI 下界至少 `+0.03 dB`，LPIPS CI 上界不大于 0。
4. 五档 SNR 的 fusion−B1 PSNR 均非负。
5. majority new 不多于 repair，fusion majority failure 不多于 control。

全部通过才升级为 pristine-population positive replication。否则如实记录负结果，不更换 seed、图片或门槛。

## 执行顺序

1. 只读 dry-run 核对候选与排除集合。
2. materialize 512 张原图并冻结 source manifest SHA。
3. 更新 config 为 cache 阶段，生成 canonical-noise B0/matched-diffusion cache，冻结 cache manifest SHA。
4. 更新 config 为 evaluation 阶段，一次性运行 512×5 指标、三分类器与 10,000 次 source-image cluster bootstrap。

全程使用本地 COCO/checkpoint/cache；无联网、无下载，official Imagenette validation 保持封存。
