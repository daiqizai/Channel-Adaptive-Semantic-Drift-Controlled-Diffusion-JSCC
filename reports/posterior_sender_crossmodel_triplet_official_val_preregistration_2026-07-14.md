# Cross-model triplet official-val 一次性审计预注册

日期：2026-07-14。审计编号：`ANALYSIS-PC-SENDER-CROSSMODEL-OFFICIAL-VAL-001`。

> **状态：CANCELLED BEFORE OUTCOME。** 后续严格统计审计推翻了本协议的启动前提：UInt4 seed20260727 相对 paired unpunctured M2 的 system new-error 为 7 clusters/1690、upper95 `0.7766%>0.5%`，且 1 dB failure `32→34`。因此本协议不得执行；official consumed marker、canonical output 和模型 outcome 均未生成。本文其余内容只保留为一次性审计工程设计记录，不再授权 official-val 访问。

## 审计目的与证据隔离

本审计只回答一个问题：在 policy-dev 上冻结、并已通过全新 channel seed `20260727` 的 cross-model triplet M3，能否迁移到完全不同的 Imagenette 图像总体。方法、controller、载荷、posterior correction、checkpoint、SNR、seed、统计总体和成功门槛都必须在任何 official-val 模型 outcome 产生前锁定。

official Imagenette validation 共 3925 张，之前没有模型推理结果、方法指标、final 输出或 `official_val_accessed=true` 记录。封存定义需作一项透明修正：早期 archive/integrity 审计曾读取验证成员字节；2026-07-14 的只读仓库搜索又因排除路径写错，对 3925 个 JPEG 做了二进制探测并更新 atime。两者都没有产生或展示方法 outcome，本审计因此称为 **outcome-sealed**，不再称为 byte-sealed。

## 一次性执行约束

- canonical output：`outputs/analysis/imagenette_supervised_final/`；
- consumed marker：`outputs/analysis/.imagenette_supervised_final.OFFICIAL_VAL_CONSUMED.json`；
- staging：`outputs/analysis/.imagenette_supervised_final.staging/`；
- marker 必须在 official-val 模型推理和 outcome 计算前以 `O_CREAT|O_EXCL` 写入；
- 中断、异常或 gate 失败都视为 official-val outcome 已消费，不得删除 marker 后重跑；
- final 入口不提供 `dry-run`、subset、batch-start、output override 或 overwrite；
- 只有完整 row grid、产物校验和 metrics 写完后，staging 才原子发布为 canonical output。

正式入口在运行前逐文件校验脚本、配置、checkpoint、manifest、LPIPS 权重和关键源码 SHA-256，并检查第三方 DeepJSCC 仓库 commit/clean 状态。由于项目根工作树存在大量未提交/未跟踪研究文件，不能只用根 commit 代替逐文件哈希。

## 冻结总体与随机性

- 图像：全部 3925 张 official `imagenette2-320/val`，按 `official_val/<WNID>/<filename>` 排序；
- official manifest 在 consumed marker 写入后由既有 sealed-manifest builder 构建；逐文件与官方归档成员 SHA-256 对齐，并排除 train/val 精确内容重复；
- channel seeds：沿用 2026-07-10 就已声明、且从未产生 official-val outcome 的 `[20260711, 20260712, 20260713]`；
- SNR：`[1,4,7,13,19] dB`，primary 为 `[1,4,7] dB`；
- batch size：8；row grid 固定为 `3925 × 3 × 5 = 58,875`，必须完整且键唯一；
- 几何变换：`Resize(256) → CenterCrop(256) → ToTensor`，不新增 PNG 量化或 TTA；
- clean-correct：`T_cls(original)==WNID` 且 calibrated confidence `>=0.50`；至少 2500 张、每类至少 150 张。

## 冻结方法

通信与 restoration 主体保持不变：

- DeepJSCC `c=8`、S13 B1 anchor、S14 6-step residual-shift diffusion；
- received-latent posterior correction 为 3 步、normalized step `0.001`；
- 10 维 `G_aux(source)` 每类 UInt4，共 40 bit；BPSK ×4；
- 总 65,536 实符号，其中 payload 160、图像 65,376；总 CBR 始终 `1/6`；
- payload 与图像使用同一次 punctured-arm AWGN；receiver 擦除 payload 位置，posterior consistency 排除相同位置；
- `T_cls` 和真实标签只做 outcome audit，不进入 controller。

posterior 仅在以下三项同时成立时被接受：

```text
JS(q_recovered, G_aux(posterior)) - JS(q_recovered, G_aux(anchor)) <= 0
argmax(q_recovered) == argmax(G_gate(anchor))
argmax(G_gate(anchor)) == argmax(G_gate(posterior))
```

## Paired M2 公平对照

旧 strict-rate 入口依赖提前生成 M2 reference CSV，不适合一次性 final。新内核在同一个不可交互 invocation、同一 batch 内同时计算 unpunctured M2 和 punctured M3：

```text
latent = DeepJSCC.encode(source)

manual_seed(derived_seed(seed, snr, batch_start))
M2_received = AWGN(latent)

manual_seed(同一个 derived_seed)
M3_received = AWGN(embed_payload(latent))
```

两臂复用同一标准高斯随机 draw；由于 channel 会按各臂实测发送功率缩放，不能表述成最终 additive-noise tensor bit-exact，只能称为 numerically paired standardized draw。M3 payload 和图像仍只经过一次共同 channel call。M2 与 M3 outcome 在整次 final 完成前不单独发布。正式访问前，新 paired kernel 必须先在已暴露 policy-dev `20260727` 全量回放，并证明 M3 离散决策逐键一致、连续指标在冻结数值容差内复现；回放本身不构成新泛化证据。

该前置回放已在最终 kernel 版本上完成：9470 行、66 列、完整键空间一致；除允许重定义的 `reference_final_{correct,psnr,lpips}` 外，所有 M3 字段和 M2 anchor/raw 字段逐字符串完全相同。允许差异来自旧 reference 表把 noiseless sender feasibility controller 的 final 写入 `reference_final_*`，而新 paired protocol 明确令公平 reference final 等于无 controller 的 M2 raw；正式 gate 从始至终只比较 `reference_raw_*`。机器校验输出：`outputs/analysis/pc_imagenette_sender_crossmodel_triplet_seed20260727_paired_replay_verification/verification.json`。

## 一次性成功门槛

本协议明确是对 2026-07-10 原 official-val estimand 的方法修订，而不是悄悄放宽旧 gate。当前 M3 的目标从旧 edge-M3 的“每个 SNR 都有正 PSNR”改为“固定总码率下的 semantic-tail/coverage tradeoff”，这是在已知 policy-dev 低中 SNR PSNR 回吐后形成的 development-outcome-informed estimand。为避免选择性结论，正式输出同时给出两层 verdict：

- `tradeoff_gates_pass`：当前方法特定的方向性 tradeoff 门槛；
- `strict_promotion_gates_pass`：迁移 2026-07-10 原协议的 image-cluster inference、逐 SNR 质量和 worst-class guardrail。只有两层都通过，顶层 verdict 才能为 `POSITIVE`；若仅前者通过，必须写成 `PARTIAL_TRADEOFF_POSITIVE`，不得称为 supervised-safe 或通过原 official-val 标准。

方向性 tradeoff 层要求：

1. 3925 图、58,875 行、键唯一、clean 总数/每类数达到门槛；
2. `c=8`、CBR `1/6`、65,536/160/65,376 账本精确，payload 与图像共享 punctured-arm AWGN；
3. 每个 seed×SNR 的 recovered top-1 agreement、cosine 和 40-bit exact-vector rate 均 `>=0.95`；
4. 每个 seed×SNR 的 masked data-consistency 均下降；
5. primary aggregate 及每个 seed 的 final failure 不高于 paired unpunctured M2 raw；
6. primary aggregate、每个 SNR、每个 seed、每个 seed×SNR 的 final new-error 不高于 in-budget raw；
7. primary final new-error 图像簇单侧 95% Clopper-Pearson upper `<=0.5%`；
8. 五 SNR、三 seed 聚合 final-minus-M2 PSNR `>0`，LPIPS `<=0`。

严格 promotion 层在上述条件外再要求：

1. 用固定 seed `161803` 做 10,000 次 `sample_id` cluster bootstrap；每次抽中一张图时保留该图全部 3 seed×5 SNR 行；
2. primary failure-rate delta（final − paired M2 raw）的双侧 95% CI upper 严格 `<0`；
3. 全五 SNR PSNR delta 的 95% CI lower 严格 `>0`；
4. 全五 SNR LPIPS delta 的 95% CI upper `<=0`；
5. 每个 SNR 的 final-minus-M2 PSNR point estimate 均 `>0`；
6. 最差 WNID 的 primary failure-rate delta `<=2 pp`。

bootstrap 的 failure cluster population 是 clean-correct 图像；每图统计保留其全部 3 seed×3 primary-SNR 行。new-error Clopper-Pearson endpoint 的分母是 primary 范围内至少一次同-row anchor-correct 的 unique image，事件是至少一次同-row `anchor-correct→final-wrong`，因此不会把 9 个重复行当作独立图像。

逐 SNR 的 PSNR 回吐、coverage、payload BER、perfect-payload 反事实、repair 与 failure 都必须完整报告。即使 aggregate 通过，也不能据此声称每个 SNR 全面质量领先；任一 gate 失败则原样记录为 final negative，不得调 threshold、coverage、位宽、重复数、seed 或样本子集。

## 审计后的分支

- 若通过：当前方法取得 image-population 阶段正结果，随后把 controller-free received-latent proximal correction 固化为 `DDNM-inspired mechanism baseline（非作者复现）`，并准备一个真正外部方法的统一合同接入；
- 若失败：保留 diffusion posterior restoration 的机制结论与本次 final negative，停止在 Imagenette official val 调方法；外部基线仍可做，但不得用 final outcome 反向选择 M3。
