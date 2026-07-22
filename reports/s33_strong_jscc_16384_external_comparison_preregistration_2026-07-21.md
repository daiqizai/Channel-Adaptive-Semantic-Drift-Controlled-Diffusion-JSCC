# S33：16,384-real Strong vs author-JSCC 等码率比较预注册

日期：2026-07-21
状态：在 S33 strong 首次访问冻结 S32 policy-dev population 前冻结。

## 冻结输入

S33b 完成 8/8 FP32 continuation epochs，全部 finite。最终 checkpoint 只由 COCO val2017 固定 512 图五档 aggregate PSNR/MS-SSIM 选择，不访问 S32 population：

- checkpoint：`outputs/train/EXP-S33B-STRONG-JSCC-16384-FP32-001/checkpoints/best.pt`
- epoch：`7`
- SHA-256：`2daad9e73df9bca049e02800d32e4f34298bab6452dcf32634f6320881dd5bfb`
- COCO aggregate：`29.415098 dB / 0.966782 MS-SSIM`
- native rate：`16,384 real / 8,192 complex uses`
- trainable parameters：`31,028,163`

author-JSCC 与 S30 `per_sample.csv`、S20 population、T_cls 和 canonical seeds 全部复用冻结 SHA。S32 的 `19,712-real` strong 和 author outcome 已知，但 S33 `16,384-real` strong 在该 population 上的任何质量/语义 outcome 当前未知。official Imagenette validation 继续封存。

## 同噪声合同

S30 每个 key 保存的是完整 `19,712-real` canonical standard-normal 向量的 SHA。S33 对每个 key 必须：

1. 重新生成同一 `19,712-real` 向量并验证其 SHA 与 S30 完全一致；
2. 只取其前 `16,384` 个实坐标作为 strong 信道噪声；
3. 对该 prefix 另存 SHA；
4. author-JSCC 与 strong 因而使用同一 canonical noise prefix，双方均严格 `16,384 real`。

禁止直接生成一个未通过完整向量 SHA 审计的 16,384 维噪声并声称同 realization。

## 评估与判定

population 为 64 张已知 Imagenette policy-dev 图；channel seeds=`[20260748,20260749,20260750]`，SNR=`[1,4,7,13,19]`，每方法 960 行。primary image quantization 继续使用 S30/S32 的 floor-uint8；float 输出只做 sensitivity。

必须输出 per-SNR/aggregate PSNR、MS-SSIM、LPIPS、T_cls failure、新错/修复，以及 strong−author 的 source-image-cluster 10,000 次 bootstrap 95% CI。13/19 dB 单列。

用户冻结的 PSNR 判定：

- 95% CI 下界 `>0`：显著超过；
- 下界在 `(-0.10,0] dB`：在 0.10 dB margin 下追平/非劣；
- 下界 `<-0.10 dB`：劣于；
- 恰为 `-0.10 dB`：边界不确定。

LPIPS、MS-SSIM 或 semantic failure 与 PSNR 冲突时只称 Pareto，不称全面超过。本分析仍是已知 policy-dev 上的严格等码率 gate，不是 independent final test。分析结束后停止，本轮不启动 S34--S36。
