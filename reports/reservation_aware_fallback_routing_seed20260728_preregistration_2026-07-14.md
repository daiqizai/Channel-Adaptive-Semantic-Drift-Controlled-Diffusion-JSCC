# 预留感知三路 fallback：seed20260728 信道复核预注册

日期：2026-07-14。阶段：S15 exploratory independent-channel replication。

## 选择依据与限制

在已暴露的 policy-dev / channel seed `20260727` 上，预留感知 B1 加冻结 S14 diffusion 的系统端点为 7 个 `reference M2 correct -> final wrong` 与 6 个 repair。7 个新增错误中，6 个样本的 in-budget raw 与 posterior 都保持正确，但旧二路规则在拒绝后退回错误 anchor；其中 5 个还出现 recovered sender top-1 与 receiver anchor top-1 不一致。

据此只冻结一个无需原图、无需 T_cls、无需新阈值的三路规则：

1. cross-model triplet 全部通过：输出 posterior；
2. 未通过且 recovered sender top-1 与 G_gate(anchor) top-1 不同：输出 diffusion raw；
3. 其余未通过：输出 anchor。

seed20260727 的离线诊断只用于选择规则，不能作为独立证据；该规则在读取 seed20260728 结果前写入代码、测试和本文件，后续不按 seed20260728 结果修改。

## 冻结复核

- 图像：同一 Imagenette `policy_dev` 1894 张；这是已暴露开发集，不包装成未见数据泛化；
- 信道：只使用新 AWGN channel seed `20260728`，SNR `[1,4,7,13,19]`；
- 总码率：DeepJSCC `c=8`、CBR `1/6`，UInt2 BPSK×4 payload 80 个实符号，图像部分 65456 个实符号；
- B1：`outputs/EXP-S15-001/checkpoints/best.pt`，reservation-aware；
- diffusion：冻结 `outputs/EXP-S14-001/checkpoints/best.pt`；
- reference M2：相同 AWGN noise 下的 paired unpunctured reference，使用同一 reservation-aware B1 与冻结 diffusion；
- official Imagenette val 不访问。

## 判据

沿用严格 promotion gates：payload 恢复门槛、masked data-consistency、逐 SNR/逐 seed 失败率、system new-error cluster 上界、五档 PSNR 正增益、图像 cluster bootstrap 的 PSNR/LPIPS/failure CI。只有 seed20260728 原样通过才记为本次 channel replication 的 POSITIVE；否则保留负结果，不继续在该 seed 上调路由。
