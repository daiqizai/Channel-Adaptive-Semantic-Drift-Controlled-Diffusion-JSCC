# UInt4-BPSK 固定码率发送端语义控制开发协议

日期：2026-07-14。分析编号：`ANALYSIS-PC-SENDER-DIGITAL-DEV-001`。

## 为什么只测试这一种新编码

`ANALYSIS-PC-SENDER-RATE-DEV-001` 已按冻结协议判为 `NEGATIVE`：模拟概率载荷虽然有 99.84% top-1 恢复率，但 40.50% 的接受/拒绝决策相对完美载荷发生翻转。原因是自然零阈值比较两个很小的 JS 差值，top-1 正确并不等于连续分数稳定。本轮不扫描模拟 repetitions 或阈值，而是测试一个预先固定的、结构上不同的数字编码候选。

本轮仍使用已暴露的 policy-dev seed `20260725`，因此只是方法开发，不是独立验证；官方 Imagenette validation 继续封存。

## 冻结编码与系统

- sender 的 `G_aux` 10 维概率逐维均匀量化为 4 bit，共 40 raw bits，量化后重新归一化；
- 40 bit 映射为 BPSK `{-1,+1}`，每 bit 重复 4 次，固定占用 160 个实符号；
- receiver 对每 bit 的 4 次接收值取均值并作零阈值硬判决，再恢复 10 个 UInt4 code 和概率单纯形；
- 160 个符号仍覆盖在同一个 `c=8` latent 内，与图像载荷共同经过一次 AWGN；总实符号 65536、总 CBR `1/6` 均不变；
- receiver 擦除保留位置后运行 B1、S14 六步 diffusion 和三步 masked posterior correction；
- controller 仍使用 `r_JS<=0` 接受 posterior，否则回退 in-budget anchor；
- 不扫描 bits/class、重复数、threshold 或 per-SNR 规则。

## 门槛

沿用模拟固定码率实验全部质量与语义门槛，并新增：每个 SNR 的 40-bit payload vector 完整无误率不低于 95%。只有所有门槛通过才允许把 `UInt4+BPSK×4+zero veto` 原样冻结到新 channel seed。失败则与模拟载荷一样记录为负结果，不在 seed `20260725` 上补救。
