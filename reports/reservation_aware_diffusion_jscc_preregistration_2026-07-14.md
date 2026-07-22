# Reservation-aware diffusion-JSCC pilot 预注册

日期：2026-07-14。阶段：S15 exploratory pilot。

## 动机

UInt2×BPSK-r4 把语义 payload 从 160 降到 80 个实符号后，policy-dev 上五个 SNR 的 M3-minus-M2 PSNR 已全部为正，但仍存在 7 个 M2-correct→M3-final-wrong image clusters。逐事件诊断显示主要风险来自 post-hoc reservation 后的错误 anchor fallback，而非 diffusion posterior 被错误接受。因此下一步保留 S14 diffusion，先让 B1 anchor 在训练时看到与部署完全一致的 80-symbol reservation/erase 分布。

## 冻结 pilot

- 数据：COCO train2017 SHA-rank 前 2000 张训练、随后 200 张内部验证；不访问 Imagenette official val；
- 信道：AWGN，SNR `[1,4,7,13,19]`，DeepJSCC `c=8`、总 CBR `1/6`；
- payload：20-bit UInt2 BPSK×4，共 80 符号；位置按 evenly-spread-floor；receiver 解码前擦除；
- cache payload 使用固定 balanced BPSK。由于 payload 位置在 decoder 前全部擦除，且 BPSK 符号数和功率固定，payload 正负内容不改变 restoration 输入分布，只改变已被擦除的位置，因此不需要 COCO 标签或 Imagenette classifier；
- 先生成 2000/200×5-SNR reserved B0 cache，再从 S13 B1 checkpoint 做小学习率微调；
- 本阶段只判断 reservation-aware 训练是否改善 reserved COCO validation 的 anchor PSNR，并为后续保持 S14 diffusion 的 policy-dev 复核提供 checkpoint，不把 pilot 写成正式泛化证据。

## 决策边界

- “改善”先按完全相同 reserved validation 输入上的新旧 B1 逐样本配对比较判定；输出在计算指标前统一量化为 8-bit PNG。要求五个 SNR 的 mean `new-old PSNR` 均为正，且以图像为 cluster、跨 SNR 聚合的 20,000 次 paired bootstrap 95% CI 下界大于 0；LPIPS 同步报告但不作为本 pilot 的硬门槛；
- 若 reserved validation 的 B1 不满足上述 PSNR 判据，停止该微调分支；
- 若改善，再在已暴露 policy-dev 做一次 full exploratory replay，重点检查 M2-relative new/repair、逐 primary-SNR failure 与五 SNR PSNR；
- 只有冻结后通过新的 channel seed，才有资格重新讨论 official-val；official outcome 在此期间保持未消费。
