# B1 特征注入逐图幅度上限诊断结果（S25）

日期：2026-07-20。分析 ID：`ANALYSIS-S25-B1FA-HEADROOM-001`。

## 结论

**明确负结果：关闭 S23 one-epoch feature direction 上继续训练逐图 amplitude controller 的路线。**

即使 oracle 可以偷看原图 PSNR，并用三分类器多数票禁止相对 B1 的新增失败，逐 sample/SNR 选择 12 个 alpha 中的最优值，相对当前固定 `alpha=0.15` 也只增加：

- PSNR `+0.001365 dB`，source-image cluster 95% CI `[+0.001186,+0.001562]`；
- MS-SSIM `+0.0000629`；
- LPIPS `-0.001817`；
- majority `0 new / 10 repair`。

PSNR headroom 远低于预注册的 `+0.02 dB` 继续门槛，因此 4 项 gate 仅通过 3 项，正式判定 `continue_to_receiver_visible_controller=false`。这不是 controller 不够聪明，而是当前 feature direction 的逐图质量上限本身太低。

## 结果解释

固定 `0.15` 在 selection 上的 PSNR/LPIPS 为 `27.099880/0.185948`，majority `4 new / 5 repair`。semantic-safe PSNR oracle 为 `27.101245/0.184132`。虽然低 SNR 有 `92.19%` 行选择了不同于 `0.15` 的 alpha，变化幅度仍没有转化成有意义的 PSNR headroom。

纯 LPIPS oracle 能额外改善 LPIPS `-0.01030`，但会损失 PSNR `-0.01329 dB` 并产生 `18` 个 majority new error，说明沿这条方向继续增大幅度主要是在换取感知纹理，而不是恢复更多可靠像素信息。

## 研究决策

1. S23 保留为最小 exact-fallback 机制基线，不再细扫 alpha、threshold 或训练 receiver-only amplitude head。
2. 下一轮返回 S19 的更强 joint-fusion representation；冻结“1/4/7 dB 使用 S19 fusion，13/19 dB 强制 B1”的结构性策略，在另一 population 上直接做 frozen cross-population replication。
3. 下一轮必须同时比较 frozen S19 control，才能继续证明收益来自 diffusion 信息，而不只是更大的恢复网络。

## 边界与产物

本轮只访问已暴露的 S23 selection 256 图×5 SNR，所有 oracle 都使用原图/评估器，只是不可部署的上限诊断；未访问 holdout、未训练、未访问 official Imagenette validation、未联网或下载。

- summary SHA：`8b8bcf65e46c2d1965696e46eecdcd1ab75864c916f6a3ab9f6a4757094d8add`
- long CSV SHA：`4b88d25a3b5bd1e44f989ab53991ae7845a3adfe37dd2c037bbe81772a4dc891`
- choices CSV SHA：`f5f690461dab903e8f98bf2068e7061a0d7e296819c0cafd7aa7578f7417828b`
