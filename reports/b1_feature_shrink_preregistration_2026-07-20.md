# S23 B1 Diffusion 特征方向全局 Shrink 预注册（2026-07-20）

## 已知结果与事后性声明

注册本实验时，S22 selection 结果已经可见：10 个非零 epoch 均改善 LPIPS，但 PSNR 低于 B1；epoch1 为 `-0.01887 dB/-0.01096 LPIPS`，epoch6 的 PSNR 最接近 B1。S22 holdout 未访问。

因此 S23 是明确标注的 development follow-up，不把它伪装成事前提出的独立验证。它只回答一个窄问题：S22 的非零感知方向是否因为幅度过大而越过 PSNR 最优点。

## 冻结设计

1. 完全复用 S22 的 train/selection cache、B1、diffusion、loss、seed、batch、crop、LR；
2. 只训练固定 `1 epoch`，不扫描 epoch；选择 epoch1 是因为它是最早的非零方向，而不是因为它有最佳指标；
3. 训练完成后冻结 projection endpoint `W1`；候选为 `alpha*W1`；
4. 全局 alpha 网格在运行前固定为 `[0,.01,.025,.05,.075,.1,.15,.2,.35,.5,.75,1]`；同一 alpha 同时用于 1/4/7 dB，不搜索 per-SNR schedule；13/19 dB 仍由 envelope 严格为 B1；
5. alpha=0 作为 B1 fallback。

选择规则：候选必须同时满足 aggregate LPIPS 不高于 B1、1/4/7 dB 的 PSNR 全部不低于 B1；在可行候选中最大化 aggregate PSNR，平局选较小 alpha。只有选中非零 alpha 且 aggregate PSNR 严格高于 B1，才把缩放后的 projection 写成冻结 checkpoint、记录 SHA，并首次访问 S23 holdout。

## 停止规则

- 若只选 alpha=0：不访问 holdout，关闭 frozen-B1 单 projection 注入族；
- 若选中非零：不得再修改 alpha、endpoint 或 loss；一次性运行与 S22 相同的 256×5 holdout、三分类器 pseudo semantic 审计和 source-cluster bootstrap；
- 无论结果如何，不访问 official Imagenette validation，不下载新数据或权重。
