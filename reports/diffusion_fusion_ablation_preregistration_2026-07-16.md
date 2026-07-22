# S19：Diffusion 是否提供 B1 之外的信息——等容量融合消融预注册

日期：2026-07-16

## 1. 要回答的问题

S18 已证明冻结的 channel-matched latent diffusion 在 1/4/7 dB 有稳定收益，并通过 13/19 dB 恒等包络消除了高信噪比负尾；但它的平均 PSNR 仍低于 B1。这里不能据此得出“diffusion 没用”，因为 B1 与 diffusion 是两个独立终点，尚未测试两者的信息是否互补。

S19 只回答一个因果问题：**在恢复网络容量、初始化、训练数据、batch 和增强都相同的条件下，把 identity-controlled diffusion 作为第二观测输入，是否比只重复输入 B0 更好？**

## 2. 冻结比较

两个模型均为 9 通道输入、64 基宽、6 个残差块，参数量必须逐项相同，且都由同一个 S16-B1 checkpoint 展开：

- `control`：输入 `[B0, B0, SNR, Sobel(B0), Laplacian(B0)]`；
- `fusion`：输入 `[B0, D_identity, SNR, Sobel(B0), Laplacian(B0)]`。

展开时复制 B1 的全部已有权重；新增的第二 RGB 输入对应卷积权重置零。因此训练开始前两个模型输出必须逐像素等于 B1，且二者参数必须完全一致。两个模型在同一训练循环中消费同一个 crop、flip 和 minibatch。

`D_identity` 使用 S18 已在独立 selection 上冻结的 `hard_identity_7db`：1/4/7 dB 为完整 diffusion，13/19 dB 严格等于 B0。禁止在 S19 selection 上重选 diffusion 包络。

## 3. 新 population 与信息隔离

从本地 COCO train2017 按固定 SHA-256 排序抽取 5,512 个新源：

- train：5,000；
- selection：256；
- 一次性 holdout：256。

排除 S16 的旧 11,000 个源、S18 的 512 个源、COCO val2017 同名文件，并同时检查源路径与源文件 SHA-256。三个角色不重叠。官方 Imagenette validation 保持封存。

信道为精确总预算 19,712 real symbols、其中图像 19,632、语义 payload 80、AWGN、SNR `{1,4,7,13,19}`；每个角色使用独立的冻结 canonical noise seed。

## 4. 训练、选择与 holdout

两模型均训练 10 epoch，损失为 `MSE + 0.1 L1`，优化器和学习率相同。epoch 0 的 B1 等价初始化也纳入候选，避免把微调退化误报为方法收益；每个模型独立按 selection 平均 PSNR 选最优 epoch，同分取更早 epoch。模型 checkpoint 及 SHA-256 冻结后才能访问 holdout。

holdout 同时报：B0、identity diffusion、原始 B1、control、fusion 的 PSNR、MS-SSIM、LPIPS；AlexNet 和三分类器多数票用于伪 semantic-drift 审计。bootstrap 以源图为 cluster，对一个源跨五个 SNR 整体重采样，避免把 1,280 行当成独立图像。

## 5. 成功与解释边界

主成功条件是 `fusion-control` 平均 PSNR 的 10,000 次配对 cluster-bootstrap 95% CI 下界大于 0。附加条件包括 LPIPS 不劣、至少四个 SNR 非负，以及伪语义新增错误受控。

- 若 fusion 显著超过 control：证明 diffusion 含有 B0/B1 路径没有的互补恢复信息；后续继续做融合而不是放弃 diffusion。
- 若 fusion 不超过 control：说明当前冻结 diffusion 的收益能被同容量 CNN 从 B0 中替代，不能把 diffusion 作为主贡献。
- `fusion > B1` 是更强的工程目标，但不是证明互补信息的必要条件。

任何视觉上更好但 semantic drift 更差的结果，不计作主成功。
