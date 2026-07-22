# S31 强 JSCC 基座预注册（2026-07-21）

## 目的

S30 已确认本项目当前方法的主要短板是保真基座，而不是“有没有再加一个融合模块”。冻结外部对比中，author-JSCC 只用 `16,384 real`，仍比 current 高 `1.762457 dB` PSNR；优势还随 SNR 从 `0.97 dB` 增加到 `2.82 dB`。因此本阶段先独立建设强 JSCC，不训练 diffusion、不使用语义标签挑 checkpoint。

## 冻结设计

- 代码为本项目 clean-room 实现，不复制未声明 license 的 DiffJSCC 源码。
- 输入固定 COCO `256x256`，四级下采样后 latent 为 `77x16x16`，原生得到严格 `19,712` 个实坐标；不使用固定 mask、不补零、不传 side information。
- 连续两个实坐标解释为一次复信道使用，共 `9,856` 次；实噪声方差保持项目合同 `P/(2*10^(SNR/10))`。
- 编码器与解码器均为宽残差网络，并在每个残差块用 SNR 条件做仿射调制；计划参数量冻结在 `25M--45M`。
- 训练 SNR 从 `[1,4,7,13,19] dB` 按图像均匀采样，发送端和接收端都假设拥有完美 SNR。该假设必须在论文系统模型中显式写出。
- 第一版只优化 MSE，checkpoint 仅按五档验证 PSNR 聚合选择；不使用 LPIPS、分类器或 diffusion 损失，以免把强保真端点和生成端点混在一起。

## 数据与隔离

- 训练：完整 COCO train2017，随机裁剪和水平翻转。
- 训练期验证：固定 seed 选取的 512 张 COCO val2017，五档 SNR 使用冻结噪声种子。
- S20 Imagenette policy-dev 只在 checkpoint 冻结后用于与 B1/current/author-JSCC 的共同噪声比较，不参与模型或超参数选择。
- official Imagenette validation 继续封存。

## 分阶段门槛

1. smoke：严格 `19,712 real`、归一化功率误差不超过 `1e-5`、GPU 前向/反向有限且无 OOM。
2. first curve：训练曲线全程有限，best aggregate PSNR 必须高于随机初始化端点。
3. external comparison：冻结 checkpoint 后，复用 S20 的图像、SNR、seed 和 canonical noise，输出 PSNR/MS-SSIM/LPIPS/语义 failure；在结果出来前不预设“必然追平 author-JSCC”。
4. 只有强 JSCC 端点通过后，才重新训练与其输入合同匹配的 channel-matched diffusion 和 risk controller；旧 B1/S19 checkpoint 不直接移植。

完整配置：`configs/s31_strong_jscc_coco256_awgn.yaml`。

## 正式输出前的系统 smoke 修订

首个全尺寸 GPU smoke 在任何正式 COCO 训练输出产生前完成：默认 31.12M 模型的前向、反向、五档 SNR 和功率检查通过，归一化最大误差为 `1.19e-7`。纯系统吞吐测试表明 RTX 4090 D 上 batch `32` 的峰值 allocated VRAM 约 `10.4 GiB`、吞吐约 `176 image/s`，因此正式配置将 batch 从 `4`、累积 `4` 修订为 batch `32`、累积 `1`。该修订不读取图像质量结果，不改变模型、总码率、损失、SNR、数据或 checkpoint 选择规则；有效 batch 从原计划 16 调整为 32，以接近外部作者的训练批量并减少训练墙钟时间。
