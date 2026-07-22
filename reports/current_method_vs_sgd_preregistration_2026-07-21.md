# 当前方法与 SGD-JSCC 外部定位预注册（2026-07-21）

## 目的

在已经冻结的 S20 Imagenette policy-dev 总体上，一次性运行当前主方法（低信噪比使用冻结的 S19 diffusion fusion，13/19 dB 精确回退 B1），并与 S20 已冻结的 B1 和 SGD-JSCC 论文协议输出做逐样本配对比较。

本轮不训练、不调参、不选择阈值，也不访问 Imagenette 官方验证集。S20 的 B1/SGD 结果已经可见，因此本轮只能承担外部定位和跨数据集复现，不能包装成全新盲测。

## 冻结合同

- 图像：S20 已冻结的 64 张 Imagenette policy-dev 图像。
- 信道：AWGN，SNR 为 1/4/7/13/19 dB；三组冻结种子 20260748/20260749/20260750。
- 当前方法与 B1：总计 19,712 个实信道符号，其中 19,632 个图像坐标、80 个语义载荷坐标；扩散与融合不引入额外信道符号。
- 路由：1/4/7 dB 固定使用 S19 fusion；13/19 dB 固定逐像素回退冻结 B1。
- 模型：DeepJSCC、latent diffusion、S19 control/fusion、B1、T_cls 与 G_aux 全部使用冻结 checkpoint。
- 统计单位：以源图像为 cluster，同时聚合三组信道种子和五个 SNR；10,000 次 bootstrap。

## 对 SGD-JSCC 的解释边界

S20 的 SGD-JSCC 输出使用论文协议中的完美、免费文本。其主图像与 active-edge 已占满 19,712 个实符号，四个 caption packet 至少还需 2,144 个未保护实符号。因此：

1. 可以在同图像、同 AWGN 噪声、同图像分支预算下，把它当作对 SGD 有利的“论文协议上界”做外部定位；
2. 不可以把它声称为严格同总物理码率的公平排名；
3. 若当前方法没有同时胜过 SGD 的 PSNR、MS-SSIM、LPIPS 和语义失败率，应报告 Pareto 关系，而不是挑单一指标宣布胜负。

## 预注册检查

- 必须用冻结 B1 逐样本记录验证本轮重新生成的噪声、B1 指标和预测；合同复现失败则整轮无效。
- 13/19 dB 的当前输出必须与重新生成的 B1 逐像素一致。
- 当前方法相对 B1 的 PSNR cluster CI 下界需不小于 0，LPIPS CI 上界需不大于 0，且 T_cls failure 数不增加，才算跨数据集增益复现。
- S19 fusion 相对同容量 control 的 PSNR/LPIPS CI 也必须分别通过 0，才能继续声称增益来自 diffusion observation，而不仅是接收端容量。
- 当前方法对 SGD 只做冻结外部定位；无论结果正负均完整记录。
