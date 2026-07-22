# S22 冻结 B1 的 Diffusion 特征注入预注册（2026-07-20）

## 1. 动机

S21 在同一未访问 holdout 前完成了四个 development 结论：learned spatial gate 两次塌缩、fixed-gate output residual 饱和、B1/D 单调凸融合只选出全零 B1。它们共同否定的是输出层合并参数化，不否定 S19 已严格证明的特征级互补信息。

S22 因此不再修改输出，而是在 frozen B1 的第一层 feature 中注入 matched-diffusion 与 B0 的差值，检验最小 feature-level merge。

## 2. 冻结结构

B1 的 head、6 个 residual blocks、tail、SNR residual gate 全部冻结。新增唯一可训练模块：

`P = Conv3x3(3→64, bias=False)`，共 `1,728` 参数，权重全零初始化。

前向为：

1. `h_B1 = frozen_B1_head([B0,SNR,Sobel(B0),Laplacian(B0)])`；
2. `delta_D = D-B0`；
3. `h = h_B1 + e(SNR)×P(delta_D)`；
4. `x_final = frozen_B1_body_and_tail(h)`。

`e(1/4/7)=1`，`e(13/19)=0`。因为 `P=0`，epoch0 对任意 auxiliary 都精确等于 B1；训练后 13/19 dB 仍由 envelope 精确等于 B1。control 使用 `B0-B0=0`，对任意 P 都精确 B1。因此比较对象不再含通用额外 CNN 的混杂，唯一有效自由度是 diffusion difference feature。

## 3. 数据与训练

- 复用 S21 已冻结的 5,000/256/256 fresh population 与 `dd79fe2f...84b87` cache；此前 S21 只访问 train/selection，holdout 未访问。
- B1、DeepJSCC、matched diffusion、channel realizations 与严格 19,712-real 合同不变。
- 10 epochs、batch16、paired 128 crop/flip、Adam LR `1e-4`。
- loss：`MSE+0.10 L1+0.01 LPIPS`。
- epoch0 纳入候选；选 mean PSNR 最大且 LPIPS 不劣于 epoch0 B1 的最早 epoch，否则保留 epoch0并不访问 holdout。
- 不扫描插入层、投影宽度、LR 或 loss。

## 4. Holdout 判据

只有 selection 选中非零训练 epoch，才冻结 checkpoint SHA 并首次访问 256×5 holdout。

1. fusion−B1 PSNR source-cluster 95% CI 下界 `>0`；
2. fusion−B1 LPIPS 95% CI 上界 `<0`；
3. 1/4/7 dB PSNR `3/3` 非负；
4. 13/19 dB逐像素最大差 `≤1e-7`；
5. majority pseudo new（相对 B1）不大于 repair。

通过只证明最小 feature injection 可用；COCO pseudo semantic 不替代后续独立监督审计。
