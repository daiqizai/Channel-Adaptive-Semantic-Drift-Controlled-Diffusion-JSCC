# S19 阶段结果：Diffusion 含有 B0/B1 路径之外的互补恢复信息

日期：2026-07-16

## 1. 结论先行

本轮得到一个可以改变项目叙事的正向因果结果：在参数量、初始化、训练图像、信道样本、minibatch、crop/flip 和训练预算均相同的条件下，使用 `B0 + identity-controlled diffusion` 的 fusion 网络在全新一次性 holdout 上显著超过只使用 `B0 + B0` 的 control。

- fusion−control PSNR：`+0.05846 dB`，256-source cluster-bootstrap 95% CI `[+0.05198,+0.06423]`；
- fusion−control LPIPS：`-0.001493`，95% CI `[-0.002162,-0.000824]`；
- fusion−原始 B1 PSNR：`+0.10168 dB`，95% CI `[+0.09431,+0.10915]`；
- fusion−原始 B1 LPIPS：`-0.006394`，95% CI `[-0.007158,-0.005636]`。

因此主判据通过：**当前冻结的 channel-matched diffusion 观测包含同容量 B0-only 恢复网络无法完全替代的信息。项目不应放弃 diffusion，而应把它从独立终点改成受可靠性控制的互补观测。**

## 2. 公平因果设计

两个模型均为 450,115 参数、9 通道输入、64 基宽、6 个残差块，并由同一个 S16-B1 checkpoint 展开。B1 原 6 通道 head 被映射到新 head 的 B0/SNR/Sobel/Laplacian 位置，新增 auxiliary RGB 权重严格置零。

- control：`[B0, B0, SNR, Sobel(B0), Laplacian(B0)]`；
- fusion：`[B0, D_identity, SNR, Sobel(B0), Laplacian(B0)]`。

训练开始前，control、fusion、原 B1 在实际 selection batch 上的最大输出差均为 `0`；两个新模型的初始 state 逐项相同。训练时两分支消费同一个 batch、crop 和 flip，分别用同超参数 Adam 优化。故可识别的主要干预只有 auxiliary 是否含有 diffusion 信息。

`D_identity` 使用 S18 已冻结的 `hard_identity_7db`：1/4/7 dB 为 decoder-aware channel-matched diffusion，13/19 dB 严格等于 B0。没有在 S19 上重选包络。

## 3. 新 population 与 cache

从本地 COCO train2017 冻结 5,512 个全新源：5,000 train、256 selection、256 holdout。抽样排除 S16 旧 11,000、S18 512 和 val2017 同名文件，并执行路径与源 SHA-256 双重去重。

- source manifest SHA-256：`b73c05656865eb6023c40dd57dfde176d05141eaf9b996feff92a41894522fe9`；
- 与旧 population 的 path/SHA overlap：`0/0`；
- 排序中发现并跳过 1 个“异名同内容”的旧源；
- cache：27,560 个 sample-SNR 行、27,560 个 B0、16,536 个低 SNR diffusion PNG；
- 修复后 cache manifest SHA-256：`8d88daf70a5ad07c213674f883c02d0a5f9b84ca082ff26727a5baead60775e3`。

信道保持精确 19,712 real symbols，其中 19,632 image-active、80 payload-reserved；AWGN SNR 为 `{1,4,7,13,19}`，三个角色使用独立 canonical noise seed。

### 中断与失败记录

第一次 cache 前台进程被对话中断时，19 dB 的 `sample_001161.png` 正在写入，留下 1 个不可识别 PNG。第一次训练在读取该图时失败，没有产生 epoch 结果。损坏文件、旧 cache manifest、失败训练与 selection 目录均保留，未覆盖。

随后 cache 写入改为临时 PNG 完成后原子 rename；全量检查 49,608 张 PNG，确认坏文件为 `0`，只重生成损坏项并获得新的 cache SHA。该事件是基础设施失败，不是方法负结果。

## 4. Selection 与冻结 checkpoint

两个模型均训练 10 epoch，epoch 0 也按预注册纳入候选；各自独立最大化 selection 平均 PSNR。

| 分支 | 最佳 epoch | selection PSNR | checkpoint SHA-256 |
|---|---:|---:|---|
| control | 7 | 27.44533 | `c9eab7648bd0120e5db2820b7a94edd13251ed8e1050cec6edb6f6187fbdbcf6` |
| fusion | 9 | 27.50549 | `e7577d59e9c2362e40feb72bb030c1c7b9302b707115d3e000c9c55bbb8942c5` |

selection 上 fusion−control 为 `+0.06016 dB`。两个 checkpoint 的 SHA 在读取 holdout 前写入最终配置。

## 5. 一次性 holdout 质量结果

256 个源、5 个 SNR，共 1,280 行。数值如下：

| SNR | B0 PSNR | identity diffusion | 原 B1 | control | fusion | fusion−control | fusion−B1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.20240 | 24.91173 | 25.61938 | 25.66484 | **25.75670** | +0.09186 | +0.13732 |
| 4 | 25.60449 | 25.86957 | 26.73339 | 26.78019 | **26.89050** | +0.11031 | +0.15711 |
| 7 | 26.50408 | 26.54861 | 27.51393 | 27.55825 | **27.69232** | +0.13407 | +0.17838 |
| 13 | 27.33045 | 27.33045 | 28.23963 | **28.27891** | 28.25399 | -0.02492 | +0.01436 |
| 19 | 27.56231 | 27.56231 | 28.41774 | **28.45798** | 28.43895 | -0.01903 | +0.02121 |
| 平均 | 26.24075 | 26.44454 | 27.30482 | 27.34803 | **27.40649** | **+0.05846** | **+0.10168** |

fusion 相比 B0 的平均 PSNR/LPIPS 改善为约 `+1.1661 dB/-0.12167`；相比独立 identity diffusion 为 `+0.9620 dB/-0.08224`。说明最佳结果不是“用 diffusion 替代 B1”，而是让结构恢复网络从 B0 与 diffusion 两个不同观测中联合取证。

## 6. Semantic-drift 辅助审计

COCO 没有本任务的直接分类真值，以下仍是冻结 ImageNet 分类器相对原图预测的 pseudo 审计，不可包装成真实语义正确率。

| 方法 | majority failure | majority new | majority repair |
|---|---:|---:|---:|
| B0 | 953 | 0 | 0 |
| identity diffusion | 895 | 19 | 77 |
| 原 B1 | 752 | 69 | 270 |
| control | 737 | 60 | 276 |
| fusion | **727** | **54** | **280** |

fusion 的 majority new 不高于 repair，且少于 control；failure 也最低。AlexNet clean-confidence eligible 子集上，fusion 为 62 new / 230 repair，满足预注册的 new≤repair，但 new 比 control 的 55 多 7，需保留为分类器口径差异，不能声称绝对 semantic safe。

## 7. 预注册判据

7 项检查通过 6 项：

- PASS：fusion−control PSNR CI 下界大于 0；
- PASS：fusion−control LPIPS 不劣；
- **FAIL：fusion−control 非负 SNR 数仅 3/5，未达到 4/5**；
- PASS：fusion 平均 PSNR 高于原 B1；
- PASS：AlexNet fusion new≤repair；
- PASS：majority fusion new≤repair；
- PASS：majority fusion new≤control new。

唯一失败来自 13/19 dB：虽然 diffusion auxiliary 在这两点严格等于 B0，但 fusion 共享权重受到低 SNR 学习影响，导致相对独立 control 轻微退化。这是清晰的下一问题，而不是否定主因果结论。

## 8. 对项目叙事的更新

现在可以用下述表述概括已验证主线：

> 经过总功率归一化的 DeepJSCC 接收 latent，在 AWGN 下可严格对应扩散前向轨迹中的连续噪声状态；SNR 通过解析的 `alpha(SNR)` 完成 channel-state matching。decoder-aware 反向过程生成一个与 B0 互补的恢复观测，SNR identity envelope 在高 SNR 把它严格退化为 B0；最后由 receiver-visible 结构恢复网络融合 B0、diffusion、SNR 与局部结构证据，在控制 semantic drift 的同时改善重建。

这比“把当前 SNR 匹配到最近离散 timestep”更精确，因为当前实现使用连续 `alpha`。但现阶段不能写“文本条件限制反向去噪解空间”：文本没有进入 S17/S19 diffusion 主链；80-symbol payload 也未在本实验中作为 diffusion 条件使用。

## 9. 下一步

1. 研究只作用于 auxiliary 路径的 SNR-gated adapter 或低/高 SNR 解耦参数，消除共享权重在 13/19 dB 的轻微负迁移；必须重新用新 selection/holdout，不得修改本轮结果。
2. 保留 `hard_identity_7db`，不回到全 SNR 强制 diffusion。
3. 把 S19 fusion 纳入外部方法共同协议；在精确码率/信道/数据对齐前，不宣称超过 SGD-JSCC、DiffJSCC 或 DiT-JSCC 论文结果。
4. 若后续引入可靠传输的 semantic payload，再做“是否真正约束 diffusion 反演”的 zero/shuffled/received 因果消融；在此之前不写文本/语义条件贡献。

阶段判定：**PRIMARY PASS / 6-of-7 checks；首次严格证明 diffusion 对强 B1 有互补价值。**
