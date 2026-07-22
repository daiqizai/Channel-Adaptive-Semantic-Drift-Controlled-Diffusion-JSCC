# S32 强 JSCC 外部同总体比较预注册（2026-07-21）

## 问题

S31 只用 COCO train2017 训练，并按固定 COCO val2017 512 图的五档平均 PSNR 选择 checkpoint。训练期曲线只能回答模型是否学起来，不能回答它是否已经追上 S30 的 author-JSCC。S32 在 S31 停止规则触发、best checkpoint 与 SHA 冻结后，才首次运行强 JSCC 在既有 S20/S30 Imagenette policy-dev 总体上的输出。

S20/S28/S30 的旧方法结果在注册本协议时已经已知；S32 因而是同总体外部定位，不是独立盲测或最终测试。strong-JSCC 在该总体上的结果尚未知。official Imagenette validation 继续封存。

## 冻结比较合同

- 总体：S20 原 64 张 T_cls clean-correct Imagenette policy-dev 图像；3 个 channel seed；`[1,4,7,13,19] dB`，共 960 个键。
- 源图处理：`Resize(256) -> CenterCrop(256) -> ToTensor()`，与 S30 指标目标一致。
- 噪声：逐 `(base_seed, sample_id, SNR)` 复用 `external-common-v1` 的 19,712 维 CPU float32 canonical standard-normal；strong-JSCC 使用全向量，author-JSCC 使用 S30 已冻结的前 16,384 维结果。
- 码率：strong-JSCC 原生发送 `19,712 real = 9,856 complex uses`，不裁剪、不补零、无 side information；author-JSCC 为 `16,384 real`，差异必须保留在表中，不能写成 exact-rate matched。
- checkpoint：只能使用 S31 按 COCO 五档平均 PSNR、再以 MS-SSIM 破平局选出的 best；S32 配置必须写入 checkpoint SHA 和 epoch，任一不符即失败。
- 主指标量化：为贴近 S30 author arm，strong 输出先做 `floor(255*x)/255` 再计算 PSNR、MS-SSIM、LPIPS 和 T_cls；同时报告未量化 float 输出相对主口径的敏感性。
- 不允许按 S32 结果选择 epoch、SNR schedule、量化方法、分类阈值或模型超参数。

## 指标和统计

每个样本保存 strong 的 PSNR、MS-SSIM、LPIPS、T_cls prediction/failure、归一化功率和前向时间，并并入 S30 已冻结的 author-JSCC、完整 DiffJSCC、current、B1 指标。差值以源图为 cluster，跨 3 seed×5 SNR 聚合后做 10,000 次 bootstrap 95% CI。

主要判断预先固定为：

1. 技术通过：960 个键齐全、canonical noise SHA 全部匹配、所有指标有限、功率误差不超过 `1e-5`、checkpoint/总体/评估器 SHA 全部匹配。
2. 相对旧 current 的保真基座升级：`strong-current PSNR` 的 source-cluster CI 下界大于 0。LPIPS、MS-SSIM 和 failure 同步报告，但不因结果不理想而隐藏。
3. 相对 author-JSCC：若 `strong-author PSNR` CI 下界大于 0 且 LPIPS CI 上界小于 0，记为 strong 在两项质量轴上占优；若两个方向相反则记为 author 占优；其他情况记为 Pareto 或不确定，不强行排单一名次。
4. S32 不含 diffusion，因此无论强基座胜负，都不能据此宣布完整项目方法超过 DiffJSCC。下一阶段必须在强基座冻结后重新训练 matched diffusion 与 semantic-risk controller。

若 S31 内部末段仍明显上升，应在打开 S32 前注册并完成延长训练；一旦 S32 运行，后续因 S32 排名而产生的改模只能使用新的开发总体，不能把同一 S20 总体继续包装成未见验证。
