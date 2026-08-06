# S34D 生成式 JSCC 推理代价：纯测量实验预注册

日期：2026-07-23
分析 ID：`ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001`

## 研究问题

本轮不使用历史 100-step wall time 直接声称 diffusion 的固有代价，而是回答：

1. 在同一 RTX 4090D、batch size=1、同一计时入口下，S33、DiffJSCC 与 SGD-JSCC 的 receiver inference 到底各需多少时间？
2. DiffJSCC 从 100 步降到 50/25/10/4 步后，延迟与 LPIPS 如何变化？
3. 在冻结候选中，保持相对 S33 显著 LPIPS 优势的最少步数和实测延迟是多少？
4. 哪些成本是当前生成链实际必须执行的，哪些只是当前 checkpoint/实现/采样器造成的可优化成本？

只使用现成 checkpoint，不训练、不下载、不调 official validation。

## 公平计时入口

所有方法的主延迟均从“已经从磁盘读入主存的 256×256 RGB 图”开始，到“256×256 RGB 重建已经物化回主存”为止。统一排除：

- 模型构造和 checkpoint 加载；
- 磁盘图像读取与结果写盘；
- LPIPS、分类器等指标计算。

统一包含：

- 方法内部的 resize、patch split/merge 与 host/device transfer；
- JSCC 编码、信道和解码；
- BLIP2 caption（如果方法使用）；
- text conditioning、edge processing、所有真实执行的 diffusion denoiser evaluations；
- 所有真实执行的 VAE encode/decode；
- color correction 与输出回到 256×256。

模型加载另作一次性系统成本，不混入每图 steady-state latency。除了主 wall time，还会用 CUDA synchronize 分解组件时间。输入 batch size 均为 1 张源图；SGD 内部四个 patch 同批处理属于其原生算法，不改成“四张源图 batch”。

## 固定总体

- latency：冻结 population 顺序前 16 张、五档 SNR，共 80 行/方法或步数；前三个 key 仅 warmup，不计统计；
- DiffJSCC 质量曲线：冻结 64 张图、seed=`20260748`、五档 SNR，共 320 行/步数；
- steps：`100/50/25/10/4`；
- 相同 key 的不同步数使用相同 sampler seed derivation；
- 指标：PSNR、MS-SSIM、LPIPS、冻结 `T_cls` failure；
- CI：按 source image 聚类、10,000 次 bootstrap。

“保持感知优势”的预注册定义是：`DiffJSCC(step) − S33` 的 LPIPS 均值 `<0`，且双侧 source-cluster 95% CI 上界 `<0`。最少步数只在预注册五个候选中选择；语义 failure 同时报告，但不事后改写 LPIPS gate。

## 分辨率与码率边界

三种方法都从同一 256×256 源图入口开始，但内部保持各自官方/冻结合同：

- S33：原生 256×256；
- DiffJSCC：官方内部将短边扩到 512，本总体为 512×512，再回缩到 256；
- SGD：一张 256×256 图拆为四个 128×128 patch。

内部工作分辨率不同是系统代价的一部分，不能为追求相同 ms 而擅自改变 checkpoint 的原生输入。码率也不因本轮计时而“洗平”：S33/DiffJSCC 为 `16,384 real`，SGD paper upper 最低物理账本仍为 `21,856 real`。

## 参数量与 FLOPs

参数量按运行时实际驻留的、去重后的 parameter object 精确统计，并分组件报告；大型预训练模块不能从总量中删除。

FLOPs 使用现有 PyTorch `torch.profiler(with_flops=True)`，报告其支持的 conv/linear/mm/bmm executed FLOPs。该值对同一实现、不同步数可审计且比单卡 ms 更可迁移，但仍是**下界**：norm、activation、resize、随机采样、部分自定义 attention 和 wavelet 等未必计入。报告中不得把这个下界写成理论完整 FLOPs。

DiffJSCC 另拆为固定前端、单次 denoiser evaluation、VAE decode/color correction，从而给出随步数变化的 executed-FLOPs 公式。100 步只是曲线一个点，不定义为 diffusion 固有代价。

## 输出与停止规则

输出目录已预注册为：

`outputs/analysis/ANALYSIS-S34D-GENERATIVE-INFERENCE-COST-001/`

若资产 hash、GPU UUID、batch、输入键、100-step 历史质量回放或组件加和审计失败，必须 fail closed 并保留失败记录。不得下载替代权重，不得访问 official Imagenette validation。
