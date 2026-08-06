# S35R-P0：SGD-JSCC 自适应步数与延迟-SNR 预注册

日期：2026-07-23
分析 ID：`ANALYSIS-S35R-P0-SGD-ADAPTIVE-COST-001`

## 目的与边界

本分析只补齐 SGD-JSCC paper-protocol upper 的系统代价，不做质量排名、不训练、不下载、不访问 official Imagenette validation。

复用 S34D 已经在同一张 RTX 4090D、PyTorch `2.1.0+cu121`、batch size=1 下得到的 80 条逐图计时：五档 SNR 各16张相同源图。入口为主存中的256×256 RGB，出口为回到主存的256×256重建；模型加载、磁盘 I/O 和指标不计，BLIP2、MuGE、edge/text conditioning、所有 VAE 和所有 denoiser evaluation 均计入。

S34D 总体延迟和源码均已知，因此本轮属于 outcome-aware 的派生测量/源码审计，不伪装为盲检验；但每档聚合、`alpha_bar_channel` 映射和固定地板判据在生成结果表前冻结。

## 步数核算

按项目 canonical paired-real AWGN：

```text
gamma = 10^(SNR_dB/10)
alpha_bar_channel = 2*gamma/(2*gamma+1)
```

必须同时报告：

1. 五档 `gamma`、`alpha_bar_channel` 和 `1-alpha_bar_channel`；
2. official continuous sampler 对应的 schedule endpoint；
3. sampler 构造的点数；
4. 实际 denoiser evaluation 数量。

关键区别预先写死：step matching 的“扩散时刻”不自动等于“数值求解器调用次数”。当前 released working point 的 `step_style=continuous`、`diffusion_step=50`；作者 sampler 构造50个连续点，循环执行49次 `pred_image`，循环后再执行1次最终 `pred_image`。若源码 hash 与预注册不符则 fail closed。

## 延迟-SNR 与固定地板

每档报告端到端、BLIP2、MuGE、diffusion solver 和其余组件的 mean/median/p05/p95/std。因为五档使用同一16张图，可以直接比较每档均值。

BLIP2 和 MuGE 只有在以下两层同时成立时才称为 released pipeline 的固定开销地板：

- 结构上：每张源图在 step matching 前各执行一次，不接收 diffusion step count，五档都不跳过；
- 测量上：报告五档均值、范围及二者之和，不把 runtime jitter 误写成精确常数。

“固定”只针对当前 released SGD 管线，不代表所有生成式 JSCC 都必须使用 BLIP2 或 MuGE。

## 输出与声明

- 正式输出：`outputs/analysis/ANALYSIS-S35R-P0-SGD-ADAPTIVE-COST-001/`
- 中文结果：`reports/s35r_p0_sgd_adaptive_cost_result_2026-07-23.md`
- SGD 仍是至少 `21,856 real`、完美 caption 的 non-ranking paper upper；本轮只允许做系统代价说明。
