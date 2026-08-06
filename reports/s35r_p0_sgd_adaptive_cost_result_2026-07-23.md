# S35R-P0：SGD-JSCC 自适应步数与延迟-SNR 结果

日期：2026-07-23
分析 ID：`ANALYSIS-S35R-P0-SGD-ADAPTIVE-COST-001`

## 一、结论

**SGD-JSCC 当前 released working point 并不会随 SNR 提高而减少 denoiser 调用次数。**

它的 step matching 确实根据信道状态改变反向扩散轨迹的起点/噪声区间，但当前配置是：

```text
step_style = continuous
diffusion_step = 50
```

作者 sampler 对每张图都构造50个连续 schedule points，循环执行49次 `pred_image`，随后再执行1次最终 `pred_image`。所以 `[1,4,7,13,19] dB` 五档**实际都是50次 denoiser evaluation**。

对应端到端延迟也基本不随 SNR 变化：五档均值只有 `2043.78–2045.41 ms/图`，最大差 `1.63 ms`，约为总时间的 `0.08%`。因此 SGD 的“channel-adaptive step matching”是**轨迹区间自适应**，不是**计算量/步数自适应**。

## 二、公平测量口径

本轮不重新训练、不下载、不访问 official Imagenette validation，也没有重新加载模型跑一套可能带来环境漂移的新计时。它复用 S34D 已冻结的80条逐图原始计时：

- 同一张 RTX 4090D；
- PyTorch `2.1.0+cu121`；
- batch size=1；
- 五档 SNR 各16张相同源图；
- 从主存中的256×256 RGB开始，到256×256重建回到主存结束；
- 包含 patch split/merge、H2D/D2H、BLIP2、MuGE、main/edge channel、CLIP、所有 VAE 和所有 denoiser evaluation；
- 排除模型/checkpoint加载、磁盘 I/O 和指标计算。

SGD 仍为 `≥21,856 real`、完美 caption 的 non-ranking paper upper；此处只解释系统代价。

## 三、`alpha_bar_channel` 与实际步数

canonical paired-real half-variance AWGN 下：

```text
gamma = 10^(SNR_dB/10)
alpha_bar_channel = 2*gamma/(2*gamma+1)
```

| SNR | gamma | alpha_bar_channel | 1-alpha_bar | continuous endpoint | schedule points | 实际 denoiser evals |
|---:|---:|---:|---:|---:|---:|---:|
| 1 dB | 1.2589 | 0.715736 | 0.284264 | 0.132504 | 50 | **50** |
| 4 dB | 2.5119 | 0.833991 | 0.166009 | 0.076033 | 50 | **50** |
| 7 dB | 5.0119 | 0.909287 | 0.090713 | 0.041291 | 50 | **50** |
| 13 dB | 19.9526 | 0.975553 | 0.024447 | 0.011101 | 50 | **50** |
| 19 dB | 79.4328 | 0.993745 | 0.006255 | 0.002840 | 50 | **50** |

这里的 `continuous endpoint` 是把理想 `alpha_bar_channel` 代入作者 sigmoid inverse schedule 得到的公式映射。

必须保留一个实现边界：released paper-upper 配置实际为 `use_gt_csi=false`，运行时由 `snr_prediction_net` 从接收 latent 预测 signal scale，所以实际逐图 endpoint 可能围绕理想公式值变化，而不是直接读取表中常数。但这不影响计算量结论：continuous sampler 无论 endpoint 是公式值还是预测值，仍然构造50点并执行50次 denoiser。

## 四、SGD 延迟-SNR 曲线

| SNR | 端到端 mean ms | median ms | p05–p95 ms | BLIP2 ms | MuGE ms | BLIP2+MuGE ms | diffusion solver ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2044.877 | 2045.126 | 2020.533–2069.082 | 220.528 | 849.177 | 1069.705 | 908.273 |
| 4 | 2043.783 | 2046.605 | 2019.274–2062.767 | 220.485 | 849.343 | 1069.828 | 907.265 |
| 7 | 2044.802 | 2047.187 | 2021.235–2064.394 | 220.654 | 849.405 | 1070.059 | 908.065 |
| 13 | 2044.636 | 2047.473 | 2021.485–2065.504 | 220.445 | 849.484 | 1069.930 | 907.943 |
| 19 | 2045.410 | 2047.908 | 2022.408–2064.791 | 220.725 | 849.419 | 1070.144 | 908.543 |

五档端到端 mean 的最大差只有 `1.627 ms`；diffusion solver mean 最大差 `1.278 ms`。没有观察到随 SNR 增高而下降的延迟趋势，这与固定50次调用完全一致。

## 五、MuGE 与 BLIP2 是否为固定地板

对当前 released SGD 管线，答案是：**是结构性固定地板，但不是数学上毫秒完全不变。**

- BLIP2 每张256×256源图的四个 patch 各生成 caption，发生在 step matching 前；
- MuGE 每张图生成 soft edge，同样发生在 step matching 前；
- 两者均不接收 diffusion step count，五档都不会被跳过；
- 二者五档合计均值为 `1069.933 ms/图`，占总体 `2044.701 ms` 的约 `52.33%`；
- 五档合计均值的最大差只有 `0.440 ms`。

因此，即使假设把后续 diffusion solver 免费删除，当前 released 管线仍有约 `1.07 s/图` 的 BLIP2+MuGE 前处理地板。它们是 SGD 这一实现的固定成本，不是所有生成式 JSCC 的固有成本；未来无文本、小 edge model 或联合特征提取可以改变这一项，但那将是另一种方法，需要重新测质量与可靠性。

## 六、对新方向的启发

P0 强化了新主线的合理性：

1. SGD 的 channel matching 很自然，但当前实现没有把好信道转化成更少计算；
2. 其一半以上延迟甚至发生在 diffusion solver 之前；
3. 一个只使用 S33 RGB+SNR、没有 BLIP2/MuGE/VAE/迭代采样的轻量 receiver refiner，理论上有非常大的系统代价空间；
4. 因而论文应比较完整的“质量—语义可靠性—参数/FLOPs/延迟”，而不是只比较是否使用生成损失。

这仍不能预言 P1 一定获得显著 LPIPS 改善；P1 必须按预注册 go/no-go 实验裁决。

## 七、产物

- 预注册：`reports/s35r_p0_sgd_adaptive_cost_preregistration_2026-07-23.md`
- 配置：`configs/s35r_p0_sgd_adaptive_cost.yaml`
- 正式输出：`outputs/analysis/ANALYSIS-S35R-P0-SGD-ADAPTIVE-COST-001/`
- 五档曲线：`latency_snr_curve.csv`
- 完整分组件统计与源码 hash：`summary.json`
