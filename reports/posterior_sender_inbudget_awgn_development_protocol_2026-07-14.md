# 发送端全概率语义载荷：固定码率 AWGN 开发协议

日期：2026-07-14。分析编号：`ANALYSIS-PC-SENDER-RATE-DEV-001`。

## 目的与证据边界

上一轮 `ANALYSIS-PC-SENDER-DEV-001` 只证明 80-bit 无噪声发送端描述在已暴露的 policy-dev seed `20260725` 上具有可达性，不能证明 matched-rate 或真实信道下有效。本轮仍是事后开发证据，不访问 Imagenette 官方 validation，也不能冒充独立验证。唯一目标是判断：在 DeepJSCC `c=8` 总符号数完全不增加、语义载荷与图像主链路共同经过 AWGN 时，自然零阈值 veto 是否还值得进入新 seed 审计。

## 冻结方法

- `G_aux`、`T_cls`、DeepJSCC、S13 B1、S14 六步 diffusion、三步 posterior correction 和 seed `20260725` 均冻结；
- sender 计算 `G_aux` 的 10 维校准概率，不再声称它是额外 80 bit 数字旁路；
- 将概率向量作 L2 单位功率归一化，每维重复 16 次，共占用 `10×16=160` 个实信道符号；
- 160 个位置以固定均匀索引覆盖在原 `c=8` latent 内，总实符号仍为 65536，图像主载荷剩 65376；语义载荷占总预算 `0.244140625%`，总 CBR 仍为 `1/6`；
- 覆盖后对整个 latent 重新归一化到单位平均功率，语义和图像载荷在一次共同 AWGN channel call 中传输；
- receiver 对 16 次重复取均值、负值截零并归一化回概率单纯形，然后擦除保留位置再送入冻结 DeepJSCC decoder；
- posterior data-consistency 只在 65376 个图像主载荷位置上计算，不能要求候选图像编码复现 sender 写入的语义载荷；
- 决策分数保持不变：

```text
r_JS = JS(q_recovered, G_aux(posterior)) - JS(q_recovered, G_aux(anchor))
```

- 唯一规则仍为 `r_JS <= 0` 接受 posterior，否则回退本轮 in-budget B1 anchor；不扫描重复次数、不扫描阈值、不设 per-SNR 例外。

## 对照与成功门槛

旧的无噪声可达性逐样本表只作为同 seed 的 `c=8` unpunctured reference，必须记录文件哈希并核对所有 `(seed, SNR, sample_id)` 键。主比较为本轮最终输出相对该 reference S14 raw diffusion；同时报告本轮 in-budget anchor/raw/posterior/final 的内部语义事件。

进入新 seed 独立信道审计前，以下条件必须同时满足：

1. 速率、保留位置数和共同 AWGN channel call 契约逐项通过；
2. 每个 SNR 的 source/recovered top-1 一致率与 cosine 均不低于 `95%`；
3. primary SNR `[1,4,7]` 的 final failure 总数不高于 unpunctured reference raw；
4. primary final new-error 不高于本轮 in-budget raw，且 image-cluster 单侧 95% Clopper–Pearson upper 不超过 `0.5%`；
5. 五个 SNR 平均 final-minus-reference-raw PSNR 为正、LPIPS 不增加；
6. masked received-latent consistency 在每个 SNR 均下降。

若失败，只记录负结果；不允许在 seed `20260725` 上扫描 repetitions 或阈值挽救。若通过，下一步只允许冻结同一个 `R=16` 和零阈值，在新的 channel seed 上审计。新增结果报告使用中文。
