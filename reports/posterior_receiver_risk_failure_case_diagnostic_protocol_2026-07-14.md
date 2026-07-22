# Posterior 风险控制器失败案例诊断协议

日期：2026-07-14。分析编号：`ANALYSIS-PC-RISK-FAIL-001`。

## 定位

这是在 `ANALYSIS-PC-RISK-SEED-AUDIT-001` 已经按冻结协议判为 `NEGATIVE` 之后进行的事后诊断，不是新的独立验证，也不能改变原 verdict。诊断只回答：两例高置信漏检究竟缺少哪类信息，以及 sender semantic description 是否存在值得继续做严格计码率实现的可达信号。

官方 Imagenette validation 继续封存。输入只允许使用已生成的 seed `20260725` 决策表、policy-dev manifest、本地冻结 checkpoint 和确定性信道重放。

## 冻结输入与案例选择

- extraction config SHA-256：`c1465c585b7e2e12c246668e8b71777c831b4dcf8408dfc6b9ce130ab93d5d33`；
- frozen decisions SHA-256：`82646e4d6c9f81c292dfbcfbbb0c28b3deb20197b8624beca94ed2d94836c3d2`；
- frozen audit CSV SHA-256：`849104b88cad23edde2ace7818611bdabe4c9a5e18abae6d308d897a2785ccbc`。

只选 primary SNR `{1,4,7}` 中两类已有结果：

1. `clean_correct AND anchor_correct AND posterior_incorrect AND not rejected`，预期 2 行；
2. `clean_correct AND anchor_incorrect AND posterior_correct AND rejected`，预期 11 行。

数量不符必须停止，不允许人工增删案例。

## 精确重建与核对

按原 manifest 顺序和 batch 起点，用原 seed/SNR 的 `derived_seed` 重放整个 batch，再运行冻结 DeepJSCC、S13 B1、S14 diffusion 和三步 posterior correction。重建后的 anchor/raw/posterior correctness、PSNR、LPIPS 与 received-latent consistency 必须在 `1e-5` 内复现原 CSV，否则诊断无效。

每个案例输出：

- source、anchor、raw、posterior 和 `10×|posterior-anchor|` 图像面板；
- `G_gate`、`G_aux`、`T_cls` 在四个状态上的 top-1、置信度、真实类概率和真实类 margin；
- source-to-anchor、source-to-posterior 的 JS/CE/cosine 与 source-class log-prob 风险；
- 4-bit learned source top-1、80-bit uint8 probability description 及 4-bit true-class oracle 的可检出性；
- pixel L1/RMSE、PSNR、LPIPS 和原冻结风险分数。

`T_cls` 只用于解释已有主指标，不能被包装成可部署 sender/receiver 模型。true-class 4-bit token 只作为 oracle 上界，不能作为 learned method。

## 决策规则

诊断只允许形成以下三种结论：

1. 若 learned source top-1/full-probability 对两例均有明显独立信号，可进入严格计码率、带信道错误的 checksum pilot；
2. 若只有 true-class 或 `T_cls`-aligned oracle 能检出，说明当前 learned grounding 不足，应先训练独立任务相关 representation，不能直接做 payload 工程；
3. 若连 source-to-posterior feature 变化也很弱，且视觉/多模型均不支持明显语义改变，则当前 tail 主要是单 evaluator decision-boundary sensitivity；下一正式协议应增加 evaluator-robustness 分层，但仍保留 `T_cls` 主指标，不能删除负结果。

本诊断不得扫描新 threshold、替换 seed、重新定义 clean subset 或解锁 official validation。新增结果报告必须使用中文。
