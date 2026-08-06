# 方法终止与项目冻结报告

- 日期：2026-08-03
- 对象：`channel-adaptive diffusion + semantic control JSCC` 当前方法开发项目
- 决策：**停止继续投入并冻结方法开发**
- 决策类型：`ENGINEERING_STOP`
- 审计基准 commit：`c435f89e7cd08055e7fafb12fe7836c6d5950fc6`
- 证据索引：`audit/CLAIM_REGISTRY.csv`
- 数据与协议时间线：`audit/DATASET_PIPELINE_TIMELINE.md`

## 1. 报告地位与冻结范围

本报告是 2026-08-03 起该项目的**最终解释层和规范终止入口**。此前报告、预注册、配置、实验输出和 `verdict.json` 保留为其生成时点的历史记录，不回写、不删除、不覆盖；其中关于“下一步”“待授权”“继续主线”的表述，自本报告起全部视为被终止决策 supersede，不再构成执行授权。

本轮只做静态证据审计和文档冻结，没有运行训练、推理、评测或下载，没有访问封存的 official Imagenette validation，也没有生成新的实验结果。冻结对象是当前方法路线的继续开发，不是删除仓库、代码、数据、checkpoint 或评测基础设施。

## 2. 原始目标与最终判定

项目原始目标不是单独优化某一个轴，而是在统一、可审计的信道与码率合同下建立一套完整系统，使以下四个维度形成相对外部强基线的联合优势：

1. channel-adaptive JSCC 的 rate / fidelity；
2. diffusion 或生成式精修带来的 perception；
3. semantic control 对 semantic drift / failure 的约束；
4. 可接受的计算、时延和部署代价。

最终判定为：

> **原始完整联合优势主张未建立，项目停止继续投入。**

这不是把所有子结果判为失败。项目中的不同子假设分别处于**局部支持、被反驳、尚未建立或工程停止**状态；不能把四类状态混成一个“项目科学上已被统一证明失败”的结论。

状态含义固定如下：

- `SUPPORTED / QUALIFIED`：在明确合同和范围内存在支持，但不得越过数据、backbone、码率、信道或评测总体外推；
- `REFUTED`：给定 scope 内的具体科学命题有直接反证；
- `NOT_ESTABLISHED`：当前证据不足或必要实验未执行，不能偷换成“被反驳”；
- `ENGINEERING_STOP`：基于现有证据、预算、归因和预期边际收益停止投入，不等同于科学命题普遍无效。

## 3. H1–H5 分项结论

| 分项 | 最终状态 | 已建立内容 | 未建立、被反驳或停止内容 |
|---|---|---|---|
| **H1：matched diffusion / semantic control** | **局部支持，广义未建立** | 旧 `19,712-real` backbone 上，S27 在 fresh COCO-512、AWGN 低 SNR 合同中相对 B1 为 `+0.092662 dB PSNR / -0.007922 LPIPS`，相对等参数 control 为 `+0.065799 dB`；13/19 dB exact fallback 逐像素恒等。 | 相对 B1 仍有 `60 new / 104 repair`，不能称 semantic-safe；结果没有迁移到 S33 或外部强 backbone 协议，不能拼接成完整联合优势。停止新的 diffusion refiner、semantic gate/controller/fusion 搜索。 |
| **H2：strong channel-adaptive JSCC backbone** | **内部局部支持；全局强基线主张被反驳** | S33 在 `256×256` policy-dev、严格 `16,384 real` 合同中聚合优于 author-JSCC；它保留为码率、功率和条件机制可审计的低成本判别式端点。 | “S33 强于 SwinJSCC”在 Kodak/CLIC 外部协议上被反驳：Kodak 对 CM-SA 为 `-0.2003 dB`，CLIC 对 Base/CM 为 `-0.2631/-0.4909 dB`，感知与结构指标总体也更弱。约 `2.32–2.45×` 延迟优势和约一半峰值显存只适用于当前大图 smoke/硬件，且以质量损失为代价。 |
| **H3：rate–perception–compute–reliability tradeoff** | **当前实现下存在局部 Pareto；普遍性未建立** | S33 与 DiffJSCC 在共同 `16,384 real`、256² 总体上构成 fidelity–perception Pareto。当前 DiffJSCC checkpoint 的 25-step 点是在已测网格中最低的 LPIPS PASS 点；在 RTX 4090D、batch 1、共同 PyTorch 2.1 runtime 下相对 S33 为 `165.1×` wall latency，profiler 已支持算子的 FLOPs 下界为 `472×`。 | 25-step failure 为 `14/320`，S33 为 `4/320`，因此不是已建立的 semantic-safe 点。上述代价只属于当前 checkpoint、sampler、步数网格、实现和硬件，不能外推成所有生成式 JSCC 的结构性定律。完整 rate–perception–compute–semantic reliability 联合优势未建立。 |
| **H4：RDD / reconstruction fingerprint** | **输出指纹现象受支持；生成先验因果解释被反驳** | RDD-P0 支持重建输出存在可识别的实现/分布指纹；按 source 分组的指纹分类显著高于随机。 | 两个都无生成先验的判别式臂仍可高准确率区分，且跨分辨率偏移方向改变，直接反驳“现有指纹由各自生成先验特有地定向造成”的解释。更细的生成先验因果机制尚未建立；不得写成“RDD 完全失败”或“不存在分布偏移”。 |
| **H5：CVaR 条件尾部风险** | **局部测量成立；`END-CVAR` 为工程停止** | 当前 AWGN 合同中 `median-p10 <= 0.11 dB`、信道方差占比 `<=0.001`，没有明显 CVaR 优化对象。未匹配 Rayleigh P0 有大尾部但受信道错配和条件 OOD 混杂。Rayleigh matched pure-MSE P1 使 `median-p10`、`CVaR-10/mean MSE`、`outage(<24dB)` 和 channel variance fraction 在 5/5 档下降。 | 仓库没有训练 CVaR-10、CVaR-20 或 worst-one 模型，也没有 CVaR-vs-matched-mean 直接比较。`END-CVAR` 只表示当前项目不再投入 CVaR 模型训练，不表示“CVaR 对 JSCC 无效”或“CVaR 已被科学反驳”。 |

### 3.1 Blind diffusion 的限定结论

早期 `EXP-S2-002` 的 blind Stable Diffusion img2img pilot 中，平均 PSNR 下降 `14.7485 dB`，LPIPS 增加 `0.3877`（更差），CLIP/分类器语义代理也恶化。因此，在该 48 图、旧 DeepJSCC、固定 blind 实现的范围内，“改善感知但增加语义失败”的前提被反驳：它没有先取得感知改善。

这只能否定该 pilot 的命题，不能外推到所有 blind 设置，更不能外推到 matched、conditioned、anchored diffusion 或所有生成式 JSCC。

## 4. 为什么在此停止

停止不是由单一负结果触发，而是由证据链整体决定：

1. **完整系统证据没有闭合。** H1 的局部正效应来自旧 backbone；H2 的 S33 是另一条判别式链；H3 的外部生成式对比又处于不同训练与系统合同。它们不能事后拼接成一个已经验证的完整方法。
2. **强外部基线改变了主张上限。** S33 的 author-JSCC policy-dev 优势没有迁移成对 SwinJSCC 的外部优势，因而“强 backbone + 轻量后处理即可形成完整主方法”的预期基础明显收缩。
3. **边际收益与代价不匹配。** 已测旧 fusion 的稳定效应约为 `0.09 dB`；复杂 controller 的特定路线 headroom 很小；当前生成式端点有明显计算和代理可靠性代价。继续搜索需要新的训练、调参和协议扩张，但预期不足以建立四轴联合优势。
4. **关键泛化和最终层仍未建立。** official supervised reliability validation 保持封存，Swin convergence extension 未执行，S35R-P1 只有预注册。为“救结论”而继续解封、延训或扫模块会增加选择偏差，而不是修复当前证据结构。
5. **现有资产已足以支持诚实的局部结论和负结果报告。** 继续开发的机会成本高于预期信息增益，因此采用 `ENGINEERING_STOP`，而不是把未运行的实验写成失败。

## 5. 冻结决定

自本报告起，下列项目**不启动**：

- S35R-P1 one-batch smoke、轻量 receiver-side refiner 训练及后续 gate；
- 新 diffusion refiner、matched B1/M2/diffusion/envelope 重训或模块搜索；
- semantic gate、controller、fusion、routing 或阈值搜索；
- CVaR-10、CVaR-20、worst-one 或其他尾部风险模型训练；
- S34A SwinJSCC convergence extension、S34B 新消融、S34C 长版生成式公平重训；
- A2、S36 official validation，或任何以挽救主张为目的的新评测；
- 大量旧实验复跑、既有正式输出覆盖或 outcome-driven 新 analysis ID 搜索。

下列内容保持不可变：

- 历史实验结果、失败目录、机器生成的 `verdict.json`；
- 已冻结预注册、判定阈值和当时的运行合同；
- 既有 checkpoint、manifest、noise key/SHA、指标与样例；
- 负结果和中断记录。

如未来出现**独立的新课题授权**，必须使用新的问题定义、预算、预注册和 experiment/analysis ID；它不能被描述为本项目自然继续，也不能以覆盖本报告为前提。

## 6. 明确禁止的推论

本项目终止后，不得作以下推论：

- “原始 idea 已被严格证明在所有设定下失败”；
- “所有 channel-adaptive diffusion、semantic control、gate/controller/fusion 都没有 headroom”；
- “所有生成式 JSCC 都必然慢 `165.1×`、耗费 `472×` FLOPs 或必然不可靠”；
- “S33 全面成功、全局 SOTA”，或反向写成“S33 毫无价值”；
- “RDD 完全失败”或“不存在重建分布偏移”；
- “CVaR 对 JSCC 无效”“CVaR 模型已经失败”或“本仓库已训练 CVaR 模型”；
- “S35R-P1、S34C 长版、A2、S36 已执行并得到负结果”；
- 把 policy-dev、定向人工审计或单硬件测量写成独立 final/general law。

允许的总括表述是：

> 项目中的不同子假设分别处于局部支持、被反驳、尚未建立或工程停止状态。完整的 rate–perception–compute–semantic reliability 联合优势没有建立。

## 7. 保留并可复用的资产

冻结方法开发不等于废弃基础设施。以下资产可在遵守原始 scope、SHA、许可和数据封存边界的前提下复用：

- **判别式 JSCC 基础设施**：`src/cadsd_jscc/strong_jscc.py`、S33/S31 checkpoint、exact-rate 与单位功率合同；
- **信道与重现性基础设施**：canonical paired-real AWGN、noise key/SHA、block-fading Rayleigh + ZF 与逐 realization 诊断；
- **公平比较基础设施**：source/processing/rate manifest、actual CBR/side-information ledger、SwinJSCC adapter 和外部 baseline 合同；
- **评测基础设施**：PSNR、MS-SSIM、LPIPS、DISTS、CLIP、FID/KID、semantic failure/new-error/repair、source-cluster bootstrap；
- **系统测量基础设施**：共同 runtime 下的延迟、组件分解、参数与 profiler FLOPs 下界统计；
- **审计资产**：`audit/CLAIM_REGISTRY.csv`、`audit/DATASET_PIPELINE_TIMELINE.md`、checkpoint registry、正式报告和失败记录；
- **数据与样例资产**：冻结的 COCO/Imagenette policy-dev population、Kodak/CLIC benchmark manifest、重建样例和人工 failure-mode 审计。

复用这些资产时必须引用原始合同；不得把基础设施复用解释为本方法路线重新激活。

## 8. 最终结论

本项目未能建立 `channel-adaptive diffusion + semantic control` 相对外部强基线的完整联合优势：旧 backbone 上存在局部 diffusion 互补信息，S33 在内部严格等码率开发协议上有局部优势和低成本价值，但该 backbone 优势没有迁移为对 SwinJSCC 的外部质量优势；当前 DiffJSCC 实现展示了 fidelity–perception Pareto，同时伴随较大的计算和代理可靠性代价；RDD 只建立了输出指纹而没有建立生成先验因果解释；CVaR 路线只完成必要性与归因诊断，未测试 CVaR 模型本身。

因此，项目停止继续投入方法开发，保留全部历史结果、负结果与可复用基础设施。该决定属于基于现有证据、预算和预期边际收益的 `ENGINEERING_STOP`，**不等同于对所有 channel-adaptive diffusion、semantic control、生成式 JSCC、RDD 或 CVaR 相关科学方向的普遍反证**。
