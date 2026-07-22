# SGD-JSCC 作者原生单图复现阶段报告（2026-07-14）

## 结论

`SMOKE-EXT-SGDJSCC-001` 已一次性跑通作者发布的完整链路：BLIP2 caption、主 JSCC latent、MuGE soft edge、独立 edge-JSCC、CLIP text condition、ControlNet 和 50-step continuous diffusion 均实际参与前向。输出为有限的 `1×3×128×128` 图像，作者权重、BLIP2 分片和 CLIP 均通过精确尺寸与 SHA-256 校验，第三方 tracked source 保持只读。

这项结果只证明“作者方法已在本仓库外接成功并可被计量”，**不证明 SGD-JSCC 优于或弱于本项目方法**。当前仍只能进入 author-native 表，不能进入 common-contract 直接排名，原因是作者协议把 caption 假设为完美/免费传输，edge 路径又同时存在 dense tensor 和 nonzero-active 两种码率解释。

## 冻结协议

- 配置：`configs/external_sgdjscc_native_smoke.yaml`
- 项目侧适配器：`scripts/external_sgdjscc_native_smoke.py`
- 作者源码：`third_party/SGDJSCC`，commit `2188acc0dd2805355d3d0d2e478cbc27b46b4da5`
- 输入：S13 train2017-scale validation cache 的 `sample_010000.png`，仅作 integration smoke
- 信道：AWGN，`1 dB`，seed `2025`，`use_gt_csi=false`
- 作者条件：semantic/text/ControlNet/JSCC feature 全开，50 diffusion steps，CFG `4.0`，`pcs_1.0`
- `1 dB` 不在作者发布的离散超参数表中；0 dB 与 5 dB 的 control scale/threshold 都同为 `0.20/0.25`，因此本 smoke 使用这组共同值，没有在结果后选择。
- official Imagenette validation 未访问；单图 smoke 明确禁止效果与 semantic-safety 声明。

## 下载与运行环境

所有大文件下载命令均清空 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 及小写变量，走服务器直连：

- 作者 4 个 checkpoint：Hugging Face `MauroZMJ/SGDJSCC` revision `e95f395f5e43e570a8d65c2afa7d916860916cf0`，合计 `2,930,865,634` bytes；
- BLIP2：`Salesforce/blip2-opt-2.7b-coco` revision `cda95b9319b722f79c9451a3d8ff92eea02048dc`，只取两个 safetensors 分片及 tokenizer/processor 文件，避免重复下载 `.bin`，两分片合计 `15,496,030,352` bytes；`hf-mirror.com` 只作直连 resolver，实际大文件来自官方 `cas-bridge.xethub.hf.co`；
- OpenAI CLIP ViT-L/14：OpenAI 官方 Azure CDN，`932,768,134` bytes，SHA-256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`；
- Stable Diffusion v1.4 scheduler：commit `133a221b8aa7292a167afc5127cb63fb5005638b`，仅缓存 55-byte scheduler config。

隔离环境为 `.venv-sgdjscc`，关键版本为 Python `3.10.12`、PyTorch `2.1.0+cu121`、torchvision `0.16.0+cu121`、xFormers `0.0.22.post7`、Transformers `4.44.2`、Diffusers `0.26.3`。`pip check` 通过，RTX 4090 D 上 xFormers memory-efficient attention 实际 kernel 测试通过。

作者 MuGE 构造函数默认先下载 EfficientNet-B7 ImageNet 初始化，再立即被完整 MuGE checkpoint 覆盖。项目适配器改为 `encoder_weights=None` 后 strict-load 发布 checkpoint；1466 个 state-dict entries 全部匹配，不改变最终运行权重，也避免无意义的 254 MB 隐式下载。BLIP2 在生成 caption 后立即释放，再加载 JSCC/diffusion 重型栈，降低峰值显存；第三方源码本身未改。

## 单图结果

输出目录：`outputs/smoke/external_sgdjscc_native_snr1_seed2025_20260714/`

| 项目 | 结果 |
|---|---:|
| 状态 | `PASS` |
| caption | `a blurry picture of a baseball field with a few players on it` |
| 输出 shape | `[1,3,128,128]` |
| smoke-only PSNR | `25.055894 dB` |
| 模型加载加单图前向耗时 | `12.6837 s` |
| peak allocated GPU memory | `7234.28 MiB` |
| failure artifact | 无 |

`source_preprocessed.png` 与 `reconstruction.png` 均为合法 RGB `128×128` PNG。肉眼 sanity check 中，球场/围栏的大场景结构仍在，但若干小人物细节明显变弱；这恰好说明不能用“看起来平滑”替代 semantic-drift 审计。单图没有统计意义，不据此评价论文方法质量。

## 实测码率账本

适配器直接 hook 作者主信道和 edge 信道的输入 tensor，而不是只按配置猜测：

| 项目 | 实测值 |
|---|---:|
| source real dimensions `3×128×128` | `49,152` |
| main latent real symbols | `4,096` |
| main real-coordinate/source-dimension ratio | `0.083333` |
| main complex-use CBR（两个实坐标/复使用） | `0.0416667` |
| edge dense tensor elements | `16,384` |
| edge nonzero-active elements | `832` |
| edge active fraction | `0.05078125` |
| main + edge active symbols | `4,928` |
| active real-coordinate/source-dimension ratio | `0.1002604` |
| active complex-use CBR | `0.0501302` |
| main + edge dense elements | `20,480` |
| literal dense real-coordinate/source-dimension ratio | `0.4166667` |
| literal dense complex-use CBR | `0.2083333` |
| caption UTF-8 cost | `488 bits` |
| caption channel symbols | 未定义 |

作者 rate-adaptive edge encoder 在 dense tensor 中保留大量零值；从稀疏调度的物理实现看可以只发 active entries，但当前代码把完整 dense tensor 送入 AWGN module。两种数字都保留，不能事后挑一个对本项目有利的口径。更关键的是 caption 没有 channel model、coding 或 symbol mapping，因此 common-contract total CBR 仍未闭合。

术语更正（2026-07-15）：原表中的 `real CBR` 是“实坐标数/源实维度”，而本项目 `CBR=1/6` 按复信道使用计数；两个实坐标组成一个复使用。上表现已同时列出两种口径，原始 smoke 输出不覆盖。后续共同协议闭环见 `reports/sgdjscc_common_contract_smoke_stage_result_2026-07-15.md`。

结果文件：

- `summary.json` SHA-256：`33671c8ea83c0371bd2410f54fc4fec47f9fabeea79cc49a67dc1af923af1d4b`
- `rate_accounting.json` SHA-256：`aea2351d93a36a198f8c46ccd2e1821100dcebc5c38c2aeb78d88cc3e55f7002`

## 对项目方向的直接影响

1. 外部复现不再停留在“以后再做”：SGD-JSCC 的作者完整链已经可运行，后续 common-contract 适配可以基于真实 tensor/rate，而不是依据论文表格猜测。
2. 本项目的差异化判断得到进一步支持：SGD-JSCC 的生成链和语义条件很完整，但其 text 可靠性/成本未纳入信道协议；本项目更有价值的比较轴仍是 matched-total-rate、refinement-induced new error 和 tail risk，而不是只比单图 PSNR 或视觉平滑度。
3. 不能立刻说“我们的最好方法强于 SGD-JSCC”。本 smoke 的输入尺寸、总码率、text 假设、数据量和语义口径都与本项目正式结果不同。
4. 下一步先做 SGD-JSCC common-contract 小 split：固定同一批 COCO-256 图、同一 AWGN realization、`[1,4,7,13,19] dB`，显式实现 caption 的 channel cost，并冻结 edge active/dense 的物理解释；闭合到总 `65,536` real symbols 后才允许进入直接排名表。
5. common-contract 接口冻结后按既定排期接 SING-Zero-style 机制对照；不因本次 smoke 跑通而改变当前 diffusion + posterior consistency 主线。

## 验证

- 作者四组网络 checkpoint strict-load：全部 keys matched；
- BLIP2/CLIP/scheduler 离线加载：通过；
- combined JSCC+MuGE+ControlNet+CLIP stack：通过，常驻约 `2570.9 MiB`；
- `python3 -m unittest discover -s tests -p 'test_*.py' -v`：84 项全部通过；
- adapter 独立 pytest：4 项通过；
- `pip check`：通过。
