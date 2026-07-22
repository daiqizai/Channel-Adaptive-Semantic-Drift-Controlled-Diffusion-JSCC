# DiffJSCC 外部严格对比预注册（2026-07-21）

## 目的

把作者公开的 DiffJSCC OpenImage `C_channel=16` 权重接入已经冻结的 S20/S28 Imagenette 总体，回答两个不同问题：

1. DiffJSCC 的扩散接收端相对它自己的 DeepJSCC 初始重建是否提升；
2. DiffJSCC 相对项目当前方法和 B1，在相同源图、相同 SNR、同一标准正态噪声实现及同一评价器下处于什么位置。

本轮不训练、不选 checkpoint、不调 sampler、guidance 或阈值，也不访问 Imagenette 官方验证集。S28 的当前方法结果已经可见，因此它是外部定位，不是新的盲测确认。

## 冻结资产

- 作者仓库：`third_party/DiffJSCC`，提交 `13aeb62451b872ce41ceba132c9c30a9ca172c53`；仓库根目录未发现许可证文件。
- 作者权重：`Mingyuyang/DiffJSCC-OpenImage-CBR-1-96`，Hub revision `318d46f7f7331ae95fb162b460c1ebfd7d20b6fa`。
- `model.ckpt`：`9,859,655,693` bytes，预期 SHA-256 `ae1e6df0...2d399f94cc171d8d0ce44f851d96cb032bd7dec579`。
- 权重仓库 metadata 标注 Apache-2.0；这不能自动补足源码仓库缺失的 license。
- 作者配置 SHA-256 为 `32351c11...175fdf99bbc3771cb17981ad7db9d89d1c248f`。
- BLIP2 processor/config/tokenizer metadata 固定为 `Salesforce/blip2-opt-2.7b` revision `59a1ef6c...c6066825bcf315e3`，包含精确权重索引后的目录树摘要为 `765e8aef...0652c89fa3268a63b47073053942e4ffa53c`；metadata 与下面单列的两个 base 权重分片均不得由其他 caption 模型替代。
- DiffJSCC requirements 指定的 `open-clip-torch==2.24.0` 固定为官方 release archive SHA-256 `83d78a78...cd6fed7d22de7b6`，解包目录树摘要 `27603908...35fec3c79969c4`；只实例化空架构，禁止下载或代入其他 OpenCLIP 预训练权重。
- 独立 runtime 固定 `transformers==4.51.1`、`tokenizers==0.21.4`、`huggingface-hub==0.30.2`；BLIP2 显式保持 `use_fast=False`，避免库升级默认切换图像 processor。
- 实际执行环境固定为 `.venv-sgdjscc` 的 PyTorch `2.1.0+cu121`、torchvision `0.16.0+cu121`、Lightning `2.4.0`、xFormers `0.0.22.post7`，四个包的 METADATA SHA-256 均写入配置并由预检核对。作者 requirements 的 legacy 组合是 PyTorch `1.13.1+cu116`、Lightning `1.4.2`、xFormers `0.0.16`；本轮只允许 Lightning 旧 import path 等 API 兼容 shim，不改变模型、权重、信道、采样器或算子精度，并在结果中保留这一复现边界。

源码的 `ControlLDM.on_save_checkpoint` 明确删除所有 `blip_model.*`，所以 DiffJSCC checkpoint 缺 BLIP2 是作者设计，而不是坏包。必须另外加载作者代码硬编码的 `Salesforce/blip2-opt-2.7b` base revision `59a1ef6c...c6066825bcf315e3`：两个 safetensors 分别为 `9,996,328,120` / `4,982,879,016` bytes，SHA-256 为 `b81228c9...ae24fc` / `536bd73b...55a19`。不得用随机参数或 SGD-JSCC 的 COCO 微调权重替代。checkpoint 仍必须完整提供 OpenCLIP、JSCC、ControlNet、UNet、VAE 和 spatial condition encoder；其余缺失一律 fail-closed。

## 总体与信道

- 总体：S20 冻结的 64 张 Imagenette `policy_dev`、`T_cls` clean-correct 图像；输入严格沿用 `Resize(256)+CenterCrop(256)`。
- SNR：`[1,4,7,13,19] dB`；channel seed 为 `20260748/49/50`。
- 对每个 `sample_id/SNR/seed` 继续使用 `external-common-v1` 生成 19,712 维 CPU float32 标准正态向量，并保存完整向量 SHA-256。
- DiffJSCC 只使用该向量的前 16,384 个实坐标；实、虚部各 8,192 个坐标，噪声方差保持项目复 AWGN 的 `P/(2×SNR)` 口径。
- DiffJSCC 公开权重训练 SNR 为 `[0,14] dB`。19 dB 结果仍计算，但必须单列为外推，不得拿它单独概括作者方法。

## 码率口径

作者推理先把 256×256 输入双三次放大到 512×512。`C_channel=16`、下采样 16 倍时，运行时 latent 应为 `16×32×32=16,384` 个实坐标，即 8,192 次复信道使用。

因此同一数字必须给出两种口径：

- 作者 512×512 处理网格口径：CBR `1/96`；
- 本对比原始 256×256 源口径：CBR `8,192/(3×256×256)=1/24`。

项目当前方法为 9,856 次复使用、CBR `0.0501302`。DiffJSCC 使用其 83.1169%，少用 3,328 个实坐标。它属于“相同预算上限内”的合法对照，但不是 exact-rate match。接收端 BLIP2 caption 从带噪 JSCC 初始重建生成，不是发送端 side information，记 0 个信道符号。

## 作者推理冻结项

- 100 sampling steps，repeat 1；
- control strength 1.0，CFG scale 1.0；
- `wavelet` color fix；
- 不启用 intermediate MSE guidance；
- 输入使用作者 `auto_resize(short_edge>=512)` 和 64 倍数 padding，输出按作者实现 Lanczos 缩回 256×256；
- batch size 1；diffusion sampler seed 由 `base_seed/sample_id/SNR` 的独立 SHA-256 规则决定，不复用或消耗信道噪声 RNG。

## 阶段顺序

1. checkpoint/state-dict/rate preflight；
2. 冻结总体第一张图、1 dB、seed `20260748` 的作者原生链 smoke；
3. 64 图×5 SNR×第一个 channel seed，共 320 行阶段比较；
4. 不改任何设置，再补 seed `20260749/50` 到完整 960 行。

smoke 只验证可执行性，不能决定修改 sampler。若作者链因环境兼容失败，只允许做等价 API 兼容适配；任何算法变化都必须停止本协议并另开配置。

## 指标与结论规则

统一报告 PSNR、MS-SSIM、LPIPS、`T_cls` failure、每图耗时、峰值显存和实测符号数，并进行 source-image cluster bootstrap。

- 先报告 `DiffJSCC - author JSCC`，确认 diffusion 自身的收益和 semantic drift；
- 再报告 `current - DiffJSCC` 与 `B1 - DiffJSCC`；
- 只有 PSNR 的 cluster CI 严格为正、LPIPS 的 CI 严格为负且 failure 不更差，才能称一方严格支配另一方；
- 其余情况明确写成 Pareto 或证据不足，不按某一个喜欢的指标强行宣布胜者。

无论结果如何，不能把本轮写成“战胜所有论文方法”。DiT-JSCC 仍无作者可运行实现；其他论文也需要按同样合同逐项接入。
