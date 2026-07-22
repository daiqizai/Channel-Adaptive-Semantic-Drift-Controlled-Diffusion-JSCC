# src

存放核心代码。

计划模块：

- `channels/`：AWGN、Rayleigh 等信道层
- `jscc/`：DeepJSCC baseline 接入或实现
- `diffusion/`：blind refinement、semantic guidance、SNR-aware strength control
- `metrics/`：PSNR、MS-SSIM、LPIPS、semantic drift 指标
- `utils/`：日志、配置、图像保存、随机种子

当前新增 `cadsd_jscc/strong_jscc.py`：clean-room 四级 residual JSCC，编码器/解码器共享 SNR embedding 并在每个残差块做 FiLM 调制；默认 `256x256`/77 latent channels 原生闭合 `19,712` 个实符号码率，同时暴露 encode/normalize/transmit/decode 和 receiver observation 接口，供后续 matched diffusion 使用。

`cadsd_jscc/swinjscc_adapter.py`：S34A 项目侧 adapter；保持固定 commit 的官方 SA-only Swin block 与 Channel ModNet 拓扑不变，把 scalar-only/batch-global 官方接口改为本项目所需的逐图 SNR、逐图单位功率、paired-real AWGN 和 external canonical-noise 接口。Base 与 capacity-matched depth 由显式构造参数区分，原生固定输出 `256×64=16,384 real`。
