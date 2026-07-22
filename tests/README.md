# tests

存放单元测试和 smoke test。

第一批测试应覆盖：

- 信道层输入输出 shape
- PSNR / MS-SSIM / LPIPS 指标接口
- semantic drift 指标接口
- baseline inference 最小样例

当前标准库 `unittest` 共 122 项，覆盖 residual refiner、diffusion-fusion 等容量/初始化合同、frozen-B1 feature injection exact fallback/参数预算、Imagenette scratch/receiver-risk protocol、sender strict-rate dual-evidence/cross-model-triplet acceptance/routing primitives、official-statistics image-cluster endpoint、empirical CDF/threshold selection、精确码率、semantic sketch、short-chain bridge、channel-matched latent diffusion、SNR identity envelope、train2017 scale-up split/channel seed、外部 baseline 公平对比契约、SGD-JSCC 作者链/共同协议适配器、canonical common complex-AWGN channel，以及 DeepJSCC split-forward/逐样本 received-latent consistency/backpropagation：

channel-matched latent diffusion 单测覆盖：项目 half-variance `alpha` 公式、AWGN/forward-marginal 代数、单调 reverse schedule、完整坐标 mask、masked MSE、梯度传播和 perfect-epsilon DDIM 恢复。

SNR identity envelope 单测覆盖：smooth strength 的有界单调性、hard cutoff 的严格恒等尾、`g=0/1` codeword 端点，以及可靠性优先的 policy selector。

diffusion-fusion 单测覆盖：B1 六通道权重向九通道 head 的等价展开、control/fusion 精确同参数量（450,115）和冻结 SNR residual gate 查表。

B1 feature-injection 单测覆盖：zero-conv 初始化精确等于 B1、`D-B0=0` control 精确等于 B1、高 SNR zero-envelope 在训练后仍精确等于 B1，以及唯一 3→64 projection 的 trainable parameter budget。

新增 sender routing 单测覆盖默认 anchor fallback、source-anchor mismatch→raw 三路 fallback 和未知策略 fail-closed。

外部 baseline 契约单测覆盖：作者原生结果禁止直接排名、common adapter 必须精确闭合 65,536 实坐标、complex-use CBR 口径不可漂移、semantic new-error 指标不可删除。

SGD-JSCC native adapter 单测覆盖：冻结 smoke contract、outcome claim fail-closed、主 latent 4096 实维度、未知 text transport 阻塞 author-native ranking，以及 real-coordinate ratio / complex-use CBR 分离。

SGD-JSCC common adapter 单测覆盖：精确 rate closure、ASCII/UTF-8 packet round-trip、UTF-8 安全截断、CRC 擦除失败处理、R21 多数表决和 outcome/rate tamper fail-closed。

external common channel 单测覆盖：SHA seed 的条件区分与稳定性、CPU noise 重现、复 AWGN 每实坐标 half-variance 以及 real-coordinate 到 complex-use CBR 转换。

S31 strong-JSCC 单测覆盖：默认模型原生 `19,712` 实符号与 25M--45M 参数合同、小模型前向/反向和单位功率、项目 half-variance AWGN 公式，以及错误图像/SNR batch 的 fail-closed 行为。

```bash
PYTHONPATH=src .venv-sgdjscc/bin/pytest -q tests/test_strong_jscc.py
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py' -v
```
