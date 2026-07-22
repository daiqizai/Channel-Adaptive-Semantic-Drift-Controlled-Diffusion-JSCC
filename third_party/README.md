# third_party

存放外部代码仓库或其路径说明。

不要直接改第三方代码。需要修改时优先：

1. 记录原始 commit。
2. 在本项目代码中写 adapter。
3. 如果必须改动，保留 patch 说明。

## 已接入

### Deep-JSCC-PyTorch

- 路径：`third_party/Deep-JSCC-PyTorch`
- 来源：https://github.com/chunbaobao/Deep-JSCC-PyTorch
- clone 方式：`git clone --depth 1`
- commit：`2665e0dc6d8bf216daf9442c5d6e5d69c5ad2f06`
- 用途：阶段1 DeepJSCC baseline 候选
- 当前处理方式：不直接修改第三方源码，通过 `src/cadsd_jscc/deepjscc_adapter.py` 包装加载

### SGDJSCC

- 路径：`third_party/SGDJSCC`
- 来源：https://github.com/MauroZMJ/SGDJSCC
- clone 方式：清空代理变量后 `git clone --depth 1`
- commit：`2188acc0dd2805355d3d0d2e478cbc27b46b4da5`
- 用途：外部 diffusion-JSCC 第一复现候选；先做作者原生轨道，再做 matched-total-rate common contract
- 许可：仓库未发现 LICENSE/COPYING，GitHub metadata 无可识别 license
- 当前处理方式：第三方源码只读；在本项目侧写 adapter，不复制/修改其源码
- 权重状态：作者 4 个 checkpoint（合计 `2,930,865,634` bytes）、精确 BLIP2 两分片和 OpenAI CLIP ViT-L/14 均已下载并校验；严格物理码率与文本开销仍由项目侧 adapter 单独审计
- 审计报告：`reports/external_method_comparison_schedule_2026-07-14.md`

### DiffJSCC

- 路径：`third_party/DiffJSCC`
- 来源：https://github.com/mingyuyng/DiffJSCC
- 固定 commit：`13aeb62451b872ce41ceba132c9c30a9ca172c53`
- 用途：在冻结 S20/S28 Imagenette 总体上运行官方 OpenImage C16 diffusion-JSCC 外部对照
- 许可边界：源码仓库根目录未发现 LICENSE/COPYING；Hugging Face checkpoint card 标 Apache-2.0，不能反向当作源码许可
- 权重：`Mingyuyang/DiffJSCC-OpenImage-CBR-1-96` 的 `model.ckpt`（`9,859,655,693` bytes）；作者保存逻辑主动排除 `blip_model.*`，因此另需固定 `Salesforce/blip2-opt-2.7b` 两个 base safetensors（合计 `14,979,207,136` bytes）
- 运行依赖：官方 `open-clip-torch==2.24.0` release 源码、`transformers==4.51.1` 独立目录和 `.venv-sgdjscc` 的固定执行包；所有版本/哈希写在 `configs/s30_diffjscc_external_comparison.yaml`
- 当前处理方式：第三方算法源码只读；canonical AWGN、严格码率、评估和 legacy API shim 全部位于项目侧 `s30_diffjscc_*` 脚本
- 当前状态：DiffJSCC `model.ckpt` 与精确 base BLIP2 两分片均完整下载并通过官方 SHA-256；preflight/checkpoint audit/preload/smoke/首 seed/full 960 行全部 PASS。最终为 fidelity/perception Pareto，详细结果见 `reports/diffjscc_external_comparison_stage_result_2026-07-21.md`；历史资产与输出禁止覆盖
- 预注册：`reports/diffjscc_external_comparison_preregistration_2026-07-21.md`

### SwinJSCC

- 路径：`third_party/SwinJSCC`
- 来源：https://github.com/semcomm/SwinJSCC
- 固定 commit：`a6d0e6da53548976acbe9317839a077ef31f190f`
- 获取：清空全部代理变量后从 GitHub codeload 下载该 commit 的 `17,887` bytes tarball；tarball SHA-256=`3f837eefbc9e62431be39e3dd58cdbd0102e4c6252f81320e3db455e97821688`
- 许可：仓库根目录未发现 LICENSE/COPYING，复现实验可读取，但不得声称拥有再分发许可
- 用途：S34A `16,384-real` fixed-rate、SA-only Transformer JSCC 外部骨干比较
- 当前处理方式：官方源码保持逐文件 hash 相同，不直接修改；逐图 SNR、逐图功率、canonical AWGN 和训练记录由 `src/cadsd_jscc/swinjscc_adapter.py` 实现
