# DiffJSCC 码率可调性审计

日期：2026-07-23
结论：**方法架构可以通过重训调到`CBR=1/24`，但现有官方checkpoint不能在推理时无损“拨档”到`1/24`。**

## 为什么当前是1/96

现有官方OpenImage checkpoint配置为：

- `n_downsample=4`，空间边长缩小16倍；
- `C_channel=16`；
- paired-real AWGN，即两个real coordinates对应一个complex channel use。

对原生大图`H×W`，channel latent为`16×H/16×W/16`，因此：

```text
real symbols = 16HW/256 = HW/16
complex uses = HW/32
CBR = (HW/32)/(3HW) = 1/96
```

checkpoint目录名、内嵌配置和作者README均明确标为`DiffJSCC-OpenImage-CBR-1-96`。作者公开的OpenImage模型只有`C_channel=4 (1/384)`与`C_channel=16 (1/96)`，没有`1/24`模型。

## 调到1/24需要什么

保持四级下采样时，CBR与`C_channel`线性成正比。目标`1/24`是当前`1/96`的4倍，因此必须把：

```text
C_channel: 16 -> 64
```

这会同时改变JSCC encoder最后投影层和decoder第一输入层的权重形状。现有C16 checkpoint无法strict load到C64，不能靠修改YAML或在推理时设置一个rate参数解决。

ControlNet/diffusion主体接收的是JSCC RGB重建和SNR，而不是直接接收C-channel latent，所以理论上可以：

1. 用官方trainer训练一个C64 JSCC前端；
2. 将它接入现有生成链；
3. 至少重新做matched-distribution的ControlNet适配/验证，稳妥方案是按官方两阶段合同重训。

只重训JSCC、不适配生成阶段，虽然代码可能运行，但输入重建分布发生变化，不能作为严谨的“官方DiffJSCC@1/24”主对比。

## 哪些捷径不算真等码率

- 在C16 latent后补零：没有增加信息，不能称使用了1/24有效码率。
- 重复发送同一latent四次再平均：相当于额外重复码/约6 dB合并增益，不是作者DiffJSCC的C64码率点。
- 把大图切成256 tile，让每个tile被官方入口放大到512：通信量可凑到S33附近，但切断全局上下文，违反本项目已冻结的原生整图公平合同。
- 发送无用padding或重复符号只为“凑预算”：项目合同明确禁止。

## 对论文主对比的判断

- **现成官方权重口径：不能做S33 vs DiffJSCC真·1/24等码率胜负。** 应报告`S33@actual CBR`与`DiffJSCC-C16@1/96`的rate-quality Pareto。
- **投入重训口径：技术上可行。** 需要新建C64前端并做生成阶段matched adaptation；这是新的长实验，不属于当前A2轻量评测。
- 若论文必须把“S33 vs DiffJSCC@1/24”作为核心主表，唯一严谨路线是训练并冻结一个明确标为`DiffJSCC-C64-reproduction`的项目复现臂，同时如实说明它不是作者发布checkpoint。

## 只读证据

- `third_party/DiffJSCC/checkpoints/DiffJSCC-OpenImage-CBR-1-96/config.yaml`
- `third_party/DiffJSCC/README.md`
- `third_party/DiffJSCC/model/deepjscc_cnn.py`
- 官方源码commit：`13aeb62451b872ce41ceba132c9c30a9ca172c53`
