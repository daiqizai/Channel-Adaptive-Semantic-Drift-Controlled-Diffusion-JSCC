# 候选方向二：CVaR 尾部风险 JSCC 实验任务书

> 用途：将本文件直接交给 Codex，让其在**现有项目代码基础上**完成候选方向二的最小诊断实验，并在诊断通过后实现 CVaR 尾部风险训练。
>
> 核心原则：**先证明问题存在，再训练新模型。不要一上来重构仓库或跑完整训练。**

---

## 0. Codex 总任务

请在当前仓库中完成以下工作：

1. 审计现有 JSCC/生成式 JSCC 代码，找出：
   - 当前基础模型或 `M0` 模型；
   - 训练入口；
   - 评测入口；
   - 信道模块；
   - 数据集和验证集；
   - 已有 checkpoint；
   - 已有 PSNR、MS-SSIM、LPIPS 等指标实现。
2. 基于现有 checkpoint，先执行**条件尾部风险诊断实验**。
3. 只有当诊断结果显示明显尾部风险时，才实现并训练：
   - 普通均值训练对照；
   - 重复信道采样均值训练对照；
   - CVaR-10%；
   - CVaR-20%。
4. 输出代码、配置、运行命令、CSV、图表和结论报告。
5. 不破坏现有训练、评测和结果复现流程。

---

# 1. 研究问题

当前大多数 Deep JSCC 模型优化平均失真：

\[
\min_\theta \mathbb{E}_{x,h,n}[D(x,\hat{x})]
\]

这可能掩盖深衰落或极端噪声条件下的失败。需要回答：

> 对同一张源图像，重复采样不同信道实现时，平均训练模型的最差 10% 表现，是否明显差于其中位数和平均表现？

如果答案是否定的，则停止该方向。

如果答案是肯定的，则进一步训练风险敏感模型：

\[
\mathcal{L}
=
(1-\lambda)\mathbb{E}[D]
+
\lambda\,\mathrm{CVaR}_{\alpha}(D)
\]

其中失真 \(D\) 越大越差，建议首先测试：

- \(\alpha=0.1\)：最差 10% 信道实现；
- \(\alpha=0.2\)：最差 20% 信道实现。

---

# 2. 工程约束

## 2.1 必须遵守

- 优先复用当前仓库已有模型、数据集、checkpoint 和指标。
- 不新建一套与现有工程平行的大型框架。
- 不修改现有默认训练结果。
- 新功能应通过独立配置或独立入口启用。
- 所有实验固定随机种子并保存完整配置。
- 所有输出均写入新的实验目录，例如：

```text
outputs/cvar_tail_risk/
```

- 先运行小规模 dry-run，再运行正式诊断。
- 若当前模型只支持 AWGN，先保留 AWGN 原逻辑，再以最小侵入方式增加 Rayleigh。
- 不要默默改变信道定义、功率归一化、SNR 定义或 CSI 假设。
- 不要在诊断完成前启动长时间 CVaR 训练。
- 不要为了实现该实验重构整个仓库。

## 2.2 硬件约束

目标环境：

```text
单卡 NVIDIA RTX 4090D
显存 24 GB
```

若默认参数显存不足，应优先减小：

1. 每次参与反向传播的不同图像数 \(B\)；
2. 图像分辨率或 crop；
3. 同时采样的信道实现数 \(M\)。

不要优先删除关键对照组。

---

# 3. 第一阶段：仓库审计

先阅读仓库，不要立即改代码。

请创建：

```text
outputs/cvar_tail_risk/REPO_AUDIT.md
```

其中必须写明：

## 3.1 当前模型

- 模型类名；
- 模型文件路径；
- 编码器和解码器入口；
- 当前是否存在基础 JSCC 输出、生成式精修输出或 `M0` 输出；
- 当前 checkpoint 路径和加载方式；
- 当前模型训练损失。

## 3.2 当前信道

- 信道实现文件路径；
- 当前支持 AWGN、Rayleigh 或其他信道中的哪些；
- SNR 的定义；
- 发送信号功率如何归一化；
- Rayleigh 是否为：
  - fast fading；
  - block fading；
  - 每图一个系数；
  - 每特征一个系数；
- 接收端是否已知 CSI；
- 是否进行了均衡。

## 3.3 当前数据和指标

- 数据集名称；
- 验证集入口；
- 图像预处理；
- PSNR、MS-SSIM、LPIPS 的已有实现；
- 当前评测脚本；
- 当前结果保存格式。

## 3.4 最小改动方案

列出计划新增和修改的文件。格式示例：

```text
新增:
- experiments/cvar_tail_risk/diagnose_tail_risk.py
- experiments/cvar_tail_risk/train_cvar.py
- experiments/cvar_tail_risk/metrics.py
- configs/cvar_tail_risk/diagnostic.yaml

修改:
- channel/rayleigh.py
  原因：增加可显式传入 realization seed 的接口
```

审计完成后再开始代码实现。

---

# 4. 第二阶段：尾部风险诊断

## 4.1 诊断目标

加载现有普通平均损失训练的 checkpoint。

对每张固定源图像，重复采样多个独立信道实现：

\[
D_{i,m}=D(x_i,\hat{x}_{i,m})
\]

其中：

- \(i=1,\dots,N\)：源图像编号；
- \(m=1,\dots,M\)：同一图像的独立信道实现。

必须区分：

- 图像本身难；
- 信道随机性导致的尾部失败。

因此不能简单把全部样本混在一起取最差 10%。

---

## 4.2 默认诊断配置

先根据仓库实际情况调整，但建议默认值如下：

```yaml
num_images: 200
num_channel_realizations: 64
channel: rayleigh
csi: known
equalization: enabled
snr_db_list: [-2, 0, 2, 4, 6, 8, 10]
seed: 20260731
batch_size_images: 4
save_reconstructions: true
save_worst_examples_per_snr: 16
```

如果正式运行过慢：

```yaml
num_images: 100
num_channel_realizations: 32
```

但 dry-run 必须先使用：

```yaml
num_images: 4
num_channel_realizations: 4
snr_db_list: [0, 6]
```

---

## 4.3 Rayleigh 信道定义

优先采用当前仓库已有实现。

若仓库没有 Rayleigh，请实现清晰、可复现的最小版本，并在报告中说明假设。

建议第一版采用：

\[
y=hx+n
\]

其中：

\[
h\sim\mathcal{CN}(0,1)
\]

接收端已知 \(h\)，执行：

\[
\tilde y = \frac{h^*}{|h|^2+\epsilon}y
\]

注意：

- 不要同时引入 CSI 估计误差；
- 不要同时引入反馈、速率自适应或重传；
- 先单独研究信道随机性导致的尾部风险；
- 确保噪声方差与当前仓库 SNR 定义一致；
- 记录每次 realization 的 \(|h|^2\)。

若当前模型在信道中直接传输高维 latent，请明确：

- \(h\) 是按整张图共享；
- 按 channel 共享；
- 还是按元素独立。

第一版优先选择 **block fading**，即一张图像或一个样本共享一个复衰落系数，以便形成明显且可解释的深衰落尾部。

---

## 4.4 每次样本必须记录的数据

输出：

```text
outputs/cvar_tail_risk/diagnostic_samples.csv
```

每行至少包含：

```text
image_id
image_path_or_index
snr_db
realization_id
channel_seed
h_real
h_imag
h_power
mse
psnr
ms_ssim
lpips
```

如果某个指标当前仓库不存在，可以先留空，但：

- MSE 必须有；
- PSNR 必须有；
- LPIPS 若仓库已有则必须复用；
- 不要为了一个非关键指标阻塞诊断。

---

## 4.5 条件尾部统计

对于每张图像、每个 SNR，基于 \(M\) 次信道实现分别计算：

- mean PSNR；
- median PSNR；
- 10th-percentile PSNR；
- 5th-percentile PSNR；
- worst-10%-mean PSNR；
- mean MSE；
- CVaR-10% MSE；
- CVaR-20% MSE；
- outage probability；
- PSNR 标准差；
- 与 \(|h|^2\) 的相关关系。

注意：

- PSNR 越大越好，因此 PSNR 尾部使用最低 10%；
- MSE/LPIPS 越大越差，因此 CVaR 使用最高 10%。

建议 outage 阈值至少测试：

```text
PSNR < 20 dB
PSNR < 22 dB
PSNR < 24 dB
```

若当前任务的典型 PSNR 与这些阈值不匹配，可根据已有 baseline 结果调整，但必须在报告中说明。

---

## 4.6 Empirical CVaR 实现

请添加通用函数，例如：

```python
from __future__ import annotations

import math
import torch


def empirical_upper_cvar(
    distortion: torch.Tensor,
    tail_fraction: float,
    dim: int = -1,
) -> torch.Tensor:
    """
    计算经验上尾 CVaR。

    distortion 越大代表越差。
    例如 tail_fraction=0.1 表示取最差 10% 的均值。
    """
    if distortion.numel() == 0:
        raise ValueError("distortion must not be empty")

    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")

    count = distortion.shape[dim]
    tail_count = max(1, math.ceil(count * tail_fraction))

    worst_values = torch.topk(
        distortion,
        k=tail_count,
        dim=dim,
        largest=True,
        sorted=False,
    ).values

    return worst_values.mean(dim=dim)
```

再实现 PSNR 的低尾均值函数，或转换为失真后统一处理。

必须添加简单单元测试：

- 全部数值相同时 CVaR 等于该数值；
- `[1,2,3,4,5]` 的 worst-20% 为 5；
- worst-40% 为 `(5+4)/2`；
- 非法 tail fraction 抛出异常。

---

# 5. 诊断输出

## 5.1 汇总 CSV

生成：

```text
outputs/cvar_tail_risk/diagnostic_summary.csv
```

每行对应：

```text
snr_db
mean_psnr
median_psnr
p10_psnr
p05_psnr
worst10_mean_psnr
mean_mse
cvar10_mse
cvar20_mse
outage_psnr20
outage_psnr22
outage_psnr24
```

另外建议同时保存按图像统计：

```text
outputs/cvar_tail_risk/per_image_tail_stats.csv
```

---

## 5.2 必须生成的图

每张图单独保存，不使用 subplot。

### 图 1：平均和尾部 PSNR

```text
mean_psnr
median_psnr
p10_psnr
worst10_mean_psnr
```

随 SNR 变化。

保存为：

```text
plots/psnr_mean_vs_tail.png
```

### 图 2：Outage probability

不同 PSNR 阈值下的 outage probability 随 SNR 变化。

```text
plots/outage_probability.png
```

### 图 3：衰落强度与失真

横轴：

\[
|h|^2
\]

纵轴：

```text
MSE 或 PSNR
```

保存为：

```text
plots/fading_power_vs_distortion.png
```

### 图 4：每个 SNR 下 PSNR 分布

可使用箱线图或经验 CDF。

```text
plots/psnr_distribution_by_snr.png
```

### 图 5：最差样本可视化

每个代表性 SNR 保存若干组：

```text
original
median-realization reconstruction
worst-10% reconstruction
worst reconstruction
```

文件名中包含：

```text
image_id
snr
realization_id
h_power
psnr
```

---

# 6. 诊断报告与 kill-switch

创建：

```text
outputs/cvar_tail_risk/TAIL_RISK_DIAGNOSTIC.md
```

必须包含：

## 6.1 实验配置

- checkpoint；
- 数据集；
- 图像数量；
- 每图 realization 数；
- 信道定义；
- CSI 假设；
- SNR 列表；
- 指标；
- 运行命令；
- commit hash。

## 6.2 关键结果

至少报告：

\[
\Delta_{\text{tail}}
=
\text{median PSNR}
-
\text{p10 PSNR}
\]

以及：

\[
\Delta_{\text{worst10}}
=
\text{mean PSNR}
-
\text{worst10-mean PSNR}
\]

## 6.3 判定规则

### 继续 CVaR 训练

满足以下多数条件：

- 在至少两个有实际意义的 SNR 点：
  - median PSNR 与 p10 PSNR 相差至少约 2 dB；
- worst-10% 的重建出现明显质量崩溃；
- outage probability 不可忽略；
- 现象在不同图像和 seed 下稳定；
- 低 \(|h|^2\) 与高失真存在明确关系。

### 停止该方向

出现以下情况：

- p10 PSNR 与 median PSNR 差距通常小于约 1 dB；
- 最差 10% 与平均值差距很小；
- 所谓“尾部”主要由少数复杂图片，而不是信道随机性造成；
- 现象只在完全不可用的极低 SNR 出现；
- 当前模型已经隐式实现了足够强的信道自适应。

报告最后必须输出明确结论：

```text
Decision: GO
```

或：

```text
Decision: NO-GO
```

不要只描述结果而不做判断。

---

# 7. 第三阶段：仅在 GO 后实现 CVaR 训练

若诊断结果为 `NO-GO`，停止，不要训练 CVaR 模型。

若诊断结果为 `GO`，继续以下工作。

---

## 7.1 必须比较的四个训练组

### B0：原始 Mean baseline

现有 checkpoint 或按原始配置训练：

\[
\mathcal L_{\text{mean}}
=
\mathbb E[D]
\]

### B1：Repeated-fading mean control

每张图像重复采样 \(M\) 个信道实现，但仍然对全部失真取平均：

\[
\mathcal L_{\text{repeat-mean}}
=
\frac{1}{BM}
\sum_{i=1}^{B}\sum_{m=1}^{M}D_{i,m}
\]

这组必须存在，用于排除：

> 改善只是因为每张图像看到了更多信道样本，而不是 CVaR 本身有效。

### B2：CVaR-10%

\[
\mathcal L
=
(1-\lambda)\mathcal L_{\text{mean}}
+
\lambda\mathcal L_{\mathrm{CVaR10}}
\]

### B3：CVaR-20%

同理使用最差 20%。

可选增加：

### B4：Worst-one

每张图只使用最差一个 realization，作为极端风险训练对照。

---

## 7.2 训练 batch 组织

每个训练 step：

1. 采样 \(B\) 张不同图像；
2. 每张图像复制 \(M\) 份；
3. 为每份复制采样独立信道 realization；
4. 得到失真矩阵：

\[
D\in\mathbb R^{B\times M}
\]

5. 对每张图像分别计算 CVaR；
6. 再在 \(B\) 张图像上取平均。

默认建议：

```yaml
num_source_images_per_step: 4
num_channel_realizations_per_image: 4
tail_fraction: 0.10
risk_weight: 0.5
```

显存允许时使用：

```yaml
B: 4
M: 8
```

不要把全部 \(B\times M\) 样本混合后直接取全局最差 10%，否则会把图像内容难度和信道尾部混在一起。

---

## 7.3 CVaR 训练损失

建议实现：

```python
from __future__ import annotations

import torch


def conditional_cvar_objective(
    distortion: torch.Tensor,
    tail_fraction: float,
    risk_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    distortion shape: [B, M]
    B: 不同源图像数量
    M: 每张图像的独立信道 realization 数量
    """
    if distortion.ndim != 2:
        raise ValueError(
            f"Expected distortion with shape [B, M], got {distortion.shape}"
        )

    if not 0.0 <= risk_weight <= 1.0:
        raise ValueError("risk_weight must be in [0, 1]")

    mean_loss = distortion.mean()

    per_image_cvar = empirical_upper_cvar(
        distortion,
        tail_fraction=tail_fraction,
        dim=1,
    )
    cvar_loss = per_image_cvar.mean()

    total_loss = (
        (1.0 - risk_weight) * mean_loss
        + risk_weight * cvar_loss
    )

    stats = {
        "loss_total": total_loss.detach(),
        "loss_mean": mean_loss.detach(),
        "loss_cvar": cvar_loss.detach(),
    }
    return total_loss, stats
```

第一轮超参数只扫描：

```text
risk_weight λ ∈ {0.25, 0.5, 0.75}
tail_fraction α ∈ {0.1, 0.2}
```

不要做大规模网格搜索。

---

# 8. 训练公平性

所有主要方法必须控制：

- 相同训练数据；
- 相同模型结构；
- 相同优化器；
- 相同学习率计划；
- 相同训练 step 数；
- 相同 \(B\times M\) forward 数量；
- 相同评测 realization；
- 相同随机种子集合；
- 相同 SNR 分布；
- 相同 checkpoint 选择规则。

特别注意：

B0 普通 baseline 与 B1/B2/B3 的计算量可能不同。正式比较时应同时报告：

- 等训练 step；
- 等总信道 forward 数；

至少选一种作为主公平口径，另一种作为补充。

---

# 9. CVaR 正式评测

使用与诊断相同的条件尾部评测方式，但增加：

- 未见过的随机种子；
- 未见过的 SNR 点；
- 如果可行，增加不同 Rayleigh 参数或 Rician 信道；
- 平均性能；
- 尾部性能；
- outage；
- LPIPS 尾部；
- 推理计算量不变的证明。

核心对比表：

| Model | Mean PSNR ↑ | P10 PSNR ↑ | Worst-10% PSNR ↑ | CVaR-10 MSE ↓ | Outage ↓ |
|---|---:|---:|---:|---:|---:|
| Mean baseline | | | | | |
| Repeated-mean | | | | | |
| CVaR-10 | | | | | |
| CVaR-20 | | | | | |
| Worst-one | | | | | |

---

# 10. CVaR 成功标准

候选方向二值得继续写论文，需要至少满足：

1. CVaR 模型相对 `Repeated-fading mean control`：
   - p10 PSNR 提升接近或超过 1 dB；
   - 或 outage probability 有明显相对下降；
2. 平均 PSNR 损失不超过约 0.2–0.3 dB；
3. 改善不只出现在一个极低 SNR 点；
4. 在未见信道种子下仍成立；
5. 不是通过增加模型参数或推理成本获得；
6. CVaR-10 与 CVaR-20 呈现可解释的风险—平均性能权衡。

如果只比原始 baseline 好、但不比 `Repeated-fading mean control` 好，则不能证明 CVaR 是有效贡献。

---

# 11. 代码组织建议

请尽量适配当前仓库结构。若仓库没有明确组织，可使用：

```text
experiments/
└── cvar_tail_risk/
    ├── README.md
    ├── diagnose_tail_risk.py
    ├── train_cvar.py
    ├── evaluate_cvar.py
    ├── tail_metrics.py
    ├── plotting.py
    └── tests/
        └── test_tail_metrics.py

configs/
└── cvar_tail_risk/
    ├── diagnostic.yaml
    ├── repeat_mean.yaml
    ├── cvar10.yaml
    └── cvar20.yaml
```

如果当前仓库已有统一配置系统，应接入已有系统，不要另起炉灶。

---

# 12. 命令要求

在：

```text
experiments/cvar_tail_risk/README.md
```

中给出真实可执行命令。

至少包括：

## 12.1 环境检查

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name())"
```

## 12.2 Dry-run

```bash
python <diagnostic_entry> \
  --config <diagnostic_config> \
  --dry-run
```

## 12.3 正式诊断

```bash
python <diagnostic_entry> \
  --config <diagnostic_config>
```

## 12.4 CVaR 训练

仅在 GO 后：

```bash
python <train_entry> \
  --config <cvar10_config>
```

## 12.5 正式评测

```bash
python <eval_entry> \
  --checkpoint <checkpoint_path> \
  --config <evaluation_config>
```

命令必须使用仓库实际文件名，不能保留这里的占位符。

---

# 13. 日志要求

训练和评测日志至少记录：

```text
epoch
global_step
learning_rate
loss_total
loss_mean
loss_cvar
mean_psnr
p10_psnr
cvar10_mse
outage_probability
gpu_memory_allocated
elapsed_time
```

同时保存：

- 完整配置；
- Git commit hash；
- 环境依赖；
- checkpoint；
- 最佳 checkpoint 选择标准；
- 最终结果 CSV。

---

# 14. 验收清单

## 阶段 A：诊断

- [ ] 已完成 `REPO_AUDIT.md`
- [ ] 已完成可复现 Rayleigh realization 采样
- [ ] 同一图像可重复采样 \(M\) 个独立信道实现
- [ ] 已保存逐样本 CSV
- [ ] 已保存按图像尾部统计
- [ ] 已生成 5 类诊断图
- [ ] 已保存最差重建案例
- [ ] 已完成 `TAIL_RISK_DIAGNOSTIC.md`
- [ ] 已输出明确 `GO` 或 `NO-GO`
- [ ] 未在诊断前启动 CVaR 长训练

## 阶段 B：训练

- [ ] 已实现 `Repeated-fading mean control`
- [ ] 已实现 CVaR-10
- [ ] 已实现 CVaR-20
- [ ] CVaR 按每张源图像分别计算
- [ ] 已完成 tail metric 单元测试
- [ ] 已保证主要实验计算预算公平
- [ ] 已输出平均—尾部风险权衡
- [ ] 已给出是否值得继续写论文的结论

---

# 15. Codex 最终回复格式

完成后请按以下结构回复：

## 1. 仓库识别结果

说明当前模型、训练入口、评测入口、信道模块、数据集和 checkpoint。

## 2. 修改文件

逐一列出新增和修改文件，并说明用途。

## 3. 可执行命令

给出从环境检查、dry-run、诊断到正式训练和评测的完整命令。

## 4. 诊断结果

给出：

- mean PSNR；
- median PSNR；
- p10 PSNR；
- worst-10% PSNR；
- outage；
- `GO` 或 `NO-GO`。

## 5. CVaR 结果

仅在 GO 后给出各对照组结果。

## 6. 已知问题

如实说明：

- 未实现部分；
- 运行失败；
- 数据不足；
- 显存限制；
- 指标缺失；
- 结果尚不足以支持结论的地方。

---

# 16. 给 Codex 的执行优先级

严格按以下顺序执行：

```text
P0：仓库审计
P1：4 图 × 4 realization 的 dry-run
P2：尾部风险正式诊断
P3：输出 GO / NO-GO
P4：仅在 GO 后实现 repeated-mean 与 CVaR
P5：正式训练和公平评测
```

不要跳过 P2，直接做 P4。

---

# 17. 最重要的研究纪律

本任务不是为了证明 CVaR 一定有效，而是为了尽快、可靠地判断它是否值得投入。

以下结果都可以接受：

```text
GO：现有模型有明显衰落尾部，CVaR 值得继续。
```

或：

```text
NO-GO：现有模型尾部问题不足，不再投入。
```

严禁：

- 只报告平均 PSNR；
- 把不同图像的难度当成信道尾部；
- 只和原始 baseline 比，不和 repeated-fading mean 比；
- 看到结果不理想后修改统计口径；
- 在没有诊断证据时包装成完整论文方向。
