# paper idea1b

> **归档状态（2026-08-03 起）：** 本工作区的 Gate A0/A1 与历史计划保留为证据资产；`reports/METHOD_TERMINATION_REPORT_2026-08-03.md` 已 supersede 下方所有“未授权/待解锁/下一步”表述。不得启动 A2、S35R-P1、S36、Swin extension 或 refiner 训练。未来工作只能作为独立新课题另设问题、预算、预注册和 ID。

本目录是“冻结 COCO-S33 + 轻量 receiver-side refiner + 代价/质量/可靠性刻画”论文的历史专属工作区。

边界：

- 这里只新增本篇论文的 `configs/`、`scripts/`、`data/`、`outputs/` 和局部 `PROGRESS.md`。
- 根目录 `src/` 是共享实现唯一来源；本目录脚本通过 import 使用它，不复制模型代码。
- S33 checkpoint、canonical noise 实现、S34D harness/结果和所有带 SHA 的审计资产保持原位，只在配置中引用。
- Imagenette 继续承担监督 reliability；official validation 继续封存。
- Gate A0 与 A1 判别式部分已完成。A2 DiffJSCC、SGD、refiner训练和official validation仍未由本工作区解锁。

Gate A0：

```bash
# 下载命令必须清空代理，走服务器直连。
bash paper_idea1b/scripts/download_gate_a0_data.sh

PYTHONPATH=src python3 paper_idea1b/scripts/prepare_gate_a0.py \
  --config paper_idea1b/configs/gate_a0_benchmark_setup.yaml

PYTHONPATH=src python3 paper_idea1b/scripts/metric_identity_sanity.py \
  --config paper_idea1b/configs/gate_a0_benchmark_setup.yaml
```

正式输出目录固定为：

`paper_idea1b/outputs/GATE-A0-BENCHMARK-SETUP-001/`

存在 `STATE.json` 且状态为 `complete` 后禁止覆盖；需要重做时必须使用新的 analysis ID。

Gate A0 的完整中文结果、actual-CBR发现和A1前决策点见 `GATE_A0_RESULT.md`。

Gate A1 判别式主表：

```bash
# 以下是已完成实验的复现入口；现有输出禁止覆盖。
PYTHONPATH=src python3 paper_idea1b/scripts/a1_discriminative_smoke.py \
  --config paper_idea1b/configs/a1_discriminative_benchmark.yaml --device cuda:0

PYTHONPATH=src python3 paper_idea1b/scripts/a1_discriminative_run.py \
  --config paper_idea1b/configs/a1_discriminative_benchmark.yaml \
  --device cuda:0 --dataset kodak --resume

PYTHONPATH=src python3 paper_idea1b/scripts/a1_discriminative_run.py \
  --config paper_idea1b/configs/a1_discriminative_benchmark.yaml \
  --device cuda:0 --dataset clic2020_test --resume

PYTHONPATH=src python3 paper_idea1b/scripts/a1_discriminative_metrics.py \
  --config paper_idea1b/configs/a1_discriminative_benchmark.yaml \
  --device cuda:0 --stage full_reference --resume

PYTHONPATH=src python3 paper_idea1b/scripts/a1_discriminative_metrics.py \
  --config paper_idea1b/configs/a1_discriminative_benchmark.yaml \
  --device cuda:0 --stage distribution --resume

PYTHONPATH=src python3 paper_idea1b/scripts/a1_discriminative_metrics.py \
  --config paper_idea1b/configs/a1_discriminative_benchmark.yaml \
  --stage summarize
```

A1最终结论是S33没有战胜SwinJSCC：Kodak上只对Base-SA的PSNR追平/非劣，面对CM-SA以及在CLIC上面对两臂均劣于。完整中文结果见`A1_DISCRIMINATIVE_RESULT.md`。指标smoke首次OpenCLIP兼容性失败和第二次通过均保留；全量指标支持按键断点续跑。
