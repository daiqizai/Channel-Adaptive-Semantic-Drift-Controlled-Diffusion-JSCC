# paper_idea1b data

本目录只保存本篇论文新增的数据与指标权重，不保存或复制根目录既有模型资产。

Gate A0 本地内容：

- `kodak/`：24张Kodak图；公开mirror archive SHA-256=`44e2569b71dd0b35950ca0b0ddc36cc974d307c6990066147893008940300223`，逐文件通过Kodak官方字节数表核验。
- `clic2020_test/`：官方CLIC2020 Mobile + Professional test共428张；archive SHA-256分别为`2025f07a6c652270e534640de4271feef3b3dd3260beed4ac4821064837aa732`与`857df244fc2bfa5da28d4c66bf0db16ee99bfc79eb807be8afa89cd507852884`。
- `metric_weights/`：DISTS所需VGG16与clean-fid Inception；SHA-256分别为`397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0`与`f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4`。

实际大文件被本目录的 `.gitignore` 排除。下载及校验方式见 `../scripts/download_gate_a0_data.sh` 和 `../configs/gate_a0_benchmark_setup.yaml`。
