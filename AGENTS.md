# AI 协作规则

本项目可能由多个 AI 对话共同开发，本地文件是唯一共享记忆。

所有参与本仓库的 AI agent 都必须遵守 `PROJECT.md` 中定义的研究边界。
`MILESTONES.md` 是本项目的收敛约束，新增方法或实验前必须确认不会破坏其中的最小闭环。

## 每次开始任务必须先读

1. `AGENTS.md`
2. `PROJECT.md`
3. `MILESTONES.md`
4. `PROGRESS.md`
5. `EXPERIMENTS.md`
6. `LITERATURE.md`
7. `README.md`

## 每次结束任务必须更新

- 更新 `PROGRESS.md`。
- 若跑了实验，更新 `EXPERIMENTS.md`。
- 若发现新相关工作，更新 `LITERATURE.md`。
- 若改了运行方式，更新 `README.md`。

## 网络与流量规则

- 大模型、大数据集、CUDA/PyTorch 等大文件下载，默认必须走服务器直连流量，不走用户本机代理流量。
- 运行大下载命令前必须检查代理环境变量，若存在 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 或对应小写变量，默认用清空代理变量的命令执行。
- 只有用户明确说可以使用代理/本机流量时，才允许通过用户代理下载大文件。
- 小规模 metadata/API 查询也应尽量避免不必要联网；任何联网下载都要先说明来源、规模和是否直连。

## 禁止

- 虚构实验结果。
- 把计划写成已完成。
- 覆盖旧实验目录。
- 擅自改项目主线。
- 在 AWGN 最小闭环完成前扩展到复杂新主线。
- 只看视觉效果不看 semantic drift。
- 不记录失败实验。
- 覆盖其他 agent 的工作前不检查当前文件状态。

## 研究纪律

- 把 semantic drift 当作核心失败模式，而不是附带讨论。
- 如果 diffusion 输出视觉上更好但语义错误，不能把它算作真正提升。
- baseline 必须清楚、可比较。
- 负结果也要记录；本项目需要知道哪些方法会失控。
- 不满足 `MILESTONES.md` 成功判据的视觉增强不能包装成主要贡献。

## 代码纪律

- 优先写可复现的小脚本，不依赖只有 notebook 才能跑的流程。
- 配置必须显式记录：数据集、SNR、信道模型、diffusion strength、guidance 方法、随机种子。
- 保存指标和生成样例时，要带足够元数据，保证后续能复现。
