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

