# data

存放数据集说明或本地数据路径指针。

不要把大型数据集直接提交到仓库。后续可以在这里记录：

- CIFAR-10 下载位置
- Kodak 数据集位置
- ImageNet 子集位置
- 数据预处理脚本和版本

## 当前数据

### CIFAR-10

- 路径：`data/cifar10/`
- 下载方式：`torchvision.datasets.CIFAR10(download=True)`
- 用途：S1 DeepJSCC sanity baseline；不作为 diffusion 主实验数据集
- 当前正式 subset：`outputs/EXP-S1-001/subset_indices.json`

### COCO2017

- 目标路径：`data/coco/train2017/`, `data/coco/val2017/`, `data/coco/annotations/`
- 用途：高分辨率 DeepJSCC 重训和 diffusion 主实验
- 预处理：训练使用 random resized crop 到 `256x256`，验证使用 resize + center crop 到 `256x256`
- 当前状态：`train2017` 已下载并解压，含 118287 张图片；`val2017` 已下载并解压，含 5000 张图片；官方 `annotations_trainval2017.zip` 已下载、`unzip -t` 验证通过并解压出 captions/instances/keypoints JSON；当前机器已验证可用 RTX 4090 D 和 CUDA PyTorch
- annotations 来源：`http://images.cocodataset.org/annotations/annotations_trainval2017.zip`，大小 252907541 bytes；下载时按项目规则清空代理变量并使用服务器直连
- 注意：`data/coco/train2017.zip.possibly_corrupt_20260630_2046` 是早期双进程下载风险 partial，已改名保留，不用于训练

### COCO2017 val pilot split

- 路径：`data/coco_val_split/train/`, `data/coco_val_split/val/`
- 来源：`data/coco/val2017/`
- 切分：seed 42，4500 train + 500 val，不重叠
- manifest：`data/coco_val_split/split_manifest.json`
- 用途：在 `train2017` 下载完成前，先训练非正式高分辨率 DeepJSCC pilot checkpoint

### Imagenette2-320

- 目标路径：`data/imagenette/`
- 来源：`https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz`
- 用途：可作为带分类标签的高分辨率语义 pilot 备选
- 当前状态：官方包可 `wget --no-proxy` 直连，但实测较慢；当前仅保留 partial 文件

### Kodak

- 目标路径：`data/kodak/`
- 用途：视觉质量补充测试和样例展示
- 当前状态：尚未下载
