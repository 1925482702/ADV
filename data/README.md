# 双模态数据处理模块

## 概述

这个模块提供了双模态（RGB + IR）数据加载和增强功能，支持YOLO格式的目标检测任务。

## 主要功能

### 1. 双模态数据集 (DualModalityDataset)

- 自动加载RGB和IR图像对
- 支持YOLO格式标签
- 支持数据缓存
- 支持多种增强方法

### 2. 双模态增强类

- **PairedLetterBox**: 同时调整RGB和IR图像的大小
- **PairedMosaic**: 将4张图像拼成一张（RGB和IR同步）
- **PairedMixUp**: 混合两张图像（RGB和IR同步）
- **PairedRandomFlip**: 随机翻转（水平和垂直）
- **PairedRandomPerspective**: 随机透视变换
- **PairedAlbumentations**: Albumentations增强
- **PairedCopyPaste**: CopyPaste增强

## 快速开始

```python
from data import DualModalityDataset

# 配置超参数
hyp = {
    'mosaic': 1.0,      # Mosaic增强概率
    'mixup': 0.0,       # MixUp增强概率
    'fliplr': 0.5,      # 水平翻转概率
    'flipud': 0.0,      # 垂直翻转概率
    'degrees': 0.0,     # 旋转角度
    'translate': 0.1,   # 平移比例
    'scale': 0.5,       # 缩放比例
    'shear': 0.0,       # 剪切角度
    'perspective': 0.0, # 透视变换
}

# 数据配置
data_config = {
    'names': ['car', 'person', 'bicycle']
}

# 创建数据集
dataset = DualModalityDataset(
    img_path='path/to/images/train',
    imgsz=640,
    cache=False,
    augment=True,
    hyp=hyp,
    data=data_config
)

# 获取数据样本
sample = dataset[0]

print(f"图像形状: {sample['img'].shape}")  # 应该是 [6, H, W]
print(f"类别: {sample['cls']}")
print(f"边界框: {sample['bboxes']}")

# 创建数据加载器
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=DualModalityDataset.collate_fn
)

# 训练循环
for batch in dataloader:
    images = batch['img']      # [B, 6, H, W]
    bboxes = batch['bboxes']  # [N, 4]
    cls = batch['cls']        # [N, 1]
    batch_idx = batch['batch_idx']
    
    # 你的训练代码...
    pass
```

## 数据格式要求

### 目录结构
```
datasets/
├── images/
│   └── train/
│       ├── image_001.jpg  # RGB图像
│       ├── image_002.jpg
│       └── ...
└── images_ir/
    └── train/
        ├── image_001.jpg  # 对应的IR图像
        ├── image_002.jpg
        └── ...
```

### 标签格式
```
# 标签文件: labels/train/image_001.txt
class_id x_center y_center width height

# 例如:
0 0.5 0.5 0.3 0.4
1 0.3 0.7 0.2 0.3
```

## 图像格式

- **RGB图像**: 标准三通道图像（BGR格式）
- **IR图像**: 单通道灰度图像或三通道图像
- **输出**: 6通道图像（前3通道RGB，后3通道IR）

## 高级用法

### 自定义增强流水线

```python
from data import DualModalityDataset, build_paired_transforms
from data.augment import Compose, PairedLetterBox, Format

# 自定义增强流水线
def custom_transforms(dataset, imgsz, hyp):
    return Compose([
        PairedLetterBox(new_shape=(imgsz, imgsz), scaleup=False),
        # 添加其他自定义增强...
        Format(bbox_format='xywh', normalize=True, batch_idx=True)
    ])

# 使用自定义增强
dataset.transforms = custom_transforms(dataset, 640, hyp)
```

### 关闭Mosaic增强

```python
# 在训练后期关闭Mosaic
hyp_mosaic = hyp.copy()
hyp_mosaic['mosaic'] = 0.0
hyp_mosaic['mixup'] = 0.0

dataset.transforms = dataset.build_transforms(hyp_mosaic)
```

## 测试

运行测试脚本验证功能：

```bash
python test_data_module.py
```

## 注意事项

1. **图像对齐**: 确保RGB和IR图像文件名一致
2. **内存使用**: 启用缓存会占用较多内存
3. **增强效果**: 可以通过调整超参数控制增强强度
4. **批次大小**: Mosaic增强会改变图像大小到1280x1280

## 与M2D-LIF项目的关系

这个模块是从M2D-LIF项目中提取和精简的双模态数据处理代码，针对你的对抗蒸馏融合框架进行了优化。保留了以下核心功能：

- 双模态图像加载和合并
- 同步的RGB+IR数据增强
- YOLO格式数据集支持
- 批次整理函数

## 文件说明

- `__init__.py`: 模块初始化
- `dataset.py`: 数据集类
- `augment.py`: 增强类
- `utils.py`: 辅助函数
- `test_data_module.py`: 测试脚本

## 下一步

这个数据处理模块已经可以直接用于你的对抗蒸馏融合框架。接下来你需要：

1. 实现Teacher模型（RGB和IR单模态检测器）
2. 实现Student模型（双模态融合检测器）
3. 实现对抗训练模块
4. 实现损失函数

参考 `tasks/` 目录下的设计文档来实现这些模块。