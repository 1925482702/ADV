# 修改说明：在现有训练器中集成Shift预测

## 文件: ultralytics/models/yolo/detect/train.py

### 修改点 1: 在 DetectionTrainer 中添加 Shift 支持

```python
# 在 __init__ 方法后添加

def setup_shift_augmentation(self):
    """Setup shift augmentation if enabled in args."""
    from ultralytics.data.augment import PairedShiftAugment
    
    # Check if shift is enabled (can be added to args)
    if hasattr(self.args, 'enable_shift') and self.args.enable_shift:
        self.shift_augment = PairedShiftAugment(
            shift_range=(10, 50),
            shift_prob=getattr(self.args, 'shift_prob', 0.3)
        )
    else:
        self.shift_augment = None

def get_shift_loss_weight(self):
    """Get shift loss weight based on training progress."""
    if self.epoch < self.args.epochs * 0.2:
        # Warmup: no shift
        return 0.0
    elif self.epoch < self.args.epochs * 0.8:
        # Early: linear growth
        progress = (self.epoch - self.args.epochs * 0.2) / (self.args.epochs * 0.6)
        return 0.1 * progress
    else:
        # Late: maximum weight
        return 0.1
```

### 修改点 2: 在 preprocess_batch 中应用 Shift 增强

```python
def preprocess_batch(self, batch):
    """Preprocesses a batch of images by scaling and converting to float."""
    batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
    if self.args.multi_scale:
        imgs = batch['img']
        sz = (random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride) // self.stride *
              self.stride)
        sf = sz / max(imgs.shape[2:])
        if sf != 1:
            ns = [math.ceil(x * sf / self.stride) * self.stride
                  for x in imgs.shape[2:]]
            imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
        batch['img'] = imgs

    # ✓ 添加这些行来应用shift增强
    # Note: 实际上shift增强应该在数据加载时应用，这里只是示例
    # 确保 shift_dx, shift_dy 在batch中
    if 'shift_dx' not in batch:
        batch['shift_dx'] = torch.zeros(batch['img'].shape[0], device=self.device)
    if 'shift_dy' not in batch:
        batch['shift_dy'] = torch.zeros(batch['img'].shape[0], device=self.device)

    return batch
```

### 修改点 3: 修改 Loss 初始化以支持 Shift

```python
def init_criterion(self):
    """Initialize the loss criterion."""
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.models.shift_integration import DetectionLossWithShift
    
    base_loss = v8DetectionLoss(self.model)
    
    # Wrap with shift loss support if enabled
    if hasattr(self.args, 'enable_shift') and self.args.enable_shift:
        shift_weight = self.get_shift_loss_weight()
        loss_fn = DetectionLossWithShift(base_loss, shift_weight=shift_weight, enabled=True)
    else:
        loss_fn = base_loss
    
    return loss_fn
```

### 修改点 4: 在训练循环中更新 Loss 权重

在 BaseTrainer 或 DetectionTrainer 的训练循环中（通常在 train_one_epoch 或类似处）：

```python
def train_one_epoch(self):
    """Train one epoch."""
    # ... existing code ...

    # ✓ 更新shift loss权重（动态调度）
    if hasattr(self.criterion, 'shift_loss') and self.criterion.shift_loss is not None:
        shift_weight = self.get_shift_loss_weight()
        self.criterion.shift_loss.weight = shift_weight

    # ... rest of training code ...
```

### 修改点 5: 可视化 Loss 名称（可选）

```python
def label_loss_items(self, loss_items=None, prefix='train'):
    """Returns a loss dict with labelled training loss items."""
    if hasattr(self, 'criterion') and hasattr(self.criterion, 'shift_loss'):
        # Include shift loss in loss names
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'shift_loss'
    else:
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
    
    keys = [f'{prefix}/{x}' for x in self.loss_names]
    if loss_items is not None:
        loss_items = [round(float(x), 5) for x in loss_items]
        return dict(zip(keys, loss_items))
    else:
        return keys
```

## 文件: ultralytics/data/dataset.py

### 修改点：在数据加载时应用 Shift 增强

```python
def build_transforms(self, hyp=None):
    """Builds and appends transforms to the list."""
    if self.augment:
        hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
        hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
        transforms = v8_Pairedtransforms(self, self.imgsz, hyp)

        # ✓ 添加shift增强（如果启用）
        if hasattr(hyp, 'enable_shift') and hyp.enable_shift:
            from ultralytics.data.augment import PairedShiftAugment
            shift_augment = PairedShiftAugment(
                shift_range=getattr(hyp, 'shift_range', (10, 50)),
                shift_prob=getattr(hyp, 'shift_prob', 0.3)
            )
            # 在其他增强之后，但在Format之前应用
            # transforms.insert(-1, shift_augment)  # 插入在倒数第二个位置
```

## 使用方法

### 方法 1: 通过命令行参数

```bash
yolo detect train \
    model=/mnt/home/pyq_code/ADV/M2D-LIF/model_yaml/yolov8_naive_add.yaml \
    data=path/to/data.yaml \
    epochs=100 \
    imgsz=640 \
    enable_shift=true \
    shift_prob=0.3 \
    shift_range=[10,50]
```

### 方法 2: 通过 Python API

```python
from ultralytics import YOLO
from ultralytics.models.shift_integration import enable_shift_head_in_model

model = YOLO('/mnt/home/pyq_code/ADV/M2D-LIF/model_yaml/yolov8_naive_add.yaml')

# Enable shift head
enable_shift_head_in_model(model, enabled=True)

# Train with shift parameters
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    enable_shift=True,
    shift_prob=0.3,
)
```

### 方法 3: 通过继承 Trainer

```python
from ultralytics.models.yolo.detect.train import DetectionTrainer

class ShiftDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.setup_shift_augmentation()

    # ... 添加上述修改的方法 ...
```

## 检查清单

- [ ] 在 Detect 头中添加了 enable_shift 参数
- [ ] 创建了 ShiftHead 模块
- [ ] 在数据增强中添加了 PairedShiftAugment
- [ ] 创建了 ShiftLoss
- [ ] 修改了 PairedFormat 来保留 shift 信息
- [ ] 设置了动态调度逻辑
- [ ] 验证了所有导入正常工作
- [ ] 在示例脚本中测试了基本功能

## 下一步

1. 在实际数据集上运行训练
2. 监控 shift loss 和检测 loss 的相互作用
3. 根据结果调整超参数（平移范围、概率、loss权重）
4. 可视化模型学到的shift预测
5. 评估对检测性能的影响
