# ⚡ 快速参数速查表

## 🎛️ 所有可配置参数一览

### 【数据增强参数】最重要 ⭐⭐⭐
```python
PairedShiftAugment(
    shift_range = (10, 50),    # ← 平移像素范围
    shift_prob = 0.3           # ← 平移概率（0-1）
)
```
**位置**: 第49、97、145行  
**快速调整**: 
- 想要更强的对齐? → 增大 shift_range 或 shift_prob
- 想要更温和? → 减小 shift_range 或 shift_prob

---

### 【训练参数】
```python
model.train(
    data='data.yaml',          # ← 数据集路径
    epochs=100,                # ← 训练轮数 (50-200)
    imgsz=640,                 # ← 输入图像大小 (320-1024)
    batch=16,                  # ← 批大小 (4-64)
    device=0,                  # ← GPU号 (0,1,2...)
)
```
**位置**: 第23-30行  
**快速调整**:
- 显存不足? → 减小 batch 或 imgsz
- 训练太慢? → 减小 epochs 或 imgsz

---

### 【Loss参数】⭐⭐⭐
```python
DetectionLossWithShift(
    base_loss,
    shift_weight=0.1,          # ← Shift Loss权重 (0.01-0.5)
    enabled=True               # ← 是否启用
)
```
**位置**: 第78行  
**快速调整**:
- Loss变化剧烈? → 减小 shift_weight
- Shift没有学好? → 增大 shift_weight

---

### 【动态调度参数】⭐⭐
```python
class DynamicShiftTrainer:
    def get_shift_prob_for_epoch(self, epoch):
        if epoch < 0.2 * 100:        # ← Warmup: 0% (可改)
        elif epoch < 0.8 * 100:
            progress = ...
            return 0.3 * progress    # ← Early: 线性增长 (可改)
        else:
            return 0.3               # ← Late: 0.3 最大 (可改)
```
**位置**: 第99-116行  
**可调数字**:
- `0.2` → Warmup的百分比
- `0.8` → Early的截止百分比
- `0.3` → 最大平移概率

---

### 【验证参数】
```python
augment = PairedShiftAugment(
    shift_range=(10, 50),      # 平移范围
    shift_prob=1.0             # 100% 应用（用于测试）
)

batch = {
    'img': np.zeros((640, 640, 3)),      # ← 图像尺寸
    'img_lwir': np.zeros((640, 640, 3)),
    'bboxes': np.array([[0.2, 0.2, 0.4, 0.4], ...]),
}

for i in range(3):            # ← 重复次数
    augmented = augment(batch)
```
**位置**: 第131-153行

---

## 📊 参数影响速查

| 参数 | 增大时 | 减小时 |
|------|--------|--------|
| **shift_range** | 更强的对齐约束，但可能影响边界 | 更温和，对齐约束弱 |
| **shift_prob** | 更多数据被增强，学习更强 | 增强数据少，影响温和 |
| **shift_weight** | Loss更关注对齐，可能压制检测 | 对齐约束弱，检测priority高 |
| **epochs** | 训练时间长，可能更好性能 | 训练快，但可能欠拟合 |
| **batch** | 梯度更稳定，但显存占用多 | 显存占用少，但噪声大 |
| **imgsz** | 细节更多，大目标更清晰 | 速度快，小目标可能模糊 |

---

## 🧪 三个必做实验

### 实验1: Shift强度对比 (5分钟快速版)
```python
# 修改第49行的 shift_range
configs = [
    (5, 30),      # 弱
    (10, 50),     # 中
    (20, 80),     # 强
]

for sr in configs:
    augment = PairedShiftAugment(shift_range=sr, shift_prob=0.3)
    print(f"shift_range={sr}: 运行训练")
```

### 实验2: Loss权重对比 (最重要!)
```python
# 修改第78行的 shift_weight
weights = [0.01, 0.05, 0.1, 0.2, 0.5]

for w in weights:
    loss = DetectionLossWithShift(base_loss, shift_weight=w, enabled=True)
    print(f"shift_weight={w}: 运行训练")
```

### 实验3: Ablation (对比baseline)
```python
# 修改第78行的 enabled
loss_with_shift = DetectionLossWithShift(base_loss, enabled=True)
loss_without_shift = DetectionLossWithShift(base_loss, enabled=False)

# 分别运行并对比
```

---

## 🎯 一行代码改参数

```python
# 全部都在这三个地方改：

# 1️⃣ 数据增强强度（第49/97/145行）
PairedShiftAugment(shift_range=(20, 80), shift_prob=0.5)  # 更强

# 2️⃣ Loss权重（第78行）
DetectionLossWithShift(base_loss, shift_weight=0.2, enabled=True)  # 更强的对齐

# 3️⃣ 训练配置（第25-28行）
model.train(epochs=150, imgsz=800, batch=32)  # 更大的模型
```

---

## 📝 参数调优建议

### 如果Loss不收敛？
```python
# 减小shift_weight
shift_weight = 0.05  # 从0.1改成0.05
```

### 如果shift loss没有学好？
```python
# 增大shift_prob和shift_range
shift_prob = 0.5      # 从0.3改成0.5
shift_range = (15, 60) # 从(10,50)改成(15,60)
```

### 如果GPU爆显存？
```python
# 减小batch和imgsz
batch = 8      # 从16改成8
imgsz = 480    # 从640改成480
```

### 如果收敛太慢？
```python
# 增加batch，或改用更大的learning rate（可在args中设置）
batch = 32  # 从16改成32
```

---

## 💡 快速参考速度表

- **跑通代码**: 5分钟 (默认参数)
- **单个参数扫描**: 1-2小时 (4-6个配置)
- **完整参数搜索**: 6-12小时 (20+ 配置)
- **最优配置验证**: 2-4小时

---

**🚀 建议**: 从默认参数开始，优先改 **shift_range** 和 **shift_weight**！
