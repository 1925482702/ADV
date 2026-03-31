# 🚀 train_shift.py 快速开始指南

## 💥 最快的方式（5秒钟）

```bash
python train_shift.py --preset balanced --data /path/to/data.yaml
```

完成！开始训练。

---

## 📋 三种最常用的用法

### 1️⃣ 用预设配置（推荐）

```bash
# 平衡配置（推荐，默认参数）
python train_shift.py --preset balanced --data data.yaml

# 保守配置（温和，baseline对比）
python train_shift.py --preset conservative --data data.yaml

# 激进配置（强约束，可能更好）
python train_shift.py --preset aggressive --data data.yaml
```

### 2️⃣ 用命令行参数

```bash
# 直接指定所有参数
python train_shift.py \
    --data data.yaml \
    --shift_range 10 50 \
    --shift_prob 0.3 \
    --shift_weight 0.1 \
    --epochs 100
```

### 3️⃣ 用配置文件 + 覆盖

```bash
# 从配置文件加载，然后覆盖某些参数
python train_shift.py \
    --config configs/shift_balanced.yaml \
    --data data.yaml \
    --epochs 200 \
    --shift_weight 0.2
```

---

## 🎯 最重要的参数（复制粘贴）

```bash
# 调整这三个参数来控制对齐强度：

python train_shift.py \
    --data data.yaml \
    --shift_range 10 50      # ← 改这个：(5,30)弱 (20,80)强
    --shift_prob 0.3         # ← 改这个：0.1弱 0.5强
    --shift_weight 0.1       # ← 改这个：0.05弱 0.2强
```

---

## 🧪 快速实验模板

### 干运行（测试配置）

```bash
# 只检查参数，不运行训练
python train_shift.py --data data.yaml --dry_run
```

### 参数扫描

```bash
#!/bin/bash
# 快速对比三个配置

for preset in conservative balanced aggressive
do
    echo "运行 $preset 配置..."
    python train_shift.py --preset $preset --data data.yaml
done
```

### 两个实验对比

```bash
# 实验1：基线（无shift）
python train_shift.py --data data.yaml --shift_enabled False

# 实验2：带shift
python train_shift.py --data data.yaml --shift_enabled True
```

---

## 📊 完整参数列表（速查表）

### 数据增强
```bash
--shift_range MIN MAX    # 平移像素范围 (推荐: 10 50)
--shift_prob PROB        # 平移概率 (推荐: 0.3)
```

### Loss参数
```bash
--shift_weight WEIGHT    # Shift Loss权重 (推荐: 0.1)
--shift_enabled BOOL     # 是否启用 (推荐: True)
```

### 训练配置
```bash
--data PATH              # 数据集 [必需]
--epochs NUM             # 轮数 (默认: 100)
--batch BATCH            # 批大小 (默认: 16)
--imgsz SIZE             # 图像大小 (默认: 640)
--device ID              # GPU号 (默认: 0)
```

### 学习率
```bash
--lr0 LR                 # 初始学习率 (默认: 0.01)
--momentum M             # 动量 (默认: 0.937)
--weight_decay WD        # 权重衰减 (默认: 0.0005)
```

### 输出
```bash
--project PATH           # 保存路径 (默认: runs/shift_training)
--name NAME              # 实验名称 (默认: 自动生成)
```

### 其他
```bash
--config FILE            # 配置文件
--preset PRESET          # 预设: conservative|balanced|aggressive
--dry_run                # 只检查配置，不训练
```

---

## 💡 常用命令集锦

```bash
# 1. 快速测试
python train_shift.py --data data.yaml --dry_run

# 2. 预设配置
python train_shift.py --preset balanced --data data.yaml

# 3. 自定义参数
python train_shift.py --data data.yaml --shift_prob 0.5 --shift_weight 0.2

# 4. 自定义名称
python train_shift.py --data data.yaml --name "exp_v1"

# 5. 更多epoch
python train_shift.py --data data.yaml --epochs 200

# 6. 更大batch（如果显存够）
python train_shift.py --data data.yaml --batch 32 --imgsz 800

# 7. 显存不足
python train_shift.py --data data.yaml --batch 8 --imgsz 480
```

---

## 🎛️ 参数优先级

如果同时指定多种参数来源，优先级为（从低到高）：
```
默认值 < 预设 < 配置文件 < 命令行参数
```

例如：
```bash
# 配置文件里有 epochs=100，但命令行改为200
python train_shift.py --config config.yaml --epochs 200
# 结果：使用 epochs=200
```

---

## 📁 配置文件示例

### 保守配置 (configs/shift_conservative.yaml)
```yaml
shift_range: [5, 30]
shift_prob: 0.1
shift_weight: 0.05
epochs: 100
batch: 16
```

### 平衡配置 (configs/shift_balanced.yaml)
```yaml
shift_range: [10, 50]
shift_prob: 0.3
shift_weight: 0.1
epochs: 100
batch: 16
```

### 激进配置 (configs/shift_aggressive.yaml)
```yaml
shift_range: [20, 80]
shift_prob: 0.5
shift_weight: 0.2
epochs: 150
batch: 32
```

---

## ✅ 常见问题解决

### Q: 不知道用什么参数？
**A:** 用预设配置，从 balanced 开始
```bash
python train_shift.py --preset balanced --data data.yaml
```

### Q: 显存不够？
**A:** 减小 batch 和 imgsz
```bash
python train_shift.py --data data.yaml --batch 8 --imgsz 480
```

### Q: 只想测试配置不想训练？
**A:** 加上 --dry_run
```bash
python train_shift.py --data data.yaml --dry_run
```

### Q: 怎么做参数对比？
**A:** 用脚本循环运行
```bash
for prob in 0.1 0.3 0.5; do
    python train_shift.py --data data.yaml --shift_prob $prob
done
```

### Q: 怎么找到训练结果？
**A:** 查看 runs/shift_training/ 目录
```bash
ls -lh runs/shift_training/
```

---

## 🎓 学习路径

1. **第一步**：干运行验证配置
   ```bash
   python train_shift.py --data data.yaml --dry_run
   ```

2. **第二步**：用预设配置快速测试
   ```bash
   python train_shift.py --preset balanced --data data.yaml
   ```

3. **第三步**：根据结果调整参数
   ```bash
   python train_shift.py --data data.yaml --shift_prob 0.5 --epochs 150
   ```

4. **第四步**：进行参数扫描找最优值
   ```bash
   # 用脚本运行多个配置
   for prob in 0.1 0.3 0.5; do
       python train_shift.py --data data.yaml --shift_prob $prob
   done
   ```

---

## 📞 获取完整帮助

```bash
python train_shift.py -h
python train_shift.py --help
```

---

**准备好了吗？开始训练吧！🚀**
