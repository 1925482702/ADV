#!/usr/bin/env bash
# train_shift.py 使用说明和示例

## 📖 train_shift.py 使用指南

### 概述
`train_shift.py` 是一个完整的训练脚本，支持：
- ✅ 命令行参数传递
- ✅ 配置文件（YAML/JSON）
- ✅ 预设配置（conservative/balanced/aggressive）
- ✅ 参数覆盖（配置文件 + 命令行）
- ✅ 干运行模式（只检查配置不训练）
- ✅ 自动实验名称生成

---

## 🚀 快速开始

### 最简单的方式（使用预设配置）

```bash
# 平衡配置（推荐）
python train_shift.py --preset balanced --data /path/to/data.yaml

# 保守配置（温和）
python train_shift.py --preset conservative --data /path/to/data.yaml

# 激进配置（强约束）
python train_shift.py --preset aggressive --data /path/to/data.yaml
```

### 使用配置文件

```bash
# 从YAML配置文件加载
python train_shift.py --config configs/shift_balanced.yaml --data /path/to/data.yaml

# 从JSON配置文件加载
python train_shift.py --config my_config.json --data /path/to/data.yaml
```

### 命令行参数

```bash
# 直接指定所有参数
python train_shift.py \
    --data /path/to/data.yaml \
    --shift_range 10 50 \
    --shift_prob 0.3 \
    --shift_weight 0.1 \
    --epochs 100 \
    --batch 16 \
    --imgsz 640 \
    --device 0
```

### 配置文件 + 命令行覆盖

```bash
# 从配置文件加载，然后用命令行覆盖某些参数
python train_shift.py \
    --config configs/shift_balanced.yaml \
    --data /path/to/data.yaml \
    --epochs 200 \
    --shift_weight 0.2 \
    --batch 32
```

---

## 📋 所有可用参数

### 数据增强参数
```bash
--shift_range MIN MAX    # 平移像素范围，例如: --shift_range 10 50
--shift_prob PROB        # 平移概率 (0-1)，例如: --shift_prob 0.3
```

### Loss参数
```bash
--shift_weight WEIGHT    # Shift Loss权重，例如: --shift_weight 0.1
--shift_enabled BOOL     # 是否启用Shift Loss，例如: --shift_enabled True
```

### 训练参数
```bash
--data PATH              # 数据集yaml路径 (必需)
--model PATH             # 模型yaml路径，默认: yolov8n.yaml
--epochs NUM             # 训练轮数，例如: --epochs 100
--imgsz SIZE             # 输入图像大小，例如: --imgsz 640
--batch SIZE             # 批大小，例如: --batch 16
--device ID              # GPU设备号，例如: --device 0
--workers NUM            # 数据加载进程数，默认: 4
```

### 学习率参数
```bash
--lr0 LR                 # 初始学习率，默认: 0.01
--momentum M             # 动量，默认: 0.937
--weight_decay WD        # 权重衰减，默认: 0.0005
```

### 输出和日志
```bash
--project PATH           # 结果保存路径，默认: runs/shift_training
--name NAME              # 实验名称 (None时自动生成)
--seed SEED              # 随机种子，默认: 42
```

### 其他选项
```bash
--dry_run                # 只检查配置，不运行训练
--config FILE            # 配置文件路径
--preset {conservative,balanced,aggressive}  # 预设配置
```

---

## 💡 使用示例

### 示例1：快速测试（干运行）

```bash
# 只检查配置是否有效，不运行训练
python train_shift.py --data data.yaml --dry_run
```

输出会显示配置摘要，帮助你验证参数。

### 示例2：参数扫描（运行多个实验）

```bash
#!/bin/bash

# 创建一个参数扫描脚本
for shift_prob in 0.1 0.3 0.5
do
    for shift_weight in 0.05 0.1 0.2
    do
        echo "运行: shift_prob=$shift_prob, shift_weight=$shift_weight"
        python train_shift.py \
            --data /path/to/data.yaml \
            --shift_prob $shift_prob \
            --shift_weight $shift_weight \
            --name "sp${shift_prob}_sw${shift_weight}"
    done
done
```

### 示例3：使用配置文件并覆盖部分参数

```bash
# 从balanced配置开始，但用更多epoch和更大batch
python train_shift.py \
    --config configs/shift_balanced.yaml \
    --data /path/to/data.yaml \
    --epochs 200 \
    --batch 32 \
    --name "balanced_extended"
```

### 示例4：对比实验

```bash
#!/bin/bash

# 保守配置
python train_shift.py --preset conservative --data data.yaml --name "exp_conservative"

# 平衡配置
python train_shift.py --preset balanced --data data.yaml --name "exp_balanced"

# 激进配置
python train_shift.py --preset aggressive --data data.yaml --name "exp_aggressive"
```

### 示例5：自定义配置文件

创建 `my_config.yaml`:
```yaml
# 自定义配置
shift_range: [15, 60]
shift_prob: 0.4
shift_weight: 0.15
epochs: 120
batch: 24
imgsz: 800
```

然后运行：
```bash
python train_shift.py --config my_config.yaml --data /path/to/data.yaml
```

---

## 🔍 配置优先级

参数来源的优先级（从低到高）：
1. 代码中的默认值 (DEFAULTS)
2. 预设配置 (--preset)
3. 配置文件 (--config)
4. 命令行参数 (--xxx)

所以命令行参数会覆盖所有其他设置。

---

## 📊 实验结果保存

所有训练结果会保存到 `runs/shift_training/` 下，每个实验会创建一个目录：

```
runs/shift_training/
├── shift_sr1050_sp0.3_sw0.10_20260331_120000/
│   ├── weights/
│   │   ├── best.pt
│   │   └── last.pt
│   ├── config.json          ← 本次实验的配置已保存
│   ├── results.csv
│   └── ...
├── shift_sr2080_sp0.5_sw0.20_20260331_130000/
│   └── ...
```

配置会自动保存到 `config.json`，便于后续回顾和复现。

---

## ⚡ 常用命令速查

```bash
# 1️⃣ 快速测试 (干运行)
python train_shift.py --data data.yaml --dry_run

# 2️⃣ 用预设配置
python train_shift.py --preset balanced --data data.yaml

# 3️⃣ 自定义参数
python train_shift.py --data data.yaml --shift_prob 0.5 --shift_weight 0.2

# 4️⃣ 从配置文件 + 覆盖
python train_shift.py --config config.yaml --data data.yaml --epochs 200

# 5️⃣ 参数扫描
for prob in 0.1 0.3 0.5; do
    python train_shift.py --data data.yaml --shift_prob $prob --name "prob_$prob"
done
```

---

## 🛠️ 故障排除

### 问题1：参数验证失败

```
❌ 参数验证失败: 必须指定 'data' 参数
```

**解决**: 必须指定 `--data` 参数

```bash
python train_shift.py --data /path/to/data.yaml --dry_run
```

### 问题2：显存不足

```
CUDA out of memory
```

**解决**: 减小 `--batch` 或 `--imgsz`

```bash
python train_shift.py \
    --data data.yaml \
    --batch 8 \
    --imgsz 480
```

### 问题3：找不到数据集文件

```
❌ 参数验证失败: 数据集文件不存在
```

**解决**: 检查 `--data` 路径是否正确

```bash
python train_shift.py --data /absolute/path/to/data.yaml --dry_run
```

---

## 📝 配置文件格式

### YAML格式 (推荐)

```yaml
# shift_custom.yaml
shift_range: [10, 50]
shift_prob: 0.3
shift_weight: 0.1
epochs: 100
batch: 16
imgsz: 640
```

### JSON格式

```json
{
  "shift_range": [10, 50],
  "shift_prob": 0.3,
  "shift_weight": 0.1,
  "epochs": 100,
  "batch": 16,
  "imgsz": 640
}
```

---

## 🎯 最佳实践

1. **从干运行开始**
   ```bash
   python train_shift.py --data data.yaml --dry_run
   ```

2. **使用预设配置**
   ```bash
   python train_shift.py --preset balanced --data data.yaml
   ```

3. **保存配置文件用于复现**
   - 配置会自动保存到结果目录的 `config.json`

4. **进行参数扫描时用脚本**
   ```bash
   python shift_param_scan.py  # 生成配置组合
   ```

5. **定期检查 `runs/shift_training/` 中的结果**

---

## 📞 获取帮助

```bash
# 显示完整帮助信息
python train_shift.py -h
python train_shift.py --help
```

---

**祝您训练顺利！🚀**
