# 🎯 Shift 模块验证方案（4 阶段递进）

## 📊 当前基线
- RGB 单独: 0.347 (map50-95)
- IR 单独: 0.441
- RGB+IR (naive_add): 0.441 （接近没有提升，存在"模型偷懒"现象）
- RGB+IR + 对抗: 0.449
- 他人光照+蒸馏: 0.461

**目标**: 验证 Shift 模块能否超过 0.461，达到 SOTA

---

## 🏗️ 4 阶段计划

### 阶段 1：最小化可行方案（图像级全局平移）
**目标**: 快速验证 shift idea 的可行性，降低实现复杂度
**预计收益**: +0.005 ~ +0.015 mAP
**耗时**: 1-2 天

#### 任务 1.1: 实现图像级数据增强
- **文件**: `ADV/data/shift_augment.py` (新建)
- **功能**:
  - 对整张 RGB 图做随机平移 (dx, dy)，IR 保持不变
  - 计算相对偏移 GT: (dx, dy)
  - 简单空洞填充：边界均值 + 高斯模糊
  - 支持动态启用/禁用（通过参数）

#### 任务 1.2: 实现轻量级 Shift 损失
- **文件**: `ADV/losses/shift_loss.py` (新建)
- **功能**:
  - Smooth L1 损失，预测 (dx, dy)
  - 权重 λ = 0.1（超参数，后续可调）
  - 与检测损失加权组合: L_total = L_det + λ * L_shift

#### 任务 1.3: 修改检测头支持 shift 输出
- **文件**: `ADV/models/student.py` (修改)
- **功能**:
  - 在 Detect 头后额外输出 (dx, dy) 预测
  - 不改变原有的检测逻辑
  - 支持 shift_enable 参数控制

#### 任务 1.4: 训练和评估
```bash
# 启用 shift，固定平移比例 0.3
python train.py \
    --model yolov8_naive_add.yaml \
    --shift_enable \
    --shift_prob 0.3 \
    --shift_weight 0.1 \
    --epochs 100
```
- **预期**: 如果有效，mAP50-95 应该上升到 ~0.445-0.450

---

### 阶段 2：渐进式学习策略（Dynamic Scheduling）
**目标**: 防止 shift 任务压倒主任务，利用课程学习思想
**预计收益**: 相比阶段1 +0.005 ~ +0.010 mAP
**耗时**: 3-5 天

#### 任务 2.1: 实现动态调度
- **文件**: `ADV/trainers/shift_trainer.py` (新建)
- **功能**:
  - Epoch 进度 0-20%: shift_prob = 0, λ = 0 (纯净训练)
  - Epoch 进度 20-80%: shift_prob 和 λ 线性增长（0 → 0.3 和 0 → 0.1）
  - Epoch 进度 80-100%: shift_prob = 0.3, λ = 0.1 (稳定阶段)

#### 任务 2.2: 蒸馏阶段（可选但推荐）
- Epoch 80-100 的最后 10 个 epoch：关闭 shift，用纯净数据训练
- 防止模型对"被平移的数据"过度适应

#### 任务 2.3: 训练和评估
```bash
python train.py \
    --model yolov8_naive_add.yaml \
    --shift_enable \
    --shift_schedule dynamic \
    --epochs 100
```
- **预期**: mAP50-95 上升到 ~0.450-0.456

---

### 阶段 3：特征级平移（RoI Align）
**目标**: 避免像素级伪影，增强跨模态语义对齐
**预计收益**: +0.010 ~ +0.020 mAP
**耗时**: 5-7 天

#### 核心思路
- 不再在图像级做平移，改在特征图级别操作
- 使用 RoI Align 从特征图中精准提取 BBox 对应的特征块
- 在特征块上做空间偏移，再拼回融合特征图

#### 任务 3.1: 实现特征级平移
- **文件**: `ADV/modules/feature_shift.py` (新建)
- **功能**:
  - 在融合层（fusion）之后，检测头之前插入
  - 使用 RoI Align 提取 BBox 特征块
  - 对特征块进行空间偏移（通过 torch.roll 或 grid_sample）
  - 拼回原特征图

#### 任务 3.2: 修改融合和检测流程
- **文件**: `ADV/models/student.py` (修改)
- **功能**:
  - 在 Detect 头前插入 FeatureShiftModule
  - 从检测框坐标计算梯度回传

#### 任务 3.3: 增加特征相似度损失（可选）
- **文件**: `ADV/losses/shift_loss.py` (修改)
- **功能**:
  - L_shift = L_offset + α * L_cosine
  - 要求平移对齐后的 RGB 和 IR 特征相似度最大化

#### 任务 3.4: 训练和评估
```bash
python train.py \
    --model yolov8_naive_add.yaml \
    --shift_enable \
    --shift_mode feature \
    --shift_schedule dynamic \
    --epochs 100
```
- **预期**: mAP50-95 上升到 ~0.456-0.465

---

### 阶段 4：高级优化（多模态对齐增强）
**目标**: 进一步增强跨模态对齐和融合质量
**预计收益**: +0.005 ~ +0.015 mAP
**耗时**: 7-10 天

#### 任务 4.1: 引入目标级别的平移
- 不是平移整张 RGB 图，而是平移单个 BBox 内的目标
- 增强小目标的学习

#### 任务 4.2: 多尺度平移
- 在 FPN 的多个层级上应用平移
- 根据目标大小自适应选择平移层级

#### 任务 4.3: 自适应平移范围
- 根据图像和 BBox 大小动态调整平移范围
- 防止过度平移导致的目标丢失

#### 任务 4.4: 融合优化
- 探索不同的融合策略（concat vs add vs attention）
- 联合优化融合层和 shift 预测

---

## 🔑 关键参数配置

### 阶段 1 推荐配置
```yaml
shift:
  enable: true
  mode: 'image'          # 图像级平移
  prob: 0.3              # 平移概率（固定）
  range: [10, 50]        # 平移范围（像素）
  fill_method: 'mean_gaussian'
  loss_weight: 0.1
  schedule: null         # 无动态调度
```

### 阶段 2 推荐配置
```yaml
shift:
  enable: true
  mode: 'image'
  prob: 0.3              # 最终概率
  range: [10, 50]
  fill_method: 'mean_gaussian'
  loss_weight: 0.1
  schedule: 'dynamic'    # 动态调度
  schedule_config:
    warmup_epochs: 20
    growth_epochs: 60
    final_epochs: 20
```

### 阶段 3 推荐配置
```yaml
shift:
  enable: true
  mode: 'feature'        # 特征级平移
  prob: 0.3
  range: [10, 50]        # 特征图像素
  loss_weight: 0.1
  similarity_weight: 0.05  # 特征相似度权重
  schedule: 'dynamic'
```

---

## 📈 预期收益曲线

```
阶段1 (图像级全局平移)    → 0.445-0.450 (+0.005~0.010)
阶段2 (渐进式调度)        → 0.450-0.456 (+0.005~0.010)
阶段3 (特征级平移)        → 0.456-0.465 (+0.010~0.020)
阶段4 (高级优化)          → 0.465-0.475 (+0.005~0.015) ✨ SOTA 候选
```

---

## ⚠️ 关键风险点和解决方案

| 风险 | 表现 | 解决方案 |
|------|------|--------|
| shift 任务压倒检测 | 检测 mAP 下降 | 使用 λ = 0.1 或更小；采用动态调度 |
| 空洞填充产生伪影 | 模型学习边界特征 | 使用高斯模糊；改用特征级平移 |
| 过度平移目标丢失 | shift 损失无法学习 | 限制平移范围；使用边界检查 |
| 特征级实现复杂度 | 开发时间长 | 从简单的图像级开始，逐步过渡 |

---

## 📋 实现检查清单

### 阶段 1
- [ ] `ADV/data/shift_augment.py` 完成
- [ ] `ADV/losses/shift_loss.py` 完成
- [ ] `ADV/models/student.py` 修改完成
- [ ] 至少 1 次完整训练验证
- [ ] 记录 mAP 提升幅度

### 阶段 2
- [ ] `ADV/trainers/shift_trainer.py` 完成
- [ ] 动态调度逻辑验证
- [ ] 至少 2 次完整训练验证
- [ ] 对比固定 vs 动态调度的效果

### 阶段 3
- [ ] `ADV/modules/feature_shift.py` 完成
- [ ] RoI Align 集成
- [ ] 特征相似度损失（可选）
- [ ] 至少 2 次完整训练验证
- [ ] 对比图像级 vs 特征级的效果

### 阶段 4
- [ ] 目标级平移实现
- [ ] 多尺度平移实现
- [ ] 最终性能评估
- [ ] 与 SOTA 对标

---

## 🚀 快速开始命令

```bash
# 阶段 1：图像级固定平移
python train.py --model yolov8_naive_add.yaml --shift_enable \
  --shift_prob 0.3 --shift_weight 0.1 --epochs 100

# 阶段 2：动态调度
python train.py --model yolov8_naive_add.yaml --shift_enable \
  --shift_schedule dynamic --epochs 100

# 阶段 3：特征级平移
python train.py --model yolov8_naive_add.yaml --shift_enable \
  --shift_mode feature --shift_schedule dynamic --epochs 100
```

---

## 📝 备注

- 每个阶段都是可选的，但推荐按顺序进行
- 每个阶段至少训练 2 次以验证稳定性
- 保存每个阶段的最佳模型用于对比和后续分析
- 记录 loss 曲线和 mAP 进展
