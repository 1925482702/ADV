#!/bin/bash
# ========================================
# 简化版训练脚本
# 目的：验证"双路特征融合"是否优于"早期融合"
# 
# 关闭的高级策略：
# - ResidualBlock (use_residual=False)
# - 蒸馏训练 (lambda_distill=0)
# - 对抗训练 (epsilon=0)
# - 预训练权重 (全网络随机初始化)
# ========================================

# 配置路径
TEACHER_RGB="/root/autodl-tmp/ADV/checkpoint/monomodal/FLIR_rgb.pt"
TEACHER_IR="/root/autodl-tmp/ADV/checkpoint/monomodal/FLIR_ir.pt"
DATA_YAML="/root/autodl-tmp/ADV/yaml/data/FLIR.yaml"
SAVE_DIR="/root/autodl-tmp/ADV/checkpoint/student_baseline"
LOG_DIR="/root/autodl-tmp/ADV/runs/student_baseline"

# 训练参数
BATCH_SIZE=16
EPOCHS=100
LR=0.01
IMG_SIZE=640
NUM_CLASSES=3

echo "========================================"
echo "简化版 Student 训练"
echo "========================================"
echo "关键配置:"
echo "  - ResidualBlock: 关闭"
echo "  - 蒸馏训练: 关闭"
echo "  - 对抗训练: 关闭"
echo "  - 权重初始化: 全随机"
echo "  - 融合模式: add (直接相加)"
echo "========================================"

python scripts/train_stage2.py \
    --teacher_rgb $TEACHER_RGB \
    --teacher_ir $TEACHER_IR \
    --data $DATA_YAML \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --img_size $IMG_SIZE \
    --num_classes $NUM_CLASSES \
    --fusion_mode add \
    --save_dir $SAVE_DIR \
    --log_dir $LOG_DIR \
    --max_epsilon 0.0 \
    --lambda_distill 0.0 \
    --lambda_det 1.0 \
    --optimizer sgd \
    --warmup_epochs 3 \
    --lr_scheduler cosine \
    --amp \
    --device cuda

echo "========================================"
echo "训练完成！"
echo "结果保存在: $SAVE_DIR"
echo "========================================"
