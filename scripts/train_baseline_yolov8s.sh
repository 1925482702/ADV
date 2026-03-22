#!/bin/bash
# ========================================
# Baseline 训练脚本 (YOLOv8s scale)
# 对应模型: yolov8_naive_add.yaml
# 数据: FLIR 双模态 (6通道输入)
# ========================================

cd /root/autodl-tmp/ADV/M2D-LIF

# 使用 YOLOv8s scale
MODEL="model_yaml/yolov8_naive_add.yaml"
DATA="data/FLIR.yaml"
SCALE="s"  # 使用 s scale，和 Teacher 一致

echo "========================================"
echo "Baseline 训练 (YOLOv8s scale)"
echo "========================================"
echo "模型: $MODEL"
echo "数据: $DATA"
echo "Scale: $SCALE"
echo "========================================"

python train_baseline.py \
    --model $MODEL \
    --data $DATA \
    --scale $SCALE \
    --epochs 100 \
    --batch 8 \
    --imgsz 640 \
    --device 0 \
    --project runs/baseline \
    --name yolov8s_naive_add \
    --amp

echo "========================================"
echo "训练完成！"
echo "结果保存在: runs/baseline/yolov8s_naive_add"
echo "========================================"
