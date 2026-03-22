#!/bin/bash
#
# ADV 对抗蒸馏训练启动脚本
#
# 使用方法:
#   ./scripts/train_adv.sh [scale]
#
# 示例:
#   ./scripts/train_adv.sh s    # 使用 s scale 训练
#   ./scripts/train_adv.sh m    # 使用 m scale 训练
#

# 默认参数
SCALE=${1:-s}
EPOCHS=100
BATCH=16
DEVICE=0

# 路径配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 🚨 Teacher 路径：已训练好的 Baseline 双模态模型
TEACHER_PATH="/root/autodl-tmp/ADV/runs/baseline/yolov8s_naive_add5/weights/best.pt"

# 检查 Teacher 模型
if [ ! -f "$TEACHER_PATH" ]; then
    echo "⚠️  Baseline Teacher 不存在: $TEACHER_PATH"
    echo "请先训练 Baseline 模型:"
    echo "  python train_baseline.py --scale $SCALE --epochs 100"
    exit 1
fi

echo "=============================================="
echo "ADV 对抗蒸馏训练"
echo "=============================================="
echo "Scale: $SCALE"
echo "Epochs: $EPOCHS"
echo "Batch: $BATCH"
echo "Device: $DEVICE"
echo "Teacher: $TEACHER_PATH"
echo "=============================================="

cd "$PROJECT_ROOT"

python train_adv.py \
    --baseline_teacher "$TEACHER_PATH" \
    --model "yolov8_naive_add.yaml" \
    --scale "$SCALE" \
    --epochs $EPOCHS \
    --batch $BATCH \
    --device $DEVICE \
    --epsilon_max 0.05 \
    --lambda_distill_start 0.8 \
    --early_layer_idx 2 \
    --attack_method fgsm \
    --warmup_epochs 5 \
    --lr0 0.01 \
    --momentum 0.937 \
    --weight_decay 5e-4 \
    --optimizer SGD \
    --data "/root/autodl-tmp/ADV/M2D-LIF/data/FLIR.yaml" \
    --project "./runs/adv" \
    --name "adv_yolov8${SCALE}" \
    --amp

echo "=============================================="
echo "训练完成!"
echo "=============================================="
