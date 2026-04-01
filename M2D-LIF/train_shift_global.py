"""
Shift Detect 训练脚本

使用自定义 Trainer 训练带有全局 shift 预测的 YOLOv8 模型

使用方法:
    python train_shift_global.py --scale m --epochs 100 --batch 16 --device 0

参数说明:
    --scale: 模型规模 (n/s/m/l/x)
    --epochs: 训练轮数
    --batch: 批次大小
    --device: GPU 设备号
    --shift_weight: shift loss 的权重 λ (默认 1.0)
    --data: 数据集配置文件路径
"""

import argparse
import os
import sys
from pathlib import Path
from copy import copy

# 添加 ultralytics 到路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER, DEFAULT_CFG_DICT, RANK
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import ShiftDetectionModel


class ShiftDetectionTrainer(DetectionTrainer):
    """
    自定义 Trainer，使用 ShiftDetectionModel 和 v8ShiftDetectionLoss
    """
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 ShiftDetectionModel"""
        model = ShiftDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def set_model_attributes(self):
        """设置模型属性，包括 shift_weight"""
        super().set_model_attributes()
        # 添加 shift_weight 到 model.args（已在 default.yaml 中注册）
        self.model.args.shift_weight = self.args.shift_weight
    
    def get_validator(self):
        """返回 DetectionValidator，过滤掉自定义参数，并对齐 loss 维度"""
        from ultralytics.models import yolo
        from types import SimpleNamespace
        
        custom_args = ['shift_weight', 'shift_ratio', 'max_shift']
        validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
        validator_args = SimpleNamespace(**validator_args_dict)
        
        # 1. 实例化官方的 Validator
        validator = yolo.detect.DetectionValidator(self.test_loader,
                                                   save_dir=self.save_dir,
                                                   args=validator_args,
                                                   _callbacks=self.callbacks)
        
        # 2. 覆盖验证器的 loss_names，使其与输出的 4 个 loss 对应
        validator.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'shift_loss']
        
        return validator
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        返回带标签的损失项
        ShiftDetect 的损失包括: box, cls, dfl, shift
        """
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'shift_loss'
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Shift Detection')
    parser.add_argument('--scale', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model scale')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--shift_weight', type=float, default=1.0,
                        help='Weight for shift loss (λ)')
    parser.add_argument('--data', type=str, default='./data/FLIR.yaml',
                        help='Path to data config file')
    parser.add_argument('--project', type=str, default='./runs/shift_global',
                        help='Project name')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 模型配置文件路径
    model_yaml = f"./model_yaml/yolov8_shift_global.yaml"
    
    if not os.path.exists(model_yaml):
        LOGGER.error(f"❌ 模型配置文件不存在: {model_yaml}")
        raise FileNotFoundError(f"Model config not found: {model_yaml}")
    
    LOGGER.info("=" * 60)
    LOGGER.info("开始训练 ShiftDetect 模型")
    LOGGER.info("=" * 60)
    LOGGER.info(f"模型配置: {model_yaml}")
    LOGGER.info(f"Scale: {args.scale}")
    LOGGER.info(f"数据集: {args.data}")
    LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    LOGGER.info(f"Shift weight: {args.shift_weight}")
    LOGGER.info(f"设备: cuda:{args.device}" if args.device >= 0 else "设备: cpu")
    
    # 准备训练参数（现在 shift_weight 可以直接传递）
    train_args = {
        'model': model_yaml,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'device': f'cuda:{args.device}' if args.device >= 0 else 'cpu',
        'batch': args.batch,
        'workers': args.workers,
        'project': args.project,
        'name': f'yolov8{args.scale}_shift_global',
        'lr0': args.lr,
        'shift_weight': args.shift_weight,
        'patience': 50,
        'save_period': 10,
        'close_mosaic': 10,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'optimizer': 'SGD',
        'momentum': 0.937,
        'weight_decay': 0.0005,
    }
    
    # 创建 ShiftDetectionTrainer
    try:
        trainer = ShiftDetectionTrainer(overrides=train_args)
        trainer.train()
        
        LOGGER.info("=" * 60)
        LOGGER.info("✓ 训练完成！")
        LOGGER.info(f"结果保存在: {trainer.save_dir}")
        
    except Exception as e:
        LOGGER.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
