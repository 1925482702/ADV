#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
冻结 Backbone 版的 Shift 训练脚本

用法：
    python train_shift_frozen.py \
        --data data/LLVIP.yaml \
        --rgb_teacher runs/teacher_rgb/best.pt \
        --ir_teacher runs/teacher_ir/best.pt \
        --shift_schedule l_shape
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import ShiftDetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.shift_scheduler import UShapeShiftScheduler
from ultralytics.utils.shift_params import shift_param_manager


class FrozenShiftTrainer(DetectionTrainer):
    """冻结 Backbone 的 Shift 训练器"""

    def __init__(self, rgb_teacher=None, ir_teacher=None, **kwargs):
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        # 标记：冻结 backbone 模式
        self._freeze_backbone_mode = True
        super().__init__(**kwargs)

    def _setup_train(self, world_size):
        """重写 _setup_train，保持已冻结参数的冻结状态"""
        # 🔥 self.args.freeze 已在 get_model 中设置，直接调用父类
        super()._setup_train(world_size)
        
        # 🔥 关键：重建优化器，只包含可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        LOGGER.info(f"可训练参数: {trainable_count:,}, 冻结参数: {frozen_count:,}")
        
        # 重建优化器
        if trainable_params:
            self.optimizer = self.optimizer.__class__(trainable_params, **self.optimizer.defaults)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 ShiftDetectionModel 并加载冻结的 backbone 权重"""
        model = ShiftDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        # 加载冻结 backbone 权重
        if self.rgb_teacher and self.ir_teacher:
            self._load_frozen_backbones(model)
            
            # 🔥 关键：立即设置 self.args.freeze，让父类 _setup_train 知道哪些层要冻结
            if hasattr(self, '_frozen_layer_indices'):
                existing_freeze = []
                if hasattr(self.args, 'freeze'):
                    if isinstance(self.args.freeze, list):
                        existing_freeze = self.args.freeze
                    elif isinstance(self.args.freeze, int):
                        existing_freeze = list(range(self.args.freeze))
                self.args.freeze = list(set(existing_freeze + list(self._frozen_layer_indices)))
                print(f"  ✓ 设置 args.freeze = {sorted(self.args.freeze)}")

        return model

    def _load_frozen_backbones(self, model):
        """加载并冻结 backbone 权重"""
        print(f"\n🔄 加载冻结 Backbone 权重...")
        print(f"  RGB teacher: {self.rgb_teacher}")
        print(f"  IR teacher: {self.ir_teacher}")

        rgb_ckpt = torch.load(self.rgb_teacher, map_location='cpu', weights_only=False)
        rgb_state = rgb_ckpt['model'].state_dict() if 'model' in rgb_ckpt else rgb_ckpt

        ir_ckpt = torch.load(self.ir_teacher, map_location='cpu', weights_only=False)
        ir_state = ir_ckpt['model'].state_dict() if 'model' in ir_ckpt else ir_ckpt

        # Teacher -> Student 层映射
        teacher_to_rgb = {0:3, 1:5, 2:7, 3:9, 4:11, 5:14, 6:16, 7:19, 8:21}
        teacher_to_ir = {0:4, 1:6, 2:8, 3:10, 4:12, 5:15, 6:17, 7:20, 8:22}

        loaded_count = 0
        frozen_count = 0
        
        # 🔥 记录冻结的层索引（传给父类的 freeze 参数）
        self._frozen_layer_indices = set(teacher_to_rgb.values()) | set(teacher_to_ir.values())

        # 🔥 直接遍历模型的参数，加载权重
        for name, param in model.named_parameters():
            parts = name.split('.')
            # 期望格式: model.3.conv.weight 或 3.conv.weight
            if len(parts) < 2:
                continue
            
            # 提取层索引
            try:
                if parts[0] == 'model':
                    layer_idx = int(parts[1])
                else:
                    layer_idx = int(parts[0])
            except ValueError:
                continue

            # RGB 分支
            for t_idx, s_idx in teacher_to_rgb.items():
                if layer_idx == s_idx:
                    # 构建 teacher key
                    if parts[0] == 'model':
                        teacher_key = f'model.{t_idx}.' + '.'.join(parts[2:])
                    else:
                        teacher_key = f'model.{t_idx}.' + '.'.join(parts[1:])
                    if teacher_key in rgb_state:
                        w = rgb_state[teacher_key]
                        if w.shape == param.data.shape:
                            param.data.copy_(w)
                            loaded_count += 1
                    param.requires_grad = False
                    frozen_count += 1
                    break

            # IR 分支
            for t_idx, s_idx in teacher_to_ir.items():
                if layer_idx == s_idx:
                    if parts[0] == 'model':
                        teacher_key = f'model.{t_idx}.' + '.'.join(parts[2:])
                    else:
                        teacher_key = f'model.{t_idx}.' + '.'.join(parts[1:])
                    if teacher_key in ir_state:
                        w = ir_state[teacher_key]
                        if w.shape == param.data.shape:
                            param.data.copy_(w)
                            loaded_count += 1
                    param.requires_grad = False
                    frozen_count += 1
                    break

        print(f"  ✓ 加载了 {loaded_count} 个 backbone 参数")
        print(f"  ✓ 冻结了 {frozen_count} 个 backbone 参数")
        print(f"  ✓ 冻结层索引: {sorted(self._frozen_layer_indices)}")

    def get_validator(self):
        from ultralytics.models import yolo
        custom_args = ['shift_weight', 'shift_ratio', 'shift_mask_weight']
        validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
        validator_args = SimpleNamespace(**validator_args_dict)
        validator = yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir,
                                                   args=validator_args, _callbacks=self.callbacks)
        validator.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'shift_loss']
        return validator

    def label_loss_items(self, loss_items=None, prefix='train'):
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'shift_loss'
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys


def create_shift_scheduler(args, total_epochs):
    if args.shift_schedule_config:
        config_path = args.shift_schedule_config
    elif args.shift_schedule in ['u_shape', 'l_shape']:
        config_path = str(Path(__file__).parent / 'configs' / f'shift_schedule_{args.shift_schedule}.yaml')
    else:
        return None

    if os.path.exists(config_path):
        scheduler = UShapeShiftScheduler(config_path=config_path, logger=LOGGER)
        scheduler.config['total_epochs'] = total_epochs
        scheduler.print_schedule_summary()
        return scheduler
    LOGGER.error(f"调度配置文件不存在: {config_path}")
    return None


def on_train_epoch_start(trainer):
    if not hasattr(trainer, 'shift_scheduler') or trainer.shift_scheduler is None:
        return
    config = trainer.shift_scheduler.get_config_for_epoch(trainer.epoch)
    shift_param_manager.update_params(ratio=config['shift_ratio'], weight=config['shift_weight'],
                                      mask_weight=config['shift_mask_weight'])
    if hasattr(trainer.model, 'args'):
        trainer.model.args.shift_weight = config['shift_weight']
        trainer.model.args.shift_ratio = config['shift_ratio']
        trainer.model.args.shift_mask_weight = config['shift_mask_weight']
    LOGGER.info(f"Epoch {trainer.epoch:3d}: ratio={config['shift_ratio']:.1f}, weight={config['shift_weight']:.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 Shift with Frozen Backbone')
    parser.add_argument('--rgb_teacher', type=str, required=True, help='RGB teacher 权重路径')
    parser.add_argument('--ir_teacher', type=str, required=True, help='IR teacher 权重路径')
    parser.add_argument('--data', type=str, required=True, help='数据集配置文件')
    parser.add_argument('--model', type=str, default='./model_yaml/yolov8m_shift_Fusion.yaml', help='模型配置文件')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--project', type=str, default='./runs/shift_frozen')
    parser.add_argument('--name', type=str, default='train')
    parser.add_argument('--shift_schedule', type=str, default='l_shape', choices=['u_shape', 'l_shape'])
    parser.add_argument('--shift_schedule_config', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        LOGGER.error(f"模型配置文件不存在: {args.model}")
        sys.exit(1)

    LOGGER.info("=" * 60)
    LOGGER.info("冻结 Backbone Shift 训练")
    LOGGER.info("=" * 60)
    LOGGER.info(f"RGB teacher: {args.rgb_teacher}")
    LOGGER.info(f"IR teacher:  {args.ir_teacher}")
    LOGGER.info(f"模型配置: {args.model}")
    LOGGER.info(f"数据集: {args.data}")
    LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")

    train_args = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'device': f'cuda:{args.device}' if args.device >= 0 else 'cpu',
        'batch': args.batch,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'lr0': args.lr,
        'shift_weight': 1.0,
        'shift_ratio': 0.3,
        'shift_mask_weight': 0.2,
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

    try:
        trainer = FrozenShiftTrainer(overrides=train_args, rgb_teacher=args.rgb_teacher, ir_teacher=args.ir_teacher)
        trainer.shift_scheduler = create_shift_scheduler(args, args.epochs)
        trainer.add_callback("on_train_epoch_start", on_train_epoch_start)
        trainer.train()
        LOGGER.info("=" * 60)
        LOGGER.info(f"✓ 训练完成！结果保存在: {trainer.save_dir}")
    except Exception as e:
        LOGGER.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()