"""
Shift Detect OBB 训练脚本

使用自定义 Trainer 训练带有全局 shift 预测的 YOLOv8 OBB 模型

使用方法:
    python train_shift_global_obb.py --scale m --epochs 120 --batch 16 --device 0 --shift_schedule l_shape
"""

import argparse
import os
import sys
from pathlib import Path
from copy import copy

# 添加 ultralytics 到路径
sys.path.insert(0, str(Path(__file__).parent))

import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER, DEFAULT_CFG_DICT, RANK
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel, ShiftOBBModel
from ultralytics.utils.shift_params import shift_param_manager
from ultralytics.utils.shift_scheduler import UShapeShiftScheduler


class ShiftOBBTrainer(OBBTrainer):
    """
    自定义 OBB Trainer，支持 Shift 训练策略
    继承自 OBBTrainer，保留 OBB 的所有功能
    """
    
    def __init__(self, cfg=DEFAULT_CFG_DICT, overrides=None, _callbacks=None):
        """初始化 ShiftOBBTrainer"""
        if overrides is None:
            overrides = {}
        # 添加 shift 相关参数
        overrides.setdefault('shift_weight', 1.0)
        overrides.setdefault('shift_ratio', 0.3)
        overrides.setdefault('shift_mask_weight', 0.2)
        super().__init__(cfg, overrides, _callbacks)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 ShiftOBBModel（支持 shift loss）"""
        model = ShiftOBBModel(cfg, ch=6, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def set_model_attributes(self):
        """设置模型属性，包括 shift_weight"""
        super().set_model_attributes()
        # 添加 shift 相关参数到 model.args
        if hasattr(self.model, 'args'):
            self.model.args.shift_weight = self.args.get('shift_weight', 1.0)
            self.model.args.shift_ratio = self.args.get('shift_ratio', 0.3)
            self.model.args.shift_mask_weight = self.args.get('shift_mask_weight', 0.2)
    
    def get_validator(self):
        """返回 OBBValidator"""
        from ultralytics.models import yolo
        from types import SimpleNamespace
        
        # 过滤掉自定义参数
        custom_args = ['shift_weight', 'shift_ratio', 'shift_mask_weight', 'max_shift']
        validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
        validator_args = SimpleNamespace(**validator_args_dict)
        
        validator = yolo.obb.OBBValidator(self.test_loader,
                                          save_dir=self.save_dir,
                                          args=validator_args,
                                          _callbacks=self.callbacks)
        # OBB 的 loss_names（包含 shift_loss）
        validator.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'shift_loss']
        return validator
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """返回带标签的损失项"""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'shift_loss'
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 OBB with Shift Detection')
    parser.add_argument('--scale', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model scale')
    parser.add_argument('--epochs', type=int, default=120,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--shift_weight', type=float, default=1.0,
                        help='Weight for shift loss (will be overridden by schedule)')
    parser.add_argument('--shift_ratio', type=float, default=0.3,
                        help='Ratio of objects to shift (will be overridden by schedule)')
    parser.add_argument('--shift_mask_weight', type=float, default=0.2,
                        help='Weight for unshifted samples (will be overridden by schedule)')
    parser.add_argument('--shift_schedule', type=str, default='l_shape',
                        choices=['u_shape', 'l_shape', 'fixed'],
                        help='Shift training schedule')
    parser.add_argument('--shift_schedule_config', type=str, default=None,
                        help='Path to custom shift schedule config file (YAML format)')
    parser.add_argument('--data', type=str, default='./data/DroneVehicle.yaml',
                        help='Path to data config file (OBB format)')
    parser.add_argument('--project', type=str, default='./runs/shift_obb_l_shape',
                        help='Project name')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--model', type=str, default='./model_yaml_obb/yolov8_obb_shift.yaml',
                        help='Path to model config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint path (e.g., path/to/last.pt)')
    return parser.parse_args()


def create_shift_scheduler(args, total_epochs):
    """
    创建 Shift 调度器
    """
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
    else:
        LOGGER.error(f"调度配置文件不存在: {config_path}")
        return None


def on_train_epoch_start(trainer):
    """
    每个 Epoch 开始时触发的回调函数
    动态调整 Shift 参数 + TensorBoard 可视化
    """
    if not hasattr(trainer, 'shift_scheduler') or trainer.shift_scheduler is None:
        return

    epoch = trainer.epoch
    scheduler = trainer.shift_scheduler

    # 获取当前阶段的配置
    config = scheduler.get_config_for_epoch(epoch)

    # 更新共享内存参数
    shift_param_manager.update_params(
        ratio=config['shift_ratio'],
        weight=config['shift_weight'],
        mask_weight=config['shift_mask_weight']
    )

    # 更新模型的 args
    if hasattr(trainer.model, 'args'):
        trainer.model.args.shift_weight = config['shift_weight']
        trainer.model.args.shift_ratio = config['shift_ratio']
        trainer.model.args.shift_mask_weight = config['shift_mask_weight']

    # TensorBoard 可视化
    if not hasattr(trainer, 'metrics'):
        trainer.metrics = {}
    trainer.metrics['custom/shift_ratio'] = config['shift_ratio']
    trainer.metrics['custom/shift_weight'] = config['shift_weight']
    trainer.metrics['custom/shift_mask_weight'] = config['shift_mask_weight']

    LOGGER.info(f"Epoch {epoch:3d}: ratio={config['shift_ratio']:.1f}, weight={config['shift_weight']:.1f}, mask_weight={config['shift_mask_weight']:.1f}")


def main():
    args = parse_args()

    model_yaml = args.model
    if not os.path.exists(model_yaml):
        LOGGER.error(f"模型配置文件不存在: {model_yaml}")
        raise FileNotFoundError(f"Model config not found: {model_yaml}")

    LOGGER.info("=" * 60)
    LOGGER.info(f"开始训练 Shift OBB 模型 ({args.shift_schedule} 策略)")
    LOGGER.info("=" * 60)
    LOGGER.info(f"模型配置: {model_yaml}")
    LOGGER.info(f"Scale: {args.scale}")
    LOGGER.info(f"数据集: {args.data}")
    LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    LOGGER.info(f"Shift schedule: {args.shift_schedule}")
    LOGGER.info(f"设备: cuda:{args.device}" if args.device >= 0 else "设备: cpu")

    # 动态实验名称
    exp_name = f'yolov8{args.scale}_shift_obb_{args.shift_schedule}'

    train_args = {
        'model': model_yaml,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'device': f'cuda:{args.device}' if args.device >= 0 else 'cpu',
        'batch': args.batch,
        'workers': args.workers,
        'project': args.project,
        'name': exp_name,
        'lr0': args.lr,
        'shift_weight': args.shift_weight,
        'shift_ratio': args.shift_ratio,
        'shift_mask_weight': args.shift_mask_weight,
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

    # 支持 resume
    if args.resume:
        train_args['resume'] = args.resume
        LOGGER.info(f"从检查点恢复训练: {args.resume}")

    try:
        trainer = ShiftOBBTrainer(overrides=train_args)

        # 创建调度器
        trainer.shift_scheduler = create_shift_scheduler(args, args.epochs)
        trainer.add_callback("on_train_epoch_start", on_train_epoch_start)

        LOGGER.info("")
        LOGGER.info(f"开始 {args.shift_schedule} 调度训练...")
        trainer.train()

        if hasattr(trainer, 'shift_scheduler') and trainer.shift_scheduler:
            LOGGER.info("")
            LOGGER.info("=" * 70)
            LOGGER.info("阶段转换摘要")
            LOGGER.info("=" * 70)
            transitions = trainer.shift_scheduler.get_stage_summary()
            for transition in transitions:
                LOGGER.info(f"Epoch {transition['epoch']:3d}: {transition['from_stage']} → {transition['to_stage']}")
            LOGGER.info("=" * 70)

        LOGGER.info("=" * 60)
        LOGGER.info("训练完成！")
        LOGGER.info(f"结果保存在: {trainer.save_dir}")

    except Exception as e:
        LOGGER.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
