# """
# Shift Detect 训练脚本

# 使用自定义 Trainer 训练带有全局 shift 预测的 YOLOv8 模型

# 使用方法:
#     python train_shift_global.py --scale m --epochs 100 --batch 16 --device 0

# 参数说明:
#     --scale: 模型规模 (n/s/m/l/x)
#     --epochs: 训练轮数
#     --batch: 批次大小
#     --device: GPU 设备号
#     --shift_weight: shift loss 的权重 λ (默认 1.0)
#     --data: 数据集配置文件路径
# """

# import argparse
# import os
# import sys
# from pathlib import Path
# from copy import copy

# # 添加 ultralytics 到路径
# sys.path.insert(0, str(Path(__file__).parent))

# import torch
# import yaml
# from ultralytics import YOLO
# from ultralytics.utils import LOGGER, DEFAULT_CFG_DICT, RANK
# from ultralytics.models.yolo.detect import DetectionTrainer
# from ultralytics.nn.tasks import ShiftDetectionModel
# from ultralytics.utils.shift_params import shift_param_manager
# from ultralytics.utils.shift_scheduler import UShapeShiftScheduler


# class ShiftDetectionTrainer(DetectionTrainer):
#     """
#     自定义 Trainer，使用 ShiftDetectionModel 和 v8ShiftDetectionLoss
#     """
    
#     def get_model(self, cfg=None, weights=None, verbose=True):
#         """返回 ShiftDetectionModel"""
#         model = ShiftDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
#         if weights:
#             model.load(weights)
#         return model
    
#     def set_model_attributes(self):
#         """设置模型属性，包括 shift_weight"""
#         super().set_model_attributes()
#         # 添加 shift_weight 到 model.args（已在 default.yaml 中注册）
#         self.model.args.shift_weight = self.args.shift_weight
    
#     def get_validator(self):
#         """返回 DetectionValidator，过滤掉自定义参数，并对齐 loss 维度"""
#         from ultralytics.models import yolo
#         from types import SimpleNamespace
        
#         custom_args = ['shift_weight', 'shift_ratio', 'shift_mask_weight', 'max_shift']
#         validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
#         validator_args = SimpleNamespace(**validator_args_dict)
        
#         # 1. 实例化官方的 Validator
#         validator = yolo.detect.DetectionValidator(self.test_loader,
#                                                    save_dir=self.save_dir,
#                                                    args=validator_args,
#                                                    _callbacks=self.callbacks)
        
#         # 2. 覆盖验证器的 loss_names，使其与输出的 4 个 loss 对应
#         validator.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'shift_loss']
        
#         return validator
    
#     def label_loss_items(self, loss_items=None, prefix='train'):
#         """
#         返回带标签的损失项
#         ShiftDetect 的损失包括: box, cls, dfl, shift
#         """
#         self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'shift_loss'
#         keys = [f'{prefix}/{x}' for x in self.loss_names]
#         if loss_items is not None:
#             loss_items = [round(float(x), 5) for x in loss_items]
#             return dict(zip(keys, loss_items))
#         else:
#             return keys


# def parse_args():
#     parser = argparse.ArgumentParser(description='Train YOLOv8 with Shift Detection')
#     parser.add_argument('--scale', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
#                         help='Model scale')
#     parser.add_argument('--epochs', type=int, default=120,
#                         help='Number of training epochs (U-shape: 120 recommended)')
#     parser.add_argument('--batch', type=int, default=16,
#                         help='Batch size')
#     parser.add_argument('--device', type=int, default=0,
#                         help='GPU device id')
#     parser.add_argument('--imgsz', type=int, default=640,
#                         help='Image size')
#     parser.add_argument('--shift_weight', type=float, default=1.0,
#                         help='Weight for shift loss (λ, will be overridden by schedule)')
#     parser.add_argument('--shift_ratio', type=float, default=0.3,
#                         help='Ratio of objects to shift (will be overridden by schedule)')
#     parser.add_argument('--shift_mask_weight', type=float, default=0.2,
#                         help='Weight for unshifted samples (will be overridden by schedule)')
#     parser.add_argument('--shift_schedule', type=str, default='u_shape',
#                         choices=['u_shape', 'fixed'],
#                         help='Shift training schedule (u_shape recommended)')
#     parser.add_argument('--shift_schedule_config', type=str, default=None,
#                         help='Path to custom shift schedule config file (YAML format)')
#     parser.add_argument('--data', type=str, default='./data/FLIR.yaml',
#                         help='Path to data config file')
#     parser.add_argument('--project', type=str, default='./runs/shift_u_shape',
#                         help='Project name')
#     parser.add_argument('--name', type=str, default='train',
#                         help='Experiment name')
#     parser.add_argument('--lr', type=float, default=0.01,
#                         help='Initial learning rate')
#     parser.add_argument('--workers', type=int, default=8,
#                         help='Number of dataloader workers')
#     parser.add_argument('--model', type=str, default='./model_yaml/yolov8_shift_v2.yaml',
#                         help='Path to model config file')
#     return parser.parse_args()


# def create_u_shape_scheduler(args, total_epochs):
#     """
#     创建 U 型调度器

#     Args:
#         args: 命令行参数
#         total_epochs: 总训练轮数

#     Returns:
#         UShapeShiftScheduler 实例，如果 args 中指定了自定义配置则使用自定义配置
#     """
#     if args.shift_schedule_config:
#         # 使用自定义配置文件
#         scheduler = UShapeShiftScheduler(
#             config_path=args.shift_schedule_config,
#             logger=LOGGER
#         )
#     elif args.shift_schedule == 'u_shape':
#         # 使用预定义的 U 型策略
#         config_path = Path(__file__).parent / 'configs' / 'shift_schedule_u_shape.yaml'
#         if config_path.exists():
#             scheduler = UShapeShiftScheduler(
#                 config_path=str(config_path),
#                 logger=LOGGER
#             )
#             # 覆盖配置文件中的 total_epochs（以命令行参数为准）
#             scheduler.config['total_epochs'] = total_epochs
#         else:
#             LOGGER.error(f"U型调度配置文件不存在: {config_path}")
#             return None
#     else:
#         # 使用默认行为（固定参数）
#         return None

#     # 打印训练计划摘要
#     if scheduler:
#         scheduler.print_schedule_summary()

#     return scheduler


# def on_train_epoch_start(trainer):
#     """
#     每个 Epoch 开始时触发的回调函数
#     动态调整 Shift 参数（U型策略）+ TensorBoard 可视化
#     """
#     if not hasattr(trainer, 'shift_scheduler') or trainer.shift_scheduler is None:
#         return

#     epoch = trainer.epoch
#     scheduler = trainer.shift_scheduler

#     # 获取当前阶段的配置（无状态方法）
#     config = scheduler.get_config_for_epoch(epoch)

#     # 🔥 关键：更新共享内存参数（跨进程安全）
#     shift_param_manager.update_params(
#         ratio=config['shift_ratio'],
#         weight=config['shift_weight'],
#         mask_weight=config['shift_mask_weight']
#     )

#     # 更新模型的 args（防止某些地方从 args 读取）
#     if hasattr(trainer.model, 'args'):
#         trainer.model.args.shift_weight = config['shift_weight']
#         trainer.model.args.shift_ratio = config['shift_ratio']
#         trainer.model.args.shift_mask_weight = config['shift_mask_weight']

#     # 🎨 TensorBoard 可视化：把参数塞进 metrics
#     if not hasattr(trainer, 'metrics'):
#         trainer.metrics = {}
#     trainer.metrics['custom/shift_ratio'] = config['shift_ratio']
#     trainer.metrics['custom/shift_weight'] = config['shift_weight']
#     trainer.metrics['custom/shift_mask_weight'] = config['shift_mask_weight']

#     # 📊 输出简洁的状态信息
#     LOGGER.info(f"Epoch {epoch:3d}: ratio={config['shift_ratio']:.1f}, weight={config['shift_weight']:.1f}, mask_weight={config['shift_mask_weight']:.1f}")


# def main():
#     args = parse_args()

#     # 模型配置文件路径
#     model_yaml = args.model

#     if not os.path.exists(model_yaml):
#         LOGGER.error(f"❌ 模型配置文件不存在: {model_yaml}")
#         raise FileNotFoundError(f"Model config not found: {model_yaml}")

#     LOGGER.info("=" * 60)
#     LOGGER.info("开始训练 ShiftDetect 模型（U型策略）")
#     LOGGER.info("=" * 60)
#     LOGGER.info(f"模型配置: {model_yaml}")
#     LOGGER.info(f"Scale: {args.scale}")
#     LOGGER.info(f"数据集: {args.data}")
#     LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
#     LOGGER.info(f"Shift schedule: {args.shift_schedule}")
#     LOGGER.info(f"设备: cuda:{args.device}" if args.device >= 0 else "设备: cpu")

#     # 准备训练参数
#     train_args = {
#         'model': model_yaml,
#         'data': args.data,
#         'epochs': args.epochs,
#         'imgsz': args.imgsz,
#         'device': f'cuda:{args.device}' if args.device >= 0 else 'cpu',
#         'batch': args.batch,
#         'workers': args.workers,
#         'project': args.project,
#         'name': f'yolov8{args.scale}_shift_u_shape',
#         'lr0': args.lr,
#         'shift_weight': args.shift_weight,
#         'shift_ratio': args.shift_ratio,
#         'shift_mask_weight': args.shift_mask_weight,
#         'patience': 50,
#         'save_period': 10,
#         'close_mosaic': 10,
#         'mosaic': 1.0,
#         'mixup': 0.0,
#         'copy_paste': 0.0,
#         'optimizer': 'SGD',
#         'momentum': 0.937,
#         'weight_decay': 0.0005,
#     }

#     # 创建 ShiftDetectionTrainer
#     try:
#         trainer = ShiftDetectionTrainer(overrides=train_args)

#         # 创建 U 型调度器
#         trainer.shift_scheduler = create_u_shape_scheduler(args, args.epochs)

#         # 添加回调
#         trainer.add_callback("on_train_epoch_start", on_train_epoch_start)

#         LOGGER.info("")
#         LOGGER.info("🚀 开始 U 型训练...")
#         trainer.train()

#         # 训练完成后，输出阶段转换摘要
#         if hasattr(trainer, 'shift_scheduler') and trainer.shift_scheduler:
#             LOGGER.info("")
#             LOGGER.info("=" * 70)
#             LOGGER.info("📊 阶段转换摘要")
#             LOGGER.info("=" * 70)
#             transitions = trainer.shift_scheduler.get_stage_summary()
#             for transition in transitions:
#                 LOGGER.info(f"Epoch {transition['epoch']:3d}: {transition['from_stage']} → {transition['to_stage']}")
#             LOGGER.info("=" * 70)

#         LOGGER.info("=" * 60)
#         LOGGER.info("✓ 训练完成！")
#         LOGGER.info(f"结果保存在: {trainer.save_dir}")

#     except Exception as e:
#         LOGGER.error(f"❌ 训练失败: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == '__main__':
#     main()


"""
Shift Detect 训练脚本

使用自定义 Trainer 训练带有全局 shift 预测的 YOLOv8 模型

使用方法:
    python train_shift_global.py --scale m --epochs 120 --batch 16 --device 0 --shift_schedule l_shape
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
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import ShiftDetectionModel
from ultralytics.utils.shift_params import shift_param_manager
# 注意：即使导入的名字叫 UShapeShiftScheduler，它实际上是一个通用的多阶段无状态调度器
from ultralytics.utils.shift_scheduler import UShapeShiftScheduler


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
        # 添加 shift_weight 到 model.args
        self.model.args.shift_weight = self.args.shift_weight
    
    def get_validator(self):
        """返回 DetectionValidator，过滤掉自定义参数，并对齐 loss 维度"""
        from ultralytics.models import yolo
        from types import SimpleNamespace
        
        custom_args = ['shift_weight', 'shift_ratio', 'shift_mask_weight', 'max_shift']
        validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
        validator_args = SimpleNamespace(**validator_args_dict)
        
        validator = yolo.detect.DetectionValidator(self.test_loader,
                                                   save_dir=self.save_dir,
                                                   args=validator_args,
                                                   _callbacks=self.callbacks)
        
        # 覆盖验证器的 loss_names
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
    parser = argparse.ArgumentParser(description='Train YOLOv8 with Shift Detection')
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
                        help='Weight for shift loss (λ, will be overridden by schedule)')
    parser.add_argument('--shift_ratio', type=float, default=0.3,
                        help='Ratio of objects to shift (will be overridden by schedule)')
    parser.add_argument('--shift_mask_weight', type=float, default=0.2,
                        help='Weight for unshifted samples (will be overridden by schedule)')
    # 🔥 核心修改 1：支持 l_shape，并将默认设为 l_shape
    parser.add_argument('--shift_schedule', type=str, default='l_shape',
                        choices=['u_shape', 'l_shape', 'fixed'],
                        help='Shift training schedule (l_shape or u_shape)')
    parser.add_argument('--shift_schedule_config', type=str, default=None,
                        help='Path to custom shift schedule config file (YAML format)')
    parser.add_argument('--data', type=str, default='./data/FLIR.yaml',
                        help='Path to data config file')
    # 🔥 核心修改 2：默认保存目录更改，避免和之前的 u_shape 混淆
    parser.add_argument('--project', type=str, default='./runs/shift_l_shape',
                        help='Project name')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--model', type=str, default='./model_yaml/yolov8_shift_v2.yaml',
                        help='Path to model config file')
    return parser.parse_args()


def create_shift_scheduler(args, total_epochs):
    """
    🔥 核心修改 3：通用的调度器创建函数
    根据传入的参数动态寻找 yaml 文件
    """
    if args.shift_schedule_config:
        config_path = args.shift_schedule_config
    elif args.shift_schedule in ['u_shape', 'l_shape']:
        # 动态拼接文件名：configs/shift_schedule_l_shape.yaml
        config_path = str(Path(__file__).parent / 'configs' / f'shift_schedule_{args.shift_schedule}.yaml')
    else:
        return None

    if os.path.exists(config_path):
        scheduler = UShapeShiftScheduler(config_path=config_path, logger=LOGGER)
        scheduler.config['total_epochs'] = total_epochs
        scheduler.print_schedule_summary()
        return scheduler
    else:
        LOGGER.error(f"❌ 调度配置文件不存在: {config_path}")
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

    # 获取当前阶段的配置（无状态方法）
    config = scheduler.get_config_for_epoch(epoch)

    # 更新共享内存参数（跨进程安全）
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
        LOGGER.error(f"❌ 模型配置文件不存在: {model_yaml}")
        raise FileNotFoundError(f"Model config not found: {model_yaml}")

    LOGGER.info("=" * 60)
    # 🔥 核心修改 4：日志动态打印
    LOGGER.info(f"开始训练 ShiftDetect 模型 ({args.shift_schedule} 策略)")
    LOGGER.info("=" * 60)
    LOGGER.info(f"模型配置: {model_yaml}")
    LOGGER.info(f"Scale: {args.scale}")
    LOGGER.info(f"数据集: {args.data}")
    LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    LOGGER.info(f"Shift schedule: {args.shift_schedule}")
    LOGGER.info(f"设备: cuda:{args.device}" if args.device >= 0 else "设备: cpu")

    # 动态实验名称
    exp_name = f'yolov8{args.scale}_shift_{args.shift_schedule}'

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

    try:
        trainer = ShiftDetectionTrainer(overrides=train_args)

        # 创建通用调度器
        trainer.shift_scheduler = create_shift_scheduler(args, args.epochs)
        trainer.add_callback("on_train_epoch_start", on_train_epoch_start)

        LOGGER.info("")
        LOGGER.info(f"🚀 开始 {args.shift_schedule} 调度训练...")
        trainer.train()

        if hasattr(trainer, 'shift_scheduler') and trainer.shift_scheduler:
            LOGGER.info("")
            LOGGER.info("=" * 70)
            LOGGER.info("📊 阶段转换摘要")
            LOGGER.info("=" * 70)
            transitions = trainer.shift_scheduler.get_stage_summary()
            for transition in transitions:
                LOGGER.info(f"Epoch {transition['epoch']:3d}: {transition['from_stage']} → {transition['to_stage']}")
            LOGGER.info("=" * 70)

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
