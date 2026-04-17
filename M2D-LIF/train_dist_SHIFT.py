"""
M²D 蒸馏 + Shift 训练脚本

结合官方 M²D-LIF 的蒸馏方法（M²D）和 Shift 预测模块：
- 保留 M²D 蒸馏：用预训练的单模态 teacher 蒸馏多模态 backbone
- 关闭 LIF：使用简单的 Add 融合（通过 yolov8_shift_v2.yaml）
- 引入 Shift 模块：预测 RGB-IR 之间的偏移

使用方法:
    python train_dist_SHIFT.py --scale m --epochs 120 --batch 8 --device 0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import attempt_load_one_weight, ShiftDetectionModel
from ultralytics.utils import LOGGER, DEFAULT_CFG, RANK
from ultralytics.utils.shift_params import shift_param_manager
from ultralytics.utils.shift_scheduler import UShapeShiftScheduler


class DistillShiftDetectionTrainer(DetectionTrainer):
    """
    支持 M²D 蒸馏 + Shift 预测的 Trainer
    
    继承 DetectionTrainer，复用其蒸馏功能（在 BaseTrainer 中实现）
    同时支持 Shift 相关的参数调度
    
    注意：蒸馏训练时 loss_items 会有 6 项（4检测 + 2蒸馏），
    但验证时模型只返回 4 项，需要特殊处理
    """
    
    # 定义基础 loss 名称（不含蒸馏 loss）
    BASE_LOSS_NAMES = ('box_loss', 'cls_loss', 'dfl_loss', 'shift_loss')
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 ShiftDetectionModel"""
        model = ShiftDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model
    
    def set_model_attributes(self):
        """设置模型属性，包括 shift_weight"""
        super().set_model_attributes()
        self.model.args.shift_weight = self.args.shift_weight
    
    def get_validator(self):
        """返回 DetectionValidator，过滤掉自定义参数"""
        from ultralytics.models import yolo
        from types import SimpleNamespace
        
        custom_args = ['shift_weight', 'shift_ratio', 'shift_mask_weight', 'max_shift']
        validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
        validator_args = SimpleNamespace(**validator_args_dict)
        
        validator = yolo.detect.DetectionValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks
        )
        validator.loss_names = list(self.BASE_LOSS_NAMES)
        return validator
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        返回带标签的损失项
        
        训练时可能有 6 项（含蒸馏 loss），但只显示前 4 项基础 loss
        """
        if loss_items is not None:
            # 只取前 4 项基础 loss（不含蒸馏 loss）
            if len(loss_items) > 4:
                loss_items = loss_items[:4]
            loss_items = [round(float(x), 5) for x in loss_items]
            keys = [f'{prefix}/{x}' for x in self.BASE_LOSS_NAMES]
            return dict(zip(keys, loss_items))
        else:
            return [f'{prefix}/{x}' for x in self.BASE_LOSS_NAMES]


def parse_args():
    parser = argparse.ArgumentParser(description='Train with M²D Distillation + Shift')
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
    
    # Shift 相关参数
    parser.add_argument('--shift_weight', type=float, default=1.0,
                        help='Weight for shift loss')
    parser.add_argument('--shift_ratio', type=float, default=0.3,
                        help='Ratio of objects to shift')
    parser.add_argument('--shift_mask_weight', type=float, default=0.2,
                        help='Weight for unshifted samples')
    parser.add_argument('--shift_schedule', type=str, default='l_shape',
                        choices=['u_shape', 'l_shape', 'fixed'],
                        help='Shift training schedule')
    parser.add_argument('--shift_schedule_config', type=str, default=None,
                        help='Path to custom shift schedule config file')
    
    # 蒸馏相关参数
    parser.add_argument('--distill_weight', type=float, default=0.8,
                        help='Distillation loss weight')
    parser.add_argument('--loss_type', type=str, default='CWD',
                        choices=['CWD', 'PKD'],
                        help='Distillation loss type')
    
    # 数据和模型
    parser.add_argument('--data', type=str, default='./data/FLIR.yaml',
                        help='Path to data config file')
    parser.add_argument('--model', type=str, default='./model_yaml/yolov8_shift_v2.yaml',
                        help='Path to model config file')
    parser.add_argument('--teacher_rgb', type=str, default=None,
                        help='Path to RGB teacher model')
    parser.add_argument('--teacher_ir', type=str, default=None,
                        help='Path to IR teacher model')
    parser.add_argument('--project', type=str, default='./runs/dist_shift',
                        help='Project name')
    parser.add_argument('--name', type=str, default='train',
                        help='Experiment name')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    
    return parser.parse_args()


def create_shift_scheduler(args, total_epochs):
    """创建 Shift 调度器"""
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
    """每个 Epoch 开始时调整 Shift 参数"""
    if not hasattr(trainer, 'shift_scheduler') or trainer.shift_scheduler is None:
        return

    epoch = trainer.epoch
    scheduler = trainer.shift_scheduler
    config = scheduler.get_config_for_epoch(epoch)

    # 更新共享内存参数
    shift_param_manager.update_params(
        ratio=config['shift_ratio'],
        weight=config['shift_weight'],
        mask_weight=config['shift_mask_weight']
    )

    # 更新模型 args
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

    # 检查 teacher 模型路径
    if not args.teacher_rgb or not args.teacher_ir:
        LOGGER.error("必须提供 --teacher_rgb 和 --teacher_ir 参数")
        sys.exit(1)
    
    if not os.path.exists(args.teacher_rgb):
        LOGGER.error(f"RGB teacher 模型不存在: {args.teacher_rgb}")
        sys.exit(1)
    
    if not os.path.exists(args.teacher_ir):
        LOGGER.error(f"IR teacher 模型不存在: {args.teacher_ir}")
        sys.exit(1)

    if not os.path.exists(args.model):
        LOGGER.error(f"模型配置文件不存在: {args.model}")
        sys.exit(1)

    # 加载 teacher 模型
    LOGGER.info("=" * 70)
    LOGGER.info("加载 Teacher 模型...")
    LOGGER.info("=" * 70)
    
    _, model_t_rgb = attempt_load_one_weight(args.teacher_rgb)
    model_t_rgb["model"].info()
    
    _, model_t_ir = attempt_load_one_weight(args.teacher_ir)
    model_t_ir["model"].info()

    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info(f"M²D 蒸馏 + Shift 训练 ({args.shift_schedule} 策略)")
    LOGGER.info("=" * 70)
    LOGGER.info(f"模型配置: {args.model}")
    LOGGER.info(f"Scale: {args.scale}")
    LOGGER.info(f"数据集: {args.data}")
    LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    LOGGER.info(f"蒸馏参数: weight={args.distill_weight}, loss_type={args.loss_type}")
    LOGGER.info(f"Shift schedule: {args.shift_schedule}")
    LOGGER.info(f"设备: cuda:{args.device}" if args.device >= 0 else "设备: cpu")

    exp_name = f'yolov8{args.scale}_dist_shift_{args.shift_schedule}'
    
    # 训练参数
    train_args = {
        # 模型和数据
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'device': f'cuda:{args.device}' if args.device >= 0 else 'cpu',
        'batch': args.batch,
        'workers': args.workers,
        'project': args.project,
        'name': exp_name,
        'lr0': args.lr,
        
        # Shift 参数
        'shift_weight': args.shift_weight,
        'shift_ratio': args.shift_ratio,
        'shift_mask_weight': args.shift_mask_weight,
        
        # 蒸馏参数
        'Distillation': 'MultiDistillation',
        'distill_weight': args.distill_weight,
        'Teacher_Model_RGB': model_t_rgb["model"],
        'Teacher_Model_IR': model_t_ir["model"],
        'loss_type': args.loss_type,
        'online': False,
        
        # 其他训练参数
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
        # 创建 Trainer
        trainer = DistillShiftDetectionTrainer(overrides=train_args)

        # 创建 Shift 调度器
        trainer.shift_scheduler = create_shift_scheduler(args, args.epochs)
        trainer.add_callback("on_train_epoch_start", on_train_epoch_start)

        DEFAULT_CFG.save_dir = trainer.save_dir

        LOGGER.info("")
        LOGGER.info(f"开始训练...")
        trainer.train()

        LOGGER.info("=" * 70)
        LOGGER.info("训练完成！")
        LOGGER.info(f"结果保存在: {trainer.save_dir}")

    except Exception as e:
        LOGGER.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
