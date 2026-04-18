#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版 Shift 训练脚本（使用 SimpleShiftHead，无注意力机制）

用法：
    python train_shift_simple.py \
        --data data/LLVIP.yaml \
        --rgb_teacher teacherTraining/runs/LLVIP_RGB_SHIFT_ADV/weights/best.pt \
        --ir_teacher teacherTraining/runs/LLVIP_IR_SHIFT_ADV/weights/best.pt \
        --shift_schedule l_shape
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import ShiftDetectionModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.shift_scheduler import UShapeShiftScheduler
from ultralytics.utils.shift_params import shift_param_manager


class SimpleShiftTrainer(DetectionTrainer):
    """简化版 Shift 训练器（使用 SimpleShiftHead）"""

    def __init__(self, rgb_teacher=None, ir_teacher=None, **kwargs):
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        super().__init__(**kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 ShiftDetectionModel 并加载冻结的 backbone 权重"""
        model = ShiftDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        # 加载冻结 backbone 权重
        if self.rgb_teacher and self.ir_teacher:
            self._load_frozen_backbones(model)

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

        teacher_to_rgb = {0:3, 1:5, 2:7, 3:9, 4:11, 5:14, 6:16, 7:19, 8:21}
        teacher_to_ir = {0:4, 1:6, 2:8, 3:10, 4:12, 5:15, 6:17, 7:20, 8:22}

        loaded_count = 0
        frozen_count = 0

        inner_model = model.model if hasattr(model, 'model') else model

        for name, param in inner_model.named_parameters():
            parts = name.split('.')
            if len(parts) < 2:
                continue
            try:
                layer_idx = int(parts[0])
            except ValueError:
                continue

            for t_idx, s_idx in teacher_to_rgb.items():
                if layer_idx == s_idx:
                    teacher_key = f'model.{t_idx}.' + '.'.join(parts[1:])
                    if teacher_key in rgb_state:
                        w = rgb_state[teacher_key]
                        if w.shape == param.data.shape:
                            param.data.copy_(w)
                            loaded_count += 1
                    param.requires_grad = False
                    frozen_count += 1
                    break

            for t_idx, s_idx in teacher_to_ir.items():
                if layer_idx == s_idx:
                    teacher_key = f'model.{t_idx}.' + '.'.join(parts[1:])
                    if teacher_key in ir_state:
                        w = ir_state[teacher_key]
                        if w.shape == param.data.shape:
                            param.data.copy_(w)
                            loaded_count += 1
                    param.requires_grad = False
                    frozen_count += 1

        print(f"  ✓ 加载了 {loaded_count} 个 backbone 参数")
        print(f"  ✓ 冻结了 {frozen_count} 个 backbone 参数")


def create_scheduler(args, total_epochs):
    """创建训练策略调度器"""
    if args.shift_schedule == "none":
        return None
    
    config_path = str(Path(__file__).parent / 'configs' / f'shift_schedule_{args.shift_schedule}.yaml')
    
    if Path(config_path).exists():
        scheduler = UShapeShiftScheduler(config_path=config_path, logger=LOGGER)
        scheduler.config['total_epochs'] = total_epochs
        scheduler.print_schedule_summary()
        return scheduler
    else:
        LOGGER.warning(f"调度配置文件不存在: {config_path}")
        return None


def on_train_epoch_start(trainer):
    """每个 Epoch 开始时更新 Shift 参数"""
    if not hasattr(trainer, 'shift_scheduler') or trainer.shift_scheduler is None:
        return

    epoch = trainer.epoch
    config = trainer.shift_scheduler.get_config_for_epoch(epoch)

    shift_param_manager.update_params(
        ratio=config['shift_ratio'],
        weight=config['shift_weight'],
        mask_weight=config['shift_mask_weight']
    )

    if hasattr(trainer.model, 'args'):
        trainer.model.args.shift_weight = config['shift_weight']
        trainer.model.args.shift_ratio = config['shift_ratio']
        trainer.model.args.shift_mask_weight = config['shift_mask_weight']

    LOGGER.info(f"Epoch {epoch:3d}: ratio={config['shift_ratio']:.1f}, weight={config['shift_weight']:.1f}")


def main():
    parser = argparse.ArgumentParser(description="简化版 Shift 训练脚本")
    parser.add_argument("--rgb_teacher", type=str, required=True)
    parser.add_argument("--ir_teacher", type=str, required=True)
    parser.add_argument("--shift_schedule", type=str, default="l_shape", choices=["l_shape", "u_shape", "none"])
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, default="./model_yaml/yolov8_simple_shift.yaml")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--close_mosaic", type=int, default=10)
    parser.add_argument("--project", type=str, default="runs/shift_simple")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    print("\n" + "="*70)
    print("简化版 Shift 训练 (SimpleShiftHead - 无注意力)")
    print("="*70)
    print(f"  Model:      {args.model}")
    print(f"  Data:       {args.data}")
    print(f"  RGB Teacher: {args.rgb_teacher}")
    print(f"  IR Teacher:  {args.ir_teacher}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch:      {args.batch}")
    print(f"  Schedule:   {args.shift_schedule}")
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"simple_shift_{args.shift_schedule}_{timestamp}" if args.name is None else args.name

    train_args = {
        'model': args.model,
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': f'cuda:{args.device}' if args.device >= 0 else 'cpu',
        'workers': args.workers,
        'lr0': args.lr0,
        'project': args.project,
        'name': exp_name,
        'patience': args.patience,
        'close_mosaic': args.close_mosaic,
    }

    try:
        trainer = SimpleShiftTrainer(
            rgb_teacher=args.rgb_teacher,
            ir_teacher=args.ir_teacher,
            overrides=train_args
        )

        trainer.shift_scheduler = create_scheduler(args, args.epochs)
        if trainer.shift_scheduler:
            trainer.add_callback("on_train_epoch_start", on_train_epoch_start)

        trainer.train()

        print("\n" + "="*70)
        print("✅ 训练完成!")
        print(f"  结果保存在: {trainer.save_dir}")
        print("="*70)

    except Exception as e:
        LOGGER.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()