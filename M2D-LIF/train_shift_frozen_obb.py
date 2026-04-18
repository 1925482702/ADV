#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train the Shift OBB model with frozen dual-branch backbones."""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import ShiftOBBModel
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.shift_params import shift_param_manager
from ultralytics.utils.shift_scheduler import UShapeShiftScheduler


# Teacher -> student mappings for model_yaml_obb/yolov8_obb_shift.yaml.
TEACHER_TO_RGB = {0: 3, 1: 5, 2: 7, 3: 10, 4: 12, 5: 15, 6: 17, 7: 20, 8: 22}
TEACHER_TO_IR = {0: 4, 1: 6, 2: 8, 3: 11, 4: 13, 5: 16, 6: 18, 7: 21, 8: 23}

STUDENT_RGB_TO_TEACHER = {student_idx: teacher_idx for teacher_idx, student_idx in TEACHER_TO_RGB.items()}
STUDENT_IR_TO_TEACHER = {student_idx: teacher_idx for teacher_idx, student_idx in TEACHER_TO_IR.items()}

FROZEN_LAYER_INDICES = sorted(set(TEACHER_TO_RGB.values()) | set(TEACHER_TO_IR.values()))


def _extract_layer_idx(name):
    """Extract a module index from parameter names like model.3.conv.weight."""
    parts = name.split(".")
    if len(parts) < 2:
        return None
    if parts[0] == "model" and parts[1].isdigit():
        return int(parts[1])
    if parts[0].isdigit():
        return int(parts[0])
    return None


def _build_teacher_key(student_param_name, teacher_layer_idx):
    """Convert a student parameter name into the corresponding teacher key."""
    parts = student_param_name.split(".")
    suffix = parts[2:] if parts[0] == "model" else parts[1:]
    return f"model.{teacher_layer_idx}." + ".".join(suffix)


def _format_examples(examples):
    """Format missing or mismatched examples for logging."""
    formatted = []
    for example in examples:
        if example["type"] == "missing":
            formatted.append(
                f"student='{example['student_param']}', teacher='{example['teacher_key']}'"
            )
        else:
            formatted.append(
                "student='{student}', teacher='{teacher}', student_shape={s_shape}, teacher_shape={t_shape}".format(
                    student=example["student_param"],
                    teacher=example["teacher_key"],
                    s_shape=example["student_shape"],
                    t_shape=example["teacher_shape"],
                )
            )
    return formatted


class FrozenShiftOBBTrainer(OBBTrainer):
    """OBB trainer for Shift models with frozen dual-branch backbones."""

    def __init__(self, rgb_teacher=None, ir_teacher=None, **kwargs):
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self._freeze_backbone_mode = True
        self._frozen_layer_indices = list(FROZEN_LAYER_INDICES)
        self._backbone_load_summary = {}
        super().__init__(**kwargs)

    def _setup_train(self, world_size):
        """Run standard setup, then rebuild optimizer and scheduler for trainable params only."""
        super()._setup_train(world_size)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_params)
        frozen_count = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        LOGGER.info(f"Trainable parameters: {trainable_count:,}, frozen parameters: {frozen_count:,}")

        if not trainable_params or self.optimizer is None:
            return

        rebuilt = self._rebuild_optimizer_for_trainable_params()
        if rebuilt and self.scheduler is not None:
            scheduler_last_epoch = self.scheduler.last_epoch
            self._setup_scheduler()
            self.scheduler.last_epoch = scheduler_last_epoch
            LOGGER.info(
                "Rebuilt optimizer for trainable params and refreshed scheduler "
                f"(last_epoch={scheduler_last_epoch})."
            )

    def _rebuild_optimizer_for_trainable_params(self):
        """Filter frozen parameters out of the optimizer while preserving param-group settings."""
        old_optimizer = self.optimizer
        filtered_groups = []

        for group in old_optimizer.param_groups:
            params = [p for p in group["params"] if p.requires_grad]
            if not params:
                continue
            filtered_groups.append(
                {
                    "params": params,
                    "settings": {k: v for k, v in group.items() if k in old_optimizer.defaults and k != "params"},
                    "lr": group.get("lr"),
                    "initial_lr": group.get("initial_lr"),
                }
            )

        if not filtered_groups:
            LOGGER.warning("No trainable optimizer param groups remain after freezing.")
            return False

        optimizer_cls = old_optimizer.__class__
        first_group = filtered_groups[0]
        init_kwargs = old_optimizer.defaults.copy()
        init_kwargs.update(first_group["settings"])
        new_optimizer = optimizer_cls(first_group["params"], **init_kwargs)

        for group in filtered_groups[1:]:
            new_optimizer.add_param_group({"params": group["params"], **group["settings"]})

        for new_group, old_group in zip(new_optimizer.param_groups, filtered_groups):
            if old_group["lr"] is not None:
                new_group["lr"] = old_group["lr"]
            new_group["initial_lr"] = (
                old_group["initial_lr"] if old_group["initial_lr"] is not None else new_group["lr"]
            )

        for group in new_optimizer.param_groups:
            for param in group["params"]:
                if param in old_optimizer.state:
                    new_optimizer.state[param] = old_optimizer.state[param]

        LOGGER.info(
            "Optimizer param groups filtered from %d to %d.",
            len(old_optimizer.param_groups),
            len(new_optimizer.param_groups),
        )
        self.optimizer = new_optimizer
        return True

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Create the ShiftOBBModel and load the frozen teacher backbones."""
        model = ShiftOBBModel(cfg, ch=6, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        if self.rgb_teacher and self.ir_teacher:
            self._load_frozen_backbones(model)
            existing_freeze = []
            if hasattr(self.args, "freeze"):
                if isinstance(self.args.freeze, list):
                    existing_freeze = self.args.freeze
                elif isinstance(self.args.freeze, int):
                    existing_freeze = list(range(self.args.freeze))
            self.args.freeze = sorted(set(existing_freeze + list(self._frozen_layer_indices)))
            LOGGER.info(f"args.freeze = {self.args.freeze}")

        return model

    def _load_frozen_backbones(self, model):
        """Load and freeze the dual-branch backbone weights from RGB and IR teachers."""
        LOGGER.info("")
        LOGGER.info("Loading frozen backbone weights...")
        LOGGER.info(f"  RGB teacher: {self.rgb_teacher}")
        LOGGER.info(f"  IR teacher:  {self.ir_teacher}")
        LOGGER.info("  Freeze boundary: dual RGB/IR branches only; shared SPPF layer model.25 stays trainable.")

        rgb_ckpt = torch.load(self.rgb_teacher, map_location="cpu", weights_only=False)
        rgb_state = rgb_ckpt["model"].state_dict() if "model" in rgb_ckpt else rgb_ckpt

        ir_ckpt = torch.load(self.ir_teacher, map_location="cpu", weights_only=False)
        ir_state = ir_ckpt["model"].state_dict() if "model" in ir_ckpt else ir_ckpt

        layer_stats = defaultdict(lambda: {"loaded": 0, "frozen": 0, "missing": 0, "shape_mismatch": 0, "examples": []})
        loaded_count = 0
        frozen_count = 0
        missing_count = 0
        shape_mismatch_count = 0

        for name, param in model.named_parameters():
            layer_idx = _extract_layer_idx(name)
            if layer_idx is None:
                continue

            teacher_state = None
            teacher_idx = None
            if layer_idx in STUDENT_RGB_TO_TEACHER:
                teacher_state = rgb_state
                teacher_idx = STUDENT_RGB_TO_TEACHER[layer_idx]
            elif layer_idx in STUDENT_IR_TO_TEACHER:
                teacher_state = ir_state
                teacher_idx = STUDENT_IR_TO_TEACHER[layer_idx]
            else:
                continue

            teacher_key = _build_teacher_key(name, teacher_idx)
            layer_stats[layer_idx]["frozen"] += 1
            frozen_count += 1
            param.requires_grad = False

            if teacher_key not in teacher_state:
                missing_count += 1
                layer_stats[layer_idx]["missing"] += 1
                if len(layer_stats[layer_idx]["examples"]) < 3:
                    layer_stats[layer_idx]["examples"].append(
                        {"type": "missing", "student_param": name, "teacher_key": teacher_key}
                    )
                continue

            teacher_weight = teacher_state[teacher_key]
            if teacher_weight.shape != param.data.shape:
                shape_mismatch_count += 1
                layer_stats[layer_idx]["shape_mismatch"] += 1
                if len(layer_stats[layer_idx]["examples"]) < 3:
                    layer_stats[layer_idx]["examples"].append(
                        {
                            "type": "shape_mismatch",
                            "student_param": name,
                            "teacher_key": teacher_key,
                            "student_shape": tuple(param.data.shape),
                            "teacher_shape": tuple(teacher_weight.shape),
                        }
                    )
                continue

            param.data.copy_(teacher_weight)
            loaded_count += 1
            layer_stats[layer_idx]["loaded"] += 1

        self._backbone_load_summary = {idx: layer_stats[idx] for idx in self._frozen_layer_indices}

        LOGGER.info(f"  Loaded {loaded_count} backbone parameters")
        LOGGER.info(f"  Frozen {frozen_count} backbone parameters")
        LOGGER.info(f"  Frozen layer indices: {self._frozen_layer_indices}")
        LOGGER.info("  Backbone load summary by layer:")
        for layer_idx in self._frozen_layer_indices:
            stats = self._backbone_load_summary[layer_idx]
            LOGGER.info(
                "    model.%d: loaded=%d frozen=%d missing=%d shape_mismatch=%d",
                layer_idx,
                stats["loaded"],
                stats["frozen"],
                stats["missing"],
                stats["shape_mismatch"],
            )

        if missing_count or shape_mismatch_count:
            LOGGER.warning(
                "Backbone load had %d missing keys and %d shape mismatches.",
                missing_count,
                shape_mismatch_count,
            )
            for layer_idx in self._frozen_layer_indices:
                examples = self._backbone_load_summary[layer_idx]["examples"]
                if not examples:
                    continue
                for example in _format_examples(examples):
                    LOGGER.warning(f"    model.{layer_idx}: {example}")
        else:
            LOGGER.info("  Backbone load check passed: no missing keys and no shape mismatches.")

    def get_validator(self):
        """Create the OBB validator while stripping custom shift-only args."""
        from ultralytics.models import yolo

        custom_args = ["shift_weight", "shift_ratio", "shift_mask_weight"]
        validator_args_dict = {k: v for k, v in vars(self.args).items() if k not in custom_args}
        validator_args = SimpleNamespace(**validator_args_dict)
        validator = yolo.obb.OBBValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=validator_args,
            _callbacks=self.callbacks,
        )
        validator.loss_names = ["box_loss", "cls_loss", "dfl_loss", "shift_loss"]
        return validator

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return labeled loss items for progress reporting."""
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "shift_loss"
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            return dict(zip(keys, [round(float(x), 5) for x in loss_items]))
        return keys


def create_shift_scheduler(args, total_epochs):
    """Build the configured shift scheduler."""
    if args.shift_schedule_config:
        config_path = args.shift_schedule_config
    elif args.shift_schedule in ["u_shape", "l_shape"]:
        config_path = str(Path(__file__).parent / "configs" / f"shift_schedule_{args.shift_schedule}.yaml")
    else:
        return None

    if os.path.exists(config_path):
        scheduler = UShapeShiftScheduler(config_path=config_path, logger=LOGGER)
        scheduler.config["total_epochs"] = total_epochs
        scheduler.print_schedule_summary()
        return scheduler

    LOGGER.error(f"Shift scheduler config not found: {config_path}")
    return None


def on_train_epoch_start(trainer):
    """Update shared shift parameters at the start of each epoch."""
    if not hasattr(trainer, "shift_scheduler") or trainer.shift_scheduler is None:
        return

    config = trainer.shift_scheduler.get_config_for_epoch(trainer.epoch)
    shift_param_manager.update_params(
        ratio=config["shift_ratio"],
        weight=config["shift_weight"],
        mask_weight=config["shift_mask_weight"],
    )
    if hasattr(trainer.model, "args"):
        trainer.model.args.shift_weight = config["shift_weight"]
        trainer.model.args.shift_ratio = config["shift_ratio"]
        trainer.model.args.shift_mask_weight = config["shift_mask_weight"]
    LOGGER.info(
        f"Epoch {trainer.epoch:3d}: ratio={config['shift_ratio']:.1f}, weight={config['shift_weight']:.1f}"
    )


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 Shift OBB with Frozen Backbone")
    parser.add_argument("--rgb_teacher", type=str, required=True, help="Path to RGB teacher weights")
    parser.add_argument("--ir_teacher", type=str, required=True, help="Path to IR teacher weights")
    parser.add_argument("--data", type=str, required=True, help="Dataset config path")
    parser.add_argument(
        "--model",
        type=str,
        default="./model_yaml_obb/yolov8_obb_shift.yaml",
        help="Model config path",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", type=str, default="./runs/shift_frozen_obb")
    parser.add_argument("--name", type=str, default="train")
    parser.add_argument("--shift_schedule", type=str, default="l_shape", choices=["u_shape", "l_shape"])
    parser.add_argument("--shift_schedule_config", type=str, default=None)
    return parser.parse_args()


def main():
    """Entrypoint."""
    args = parse_args()

    if not os.path.exists(args.model):
        LOGGER.error(f"Model config not found: {args.model}")
        sys.exit(1)

    LOGGER.info("=" * 60)
    LOGGER.info("Frozen Backbone Shift OBB Training")
    LOGGER.info("=" * 60)
    LOGGER.info(f"RGB teacher: {args.rgb_teacher}")
    LOGGER.info(f"IR teacher:  {args.ir_teacher}")
    LOGGER.info(f"Model config: {args.model}")
    LOGGER.info(f"Dataset: {args.data}")
    LOGGER.info(f"Train args: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")

    train_args = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "device": f"cuda:{args.device}" if args.device >= 0 else "cpu",
        "batch": args.batch,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "lr0": args.lr,
        "shift_weight": 1.0,
        "shift_ratio": 0.3,
        "shift_mask_weight": 0.2,
        "patience": 50,
        "save_period": 10,
        "close_mosaic": 10,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "optimizer": "SGD",
        "momentum": 0.937,
        "weight_decay": 0.0005,
    }

    try:
        trainer = FrozenShiftOBBTrainer(
            overrides=train_args,
            rgb_teacher=args.rgb_teacher,
            ir_teacher=args.ir_teacher,
        )
        trainer.shift_scheduler = create_shift_scheduler(args, args.epochs)
        trainer.add_callback("on_train_epoch_start", on_train_epoch_start)
        trainer.train()
        LOGGER.info("=" * 60)
        LOGGER.info(f"Training finished. Results saved to: {trainer.save_dir}")
    except Exception as exc:
        LOGGER.error(f"Training failed: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
