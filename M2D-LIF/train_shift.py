#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
跨模态物体平移预测训练脚本
支持命令行参数和配置文件，可灵活配置所有参数并直接开始训练
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.shift_integration import enable_shift_head_in_model
from ultralytics.data.augment import PairedShiftAugment


class ShiftTrainingConfig:
    """shift训练配置管理器"""

    # 默认参数
    DEFAULTS = {
        # ========== 数据增强参数 ==========
        "shift_range": (10, 50),       # RGB和IR的平移像素范围
        "shift_prob": 0.3,              # 应用平移的概率 (0-1)

        # ========== Loss参数 ==========
        "shift_weight": 0.1,            # Shift Loss相对于Detection Loss的权重
        "shift_enabled": True,          # 是否启用Shift Loss

        # ========== 训练参数 ==========
        "data": None,                   # 数据集yaml路径 (必需)
        "model": "/mnt/home/pyq_code/ADV/M2D-LIF/model_yaml/yolov8_shift.yaml",        # 模型yaml路径
        "epochs": 100,                  # 训练轮数
        "imgsz": 640,                   # 输入图像大小
        "batch": 16,                    # 批大小
        "device": 0,                    # GPU设备号
        "workers": 4,                   # 数据加载进程数
        "lr0": 0.01,                    # 初始学习率
        "momentum": 0.937,              # 动量
        "weight_decay": 0.0005,         # 权重衰减
        "seed": 42,                     # 随机种子

        # ========== 动态调度参数 ==========
        "warmup_epochs": 20,            # Warmup轮数
        "schedule": "linear",           # 动态调度方式

        # ========== 输出和日志 ==========
        "project": "runs/shift_training",  # 结果保存路径
        "name": None,                   # 实验名称 (为None时自动生成)
        "save": True,                   # 是否保存模型
        "verbose": True,                # 是否详细输出

        # ========== 其他参数 ==========
        "patience": 50,                 # 早停的patience
        "dropout": 0.0,                 # Dropout率
        "augment": True,                # 是否使用增强
    }

    def __init__(self):
        self.config = self.DEFAULTS.copy()

    def update_from_dict(self, params: Dict[str, Any]):
        """从字典更新参数"""
        for key, value in params.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"⚠️ 警告: 未知参数 '{key}' 将被忽略")

    def update_from_file(self, config_file: str):
        """从YAML或JSON文件读取参数"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        if config_file.endswith('.yaml') or config_file.endswith('.yml'):
            with open(config_file, 'r') as f:
                params = yaml.safe_load(f) or {}
        elif config_file.endswith('.json'):
            with open(config_file, 'r') as f:
                params = json.load(f)
        else:
            raise ValueError("配置文件必须是YAML或JSON格式")

        self.update_from_dict(params)

    def validate(self):
        """验证参数有效性"""
        if self.config["data"] is None:
            raise ValueError("必须指定 'data' 参数 (数据集yaml路径)")

        if not os.path.exists(self.config["data"]):
            raise FileNotFoundError(f"数据集文件不存在: {self.config['data']}")

        if not os.path.exists(self.config["model"]):
            raise FileNotFoundError(f"模型文件不存在: {self.config['model']}")

        # 验证shift参数
        if len(self.config["shift_range"]) != 2:
            raise ValueError("shift_range 必须是 (min, max) 元组")

        if not (0.0 <= self.config["shift_prob"] <= 1.0):
            raise ValueError("shift_prob 必须在 0-1 之间")

        if self.config["shift_weight"] < 0:
            raise ValueError("shift_weight 必须 >= 0")

    def get(self, key: str, default=None):
        """获取参数值"""
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()

    def to_json(self) -> str:
        """转换为JSON字符串"""
        config_copy = self.config.copy()
        if isinstance(config_copy["shift_range"], tuple):
            config_copy["shift_range"] = list(config_copy["shift_range"])
        return json.dumps(config_copy, indent=2)

    def print_summary(self):
        """打印参数摘要"""
        print("\n" + "="*70)
        print("📋 训练配置摘要")
        print("="*70)

        sections = {
            "数据增强参数": ["shift_range", "shift_prob"],
            "Loss参数": ["shift_weight", "shift_enabled"],
            "训练参数": ["data", "model", "epochs", "imgsz", "batch", "device"],
            "学习率参数": ["lr0", "momentum", "weight_decay"],
            "动态调度": ["warmup_epochs", "schedule"],
            "输出配置": ["project", "name", "save"],
        }

        for section, keys in sections.items():
            print(f"\n【{section}】")
            for key in keys:
                value = self.config.get(key)
                print(f"  {key:20s} = {value}")

        print("\n" + "="*70 + "\n")


class ShiftTrainer:
    """shift训练器"""

    def __init__(self, config: ShiftTrainingConfig):
        self.config = config
        self.model = None
        self.results = None

    def setup_model(self):
        """设置模型"""
        print("\n🔧 设置模型...")

        model = YOLO(self.config["model"])
        print(f"  ✓ 加载模型: {self.config['model']}")

        # 启用Shift head
        if self.config["shift_enabled"]:
            enable_shift_head_in_model(model, enabled=True)
            print(f"  ✓ 启用 ShiftHead")
        else:
            print(f"  ✓ 禁用 ShiftHead")

        self.model = model

    def setup_augmentation(self):
        """设置数据增强 (可选，实际增强在数据加载时应用)"""
        print("\n📊 数据增强配置...")

        augment = PairedShiftAugment(
            shift_range=tuple(self.config["shift_range"]),
            shift_prob=self.config["shift_prob"]
        )

        print(f"  ✓ PairedShiftAugment")
        print(f"    - shift_range: {self.config['shift_range']}")
        print(f"    - shift_prob: {self.config['shift_prob']}")

        return augment

    def train(self) -> Dict[str, Any]:
        """开始训练"""
        print("\n🚀 开始训练...\n")

        # 准备训练参数
        train_params = {
            "data": self.config["data"],
            "epochs": self.config["epochs"],
            "imgsz": self.config["imgsz"],
            "batch": self.config["batch"],
            "device": self.config["device"],
            "workers": self.config["workers"],
            "lr0": self.config["lr0"],
            "momentum": self.config["momentum"],
            "weight_decay": self.config["weight_decay"],
            "seed": self.config["seed"],
            "project": self.config["project"],
            "patience": self.config["patience"],
            "dropout": self.config["dropout"],
            "augment": self.config["augment"],
            "verbose": self.config["verbose"],
            "save": self.config["save"],
        }

        # 如果指定了实验名称，使用它
        if self.config["name"]:
            train_params["name"] = self.config["name"]
        else:
            # 自动生成实验名称
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            train_params["name"] = f"shift_sr{self.config['shift_range'][0]}{self.config['shift_range'][1]}_" \
                                   f"sp{self.config['shift_prob']:.1f}_sw{self.config['shift_weight']:.2f}_{timestamp}"

        # 运行训练
        self.results = self.model.train(**train_params)

        return self.results

    def save_config(self, save_dir: Optional[str] = None):
        """保存配置文件"""
        if save_dir is None:
            save_dir = os.path.join(self.config["project"], self.config["name"] or "config")

        os.makedirs(save_dir, exist_ok=True)

        config_file = os.path.join(save_dir, "config.json")
        with open(config_file, 'w') as f:
            f.write(self.config.to_json())

        print(f"\n✅ 配置已保存到: {config_file}")

    def print_results_summary(self):
        """打印结果摘要"""
        if self.results is None:
            print("⚠️ 还没有训练结果")
            return

        print("\n" + "="*70)
        print("📊 训练结果摘要")
        print("="*70)

        print("\n✅ 训练完成!")
        print(f"  模型已保存到: {self.model.trainer.save_dir}")


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """获取预设配置"""
    presets = {
        "conservative": {
            "shift_range": (5, 30),
            "shift_prob": 0.1,
            "shift_weight": 0.05,
            "epochs": 100,
            "batch": 16,
        },
        "balanced": {
            "shift_range": (10, 50),
            "shift_prob": 0.3,
            "shift_weight": 0.1,
            "epochs": 100,
            "batch": 16,
        },
        "aggressive": {
            "shift_range": (20, 80),
            "shift_prob": 0.5,
            "shift_weight": 0.2,
            "epochs": 150,
            "batch": 32,
        },
    }

    if preset_name not in presets:
        raise ValueError(f"未知的预设配置: {preset_name}")

    return presets[preset_name]


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="跨模态物体平移预测训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从配置文件训练
  python train_shift.py --config config.yaml

  # 使用命令行参数
  python train_shift.py --data data.yaml --epochs 150 --shift_prob 0.5

  # 配置文件 + 命令行覆盖
  python train_shift.py --config config.yaml --epochs 200 --shift_weight 0.2

  # 预设配置
  python train_shift.py --preset aggressive --data data.yaml

  # 干运行 (只检查配置)
  python train_shift.py --data data.yaml --shift_prob 0.5 --dry_run
        """
    )

    # 配置文件
    parser.add_argument(
        "--config", type=str, default=None,
        help="配置文件路径 (YAML或JSON)"
    )

    # 预设配置
    parser.add_argument(
        "--preset", type=str,
        choices=["conservative", "balanced", "aggressive"],
        default=None,
        help="预设配置"
    )

    # ========== 数据增强参数 ==========
    parser.add_argument(
        "--shift_range", type=int, nargs=2, default=None,
        metavar=("MIN", "MAX"),
        help="平移像素范围 (默认: 10 50)"
    )

    parser.add_argument(
        "--shift_prob", type=float, default=None,
        help="平移概率 (默认: 0.3)"
    )

    # ========== Loss参数 ==========
    parser.add_argument(
        "--shift_weight", type=float, default=None,
        help="Shift Loss权重 (默认: 0.1)"
    )

    parser.add_argument(
        "--shift_enabled", type=bool, default=None,
        help="是否启用Shift Loss (默认: True)"
    )

    # ========== 训练参数 ==========
    parser.add_argument(
        "--data", type=str, required=True,
        help="数据集yaml路径 (必需)"
    )

    parser.add_argument(
        "--model", type=str, default=None,
        help="模型yaml路径 (默认: yolov8n.yaml)"
    )

    parser.add_argument(
        "--epochs", type=int, default=None,
        help="训练轮数 (默认: 100)"
    )

    parser.add_argument(
        "--imgsz", type=int, default=None,
        help="输入图像大小 (默认: 640)"
    )

    parser.add_argument(
        "--batch", type=int, default=None,
        help="批大小 (默认: 16)"
    )

    parser.add_argument(
        "--device", type=int, default=None,
        help="GPU设备号 (默认: 0)"
    )

    parser.add_argument(
        "--workers", type=int, default=None,
        help="数据加载进程数 (默认: 4)"
    )

    # ========== 学习率参数 ==========
    parser.add_argument(
        "--lr0", type=float, default=None,
        help="初始学习率 (默认: 0.01)"
    )

    parser.add_argument(
        "--momentum", type=float, default=None,
        help="动量 (默认: 0.937)"
    )

    parser.add_argument(
        "--weight_decay", type=float, default=None,
        help="权重衰减 (默认: 0.0005)"
    )

    # ========== 输出和日志 ==========
    parser.add_argument(
        "--project", type=str, default=None,
        help="结果保存路径"
    )

    parser.add_argument(
        "--name", type=str, default=None,
        help="实验名称 (默认: 自动生成)"
    )

    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子 (默认: 42)"
    )

    # ========== 其他 ==========
    parser.add_argument(
        "--dry_run", action="store_true",
        help="只检查配置，不运行训练"
    )

    return parser.parse_args()


def main():
    """主函数"""
    print("\n" + "╔" + "="*68 + "╗")
    print("║" + "跨模态物体平移预测训练脚本".center(68) + "║")
    print("╚" + "="*68 + "╝" + "\n")

    # 解析参数
    args = parse_arguments()

    # 创建配置
    config = ShiftTrainingConfig()

    # 加载预设配置 (如果指定)
    if args.preset:
        print(f"📋 使用预设配置: {args.preset}")
        preset_config = get_preset_config(args.preset)
        config.update_from_dict(preset_config)

    # 加载配置文件 (如果指定)
    if args.config:
        print(f"📋 加载配置文件: {args.config}")
        config.update_from_file(args.config)

    # 更新命令行参数 (覆盖配置文件)
    cli_params = {}
    for key, value in vars(args).items():
        if value is not None and key not in ["config", "preset", "dry_run"]:
            cli_params[key] = value

    if cli_params:
        print(f"📋 应用命令行参数覆盖")
        config.update_from_dict(cli_params)

    # 验证参数
    try:
        config.validate()
    except (ValueError, FileNotFoundError) as e:
        print(f"\n❌ 参数验证失败: {e}")
        sys.exit(1)

    # 打印配置摘要
    config.print_summary()

    # 干运行 (只打印配置)
    if args.dry_run:
        print("✅ 干运行模式 - 配置有效，但未运行训练")
        return

    # 设置模型和开始训练
    trainer = ShiftTrainer(config)

    try:
        # 设置模型
        trainer.setup_model()

        # 设置增强
        trainer.setup_augmentation()

        # 开始训练
        trainer.train()

        # 保存配置
        trainer.save_config()

        # 打印结果摘要
        trainer.print_results_summary()

        print("\n✅ 训练完成!")

    except KeyboardInterrupt:
        print("\n⚠️ 训练被中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
