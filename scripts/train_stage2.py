#!/usr/bin/env python3
"""
阶段2：Student对抗蒸馏训练
训练Student模型，使用双Teacher监督
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 🚨 全局调试开关
DEBUG_MODE = False  # 🚨 关闭调试模式，让训练输出更清晰
DEBUG_INTERVAL = 50  # 每隔多少个batch输出一次debug信息

def debug_print(title, data_dict, batch_idx=None, enabled=True):
    """
    统一的调试输出函数
    
    参数:
        title: 标题
        data_dict: {名称: 值} 字典
        batch_idx: 当前批次索引（可选，用于控制输出频率）
        enabled: 是否启用
    """
    if not enabled or not DEBUG_MODE:
        return
    
    # 如果指定了batch_idx，只在第0、10、50、100批次输出
    if batch_idx is not None:
        if batch_idx not in [0, 1, 10, 50, 100]:
            return
    
    print(f"\n{'='*20} 🔍 {title} {'='*20}")
    for name, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: shape={list(value.shape)}, dtype={value.dtype}, "
                  f"range=[{value.min().item():.4f}, {value.max().item():.4f}], "
                  f"mean={value.mean().item():.4f}")
        elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
            print(f"  {name}: {len(value)}个张量")
            for i, v in enumerate(value[:3]):  # 只显示前3个
                print(f"    [{i}]: shape={list(v.shape)}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
        elif isinstance(value, dict):
            print(f"  {name}: 字典，keys={list(value.keys())}")
        elif isinstance(value, (int, float)):
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {type(value).__name__}")
    print(f"{'='*50}")

from models.teacher import TeacherModel
from models.student import DualModalStudent
from losses.distill_loss import TotalLoss
from trainer.scheduler import (
    AdversarialScheduler,
    LossWeightScheduler,
    LRScheduler,
    CombinedScheduler,
)
from data.loader import create_data_loaders

# 🚨 全局调试开关
DEBUG_MODE = False  # 🚨 关闭调试模式，让训练输出更清晰

def debug_print(title, data_dict, batch_idx=None, enabled=True):
    """
    统一的调试输出函数
    
    参数:
        title: 标题
        data_dict: {名称: 值} 字典
        batch_idx: 当前批次索引（可选，用于控制输出频率）
        enabled: 是否启用
    """
    if not enabled or not DEBUG_MODE:
        return
    
    # 如果指定了batch_idx，只在第0、10、50、100批次输出
    if batch_idx is not None and batch_idx not in [0, 10, 50, 100]:
        return
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] {title}")
    print(f"{'='*60}")
    
    for name, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: shape={list(value.shape)}, "
                  f"dtype={value.dtype}, device={value.device}, "
                  f"range=[{value.min().item():.4f}, {value.max().item():.4f}], "
                  f"mean={value.mean().item():.4f}, std={value.std().item():.4f}")
        elif isinstance(value, (list, tuple)):
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"  {name}: len={len(value)}")
                for i, v in enumerate(value[:3]):  # 只显示前3个
                    print(f"    [{i}]: shape={list(v.shape)}, range=[{v.min().item():.4f}, {v.max().item():.4f}]")
            else:
                print(f"  {name}: {type(value).__name__}, len={len(value)}")
        elif isinstance(value, dict):
            print(f"  {name}: keys={list(value.keys())}")
        elif isinstance(value, (int, float)):
            print(f"  {name}: {value}")
        else:
            print(f"  {name}: {value}")


def debug_gradients(model, batch_idx=None, enabled=True):
    """
    检查模型各模块的梯度状态
    
    参数:
        model: 模型
        batch_idx: 当前批次索引
        enabled: 是否启用
    """
    if not enabled or not DEBUG_MODE:
        return
    
    if batch_idx is not None and batch_idx not in [0, 10, 50, 100]:
        return
    
    print(f"\n{'='*60}")
    print(f"[DEBUG] 梯度检查 (Batch {batch_idx})")
    print(f"{'='*60}")
    
    for name, module in [('backbone_rgb', model.backbone_rgb), 
                          ('backbone_ir', model.backbone_ir),
                          ('fusion', model.fusion),
                          ('neck_layers', model.neck_layers),
                          ('detect_head', model.detect_head)]:
        grad_norm = 0.0
        param_count = 0
        for param in module.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        grad_norm = grad_norm ** 0.5 if param_count > 0 else 0.0
        
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        has_grad = sum(1 for p in module.parameters() if p.grad is not None)
        
        status = "✓" if has_grad > 0 else "⚠️ 无梯度"
        print(f"  {name:15s}: grad_norm={grad_norm:.6f}, 有梯度参数={has_grad}, 可训练={trainable} {status}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Stage 2: Student Adversarial Distillation Training')

    # Teacher模型路径
    parser.add_argument('--teacher_rgb', type=str, required=True,
                        help='RGB Teacher模型路径')
    parser.add_argument('--teacher_ir', type=str, required=True,
                        help='IR Teacher模型路径')

    # 数据集配置
    parser.add_argument('--data', type=str, default='data/FLIR.yaml',
                        help='数据集配置文件')
    parser.add_argument('--img_size', type=int, default=640,
                        help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='数据加载工作进程数')

    # 模型配置
    parser.add_argument('--num_classes', type=int, default=80,
                        help='类别数量')
    parser.add_argument('--fusion_channels', type=int, nargs='+', default=None,
                        help='融合后特征通道数列表 [P3, P4, P5]')
    parser.add_argument('--fusion_mode', type=str, default='add',
                        choices=['concat', 'add'],
                        help='特征融合模式: concat(拼接) 或 add(相加)')

    # 训练配置
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='学习率')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adamw'],
                        help='优化器类型: sgd 或 adamw')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='warm-up轮数')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'constant'],
                        help='学习率调度器类型: cosine 或 constant (固定学习率)')

    # 对抗训练配置
    parser.add_argument('--max_epsilon', type=float, default=0.0,
                        help='最大对抗强度 (设为0关闭对抗训练)')
    parser.add_argument('--adv_schedule_type', type=str, default='progressive',
                        choices=['progressive', 'step', 'cosine'],
                        help='对抗强度调度类型')

    # 损失权重
    parser.add_argument('--lambda_det', type=float, default=1.0,
                        help='检测损失权重')
    parser.add_argument('--lambda_distill', type=float, default=0.0,
                        help='蒸馏损失权重 (设为0关闭蒸馏训练)')

    # 输出配置
    parser.add_argument('--save_dir', type=str, default='checkpoint/student',
                        help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='runs/student',
                        help='TensorBoard日志目录')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='保存频率（每N个epoch）')

    # 设备配置
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--amp', action='store_true',
                        help='是否使用混合精度训练 (AMP)')

    # 恢复训练
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')

    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_teacher(teacher_path: str, device: torch.device) -> TeacherModel:
    """
    加载Teacher模型

    参数:
        teacher_path: Teacher模型路径
        device: 设备

    返回:
        teacher: Teacher模型
    """
    print(f"正在加载Teacher模型: {teacher_path}")
    teacher = TeacherModel.load_from_checkpoint(teacher_path, device=str(device))
    teacher.eval()
    teacher.freeze()
    print(f"✓ Teacher模型加载成功")
    return teacher


def create_student(
    teacher_rgb: TeacherModel,
    teacher_ir: TeacherModel,
    num_classes: int,
    fusion_channels: list,
    fusion_mode: str,
    device: torch.device,
) -> DualModalStudent:
    """
    创建Student模型

    参数:
        teacher_rgb: RGB Teacher模型
        teacher_ir: IR Teacher模型
        num_classes: 类别数量
        fusion_channels: 融合后特征通道数
        fusion_mode: 融合模式 ('concat' 或 'add')
        device: 设备

    返回:
        student: Student模型
    """
    print("正在创建Student模型...")
    student = DualModalStudent(
        teacher_rgb=teacher_rgb,
        teacher_ir=teacher_ir,
        num_classes=num_classes,
        fusion_channels=fusion_channels,
        use_residual=True,
        fusion_mode=fusion_mode,
    ).to(device)
    print(f"✓ Student模型创建成功")

    # 🚨 计算并打印参数总量
    total_params = sum(p.numel() for p in student.parameters())
    trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"✓ 参数总量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"✓ 可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"✓ 冻结参数量: {frozen_params:,} ({frozen_params / 1e6:.2f}M)")

    return student


def create_optimizer(student: DualModalStudent, lr: float, optimizer_type: str = 'sgd') -> optim.Optimizer:
    """
    创建优化器 - 完全参照 Ultralytics 配置

    参数:
        student: Student模型
        lr: 学习率
        optimizer_type: 优化器类型 ('sgd' 或 'adamw')

    返回:
        optimizer: 优化器
    """
    # 只优化可训练参数（fusion和detect_head）
    trainable_params = student.get_trainable_params()
    
    # 🚨 关键修复：分离 bias 和其他参数，使用不同的 weight_decay
    # Ultralytics 对 bias 不使用 weight_decay
    decay_params = []
    no_decay_params = []
    for name, param in trainable_params:
        if 'bias' in name or 'bn' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': 0.0005},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]

    if optimizer_type == 'sgd':
        # SGD: YOLOv8 标准配置
        optimizer = optim.SGD(
            param_groups,
            lr=lr,
            momentum=0.937,      # YOLO 默认动量
            weight_decay=0.0005, # 会被 param_groups 覆盖
            nesterov=True
        )
        print(f"✓ 优化器创建成功 (SGD)")
    else:  # adamw
        # AdamW: 更敏感，建议使用较小的学习率
        optimizer = optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=0.0005
        )
        print(f"✓ 优化器创建成功 (AdamW)")

    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(f"✓ 可训练参数数: {trainable_param_count:,}")
    return optimizer


def create_schedulers(
    optimizer: optim.Optimizer,
    epochs: int,
    lr: float,
    warmup_epochs: int,
    max_epsilon: float,
    adv_schedule_type: str,
    lr_scheduler_type: str = 'cosine',
) -> CombinedScheduler:
    """
    创建调度器

    参数:
        optimizer: 优化器
        epochs: 总训练轮数
        lr: 学习率
        warmup_epochs: warm-up轮数
        max_epsilon: 最大对抗强度
        adv_schedule_type: 对抗强度调度类型
        lr_scheduler_type: 学习率调度器类型 ('cosine' 或 'constant')

    返回:
        scheduler: 组合调度器
    """
    # 对抗强度调度器
    adv_scheduler = AdversarialScheduler(
        warmup_epochs=warmup_epochs,
        max_epsilon=max_epsilon,
        total_epochs=epochs,
        scheduler_type=adv_schedule_type,
    )

    # 损失权重调度器
    loss_weight_scheduler = LossWeightScheduler(
        total_epochs=epochs,
        schedule_type='stage',
    )

    # 学习率调度器
    lr_scheduler = LRScheduler(
        optimizer=optimizer,
        total_epochs=epochs,
        initial_lr=lr,
        scheduler_type=lr_scheduler_type,
        warmup_epochs=warmup_epochs,
    )

    # 组合调度器
    scheduler = CombinedScheduler(
        adversarial_scheduler=adv_scheduler,
        loss_weight_scheduler=loss_weight_scheduler,
        lr_scheduler=lr_scheduler,
    )

    print(f"✓ 调度器创建成功")
    return scheduler


def train_epoch(
    student: DualModalStudent,
    train_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: TotalLoss,
    epoch: int,
    device: torch.device,
    epsilon: float,
    lambda_det: float,
    lambda_distill: float,
    writer: SummaryWriter,
    progress_bar: tqdm,
    scaler: GradScaler = None,
) -> dict:
    """
    训练一个epoch

    参数:
        student: Student模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch
        device: 设备
        epsilon: 对抗强度
        lambda_det: 检测损失权重
        lambda_distill: 蒸馏损失权重
        writer: TensorBoard写入器
        progress_bar: 进度条
        scaler: 混合精度缩放器（可选）

    返回:
        metrics: 指标字典
    """
    import numpy as np
    
    student.train()

    # 更新损失权重
    criterion.update_weights(lambda_det, lambda_distill)

    total_loss = 0.0
    det_loss_sum = 0.0
    distill_loss_sum = 0.0
    
    # ========== 🚨 Batch-level warmup (参考 Ultralytics) ==========
    nb = len(train_loader)  # number of batches per epoch
    nw = max(round(3 * nb), 100)  # warmup iterations: max(3 epochs * nb, 100)
    
    # 保存优化器的初始学习率
    if not hasattr(optimizer, '_warmup_initialized'):
        for x in optimizer.param_groups:
            x['initial_lr'] = x['lr']
        optimizer._warmup_initialized = True

    # 批次循环
    for batch_idx, batch in enumerate(train_loader):
        # ========== Warmup：动态调整学习率和动量 ==========
        ni = epoch * nb + batch_idx  # 当前总 batch 索引
        if ni <= nw:
            xi = [0, nw]
            for j, x in enumerate(optimizer.param_groups):
                # bias lr 从 0.1 升到 lr0，其他参数从 0.0 升到 lr0
                warmup_bias_lr = 0.1 if j == 0 else 0.0
                x['lr'] = np.interp(ni, xi, [warmup_bias_lr, x['initial_lr']])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [0.8, 0.937])
        
        # 🚨 关键：Student接口必须正确
        # 1. 分离RGB和IR图像
        rgb_imgs = batch['img'][:, :3, :, :].to(device)  # [B, 3, H, W]
        ir_imgs = batch['img'][:, 3:, :, :].to(device)   # [B, 3, H, W]

        # 2. 拼接为6通道输入
        x = torch.cat([rgb_imgs, ir_imgs], dim=1)  # [B, 6, H, W]

        # 3. 准备目标
        targets = batch
        
        # ======== DEBUG: 数据加载检查 ========
        debug_print("数据加载检查", {
            'rgb_imgs': rgb_imgs,
            'ir_imgs': ir_imgs,
            'x (6通道输入)': x,
            'batch_idx': batch.get('batch_idx', 'N/A'),
            'cls': batch.get('cls', 'N/A'),
            'bboxes': batch.get('bboxes', 'N/A'),
        }, batch_idx=batch_idx)

        # ======== 恢复成老实本分的独立计算图 ========
        noises_rgb = None
        noises_ir = None

        if epsilon > 0:
            from models.adversarial import AdversarialNoiseGenerator
            adv_gen = AdversarialNoiseGenerator(epsilon=epsilon)
            
            # 第一轮：专门为了生噪声单独跑一次提取 (不积累到大循环的梯度里)
            with torch.no_grad():
                # 为了防止类型转换报错，加上 autocast
                with torch.cuda.amp.autocast(enabled=True):
                    clean_rgb = student.backbone_rgb(rgb_imgs)
                    clean_ir = student.backbone_ir(ir_imgs)

            is_rgb_attack = torch.rand(1).item() < 0.5

            def custom_loss_wrapper(adv_features):
                # 🚨 使用 autocast 处理混合精度训练
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    if is_rgb_attack:
                        fused = student.fusion(adv_features, [f.detach() for f in clean_ir], apply_residual=True)
                    else:
                        fused = student.fusion([f.detach() for f in clean_rgb], adv_features, apply_residual=True)

                    # 🚨 核心修复：调用专门的方法走完 Neck 和 Head
                    preds = student.forward_from_fusion(fused)

                    return criterion.det_loss(preds, targets)['total']

            if is_rgb_attack:
                noises_rgb = adv_gen.generate_noise([f.detach() for f in clean_rgb], custom_loss_wrapper, student)
            else:
                noises_ir = adv_gen.generate_noise([f.detach() for f in clean_ir], custom_loss_wrapper, student)
        
        # ======== 第二轮：正式带梯度的训练 ========

        # 4. 前向传播和损失计算
        # 🚨 关键：Student接口不接受epsilon，只接受6通道输入
        # 也不接受分离的rgb_imgs和ir_imgs
        # 必须先拼接，然后传入

        # 🚨 使用混合精度训练
        if scaler is not None:
            with autocast():
                predictions = student(x, noises_rgb=noises_rgb, noises_ir=noises_ir)

                # 5. 获取特征（用于蒸馏）
                features_dict = student.get_latest_features()

                # ======== DEBUG: 前向传播检查 ========
                debug_print("前向传播检查", {
                    'predictions类型': type(predictions).__name__,
                    'predictions': predictions,
                    'student_base_sum': features_dict.get('student_base_sum', 'N/A'),
                    'fused_features': features_dict.get('fused_features', 'N/A'),
                    'teacher_rgb_features': features_dict.get('teacher_rgb_features', 'N/A'),
                    'teacher_ir_features': features_dict.get('teacher_ir_features', 'N/A'),
                }, batch_idx=batch_idx)

                # 6. 🚨 计算损失
                # 🚨 新接口：TotalLoss会自动从features_dict中提取fused_features和teacher特征
                loss_dict = criterion(
                    predictions=predictions,
                    targets=targets,
                    features_dict=features_dict,
                )

                loss = loss_dict['total']
                
                # ======== DEBUG: 损失检查 ========
                debug_print("损失计算检查", {
                    'total_loss': loss.item(),
                    'det_loss': loss_dict['det'].item(),
                    'distill_loss': loss_dict['distill'].item(),
                }, batch_idx=batch_idx)
        else:
            predictions = student(x, noises_rgb=noises_rgb, noises_ir=noises_ir)

            # 5. 获取特征（用于蒸馏）
            features_dict = student.get_latest_features()

            # ======== DEBUG: 前向传播检查 ========
            debug_print("前向传播检查", {
                'predictions类型': type(predictions).__name__,
                'predictions': predictions,
                'student_base_sum': features_dict.get('student_base_sum', 'N/A'),
                'fused_features': features_dict.get('fused_features', 'N/A'),
                'teacher_rgb_features': features_dict.get('teacher_rgb_features', 'N/A'),
                'teacher_ir_features': features_dict.get('teacher_ir_features', 'N/A'),
            }, batch_idx=batch_idx)

            # 6. 🚨 计算损失
            # 🚨 新接口：TotalLoss会自动从features_dict中提取fused_features和teacher特征
            loss_dict = criterion(
                predictions=predictions,
                targets=targets,
                features_dict=features_dict,
            )

            loss = loss_dict['total']
            
            # ======== DEBUG: 损失检查 ========
            debug_print("损失计算检查", {
                'total_loss': loss.item(),
                'det_loss': loss_dict['det'].item(),
                'distill_loss': loss_dict['distill'].item(),
            }, batch_idx=batch_idx)

        # 8. 反向传播
        optimizer.zero_grad()

        # 🚨 使用混合精度训练
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # 先unscale
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)  # 🚨 改为10.0，与Baseline一致
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)  # 🚨 改为10.0，与Baseline一致
            optimizer.step()

        # ======== DEBUG: 梯度检查 ========
        if batch_idx in [0, 1, 10, 50, 100]:
            print(f"\n{'='*20} 🔍 梯度检查 (Batch {batch_idx}) {'='*20}")
            for name, module in [('backbone_rgb', student.backbone_rgb), 
                                  ('backbone_ir', student.backbone_ir),
                                  ('fusion', student.fusion),
                                  ('neck_layers', student.neck_layers),
                                  ('detect_head', student.detect_head)]:
                grad_norm = 0.0
                has_grad = 0
                for param in module.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                        has_grad += 1
                grad_norm = grad_norm ** 0.5 if has_grad > 0 else 0.0
                status = "✓" if has_grad > 0 else "⚠️无梯度"
                print(f"  {name:15s}: grad_norm={grad_norm:.6f}, 有梯度参数={has_grad} {status}")
            print(f"{'='*60}")

        # 9. 累积损失
        total_loss += loss.item()
        det_loss_sum += loss_dict['det'].item()
        distill_loss_sum += loss_dict['distill'].item()

        # 10. 更新进度条
        postfix = {
            'loss': f'{loss.item():.4f}',
            'det': f'{loss_dict["det"].item():.4f}',
        }
        # 🚨 只有当蒸馏损失开启时才显示
        if lambda_distill > 0:
            postfix['distill'] = f'{loss_dict["distill"].item():.4f}'
        progress_bar.set_postfix(postfix)
        progress_bar.update(1)

    # 计算平均损失
    num_batches = len(train_loader)
    metrics = {
        'train_loss': total_loss / num_batches,
        'train_det_loss': det_loss_sum / num_batches,
    }
    
    # 🚨 只有当蒸馏损失开启时才记录
    if lambda_distill > 0:
        metrics['train_distill_loss'] = distill_loss_sum / num_batches

    # 记录到TensorBoard
    writer.add_scalar('Train/Loss', metrics['train_loss'], epoch)
    writer.add_scalar('Train/DetLoss', metrics['train_det_loss'], epoch)
    if lambda_distill > 0:
        writer.add_scalar('Train/DistillLoss', metrics['train_distill_loss'], epoch)

    return metrics


def validate(
    student: DualModalStudent,
    val_loader: torch.utils.data.DataLoader,
    criterion: TotalLoss,
    epoch: int,
    device: torch.device,
    writer: SummaryWriter,
    num_classes: int,
) -> dict:
    """
    验证（包含 mAP 计算）
    
    参考 M2D-LIF 的 DetectionValidator 实现
    关键：使用 scale_boxes 将预测框和目标框都转换到原始图像空间计算 IoU

    参数:
        student: Student模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        epoch: 当前epoch
        device: 设备
        writer: TensorBoard写入器
        num_classes: 类别数量

    返回:
        metrics: 指标字典
    """
    import numpy as np
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import DetMetrics, box_iou

    student.eval()

    total_loss = 0.0
    det_loss_sum = 0.0
    distill_loss_sum = 0.0

    # mAP 计算所需的统计信息 (参考 M2D-LIF DetectionValidator)
    stats = {
        'tp': [],
        'conf': [],
        'pred_cls': [],
        'target_cls': [],
    }
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()

    print("Validating...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False)):
            # 准备输入
            rgb_imgs = batch['img'][:, :3, :, :].to(device)
            ir_imgs = batch['img'][:, 3:, :, :].to(device)
            x = torch.cat([rgb_imgs, ir_imgs], dim=1)

            # 将 batch 数据移到 GPU
            for k in ['img', 'bboxes', 'cls', 'batch_idx']:
                if k in batch:
                    batch[k] = batch[k].to(device)

            # 前向传播
            predictions = student(x)

            # 处理 YOLOv8 的返回值
            if isinstance(predictions, tuple):
                decoded_predictions = predictions[0]  # [B, 4+nc, 8400]
                raw_predictions = predictions[1]
            else:
                continue

            # 获取特征并计算损失
            features_dict = student.get_latest_features()
            loss_dict = criterion(
                predictions=raw_predictions,
                targets=batch,
                features_dict=features_dict,
                teacher_rgb_features=features_dict['teacher_rgb_features'],
                teacher_ir_features=features_dict['teacher_ir_features'],
            )

            total_loss += loss_dict['total'].item()
            det_loss_sum += loss_dict['det'].item()
            distill_loss_sum += loss_dict['distill'].item()

            # ========== mAP 计算 (参考 M2D-LIF DetectionValidator) ==========
            # 1. NMS 后处理
            preds_nms = ops.non_max_suppression(
                decoded_predictions,
                conf_thres=0.001,
                iou_thres=0.6,
                multi_label=True,
                agnostic=False,
                max_det=300,
            )

            # 2. 更新统计信息 (参考 update_metrics 方法)
            for si, pred in enumerate(preds_nms):
                npr = len(pred)
                stat = dict(
                    conf=torch.zeros(0, device=device),
                    pred_cls=torch.zeros(0, device=device),
                    tp=torch.zeros(npr, niou, dtype=torch.bool, device=device)
                )

                # 准备目标 (参考 _prepare_batch 方法)
                idx = batch['batch_idx'] == si
                cls = batch['cls'][idx].view(-1)  # [N]
                bbox = batch['bboxes'][idx]  # [N, 4] xywh normalized
                
                nl = len(cls)
                stat['target_cls'] = cls

                if npr == 0:
                    if nl:
                        for k in stats.keys():
                            stats[k].append(stat[k])
                    continue

                # 准备预测框 (参考 _prepare_pred 方法)
                # NMS 返回的 pred 是像素坐标，需要缩放到原始图像空间
                predn = pred.clone()
                if 'ori_shape' in batch and 'ratio_pad' in batch:
                    ori_shape = batch['ori_shape'][si]
                    ratio_pad = batch['ratio_pad'][si]
                    imgsz = batch['img'].shape[2:]  # (H, W)
                    ops.scale_boxes(imgsz, predn[:, :4], ori_shape, ratio_pad=ratio_pad)

                stat['conf'] = predn[:, 4]
                stat['pred_cls'] = predn[:, 5]

                # 计算 TP (参考 _process_batch 方法)
                if nl:
                    # 准备目标框 (参考 _prepare_batch 方法)
                    # bbox 是归一化的 xywh，需要转换到原始图像空间
                    gt_bbox = ops.xywh2xyxy(bbox)  # 转为 xyxy
                    
                    # 缩放到原始图像空间
                    if 'ori_shape' in batch and 'ratio_pad' in batch:
                        ori_shape = batch['ori_shape'][si]
                        ratio_pad = batch['ratio_pad'][si]
                        imgsz = batch['img'].shape[2:]
                        # 先缩放到当前图像尺寸
                        gt_bbox = gt_bbox * torch.tensor(imgsz, device=device)[[1, 0, 1, 0]]
                        # 再缩放到原始图像空间
                        ops.scale_boxes(imgsz, gt_bbox, ori_shape, ratio_pad=ratio_pad)

                    # 计算 IoU 并匹配
                    iou = box_iou(gt_bbox, predn[:, :4])
                    stat['tp'] = _match_predictions(predn[:, 5], cls, iou, iouv)

                for k in stats.keys():
                    stats[k].append(stat[k])

    # 计算平均损失
    num_batches = len(val_loader)
    metrics = {
        'val_loss': total_loss / num_batches,
        'val_det_loss': det_loss_sum / num_batches,
    }
    
    if criterion.lambda_distill > 0:
        metrics['val_distill_loss'] = distill_loss_sum / num_batches

    # 计算 mAP (参考 get_stats 方法)
    if len(stats['tp']) > 0:
        stats_np = {k: torch.cat(v, 0).cpu().numpy() for k, v in stats.items()}
        
        if len(stats_np['tp']) and stats_np['tp'].any():
            # 获取类别名称
            try:
                import yaml
                config_path = '/root/autodl-tmp/ADV/yaml/data/FLIR.yaml'
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = yaml.safe_load(f)
                        names = config.get('names', {i: f'class_{i}' for i in range(num_classes)})
                else:
                    names = {i: f'class_{i}' for i in range(num_classes)}
            except:
                names = {i: f'class_{i}' for i in range(num_classes)}
            
            if isinstance(names, (list, tuple)):
                names = {i: name for i, name in enumerate(names)}

            det_metrics = DetMetrics(save_dir=None, plot=False, names=names)
            det_metrics.process(**stats_np)

            mean_results = det_metrics.mean_results()
            metrics['val_precision'] = mean_results[0]
            metrics['val_recall'] = mean_results[1]
            metrics['val_map50'] = mean_results[2]
            metrics['val_map50_95'] = mean_results[3]

            print(f"\n{'='*60}")
            print(f"Validation Results - Epoch {epoch + 1}")
            print(f"{'='*60}")
            print(f"mAP50:      {metrics['val_map50']:.4f}")
            print(f"mAP50-95:   {metrics['val_map50_95']:.4f}")
            print(f"Precision:  {metrics['val_precision']:.4f}")
            print(f"Recall:     {metrics['val_recall']:.4f}")
            print(f"{'='*60}\n")

            writer.add_scalar('Val/mAP50', metrics['val_map50'], epoch)
            writer.add_scalar('Val/mAP50-95', metrics['val_map50_95'], epoch)
            writer.add_scalar('Val/Precision', metrics['val_precision'], epoch)
            writer.add_scalar('Val/Recall', metrics['val_recall'], epoch)

    # 记录到 TensorBoard
    writer.add_scalar('Val/Loss', metrics['val_loss'], epoch)
    writer.add_scalar('Val/DetLoss', metrics['val_det_loss'], epoch)
    if criterion.lambda_distill > 0:
        writer.add_scalar('Val/DistillLoss', metrics['val_distill_loss'], epoch)

    return metrics


def _match_predictions(pred_cls, gt_cls, iou, iouv):
    """
    匹配预测和目标，返回 TP 矩阵
    
    参考 M2D-LIF 的 _process_batch 方法
    """
    npr = len(pred_cls)
    niou = len(iouv)
    tp = torch.zeros(npr, niou, dtype=torch.bool, device=iou.device)
    
    # 对每个类别分别处理
    unique_classes = torch.unique(gt_cls)
    
    for cls_idx in unique_classes:
        # 找到同类别的预测和目标
        pred_mask = pred_cls == cls_idx
        gt_mask = gt_cls == cls_idx
        
        if not pred_mask.any() or not gt_mask.any():
            continue
        
        # 提取该类别的 IoU 矩阵
        cls_iou = iou[gt_mask][:, pred_mask]
        
        # 对每个预测框找到最佳匹配
        best_iou, best_gt_idx = cls_iou.max(dim=0)
        
        # 标记 TP
        pred_indices = torch.where(pred_mask)[0]
        for pi, iou_val, gi in zip(pred_indices, best_iou, best_gt_idx):
            for iou_idx, iou_thresh in enumerate(iouv):
                if iou_val > iou_thresh:
                    tp[pi, iou_idx] = True
    
    return tp


def save_checkpoint(
    student: DualModalStudent,
    optimizer: optim.Optimizer,
    scheduler: CombinedScheduler,
    epoch: int,
    save_dir: str,
    filename: str,
    scaler: GradScaler = None,
):
    """
    保存检查点

    参数:
        student: Student模型
        optimizer: 优化器
        scheduler: 调度器
        epoch: 当前epoch
        save_dir: 保存目录
        filename: 文件名
        scaler: 混合精度缩放器（可选）
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    # 保存Student可训练参数
    checkpoint = {
        'epoch': epoch,
        'student_state_dict': {k: v for k, v in student.state_dict().items() if v.requires_grad},
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.lr_scheduler.scheduler.state_dict(),
    }

    # 保存 scaler 状态（如果存在）
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()

    torch.save(checkpoint, save_path)
    print(f"✓ 检查点已保存: {save_path}")


def main():
    """主函数"""
    args = parse_args()

    print("=" * 60)
    print("Stage 2: Student Adversarial Distillation Training")
    print("=" * 60)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 设置随机种子
    set_seed(args.seed)

    # 创建输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 创建TensorBoard写入器
    writer = SummaryWriter(args.log_dir)

    # 加载Teacher模型
    print("\n" + "=" * 60)
    print("Loading Teacher models...")
    print("=" * 60)
    teacher_rgb = load_teacher(args.teacher_rgb, device)
    teacher_ir = load_teacher(args.teacher_ir, device)

    # 创建Student模型
    print("\n" + "=" * 60)
    print("Creating Student model...")
    print("=" * 60)
    student = create_student(
        teacher_rgb=teacher_rgb,
        teacher_ir=teacher_ir,
        num_classes=args.num_classes,
        fusion_channels=args.fusion_channels,
        fusion_mode=args.fusion_mode,
        device=device,
    )

    # 创建优化器
    optimizer = create_optimizer(student, args.lr, args.optimizer)

    # 创建调度器
    scheduler = create_schedulers(
        optimizer=optimizer,
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        max_epsilon=args.max_epsilon,
        adv_schedule_type=args.adv_schedule_type,
        lr_scheduler_type=args.lr_scheduler,
    )

    # 创建损失函数
    criterion = TotalLoss(
        model=student,
        lambda_det=args.lambda_det,
        lambda_distill=args.lambda_distill,
        distill_loss_type='mse',
    )

    # 创建数据加载器
    print("\n" + "=" * 60)
    print("Creating data loaders...")
    print("=" * 60)
    train_loader, val_loader = create_data_loaders({
        'train_yaml': args.data,
        'val_yaml': args.data,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    })

    # 🚨 初始化混合精度缩放器（必须在恢复训练之前）
    scaler = GradScaler() if args.amp else None
    if args.amp:
        print("✓ 已启用混合精度训练 (AMP)")

    # 恢复训练
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"\n恢复训练从: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        student.load_state_dict(checkpoint['student_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.lr_scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        # 恢复 scaler 状态（如果存在）
        if 'scaler_state_dict' in checkpoint and args.amp:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print(f"✓ 已恢复混合精度训练状态")
        print(f"✓ 从epoch {start_epoch}恢复训练")

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 获取当前调度状态
        state = scheduler.get_state(epoch)
        print(f"Learning rate: {state['lr']:.6f}")
        print(f"Epsilon: {state['epsilon']:.4f}")
        print(f"Loss weights: {state['loss_weights']}")

        # 训练
        with tqdm(train_loader, desc=f'Train Epoch {epoch + 1}') as pbar:
            train_metrics = train_epoch(
                student=student,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch,
                device=device,
                epsilon=state['epsilon'],
                lambda_det=args.lambda_det,
                lambda_distill=args.lambda_distill,
                writer=writer,
                progress_bar=pbar,
                scaler=scaler,
            )

        # 🚨 关键修改 3：在训练完成后更新调度器（必须在optimizer.step()之后）
        scheduler.step(epoch)

        # 验证
        print(f"\nValidating...")
        val_metrics = validate(
            student=student,
            val_loader=val_loader,
            criterion=criterion,
            epoch=epoch,
            device=device,
            writer=writer,
            num_classes=args.num_classes,
        )

        # 打印结果
        print(f"Train Loss: {train_metrics['train_loss']:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")

        # 保存最新模型 (last.pt)
        save_checkpoint(
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            save_dir=args.save_dir,
            filename='last.pt',
            scaler=scaler,
        )

        # 保存最佳模型 (best.pt) - 基于 val_loss
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(
                student=student,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                save_dir=args.save_dir,
                filename='best.pt',
                scaler=scaler,
            )
            print(f"✓ 保存最佳模型: val_loss={best_val_loss:.4f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    writer.close()


if __name__ == '__main__':
    main()