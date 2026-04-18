"""
单模态 OBB 对抗训练脚本

功能：
- 完整对抗训练逻辑（两次前向传播）
- 三阶段训练策略
- 支持与物体平移增强同时使用
- 支持 OBB（旋转框）数据格式

使用方法:
    python train_adv_obb.py --data ./DroneVehicle.yaml --epochs 100 --epsilon_max 0.05
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER, RANK, DEFAULT_CFG, TQDM
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import OBBModel
from ultralytics.utils.adversarial import create_single_adv_generator, set_bn_eval, set_bn_train
from ultralytics.utils.scheduler import create_single_adv_scheduler


class AdversarialOBBTrainer(OBBTrainer):
    """
    带对抗训练的 OBB 检测训练器
    
    核心逻辑：
    - 第一次前向：干净特征，获取早期特征梯度
    - 生成对抗噪声
    - 第二次前向：加噪特征，真正训练
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        # 提取对抗训练参数
        overrides = overrides or {}
        self.adv_enabled = overrides.pop('adv_enabled', True)
        self.epsilon_max = overrides.pop('epsilon_max', 0.05)
        self.early_layer_idx = overrides.pop('early_layer_idx', 2)
        
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        
        if self.adv_enabled:
            # 创建对抗噪声生成器
            self.adv_noise_gen = create_single_adv_generator(epsilon=self.epsilon_max)
            # 创建对抗训练调度器
            self.adv_scheduler = create_single_adv_scheduler(
                total_epochs=self.epochs,
                epsilon_max=self.epsilon_max
            )
            LOGGER.info(f"🔥 OBB 对抗训练已启用: epsilon_max={self.epsilon_max}, early_layer_idx={self.early_layer_idx}")
        else:
            self.adv_noise_gen = None
            self.adv_scheduler = None
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """返回 OBB 检测模型，并标记早期层索引"""
        model = OBBModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        
        # 标记早期层索引
        model.early_layer_idx = self.early_layer_idx
        
        return model
    
    def _do_train(self, world_size=1):
        """执行训练，实现两次前向传播逻辑"""
        if not self.adv_enabled:
            # 不启用对抗训练，使用标准训练流程
            super()._do_train(world_size)
            return
        
        # === 标准初始化 ===
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        
        nb = len(self.train_loader)
        nw = max(round(3 * nb), 100)  # YOLO 标准的 3 epoch warmup
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        self.run_callbacks('on_train_start')
        LOGGER.info(f'🔥 OBB 对抗训练开始，共 {self.epochs} epochs...')
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            
            # 更新调度器
            schedule_state = self.adv_scheduler.step(epoch)
            current_epsilon = schedule_state['epsilon']
            current_stage = schedule_state['stage']
            
            LOGGER.info(f"Epoch {epoch+1}: stage={current_stage}, eps={current_epsilon:.4f}")
            
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            # 关闭 mosaic
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            
            pbar = TQDM(enumerate(self.train_loader), total=nb) if RANK in (-1, 0) else enumerate(self.train_loader)
            self.tloss = None
            self.optimizer.zero_grad()
            
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                
                # 学习率 warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    # 根据阶段选择训练方式
                    if current_epsilon > 0.001 and current_stage == "adv_train":
                        loss, loss_items = self._adv_train_step(batch, current_epsilon)
                    else:
                        loss, loss_items = self._clean_train_step(batch)
                    
                    if RANK != -1:
                        loss *= world_size
                    
                    self.loss = loss
                    self.loss_items = loss_items
                    self.tloss = (
                        (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                    )
                
                # Backward
                self.scaler.scale(self.loss).backward()
                
                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                
                # Log
                if RANK in (-1, 0):
                    mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"
                    loss_len = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_len))
                        % (f"{epoch + 1}/{self.epochs}", mem, *losses, batch["cls"].shape[0], batch["img"].shape[-1])
                    )
                
                self.run_callbacks('on_train_batch_end')
            
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in (-1, 0):
                # Validation
                self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                
                # Save model
                self.save_model()
            
            torch.cuda.empty_cache()
        
        if RANK in (-1, 0):
            LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in "
                       f"{(time.time() - self.train_time_start) / 3600:.3f} hours.")
            self.final_eval()
    
    def _adv_train_step(self, batch, epsilon):
        """
        对抗训练步骤：两次前向传播
        
        参数:
            batch: 数据批次
            epsilon: 当前噪声强度
        
        返回:
            loss: 总损失
            loss_items: 损失项张量
        """
        x = batch['img']
        
        # ========== 第一次前向：干净特征，获取梯度 ==========
        set_bn_eval(self.model)
        self.model.train()
        
        # 提取早期特征
        early_feat_clean, y_cache = self._extract_early_feature(x)
        early_feat_clean.requires_grad_(True)
        
        # 继续前向到检测头
        predictions = self._forward_from_early(early_feat_clean, y_cache)
        
        # 计算损失
        det_loss, det_loss_items = self.model.loss(batch, predictions)
        
        # 反向传播获取梯度
        det_loss.backward()
        
        # ========== 提取梯度并生成噪声 ==========
        if early_feat_clean.grad is not None:
            grad = early_feat_clean.grad.clone().detach()
            feat_scale = early_feat_clean.abs().mean().detach()
            
            # 归一化梯度
            grad_norm = grad / (grad.norm() + 1e-8)
            
            # 生成噪声
            noise = epsilon * feat_scale * grad_norm
        else:
            noise = None
            LOGGER.warning("早期特征梯度为 None，跳过噪声生成")
        
        # 清理第一次前向的计算图
        del predictions, det_loss, early_feat_clean
        torch.cuda.empty_cache()
        
        # 重置
        self.optimizer.zero_grad()
        set_bn_train(self.model)
        
        # ========== 第二次前向：加噪特征，真正训练 ==========
        self.model.train()
        
        # 重新提取早期特征
        early_feat_clean, y_cache = self._extract_early_feature(x)
        
        # 应用噪声
        if noise is not None:
            early_feat_noisy = early_feat_clean + noise.to(early_feat_clean.dtype)
        else:
            early_feat_noisy = early_feat_clean
        
        # 从加噪的早期特征继续前向
        predictions = self._forward_from_early(early_feat_noisy, y_cache)
        
        # 计算最终损失
        loss, loss_items = self.model.loss(batch, predictions)
        
        return loss, loss_items
    
    def _clean_train_step(self, batch):
        """
        干净训练步骤（标准单次前向传播）
        
        参数:
            batch: 数据批次
        
        返回:
            loss: 总损失
            loss_items: 损失项张量
        """
        x = batch['img']
        
        # 标准前向传播
        predictions = self.model(x)
        
        # 计算损失
        loss, loss_items = self.model.loss(batch, predictions)
        
        return loss, loss_items
    
    def _extract_early_feature(self, x):
        """
        提取早期特征
        
        参数:
            x: 输入图像 [B, 3, H, W]
        
        返回:
            early_feat: 早期特征
            y_cache: 特征缓存（用于跳跃连接）
        """
        early_layer_idx = getattr(self.model, 'early_layer_idx', 2)
        
        # 获取模型的层列表
        if hasattr(self.model, 'model'):
            layers = list(self.model.model.children())
        else:
            layers = list(self.model.children())
        
        # 前向传播到早期层
        y_cache = [None] * len(layers)
        curr = x
        
        for i in range(min(early_layer_idx + 1, len(layers))):
            curr = layers[i](curr)
            y_cache[i] = curr
        
        return curr, y_cache
    
    def _forward_from_early(self, early_feat, y_cache):
        """
        从早期特征继续前向传播
        
        参数:
            early_feat: 早期特征
            y_cache: 特征缓存
        
        返回:
            predictions: 最终预测
        """
        early_layer_idx = getattr(self.model, 'early_layer_idx', 2)
        
        # 获取模型的层列表
        if hasattr(self.model, 'model'):
            layers = list(self.model.model.children())
        else:
            layers = list(self.model.children())
        
        # 从早期层继续前向传播
        curr = early_feat
        
        for i in range(early_layer_idx + 1, len(layers)):
            layer = layers[i]
            
            # 处理跳跃连接
            if hasattr(layer, 'f') and layer.f != -1:
                if isinstance(layer.f, int):
                    curr = layer(y_cache[layer.f] if layer.f >= 0 else curr)
                elif isinstance(layer.f, list):
                    inputs = [y_cache[f] if f >= 0 else curr for f in layer.f]
                    curr = layer(inputs)
            else:
                curr = layer(curr)
            
            y_cache[i] = curr
        
        return curr


def parse_args():
    parser = argparse.ArgumentParser(description='Adversarial OBB Training for YOLOv8')
    parser.add_argument('--model', type=str, default='./ultralytics/cfg/models/v8/yolov8m-obb.yaml',
                        help='Model config file')
    parser.add_argument('--data', type=str, default='./DroneVehicle.yaml',
                        help='Dataset config file (OBB format)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--project', type=str, default='./runs/adv_obb_train',
                        help='Project name')
    parser.add_argument('--name', type=str, default='adv_obb',
                        help='Experiment name')
    
    # 对抗训练参数
    parser.add_argument('--adv_enabled', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable adversarial training')
    parser.add_argument('--epsilon_max', type=float, default=0.05,
                        help='Maximum noise strength')
    parser.add_argument('--early_layer_idx', type=int, default=2,
                        help='Early layer index for noise injection')
    
    # 物体平移增强参数
    parser.add_argument('--object_shift', type=float, default=0.5,
                        help='Probability of applying object shift (0.0=disable)')
    parser.add_argument('--max_shift_px', type=int, default=40,
                        help='Maximum shift in pixels')
    parser.add_argument('--min_shift_px', type=int, default=8,
                        help='Minimum shift in pixels')
    parser.add_argument('--shift_ratio', type=float, default=0.3,
                        help='Ratio of objects to shift')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    LOGGER.info("=" * 60)
    LOGGER.info("🔥 单模态 OBB 对抗训练")
    LOGGER.info("=" * 60)
    LOGGER.info(f"模型配置: {args.model}")
    LOGGER.info(f"数据集: {args.data}")
    LOGGER.info(f"训练参数: epochs={args.epochs}, batch={args.batch}, lr={args.lr}")
    LOGGER.info(f"对抗训练: epsilon_max={args.epsilon_max}, early_layer_idx={args.early_layer_idx}")
    LOGGER.info(f"设备: cuda:{args.device}" if args.device >= 0 else "设备: cpu")
    
    # 准备训练参数
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
        'patience': 50,
        'save_period': 10,
        'close_mosaic': 10,
        'optimizer': 'SGD',
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # 对抗训练参数
        'adv_enabled': args.adv_enabled,
        'epsilon_max': args.epsilon_max,
        'early_layer_idx': args.early_layer_idx,
        
        # 物体平移增强参数
        'object_shift': args.object_shift,
        'max_shift_px': args.max_shift_px,
        'min_shift_px': args.min_shift_px,
        'shift_ratio': args.shift_ratio,
    }
    
    try:
        trainer = AdversarialOBBTrainer(overrides=train_args)
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
