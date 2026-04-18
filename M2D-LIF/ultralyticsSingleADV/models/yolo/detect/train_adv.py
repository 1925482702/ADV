"""
单模态对抗训练器

核心设计理念：
1. 两次前向传播：干净前向获取梯度 → 加噪前向真正训练
2. 早期层噪声注入，让噪声随网络传播
3. 三阶段训练：Warmup → ADV Train → Clean Finetune

训练流程：
  Stage 0 (0% ~ 5% epochs): Warmup
    - 纯检测训练，不加噪
    - 先建立基本检测能力

  Stage 1 (5% ~ 80% epochs): ADV Train
    - 干净前向传播 → 获取早期特征梯度
    - 根据梯度生成对抗噪声 (FGSM)
    - 加噪前向传播 → 真正训练
    - epsilon 线性增长

  Stage 2 (80% ~ 100% epochs): Clean Finetune
    - 不再加噪，干净前向传播
    - 纯检测损失优化，巩固检测能力
"""

import time
from copy import copy

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.torch_utils import de_parallel


class SingleModalADVTrainer(DetectionTrainer):
    """
    单模态对抗训练器
    
    在标准 YOLOv8 训练基础上，添加对抗噪声注入：
    1. 第一次前向：干净特征 → 计算损失 → 获取早期特征梯度
    2. 根据梯度生成对抗噪声
    3. 第二次前向：加噪特征 → 计算损失 → 反向传播
    
    关键参数（通过 overrides 传入）:
        adv_enabled: 是否启用对抗训练（默认 True）
        epsilon_max: 最大噪声强度（默认 0.05）
        early_layer_idx: 早期层索引，噪声注入位置（默认 2）
        warmup_epochs: warmup 轮数（默认 5）
        stage2_ratio: 干净微调阶段占比（默认 0.2）
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化单模态对抗训练器
        """
        if overrides is None:
            overrides = {}
        
        # 提取自定义对抗训练参数
        custom_keys = [
            'adv_enabled', 'epsilon_max', 'early_layer_idx',
            'warmup_epochs', 'stage2_ratio'
        ]
        
        self.adv_cfg = {}
        for k in custom_keys:
            if k in overrides:
                self.adv_cfg[k] = overrides.pop(k)
        
        # 调用父类初始化
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        # 从 adv_cfg 读取配置
        self.adv_enabled = self.adv_cfg.get('adv_enabled', True)
        
        if self.adv_enabled:
            self.epsilon_max = self.adv_cfg.get('epsilon_max', 0.05)
            self.early_layer_idx = self.adv_cfg.get('early_layer_idx', 2)
            warmup_epochs = self.adv_cfg.get('warmup_epochs', 5)
            stage2_ratio = self.adv_cfg.get('stage2_ratio', 0.2)
            
            # 创建调度器
            from ultralytics.utils.scheduler import create_single_adv_scheduler
            self.adv_scheduler = create_single_adv_scheduler(
                total_epochs=self.epochs,
                epsilon_max=self.epsilon_max,
            )
            
            # 创建对抗噪声生成器
            from ultralytics.utils.adversarial import create_single_adv_generator
            self.adv_noise_gen = create_single_adv_generator(epsilon=self.epsilon_max)
            
            LOGGER.info(f"单模态对抗训练已启用: epsilon_max={self.epsilon_max}, "
                       f"early_layer_idx={self.early_layer_idx}")
        
        # 禁用蒸馏逻辑
        self.Distillation = None
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        返回检测模型
        
        添加 early_layer_idx 属性，用于确定噪声注入位置
        """
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        
        # 标记早期层索引
        model.early_layer_idx = getattr(self, 'early_layer_idx', 2)
        
        return model
    
    def _do_train(self, world_size=1):
        """
        执行训练，实现两次前向传播逻辑
        
        核心流程：
        1. 第一次前向：干净特征，获取早期特征梯度
        2. 生成对抗噪声
        3. 第二次前向：加噪特征，真正训练
        """
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
        LOGGER.info(f'单模态对抗训练开始，共 {self.epochs} epochs...')
        
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
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                # 优化器步进
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                
                # 更新损失统计
                self.tloss = (self.tloss * i + loss_items) / (i + 1) if self.tloss is not None else loss_items
                
                # 日志
                if RANK in (-1, 0):
                    mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                        (f'{epoch + 1}/{self.epochs}', mem, *self.tloss.tolist()[:3], 
                         batch['cls'].shape[0], batch['img'].shape[-1]))
                
                self.run_callbacks('on_train_batch_end')
            
            # Epoch 结束
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                
                # 验证
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness)
                
                # 保存模型
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks('on_model_save')
            
            # 学习率调度
            self.scheduler.step()
            torch.cuda.empty_cache()
        
        if RANK in (-1, 0):
            self.final_eval()
            self.run_callbacks('on_train_end')
        
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
    
    def _adv_train_step(self, batch, epsilon):
        """
        对抗训练步骤（两次前向传播）
        
        参数:
            batch: 数据批次
            epsilon: 当前噪声强度
        
        返回:
            loss: 总损失
            loss_items: 损失项张量
        """
        from ultralytics.utils.adversarial import set_bn_eval, set_bn_train
        
        x = batch['img']  # [B, 3, H, W]
        
        # ========== 第一次前向：干净特征，获取梯度 ==========
        set_bn_eval(self.model)
        
        # 提取早期特征并标记需要梯度
        early_feat, y_cache = self._extract_early_feature(x)
        early_feat.requires_grad_(True)
        early_feat.retain_grad()
        
        # 从早期特征继续前向
        predictions = self._forward_from_early(early_feat, y_cache)
        
        # 计算检测损失
        self.loss, self.loss_items = self.model.loss(batch, predictions)
        det_loss = self.loss
        
        # 反向传播获取早期特征梯度
        det_loss.backward()
        
        # 提取梯度并生成噪声
        if early_feat.grad is not None:
            grad = early_feat.grad.clone().detach()
            feat_scale = early_feat.abs().mean().detach()
            
            # 归一化梯度
            grad_norm = grad / (grad.norm() + 1e-8)
            
            # 生成噪声（按当前 epsilon 缩放）
            noise = epsilon * feat_scale * grad_norm
        else:
            noise = None
            LOGGER.warning("早期特征梯度为 None，跳过噪声生成")
        
        # 清理第一次前向的计算图
        del predictions, det_loss, early_feat
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
        self.loss, self.loss_items = self.model.loss(batch, predictions)
        
        return self.loss, self.loss_items
    
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
        self.loss, self.loss_items = self.model.loss(batch, predictions)
        
        return self.loss, self.loss_items
    
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
            predictions: 检测预测结果
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
                    curr = y_cache[layer.f]
                else:
                    curr = [curr if j == -1 else y_cache[j] for j in layer.f]
            
            curr = layer(curr)
            y_cache[i] = curr
        
        return curr
    
    def save_model(self):
        """保存模型，添加对抗训练调度状态"""
        import pandas as pd
        from copy import deepcopy
        from datetime import datetime
        from ultralytics.utils.torch_utils import de_parallel
        
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.model)).half(),
            'ema': deepcopy(self.ema.ema).half() if self.ema else None,
            'updates': self.ema.updates if self.ema else 0,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'adv_schedule': self.adv_scheduler.get_state() if self.adv_enabled else None,
        }
        
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
