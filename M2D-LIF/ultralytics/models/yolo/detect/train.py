# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
import time
import warnings
from copy import copy

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first, EarlyStopping, one_cycle


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """
    def __init__(self, overrides=None, _callbacks=None):
        super().__init__(overrides=overrides)
        self._callbacks = _callbacks

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        if self.args.multi_scale:
            imgs = batch['img']
            sz = (random.randrange(self.args.imgsz * 0.5, self.args.imgsz * 1.5 + self.stride) // self.stride *
                  self.stride)  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.stride) * self.stride
                      for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
            batch['img'] = imgs
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        # 是否蒸馏
        if self.Distillation is not None:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'lif_loss', 'im_loss', 'cm_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader,
                                              save_dir=self.save_dir,
                                              args=copy(self.args),
                                              _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        # import pdb
        # pdb.set_trace()
        return ('\n' + '%11s' *(4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'][:, :3],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)
        if batch['img'].shape[1]==6:
            plot_images(images=batch['img'][:, 3:],
                        batch_idx=batch['batch_idx'],
                        cls=batch['cls'].squeeze(-1),
                        bboxes=batch['bboxes'],
                        paths=batch['im_file'],
                        fname=self.save_dir / f'train_{ni}.jpg',
                        on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)


# ============================================================================
# ShiftScheduler: 动态调度平移概率和 Loss 权重
# ============================================================================
class ShiftScheduler:
    """
    动态调度器：平移概率 + Loss 权重
    
    课程学习策略：
    - Warmup 阶段：不启用平移，让模型学习基本检测
    - 训练阶段：逐渐增加平移概率和 loss 权重
    """
    
    def __init__(self, warmup_epochs=10, p_start=0.3, p_end=0.8,
                 lambda_start=0.1, lambda_end=0.5):
        """
        Args:
            warmup_epochs: 预热 epoch 数
            p_start: 初始平移概率
            p_end: 最终平移概率（最大 0.8，保留 20% 不平移样本）
            lambda_start: 初始 loss 权重
            lambda_end: 最终 loss 权重
        """
        self.warmup_epochs = warmup_epochs
        self.p_start = p_start
        self.p_end = min(p_end, 0.8)  # 最高 0.8，确保 20% 样本不平移
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
    
    def get_params(self, epoch, total_epochs):
        """
        获取当前 epoch 的调度参数
        
        Returns:
            p: 平移概率
            lambda_shift: loss 权重
        """
        if epoch < self.warmup_epochs:
            return 0.0, 0.0
        
        progress = (epoch - self.warmup_epochs) / max(total_epochs - self.warmup_epochs, 1)
        
        p = self.p_start + (self.p_end - self.p_start) * progress
        lambda_shift = self.lambda_start + (self.lambda_end - self.lambda_start) * progress
        
        return p, lambda_shift


# ============================================================================
# ShiftHead: 偏移预测头
# ============================================================================
class ShiftHead(nn.Module):
    """
    偏移预测头
    
    输入：融合后的特征图 (B, C, H, W)
    输出：相对偏移量 (B, 2)
    """
    
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 2)  # dx, dy
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 特征图
        Returns:
            shift: (B, 2) [dx, dy]
        """
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# ============================================================================
# ShiftDetectionTrainer: 带跨模态平移预测的检测训练器
# ============================================================================
class ShiftDetectionTrainer(DetectionTrainer):
    """
    带跨模态平移预测的检测训练器
    
    继承 DetectionTrainer，增加：
    1. 动态平移概率调度
    2. 偏移预测分支
    3. 联合损失训练
    """
    
    def __init__(self, overrides=None, _callbacks=None, shift_cfg=None):
        """
        Args:
            overrides: 参数覆盖
            _callbacks: 回调函数
            shift_cfg: 平移配置，包含：
                - warmup_epochs: 预热 epoch 数
                - p_start: 初始平移概率
                - p_end: 最终平移概率
                - lambda_start: 初始 loss 权重
                - lambda_end: 最终 loss 权重
                - fusion_layer_idx: 融合层索引（用于提取特征）
                - fusion_channels: 融合层通道数
        """
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        # 平移配置
        self.shift_cfg = shift_cfg or {}
        self.shift_scheduler = ShiftScheduler(
            warmup_epochs=self.shift_cfg.get('warmup_epochs', 10),
            p_start=self.shift_cfg.get('p_start', 0.3),
            p_end=self.shift_cfg.get('p_end', 0.8),
            lambda_start=self.shift_cfg.get('lambda_start', 0.1),
            lambda_end=self.shift_cfg.get('lambda_end', 0.5),
        )
        
        # 偏移预测相关
        self.shift_head = None
        self.shift_loss_fn = None
        self.current_shift_p = 0.0
        self.current_shift_lambda = 0.0
        self.fusion_layer_idx = None  # 自动检测
        self.fusion_channels = None   # 自动检测
    
    def setup_model(self):
        """设置模型，添加偏移预测头"""
        ret = super().setup_model()
        
        # 自动检测融合层和通道数
        self._detect_fusion_layer()
        
        # 创建偏移预测头
        self.shift_head = ShiftHead(in_channels=self.fusion_channels).to(self.device)
        
        # 创建损失函数
        from ultralytics.utils.loss import ShiftLoss
        self.shift_loss_fn = ShiftLoss(max_shift=50.0).to(self.device)
        
        LOGGER.info(f'ShiftDetectionTrainer: auto-detected fusion_layer_idx={self.fusion_layer_idx}, fusion_channels={self.fusion_channels}')
        
        return ret
    
    def _detect_fusion_layer(self):
        """
        自动检测融合层
        
        策略：找到第一个 Add 或 Concat 模块（融合操作），使用该层的输出特征
        """
        model = de_parallel(self.model)
        
        # 查找融合层（Add 或 Concat）
        fusion_types = ('Add', 'Concat')
        for i, m in enumerate(model.model):
            module_type = type(m).__name__
            if module_type in fusion_types:
                self.fusion_layer_idx = i
                break
        
        # 如果没找到融合层，使用倒数第3层
        if self.fusion_layer_idx is None:
            self.fusion_layer_idx = len(model.model) - 3
        
        # 从模型配置获取输入通道数
        in_channels = getattr(model.yaml, 'ch', 6) if hasattr(model, 'yaml') else 6
        if isinstance(model.yaml, dict):
            in_channels = model.yaml.get('ch', 6)
        
        # 探测该层的通道数
        try:
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, 640, 640).to(self.device)
                y = []
                x = dummy_input
                for i, m in enumerate(model.model):
                    if m.f != -1:
                        x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
                    x = m(x)
                    y.append(x if m.i in model.save else None)
                    if i == self.fusion_layer_idx:
                        self.fusion_channels = x.shape[1]
                        return
        except Exception as e:
            LOGGER.warning(f'ShiftDetectionTrainer: Failed to detect fusion channels, using default 512. Error: {e}')
        
        # 默认值
        self.fusion_channels = 512
    
    def preprocess_batch(self, batch):
        """预处理 batch，处理 shift GT"""
        batch = super().preprocess_batch(batch)
        
        # 处理 shift GT
        if 'shift_dx' in batch:
            batch['shift_gt'] = torch.stack([
                batch['shift_dx'].to(self.device),
                batch['shift_dy'].to(self.device)
            ], dim=1).float()  # (B, 2)
        
        return batch
    
    def get_validator(self):
        """获取验证器，添加 shift_loss 到 loss_names"""
        if self.Distillation is not None:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'lif_loss', 'im_loss', 'cm_loss', 'shift_loss'
        else:
            self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'shift_loss'
        
        return yolo.detect.DetectionValidator(self.test_loader,
                                              save_dir=self.save_dir,
                                              args=copy(self.args),
                                              _callbacks=self.callbacks)
    
    def _extract_fusion_features(self, x):
        """
        提取融合层特征
        
        Args:
            x: 输入图像 (B, C, H, W)
        Returns:
            fusion_feat: 融合层特征
        """
        # 获取模型
        model = de_parallel(self.model)
        
        # 前向传播到融合层
        y = []
        for i, m in enumerate(model.model):
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in model.save else None)
            
            # 在融合层返回特征
            if i == self.fusion_layer_idx:
                return x
        
        # 如果没有找到融合层，返回最后一层特征
        return x
    
    def _do_train(self, world_size=1):
        """
        训练循环（覆盖父类，添加 shift loss 计算）
        
        注意：这个方法复制自 BaseTrainer._do_train，只添加了 shift 相关逻辑
        标记为 ### SHIFT INSERT ### 的部分是新增代码
        """
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        self._setup_scheduler()
        
        ### SHIFT INSERT A: 设置 shift head 的优化器 ###
        # 将 shift_head 参数添加到优化器（必须设置 initial_lr）
        self.optimizer.add_param_group({
            'params': self.shift_head.parameters(),
            'lr': self.args.lr0,
            'initial_lr': self.args.lr0  # 必须设置，否则 warmup 会报错
        })
        
        # 确保所有参数组都有 initial_lr
        for x in self.optimizer.param_groups:
            x.setdefault('initial_lr', self.args.lr0)
        
        nb = len(self.train_loader)
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks('on_train_start')
        LOGGER.info(
            f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
            f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.epochs
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            
            ### SHIFT INSERT B: 获取当前 epoch 的调度参数 ###
            self.current_shift_p, self.current_shift_lambda = self.shift_scheduler.get_params(epoch, self.epochs)
            if RANK in (-1, 0):
                if epoch == self.start_epoch or epoch == self.shift_scheduler.warmup_epochs:
                    LOGGER.info(f'SHIFT: p={self.current_shift_p:.2f}, lambda={self.current_shift_lambda:.2f}')
            
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                ni = i + nb * epoch
                
                # Warmup
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])
                
                # Forward + Loss
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    
                    # 检测 loss
                    self.loss, self.loss_items = self.model(batch)
                    
                    ### SHIFT INSERT C: 计算 shift loss ###
                    if self.current_shift_lambda > 0 and 'shift_gt' in batch:
                        # 提取融合层特征
                        fusion_feat = self._extract_fusion_features(batch['img'])
                        # 预测偏移
                        shift_pred = self.shift_head(fusion_feat)
                        # 计算 shift loss
                        shift_loss = self.shift_loss_fn(shift_pred, batch['shift_gt'])
                        # 添加到总 loss
                        self.loss = self.loss + self.current_shift_lambda * shift_loss
                        
                        # 调试信息：每 100 步打印一次
                        if ni % 100 == 0:
                            gt_mean = batch['shift_gt'].abs().mean().item()
                            pred_mean = shift_pred.abs().mean().item()
                            LOGGER.info(f'SHIFT DEBUG: GT mean={gt_mean:.2f}, Pred mean={pred_mean:.2f}, Loss={shift_loss.item():.4f}')
                        
                        # 更新 loss_items
                        self.loss_items = torch.cat([self.loss_items, shift_loss.detach().unsqueeze(0)])
                    else:
                        # 没有 shift loss 时，添加 0
                        self.loss_items = torch.cat([self.loss_items, torch.zeros(1, device=self.loss.device)])
                    
                    if RANK != -1:
                        self.loss *= world_size
                    
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None else self.loss_items
                
                # Backward
                self.scaler.scale(self.loss).backward()
                
                # Optimize
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            import torch.distributed as dist
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break
                
                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + len(losses.tolist()))) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses.tolist(),
                         batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)
                
                self.run_callbacks('on_train_batch_end')
            
            # Epoch end
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness)
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks('on_model_save')
            
            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                if self.args.time:
                    mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                    self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                    self._setup_scheduler()
                    self.scheduler.last_epoch = self.epoch
                    self.stop |= epoch >= self.epochs
                self.scheduler.step()
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()
            
            if RANK != -1:
                import torch.distributed as dist
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            if self.stop:
                break
        
        if RANK in (-1, 0):
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
