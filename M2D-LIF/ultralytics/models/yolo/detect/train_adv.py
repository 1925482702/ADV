"""
ADV 对抗蒸馏训练器 —— 作为 ultralytics 的插件

核心思路：
  - self.model = 标准 DetectionModel（和 baseline 完全相同架构）
  - Teacher 单独存放，不参与 EMA
  - 仅 override _do_train，在标准训练循环中插入 ADV 逻辑
  - 所有基础设施（lr warmup, EMA, 验证, 数据管线）原封不动继承
"""

import math
import time
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import distributed as dist, optim

from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA, de_parallel, one_cycle


# ---------------------------------------------------------------------------
# ADV Scheduler: 三阶段调度（warmup → adversarial → finetune）
# ---------------------------------------------------------------------------

class ADVScheduler:
    """三阶段 ADV 训练调度器。"""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.warmup_frac = cfg.get('warmup_frac', 0.10)
        self.finetune_frac = cfg.get('finetune_frac', 0.10)
        self.epsilon_max = cfg.get('epsilon_max', 4.0 / 255)
        self.lambda_max = cfg.get('lambda_max', 0.5)

    def get_params(self, epoch, total_epochs):
        """返回 (stage, epsilon, lambda_distill)"""
        frac = epoch / max(total_epochs, 1)

        # Stage 0: Warmup — 纯检测
        if frac < self.warmup_frac:
            return 'warmup', 0.0, 0.0

        # Stage 2: Finetune — 纯检测
        if frac >= (1.0 - self.finetune_frac):
            return 'finetune', 0.0, 0.0

        # Stage 1: Adversarial — cosine ramp-up
        adv_range = 1.0 - self.warmup_frac - self.finetune_frac
        adv_frac = (frac - self.warmup_frac) / max(adv_range, 1e-6)
        ramp = 0.5 * (1 - math.cos(math.pi * adv_frac))

        return 'adversarial', self.epsilon_max * ramp, self.lambda_max * ramp


# ---------------------------------------------------------------------------
# ADVDetectionTrainer
# ---------------------------------------------------------------------------

class ADVDetectionTrainer(DetectionTrainer):
    """
    对抗蒸馏训练器，继承标准 DetectionTrainer。

    与标准训练的唯一区别：
      1. 加载冻结的 Teacher 模型
      2. 在训练循环中插入 FGSM 对抗噪声（图像空间）
      3. 在训练循环中插入 MSE 蒸馏损失（fusion 层 13/18/23）
      4. 三阶段调度（warmup → adversarial → finetune）
    """

    def __init__(self, overrides=None, teacher_weights=None, adv_cfg=None, _callbacks=None):
        # 保存 ADV 配置（在 super().__init__ 之前，避免被 YOLO 参数检查拦截）
        self.teacher_weights = teacher_weights
        self.adv_scheduler = ADVScheduler(adv_cfg)

        # 禁用原有蒸馏逻辑
        if overrides is None:
            overrides = {}
        overrides['Distillation'] = None

        super().__init__(overrides=overrides, _callbacks=_callbacks)

    def get_validator(self):
        """标准验证器 + 扩展 loss 名称以包含 distill_loss。"""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss', 'distill_loss'
        val = super().get_validator()
        return val

    # ------------------------------------------------------------------
    # _do_train: 复制自 BaseTrainer._do_train，插入 4 处 ADV 逻辑
    # 标注为 ### ADV INSERT ### 的部分是新增代码
    # ------------------------------------------------------------------

    def _do_train(self, world_size=1):
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)

        # --- 标准 optimizer / scheduler 构建（不走原有蒸馏路径）---
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model, teacher_model=None, distill_loss=None,
            name=self.args.optimizer, lr=self.args.lr0,
            momentum=self.args.momentum, decay=weight_decay, iterations=iterations,
        )
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)
        else:
            self.lf = lambda x: (1 - x / self.epochs) * (1.0 - self.args.lrf) + self.args.lrf
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.scheduler.last_epoch = self.start_epoch - 1
        self.run_callbacks('on_pretrain_routine_end')

        ### ADV INSERT A: 加载 Teacher + 注册 hooks ###
        self._adv_setup()

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
            f'Starting training for {self.epochs} epochs...'
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])

        epoch = self.epochs
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
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

            ### ADV INSERT: 获取本 epoch 的调度参数 ###
            adv_stage, adv_eps, adv_lambda = self.adv_scheduler.get_params(epoch, self.epochs)
            if RANK in (-1, 0) and (epoch == self.start_epoch or adv_stage != getattr(self, '_prev_stage', None)):
                LOGGER.info(f'ADV: stage={adv_stage}, eps={adv_eps:.5f}, lambda={adv_lambda:.4f}')
                self._prev_stage = adv_stage

            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                ni = i + nb * epoch

                # --- lr warmup（标准 YOLO）---
                if ni <= nw:
                    xi = [0, nw]
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # --- Forward + Loss ---
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)

                    ### ADV INSERT B: 对抗噪声 + 蒸馏 ###
                    self.loss, self.loss_items = self._adv_forward(batch, adv_stage, adv_eps, adv_lambda)

                    if RANK != -1:
                        self.loss *= world_size

                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # --- Backward ---
                self.scaler.scale(self.loss).backward()

                # --- Optimize ---
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:
                            break

                # --- Log ---
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

            # --- epoch 结束 ---
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):
                final_epoch = epoch + 1 == self.epochs
                self.ema.update_attr(self.model,
                                     include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop |= self.stopper(epoch + 1, self.fitness)
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
                if self.args.save or final_epoch:
                    self.save_model()
                    self.run_callbacks('on_model_save')

            # --- Scheduler ---
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

        ### ADV INSERT D: 清理 ###
        self._adv_cleanup()
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    # ------------------------------------------------------------------
    # ADV 专用方法
    # ------------------------------------------------------------------

    def _adv_setup(self):
        """加载 Teacher 模型并注册 feature hooks。"""
        if self.teacher_weights is None:
            LOGGER.warning('ADV: No teacher_weights provided, distillation disabled.')
            self.teacher_model = None
            self._hook_handles = []
            return

        LOGGER.info(f'ADV: Loading teacher from {self.teacher_weights}')
        ckpt = torch.load(self.teacher_weights, map_location=self.device, weights_only=False)
        teacher = ckpt.get('ema') or ckpt.get('model')
        if teacher is None:
            raise ValueError(f'Cannot load teacher from {self.teacher_weights}: no "ema" or "model" key found')
        if hasattr(teacher, 'float'):
            teacher = teacher.float()
        self.teacher_model = teacher.to(self.device)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        LOGGER.info('ADV: Teacher loaded and frozen.')

        # Feature hooks on Add fusion layers (13, 18, 23)
        self.fusion_indices = [13, 18, 23]
        self.student_feats = []
        self.teacher_feats = []
        self._hook_handles = []

        student_raw = de_parallel(self.model)
        for idx in self.fusion_indices:
            h = student_raw.model[idx].register_forward_hook(
                lambda m, inp, out, buf=self.student_feats: buf.append(out))
            self._hook_handles.append(h)

            h = self.teacher_model.model[idx].register_forward_hook(
                lambda m, inp, out, buf=self.teacher_feats: buf.append(out))
            self._hook_handles.append(h)

        LOGGER.info(f'ADV: Feature hooks registered on layers {self.fusion_indices}')

    def _adv_forward(self, batch, stage, epsilon, lambda_d):
        """
        ADV 增强的 forward + loss。

        三种模式：
          warmup / finetune → 标准检测 forward
          adversarial        → FGSM 噪声 + 检测 + 蒸馏
        """
        imgs_clean = batch['img']

        # --- (1) 对抗噪声（仅 adversarial 阶段）---
        if stage == 'adversarial' and epsilon > 0:
            # 探测 forward 前清空 hooks 残留
            self.student_feats.clear()
            self.teacher_feats.clear()

            imgs_for_grad = imgs_clean.detach().clone().requires_grad_(True)

            # 直接调用 model 的 predict 和 criterion，避免走 model(dict) 可能的副作用
            student_raw = de_parallel(self.model)
            probe_preds = student_raw.predict(imgs_for_grad)
            probe_loss = student_raw.criterion(probe_preds, batch)[0]
            grad = torch.autograd.grad(probe_loss, imgs_for_grad, create_graph=False)[0]

            # 清空 probe 产生的 hook 特征
            self.student_feats.clear()
            self.teacher_feats.clear()

            # FGSM：随机攻击一个模态
            noise = epsilon * grad.sign()
            if torch.rand(1).item() < 0.5:
                noise[:, :3] = 0   # 只攻击 IR（后 3 通道）
            else:
                noise[:, 3:] = 0   # 只攻击 RGB（前 3 通道）

            batch['img'] = (imgs_clean + noise.detach()).clamp(0, 1)

        # --- (2) 主检测 forward ---
        self.student_feats.clear()
        self.teacher_feats.clear()
        det_loss, det_items = self.model(batch)

        # --- (3) 蒸馏损失 ---
        distill_val = torch.tensor(0.0, device=det_loss.device)
        if lambda_d > 0 and self.teacher_model is not None:
            with torch.no_grad():
                self.teacher_model.predict(imgs_clean)  # 用 predict 而非 .model()，正确处理跳跃连接

            d_loss = torch.tensor(0.0, device=det_loss.device)
            n_feats = min(len(self.student_feats), len(self.teacher_feats))
            for k in range(n_feats):
                sf = self.student_feats[k].float()
                tf = self.teacher_feats[k].float()
                d_loss = d_loss + F.mse_loss(sf, tf)
            if n_feats > 0:
                d_loss = d_loss / n_feats

            distill_val = lambda_d * d_loss
            det_loss = det_loss + distill_val

        self.student_feats.clear()
        self.teacher_feats.clear()

        # 拼接 loss_items（检测3项 + 蒸馏1项）
        loss_items = torch.cat([det_items, distill_val.detach().unsqueeze(0)])
        return det_loss, loss_items

    def _adv_cleanup(self):
        """清理 hooks 和 teacher 模型。"""
        for h in getattr(self, '_hook_handles', []):
            h.remove()
        if hasattr(self, 'teacher_model') and self.teacher_model is not None:
            del self.teacher_model
            self.teacher_model = None
