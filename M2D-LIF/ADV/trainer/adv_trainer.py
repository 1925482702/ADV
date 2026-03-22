"""
ADV 对抗蒸馏训练器

独立的训练器模块，避免循环导入问题。
"""

import time
from copy import copy

import numpy as np
import torch
from torch import nn, optim

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK, TQDM


class ADVBaseTrainer(DetectionTrainer):
    """
    对抗蒸馏训练器
    
    核心设计理念：
    1. Teacher使用Baseline模型（已训练好的双模态融合模型）
    2. 对抗噪声注入在backbone早期层，让噪声随网络传播
    3. 两次前向传播：干净前向获取梯度 → 加噪前向真正训练
    4. 残差块学习"误差修补"能力，打破Teacher上限
    5. 两阶段训练：对抗蒸馏阶段 → 干净微调阶段
    
    训练流程：
    Stage 1 (0% ~ 80% epochs): 对抗蒸馏阶段
      - 干净前向传播 → 获取早期特征梯度
      - 根据梯度生成对抗噪声 (FGSM/PGD)
      - 加噪前向传播 → 真正训练
      - 检测损失 + 蒸馏损失（Student逼近Teacher）
      - 残差块学习"误差修补"
    
    Stage 2 (80% ~ 100% epochs): 干净微调阶段
      - 不再加噪，干净前向传播
      - 蒸馏权重逐渐衰减到0
      - 纯检测损失优化，巩固检测能力
    """
    
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        初始化ADV训练器
        
        关键参数（通过overrides传入）:
            Baseline_Teacher: Baseline双模态Teacher模型路径
            epsilon_max: 最大噪声强度（默认0.05）
            lambda_distill_start: 初始蒸馏权重（默认0.8）
            early_layer_idx: 早期层索引，噪声注入位置（默认2）
            attack_method: 攻击方法（'fgsm' 或 'pgd'，默认'fgsm'）
        """
        if overrides is None:
            overrides = {}
        
        # 🚨 核心修复：分离自定义参数，避开 YOLO 官方的严格参数白名单检查
        custom_keys = [
            'ADV_enabled', 'Baseline_Teacher', 'epsilon_max',
            'lambda_distill_start', 'early_layer_idx', 'attack_method',
            'warmup_epochs', 'stage2_ratio'
        ]
        
        # 1. 把自定义参数从 overrides 里提取出来
        self.adv_cfg = {}
        for k in custom_keys:
            if k in overrides:
                self.adv_cfg[k] = overrides.pop(k)
        
        # 禁用原有的蒸馏逻辑（这是官方支持的参数，不需要移除）
        overrides['Distillation'] = None
        
        # 2. 调用父类初始化（DetectionTrainer 不接受 cfg 参数）
        super().__init__(overrides=overrides, _callbacks=_callbacks)
        
        # 3. 检查通过后，把自定义参数注入到 self.args
        for k, v in self.adv_cfg.items():
            setattr(self.args, k, v)
        
        # 从 self.args 读取配置
        self.ADV_enabled = self.adv_cfg.get('ADV_enabled', True)
        
        if self.ADV_enabled:
            self.baseline_teacher_path = self.adv_cfg.get('Baseline_Teacher')
            self.epsilon_max = self.adv_cfg.get('epsilon_max', 0.05)
            self.lambda_distill_start = self.adv_cfg.get('lambda_distill_start', 0.5)
            self.early_layer_idx = self.adv_cfg.get('early_layer_idx', 2)
            self.attack_method = self.adv_cfg.get('attack_method', 'fgsm')
        
        if self.ADV_enabled:
            # 初始化ADV调度器
            from ADV.trainer.scheduler import create_adv_scheduler
            self.adv_scheduler = create_adv_scheduler(
                total_epochs=self.epochs,
                epsilon_max=self.epsilon_max,
                lambda_distill_start=self.lambda_distill_start,
            )
            
            # ADV模块将在_setup_train中初始化
            self.adv_noise_gen = None
            self.student_model = None
            self.total_loss_fn = None
            
            LOGGER.info(f"ADV Training enabled with epsilon_max={self.epsilon_max}, "
                       f"lambda_distill_max={self.lambda_distill_start}")
    
    def _setup_train(self, world_size):
        """设置训练环境，添加ADV模块初始化"""
        # 🚨 关键修复：在调用父类之前，先把自定义参数从 self.args 暂时摘掉
        # 否则父类的 get_validator() 会因为非法参数报 SyntaxError
        _custom_keys = [
            'ADV_enabled', 'Baseline_Teacher', 'epsilon_max',
            'lambda_distill_start', 'early_layer_idx', 'attack_method',
            'warmup_epochs', 'stage2_ratio', 'baseline_teacher',
        ]
        _saved = {}
        for k in _custom_keys:
            if hasattr(self.args, k):
                _saved[k] = getattr(self.args, k)
                delattr(self.args, k)
        
        super()._setup_train(world_size)
        
        # 父类调用完毕，把自定义参数装回去，后续 ADV 逻辑还需要用
        for k, v in _saved.items():
            setattr(self.args, k, v)
        
        if not self.ADV_enabled:
            return
        
        LOGGER.info("Initializing ADV modules...")
        
        # 1. 使用 get_model 创建的 Student 模型（self.model）
        self.student_model = self.model
        self.student_model = self.student_model.to(self.device)
        
        # 2. 设置 Teacher 为评估模式（在 get_model 中已加载）
        if self.baseline_teacher is not None:
            self.baseline_teacher = self.baseline_teacher.to(self.device)
            self.baseline_teacher.eval()
            for param in self.baseline_teacher.parameters():
                param.requires_grad = False
            for module in self.baseline_teacher.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.track_running_stats = False
            LOGGER.info(f"Baseline Teacher frozen and set to eval mode")
        
        # 3. 创建对抗噪声生成器
        from ADV.models.adversarial import AdversarialNoiseGenerator
        self.adv_noise_gen = AdversarialNoiseGenerator(
            attack_method=self.attack_method,
            epsilon=self.epsilon_max,
            normalize_noise=True,
        )
        LOGGER.info(f"Adversarial noise generator created: method={self.attack_method}")
        
        # 4. 创建总损失函数
        from ADV.losses.distill_loss import TotalLoss
        self.total_loss_fn = TotalLoss(
            model=self.student_model,
            lambda_det=1.0,
            lambda_distill=self.lambda_distill_start,
            distill_loss_type='mse',
            distill_loss_scale=1.0,
        )
        
        # 5. 重建优化器
        self._rebuild_optimizer()
        
        # 6. 更新损失名称
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'distill_loss']
        
        # 7. 初始化验证器实例
        self.validator = self.get_validator()
        
        LOGGER.info("ADV modules initialization completed!")
    
    def _rebuild_optimizer(self):
        """重建优化器，包含Student模型参数"""
        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        
        for module_name, module in self.student_model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                fullname = f'{module_name}.{param_name}' if module_name else param_name
                if 'bias' in fullname:
                    g[2].append(param)
                elif isinstance(module, bn):
                    g[1].append(param)
                else:
                    g[0].append(param)
        
        if self.total_loss_fn is not None:
            for k, v in self.total_loss_fn.named_parameters():
                if not v.requires_grad:
                    continue
                if 'bias' in k:
                    g[2].append(v)
                elif 'bn' in k:
                    g[1].append(v)
                else:
                    g[0].append(v)
        
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        lr = self.args.lr0
        momentum = self.args.momentum
        optimizer_name = self.args.optimizer
        
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            optimizer = optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        
        optimizer.add_param_group({'params': g[0], 'weight_decay': weight_decay})
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
        
        self.optimizer = optimizer
        self._setup_scheduler()
        LOGGER.info("Optimizer rebuilt with Student model parameters")
    
    def _do_train(self, world_size=1):
        """执行训练，实现两次前向传播逻辑"""
        if not self.ADV_enabled:
            super()._do_train(world_size)
            return
        
        if world_size > 1:
            self._setup_ddp(world_size)
        self._setup_train(world_size)
        
        nb = len(self.train_loader)
        # 🚨 关键修复：YOLO lr warmup 必须保留，不能被 ADV 的 warmup_epochs 覆盖
        # ADV warmup_epochs 控制的是"何时开始蒸馏/加噪"，和 lr warmup 是两回事
        # 这里使用标准 YOLO 的 3 个 epoch lr warmup
        yolo_warmup_epochs = 3.0
        nw = max(round(yolo_warmup_epochs * nb), 100)
        last_opt_step = -1
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        
        self.run_callbacks('on_train_start')
        LOGGER.info(f'ADV Training starting for {self.epochs} epochs...')
        
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            
            # 更新调度器
            schedule_state = self.adv_scheduler.step(epoch)
            current_epsilon = schedule_state['epsilon']
            current_lambda_distill = schedule_state['lambda_distill']
            current_stage = schedule_state['stage']
            
            self.total_loss_fn.update_weights(lambda_distill=current_lambda_distill)
            
            LOGGER.info(f"Epoch {epoch+1}: stage={current_stage}, eps={current_epsilon:.4f}, "
                       f"lambda={current_lambda_distill:.4f}")
            
            self.student_model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()
            
            pbar = TQDM(enumerate(self.train_loader), total=nb) if RANK in (-1, 0) else enumerate(self.train_loader)
            self.tloss = None
            self.optimizer.zero_grad()
            
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                
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
                    
                    if current_epsilon > 0.001 and current_stage != "warmup":
                        loss, loss_items = self._adv_train_step(batch, current_epsilon)
                    else:
                        loss, loss_items = self._clean_train_step(batch)
                    
                    if RANK != -1:
                        loss *= world_size
                
                self.scaler.scale(loss).backward()
                
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni
                
                self.tloss = (self.tloss * i + loss_items) / (i + 1) if self.tloss is not None else loss_items
                
                if RANK in (-1, 0):
                    mem = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'
                    pbar.set_description(('%11s' * 2 + '%11.4g' * 6) %
                        (f'{epoch + 1}/{self.epochs}', mem, *self.tloss.tolist()[:4], 
                         batch['cls'].shape[0], batch['img'].shape[-1]))
                
                self.run_callbacks('on_train_batch_end')
            
            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}
            self.run_callbacks('on_train_epoch_end')
            
            if RANK in (-1, 0):
                if self.args.val or epoch + 1 == self.epochs:
                    # 🚨 核心修复：直接调用 self.validator(trainer=self)，而不是 self.validate()
                    # self.validate() 是父类方法，会走官方流程调用错误的 model
                    val_result = self.validator(trainer=self)
                    self.metrics = val_result['metrics']
                    self.fitness = val_result['fitness']
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.save_model()
            
            self.scheduler.step()
            torch.cuda.empty_cache()
        
        if RANK in (-1, 0):
            self.final_eval()
            self.run_callbacks('on_train_end')
        
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')
    
    def _adv_train_step(self, batch, epsilon):
        """对抗训练步骤（两次前向传播）"""
        from ADV.models.adversarial import set_bn_eval, set_bn_train, random_modality_selection
        
        x, targets = batch['img'], batch
        
        # 第一次前向：获取梯度
        set_bn_eval(self.student_model)
        predictions, rgb_early, ir_early, y_rgb, y_ir = self.student_model.forward_clean_for_adv(x)
        
        # 使用检测损失获取梯度
        det_loss_dict = self.total_loss_fn.det_loss(predictions, targets)
        det_loss = det_loss_dict['total']
        
        # 🚨 显存修复：不保留计算图，backward 后图自动释放
        det_loss.backward()
        
        # 🚨 显存修复：立即 clone 梯度和特征尺度，再释放第一次前向的张量
        rgb_grad = rgb_early.grad.clone().detach() if rgb_early.grad is not None else None
        ir_grad = ir_early.grad.clone().detach() if ir_early.grad is not None else None
        # 🚨 关键：保存特征尺度，用于 normalize_noise 模式的噪声生成
        rgb_feat_scale = rgb_early.abs().mean().detach() if rgb_grad is not None else None
        ir_feat_scale = ir_early.abs().mean().detach() if ir_grad is not None else None
        
        # 🚨 显存修复：释放第一次前向的计算图和中间变量
        del predictions, det_loss, det_loss_dict, rgb_early, ir_early, y_rgb, y_ir
        torch.cuda.empty_cache()
        
        # 生成噪声（用保存的梯度和特征尺度手动计算）
        batch_size = x.shape[0]
        attack_rgb = random_modality_selection(batch_size, self.device)
        
        noise_rgb = noise_ir = None
        if rgb_grad is not None:
            # 🚨 手动按 normalize_noise 公式计算：noise = epsilon * feat_scale * grad_norm
            grad_norm = rgb_grad / (rgb_grad.norm() + 1e-8)
            raw_noise = self.epsilon_max * rgb_feat_scale * grad_norm * (epsilon / self.epsilon_max)
            noise_rgb = torch.where(
                attack_rgb.view(-1, 1, 1, 1).expand_as(rgb_grad),
                raw_noise,
                torch.zeros_like(rgb_grad))
        if ir_grad is not None:
            grad_norm = ir_grad / (ir_grad.norm() + 1e-8)
            raw_noise = self.epsilon_max * ir_feat_scale * grad_norm * (epsilon / self.epsilon_max)
            noise_ir = torch.where(
                (~attack_rgb).view(-1, 1, 1, 1).expand_as(ir_grad),
                raw_noise,
                torch.zeros_like(ir_grad))
        
        # 🚨 显存修复：释放梯度和特征尺度
        del rgb_grad, ir_grad, rgb_feat_scale, ir_feat_scale
        
        self.optimizer.zero_grad()
        set_bn_train(self.student_model)
        
        # 第二次前向：加噪训练
        self.student_model.train()
        predictions, features_dict = self.student_model.forward_with_noise(x, noise_rgb, noise_ir)
        
        # 🚨 提取 Teacher 特征（用于蒸馏损失）
        self.student_model.hook_teacher.clear()
        with torch.no_grad():
            # 🚨 类型对齐：确保输入类型和 teacher 权重一致
            teacher_dtype = next(self.student_model.teacher.parameters()).dtype
            _ = self.student_model.teacher(x.to(teacher_dtype))
        teacher_features = [
            self.student_model.hook_teacher.get_features()[f'model.{i}'].detach().clone()
            for i in self.student_model.teacher_fusion_layers
        ]
        self.student_model.hook_teacher.clear()
        
        # 添加 Teacher 特征到 features_dict
        features_dict['teacher_features'] = teacher_features
        
        # 计算总损失
        loss_dict = self.total_loss_fn(predictions, targets, features_dict)
        loss = loss_dict['total']
        # TotalLoss 返回的键: det_bbox, det_cls, det_dfl, distill
        loss_items = torch.tensor([
            loss_dict.get('det_bbox', 0.0), loss_dict.get('det_cls', 0.0),
            loss_dict.get('det_dfl', 0.0), loss_dict.get('distill', 0.0),
        ], device=self.device)
        
        return loss, loss_items
    
    def _clean_train_step(self, batch):
        """干净训练步骤"""
        x, targets = batch['img'], batch
        
        predictions, features_dict = self.student_model.forward(x, return_features=True)
        
        # 计算总损失
        loss_dict = self.total_loss_fn(predictions, targets, features_dict)
        loss = loss_dict['total']
        # TotalLoss 返回的键: det_bbox, det_cls, det_dfl, distill
        loss_items = torch.tensor([
            loss_dict.get('det_bbox', 0.0), loss_dict.get('det_cls', 0.0),
            loss_dict.get('det_dfl', 0.0), loss_dict.get('distill', 0.0),
        ], device=self.device)
        
        return loss, loss_items
    
    def save_model(self):
        """保存模型"""
        import pandas as pd
        from copy import deepcopy
        from datetime import datetime
        from ultralytics.utils.torch_utils import de_parallel
        
        metrics = {**self.metrics, **{'fitness': self.fitness}}
        results = {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient='list').items()}
        
        # 🚨 核心修复：在 deepcopy 之前，清空模型里残留的计算图张量！
        if hasattr(self, 'student_model') and self.student_model is not None:
            self.student_model.fused_features = None
            self.student_model.teacher_features = None
            self.student_model.student_base_sum = None
            self.student_model.early_rgb = None
            self.student_model.early_ir = None
            if hasattr(self.student_model, 'hook_teacher'):
                self.student_model.hook_teacher.clear()
        
        ckpt = {
            'epoch': self.epoch,
            'best_fitness': self.best_fitness,
            'model': deepcopy(de_parallel(self.student_model)).half() if self.student_model else None,
            'ema': deepcopy(self.ema.ema).half() if self.ema else None,
            'updates': self.ema.updates if self.ema else 0,
            'optimizer': self.optimizer.state_dict(),
            'train_args': vars(self.args),
            'train_metrics': metrics,
            'train_results': results,
            'date': datetime.now().isoformat(),
            'adv_schedule': self.adv_scheduler.get_state() if self.ADV_enabled else None,
        }
        
        torch.save(ckpt, self.last)
        if self.best_fitness == self.fitness:
            torch.save(ckpt, self.best)
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        """
        覆盖父类的 get_model，返回我们自定义的双模态 Student 模型
        
        注意：这个方法会在 _setup_train 之前被调用，
        所以 Teacher 需要在这里加载，而不是在 _setup_train 中
        """
        from ADV.models.student import create_student_model
        from ultralytics import YOLO
        
        LOGGER.info("🚀 构建双模态 ADV Student 模型...")
        
        # 1. 加载 Baseline Teacher
        teacher_path = self.args.baseline_teacher if hasattr(self.args, 'baseline_teacher') else self.baseline_teacher_path
        if not teacher_path:
            raise ValueError("Baseline_Teacher path is required for ADV training!")
        
        LOGGER.info(f"加载 Baseline Teacher: {teacher_path}")
        teacher_model = YOLO(teacher_path)
        self.baseline_teacher = teacher_model.model
        
        # 2. 获取数据集的类别数
        nc = self.data['nc'] if self.data else getattr(self.args, 'nc', 3)
        
        # 3. 创建 Student 模型
        student_model = create_student_model(
            baseline_teacher=self.baseline_teacher,
            num_classes=nc,
            use_residual=True,
            fusion_mode='add',
        )
        
        # 将配置中的早期层索引注入模型
        student_model.early_layer_idx = getattr(self.args, 'early_layer_idx', 2)
        
        # 4. 处理预训练权重加载（用于 resume）
        if weights:
            LOGGER.info(f"加载 Student 权重: {weights}")
            ckpt = torch.load(weights, map_location='cpu')
            if 'model' in ckpt:
                student_model.load_state_dict(ckpt['model'].state_dict())
            else:
                student_model.load_state_dict(ckpt)
        
        LOGGER.info(f"Student model created with early_layer_idx={student_model.early_layer_idx}")
        return student_model
    
    def build_dataset(self, img_path, mode='train', batch=None):
        """
        构建双模态数据集
        
        🚨 核心修复：使用 DualModalityDataset 替代官方 YOLODataset
        🚨 强制 rect=False，确保验证时图像尺寸为正方形 (640x640)
        
        Args:
            img_path: 图像路径
            mode: 'train' 或 'val'
            batch: 批次大小
            
        Returns:
            DualModalityDataset 实例
        """
        from ADV.data.dataset import DualModalityDataset
        
        gs = max(int(self.model.stride.max() if self.model else 0), 32)
        
        # 🚨 修复：将 IterableSimpleNamespace 转换为字典，避免 setdefault 报错
        hyp = dict(vars(self.args)) if hasattr(self.args, '__dict__') else dict(self.args)
        
        # 🚨 关键修复：强制 rect=False，确保验证时图像尺寸一致
        # 父类 DetectionTrainer 使用 rect=mode == 'val'，导致验证图像尺寸不一致
        return DualModalityDataset(
            img_path=img_path,
            imgsz=self.args.imgsz,
            batch_size=batch,
            augment=mode == 'train',
            hyp=hyp,
            rect=False,  # 🚨 强制 False，确保正方形图像
            cache=self.args.cache or None,
            single_cls=self.args.single_cls or False,
            stride=int(gs),
            prefix=f'{mode}: ',
            classes=self.args.classes,
            data=self.data,
            fraction=self.args.fraction if mode == 'train' else 1.0,
        )
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """
        构建并返回数据加载器
        
        Args:
            dataset_path: 数据集路径
            batch_size: 批次大小
            rank: 分布式训练的 rank
            mode: 'train' 或 'val'
            
        Returns:
            DataLoader 实例
        """
        from ultralytics.data.build import build_dataloader
        
        assert mode in ['train', 'val']
        
        # 构建数据集
        dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        # 设置 shuffle
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        # 工作线程数
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)
    
    def get_validator(self):
        """返回自定义的双模态验证器 DualModalValidator"""
        from ADV.validator import DualModalValidator
        import copy
        import torch
        
        self.loss_names = ['box_loss', 'cls_loss', 'dfl_loss', 'distill_loss']
        self.loss_items = torch.zeros(len(self.loss_names), device=self.device)
        
        val_args = copy.copy(self.args)
        val_args.plots = False
        
        # 卸妆：把官方不认识的参数擦除
        custom_keys = [
            'ADV_enabled', 'baseline_teacher', 'Baseline_Teacher', 'epsilon_max',
            'lambda_distill_start', 'early_layer_idx', 'attack_method',
            'warmup_epochs', 'stage2_ratio', 'student', 'teacher'
        ]
        for k in custom_keys:
            if hasattr(val_args, k):
                delattr(val_args, k)
        
        # 🚨 核心修复：使用自定义的 DualModalValidator，而不是官方的 DetectionValidator
        # DualModalValidator 支持 6 通道输入和 forward_inference 接口
        return DualModalValidator(
            dataloader=self.test_loader,
            save_dir=self.save_dir,
            args=val_args,
        )
    
    def label_loss_items(self, loss_items=None, prefix='train'):
        """返回带标签的损失字典或键名列表 (严格对齐 Ultralytics 官方 API)"""
        # 定义我们的四个核心损失指标
        keys = [f'{prefix}/box_loss', f'{prefix}/cls_loss', f'{prefix}/dfl_loss', f'{prefix}/distill_loss']
        
        if loss_items is not None:
            # 传了数值，就返回 Dict
            loss_items = [round(float(x), 5) for x in loss_items]  # 转换为 5 位小数
            # 防止 loss 数量和 keys 不匹配
            return dict(zip(keys, loss_items + [0.0] * (len(keys) - len(loss_items))))
        else:
            # 没传数值（建表头时），乖乖返回 List！
            return keys
