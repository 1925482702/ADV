"""
双模态验证器

继承 Ultralytics 的 DetectionValidator，支持 6 通道双模态输入验证
"""

import copy
import os
from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import LOGGER, ops, TQDM
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, smart_inference_mode


class DualModalValidator(DetectionValidator):
    """
    双模态检测验证器
    
    支持 6 通道输入（RGB + IR）的 DualModalStudent 模型验证
    
    与标准 DetectionValidator 的主要区别：
    1. preprocess 不修改通道数（保持 6 通道）
    2. 支持 DualModalStudent 模型的前向传播
    3. 兼容 Ultralytics 的 mAP 计算流程
    """
    
    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """初始化双模态验证器"""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.student_model = None  # 外部注入 DualModalStudent
        LOGGER.info("DualModalValidator initialized for 6-channel dual-modality validation")
    
    def preprocess(self, batch):
        """
        预处理 batch 数据
        
        关键改动：不修改通道数，保持 6 通道输入
        DualModalStudent 需要 [B, 6, H, W] 格式的输入
        """
        # 将图像移动到设备并归一化
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
        
        # 标签移动到设备
        for k in ['batch_idx', 'cls', 'bboxes']:
            batch[k] = batch[k].to(self.device)
        
        # 保存 hybrid 标签（如果需要）
        if self.args.save_hybrid:
            height, width = batch['img'].shape[2:]
            nb = len(batch['img'])
            bboxes = batch['bboxes'] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch['cls'][batch['batch_idx'] == i], bboxes[batch['batch_idx'] == i]], dim=-1)
                for i in range(nb)
            ]
        
        return batch
    
    def init_metrics(self, model):
        """
        初始化评估指标
        
        Args:
            model: 可以是 DualModalStudent 或标准 DetectionModel
        """
        # 获取类别名称
        if hasattr(model, 'names'):
            self.names = model.names
        elif hasattr(model, 'num_classes'):
            # DualModalStudent 没有 names 属性
            self.names = {i: f'class_{i}' for i in range(model.num_classes)}
        else:
            self.names = {0: 'car', 1: 'person', 2: 'bicycle'}  # FLIR 默认类别
        
        self.nc = len(self.names)
        
        # 初始化指标
        val = self.data.get(self.args.split, '')
        self.is_coco = isinstance(val, str) and 'coco' in val and val.endswith(f'{os.sep}val2017.txt')
        
        from ultralytics.data import converter
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        self.args.save_json |= self.is_coco and not self.training
        
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        from ultralytics.utils.metrics import ConfusionMatrix
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[])
        
        LOGGER.info(f"DualModalValidator metrics initialized: nc={self.nc}, names={self.names}")
    
    def postprocess(self, preds):
        """
        后处理：应用 NMS
        
        Args:
            preds: DualModalStudent 的输出（tuple 或 tensor）
        
        Returns:
            NMS 后的检测结果列表
        """
        # DualModalStudent 的 Detect head 返回 (predictions, ...)
        # predictions 形状为 [B, 4+nc, num_anchors]
        if isinstance(preds, (list, tuple)):
            preds = preds[0] if len(preds) > 0 else preds
        
        # 应用 NMS
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )
    
    def _prepare_batch(self, si, batch):
        """
        准备单个样本的批次数据
        
        🚨 使用 ops.scale_boxes 替代手写缩放逻辑
        """
        idx = batch['batch_idx'] == si
        cls = batch['cls'][idx].squeeze(-1)
        bbox = batch['bboxes'][idx]
        ori_shape = batch['ori_shape'][si]
        imgsz = batch['img'].shape[2:]
        # 🚨 修复：不使用 ratio_pad 参数，让 ops.scale_boxes 自动从 imgsz 和 ori_shape 计算
        # ratio_pad = batch['ratio_pad'][si]  # 可能是 float 或错误的 tuple 格式
        
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]
            # 🚨 使用 ops.scale_boxes 自动计算 gain 和 pad
            ops.scale_boxes(imgsz, bbox, ori_shape)
        
        return dict(cls=cls, bbox=bbox, ori_shape=ori_shape, imgsz=imgsz)
    
    def _prepare_pred(self, pred, pbatch):
        predn = pred.clone()
        # 🚨 修复：不使用 ratio_pad 参数，让 ops.scale_boxes 自动从 imgsz 和 ori_shape 计算
        ops.scale_boxes(
            pbatch['imgsz'],
            predn[:, :4],
            pbatch['ori_shape'],
        )
        return predn
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        执行验证（兼容官方调用方式）
        
        Args:
            trainer: 训练器实例（官方调用时传入）
            model: 模型实例（独立验证时传入）
        
        Returns:
            验证结果字典
        """
        # 支持官方调用方式：validator(trainer)
        if trainer is not None:
            self.training = True
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != 'cpu'  # 训练时强制 FP16
            # 🚨 核心修复：优先用 trainer.student_model（DualModalStudent）
            # EMA 和 trainer.model 都是官方 DetectionModel，不支持 6 通道输入
            if hasattr(trainer, 'student_model') and trainer.student_model is not None:
                self.student_model = trainer.student_model
            elif hasattr(trainer, 'ema') and trainer.ema:
                self.student_model = trainer.ema.ema
            else:
                self.student_model = trainer.model
            # 🚨 修复：不整体转换 student_model，避免污染内部 teacher 的精度
            # 验证时强制用 float，避免 teacher/student 类型不一致
            self.student_model.float()
            self.student_model.eval()
        else:
            # 独立验证方式：validator(model=xxx)
            self.training = False
            self.student_model = model or self.student_model
            if self.student_model is not None:
                self.device = next(self.student_model.parameters()).device
            self.student_model.eval()
        
        # 初始化指标
        self.init_metrics(self.student_model)
        
        # 开始验证循环
        dt = Profile(), Profile(), Profile(), Profile()
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.jdict = []
        
        for batch_i, batch in enumerate(bar):
            self.run_callbacks('on_val_batch_start')
            
            with dt[0]:
                batch = self.preprocess(batch)
            
            with dt[1]:
                # 确保输入类型与模型匹配
                model_dtype = next(self.student_model.parameters()).dtype
                input_tensor = batch['img'].to(dtype=model_dtype)
                
                # 检测模型类型，使用正确的推理方法
                if hasattr(self.student_model, 'forward_inference'):
                    # DualModalStudent 使用独立推理（不需要 Teacher）
                    preds = self.student_model.forward_inference(input_tensor)
                else:
                    # Teacher 或其他模型使用标准 forward
                    preds = self.student_model(input_tensor)
            
            # 🚨 DEBUG: 检查 NMS 前的原始输出
            if batch_i == 0:
                if isinstance(preds, (list, tuple)):
                    p = preds[0] if len(preds) > 0 else None
                    if p is not None:
                        print(f"\n[DEBUG] NMS前原始输出 shape: {p.shape}")
                        if len(p.shape) == 3:
                            # p: [batch, no, num_anchors] 其中 no = 4 + nc
                            # 前4个通道是bbox坐标，后面的通道是类别置信度
                            scores = p[0, 4:, :].max(dim=0)[0]  # [num_anchors] 每个 anchor 的最大置信度
                            print(f"[DEBUG] 第一张图置信度: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
                            # 统计得分 > 阈值的数量
                            for thresh in [0.001, 0.01, 0.1, 0.25, 0.5]:
                                count = (scores > thresh).sum().item()
                                print(f"[DEBUG] 置信度 > {thresh}: {count} 个")
                        else:
                            print(f"[DEBUG] 预测格式异常: {p.shape}")
            
            with dt[2]:
                preds = self.postprocess(preds)
            
            # 🚨 DEBUG: 打印 NMS 后预测框数量和 GT 框坐标
            if batch_i == 0:
                print(f"\n[DEBUG] NMS后预测框数量: {[len(p) for p in preds]}")
                
                # 检查 GT 框坐标
                pbatch = self._prepare_batch(0, batch)
                print(f"[DEBUG] GT bbox (原图空间): shape={pbatch['bbox'].shape}")
                if len(pbatch['bbox']) > 0:
                    print(f"[DEBUG] GT bbox 前3个: {pbatch['bbox'][:3]}")
                print(f"[DEBUG] ori_shape: {pbatch['ori_shape']}")
                print(f"[DEBUG] imgsz: {pbatch['imgsz']}")
                
                # 检查预测框缩放后坐标
                if len(preds) > 0 and len(preds[0]) > 0:
                    print(f"[DEBUG] 第一张图预测框前3个 (NMS输出): {preds[0][:3]}")
                    predn = self._prepare_pred(preds[0], pbatch)
                    print(f"[DEBUG] 缩放后预测框前3个: {predn[:3]}")
                else:
                    print(f"[DEBUG] 预测框为空！")
            
            with dt[3]:
                self.update_metrics(preds, batch)
            
            self.run_callbacks('on_val_batch_end')
        
        # 计算最终指标
        results = self.get_stats()
        self.fitness = self.metrics.fitness  # fitness 是属性，不是方法
        
        # 打印结果
        self.print_results()
        
        return {
            'metrics': results,
            'fitness': self.fitness,
        }


def create_dual_modal_validator(
    dataloader=None,
    save_dir=None,
    args=None,
    device='cuda',
):
    """
    创建双模态验证器的便捷函数
    
    Args:
        dataloader: 验证数据加载器
        save_dir: 结果保存目录
        args: 验证参数
        device: 设备
    
    Returns:
        DualModalValidator 实例
    """
    validator = DualModalValidator(
        dataloader=dataloader,
        save_dir=save_dir,
        args=args,
    )
    validator.device = device
    
    return validator
