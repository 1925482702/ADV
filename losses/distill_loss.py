"""
蒸馏损失和总损失
"""

import torch
import torch.nn as nn
from .detection_loss import DetectionLoss

# 🚨 全局调试开关
DEBUG_MODE = False  # 🚨 关闭调试模式
DEBUG_LOSS_COUNT = 0  # 用于限制输出次数


def _debug_tensor(name: str, tensor: torch.Tensor, enabled: bool = True):
    """统一的张量调试输出"""
    if not enabled or not DEBUG_MODE:
        return
    global DEBUG_LOSS_COUNT
    if DEBUG_LOSS_COUNT > 5:  # 只输出前5次
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    print(f"    {name}: shape={list(tensor.shape)}, "
          f"range=[{tensor.min().item():.4f}, {tensor.max().item():.4f}], "
          f"mean={tensor.mean().item():.4f}, "
          f"NaN={has_nan}, Inf={has_inf}")


class DistillationLoss(nn.Module):
    """
    特征蒸馏损失
    
    让Student特征在P3/P4/P5三个尺度上分别与Teacher特征对齐
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        temperature: float = 1.0,
        loss_scale: float = 1.0,
    ):
        """
        参数:
            loss_type: 损失类型 ('mse', 'kl', 'cosine')
            temperature: 温度参数（用于 KL 散度）
            loss_scale: 损失缩放因子，用于调整蒸馏损失的量级
                        默认为1.0，如果检测损失远大于蒸馏损失，可以增大此值
        """
        super().__init__()
        self.loss_type = loss_type
        self.temperature = temperature
        self.loss_scale = loss_scale

        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'kl':
            self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        elif loss_type == 'cosine':
            self.loss_fn = None  # Cosine距离直接计算
        else:
            raise ValueError(f"未知的损失类型: {loss_type}")
    
    def forward(
        self,
        fused_features: list,
        teacher_rgb_features: list,
        teacher_ir_features: list,
    ) -> dict:
        """
        计算蒸馏损失

        🚨 新理念：
        让Student融合后的特征（不包括Residual）逼近Teacher特征之和

        参数:
            fused_features: Student融合后的特征 [P3, P4, P5]（不包括Residual）
            teacher_rgb_features: Teacher RGB特征 [P3, P4, P5]
            teacher_ir_features: Teacher IR特征 [P3, P4, P5]

        返回:
            loss_dict: {
                'total': 总蒸馏损失,
                'P3': P3层损失,
                'P4': P4层损失,
                'P5': P5层损失,
            }
        """
        global DEBUG_LOSS_COUNT
        
        # ======== DEBUG: 蒸馏损失计算 ========
        if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
            print(f"\n{'='*20} 🔍 蒸馏损失计算 (#{DEBUG_LOSS_COUNT}) {'='*20}")
        
        total_loss = 0.0
        layer_losses = {}

        # 🚨 逐尺度计算损失（融合特征 vs Teacher特征之和）
        for i, (fused, t_rgb, t_ir) in enumerate(
            zip(fused_features, teacher_rgb_features, teacher_ir_features)
        ):
            layer_name = f'P{i+3}'  # P3, P4, P5

            # ======== DEBUG: 特征检查 ========
            if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
                print(f"  📌 {layer_name}层:")
                _debug_tensor("Student特征", fused)
                _debug_tensor("Teacher RGB", t_rgb)
                _debug_tensor("Teacher IR", t_ir)
            
            # 🚨 关键修改：计算Teacher特征之和
            teacher_sum = t_rgb + t_ir
            
            if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
                _debug_tensor("Teacher和", teacher_sum)
            
            # 检查是否有 NaN 或 Inf
            has_nan_student = torch.isnan(fused).any().item()
            has_inf_student = torch.isinf(fused).any().item()
            has_nan_teacher = torch.isnan(teacher_sum).any().item()
            has_inf_teacher = torch.isinf(teacher_sum).any().item()
            
            if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
                print(f"    NaN检查 - Student: {has_nan_student}, Teacher: {has_nan_teacher}")
                print(f"    Inf检查 - Student: {has_inf_student}, Teacher: {has_inf_teacher}")

            # 🚨 关键修改：Student融合特征 vs Teacher特征之和
            layer_loss = self._compute_loss(fused, teacher_sum)
            
            if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
                print(f"    {layer_name}损失: {layer_loss.item():.6f}")
            
            layer_losses[layer_name] = layer_loss
            total_loss += layer_loss

        loss_dict = {
            'total': total_loss,
            **layer_losses,
        }
        
        if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
            print(f"  总蒸馏损失: {total_loss.item():.6f}")
            print(f"{'='*60}")
            DEBUG_LOSS_COUNT += 1

        return loss_dict
    
    def _compute_loss(self, student_feat, teacher_feat):
        """
        计算单个尺度的损失
        
        使用空间归一化，确保不同尺度的损失量级一致
        """
        assert student_feat.shape == teacher_feat.shape, \
            f"蒸馏损失尺寸不匹配！Student: {student_feat.shape}, Teacher: {teacher_feat.shape}"

        if self.loss_type == 'mse':
            # 🔍 调试：计算逐元素的差异
            diff = student_feat - teacher_feat
            # print(f"🔍   [MSE调试] 差异范围: [{diff.min().item():.6f}, {diff.max().item():.6f}]")
            # print(f"🔍   [MSE调试] 差异均值: {diff.mean().item():.6f}")
            # print(f"🔍   [MSE调试] 差异绝对值均值: {diff.abs().mean().item():.6f}")
            
            # 使用空间归一化：先对空间维度(H,W)求平均，再对所有元素求平均
            # 这样可以避免不同尺度特征图面积差异带来的量级偏差
            loss = torch.nn.functional.mse_loss(student_feat, teacher_feat, reduction='none')
            # print(f"🔍   [MSE调试] reduction='none'后形状: {loss.shape}")
            # print(f"🔍   [MSE调试] 逐点MSE范围: [{loss.min().item():.6f}, {loss.max().item():.6f}]")
            # print(f"🔍   [MSE调试] 逐点MSE均值: {loss.mean().item():.6f}")
            
            layer_loss = loss.mean(dim=[2, 3]).mean()  # 先按HW求均值，再对B和C求均值
            # print(f"🔍   [MSE调试] 空间平均后损失: {layer_loss.item():.6f}")
            # print(f"🔍   [MSE调试] 缩放因子: {self.loss_scale}")
            
            # 应用缩放因子，确保蒸馏损失的量级与检测损失匹配
            layer_loss = layer_loss * self.loss_scale
            # print(f"🔍   [MSE调试] 缩放后损失: {layer_loss.item():.6f}")
            
        elif self.loss_type == 'kl':
            # KL散度需要softmax
            s_prob = torch.softmax(student_feat / self.temperature, dim=1)
            t_prob = torch.softmax(teacher_feat / self.temperature, dim=1)
            layer_loss = self.loss_fn(s_prob.log(), t_prob) * self.loss_scale
            
        elif self.loss_type == 'cosine':
            # Cosine距离（空间归一化后）
            s_flat = student_feat.flatten(2).mean(dim=2)  # [B, C]
            t_flat = teacher_feat.flatten(2).mean(dim=2)  # [B, C]
            layer_loss = (1 - torch.nn.functional.cosine_similarity(s_flat, t_flat, dim=1)).mean() * self.loss_scale
        
        return layer_loss


class TotalLoss(nn.Module):
    """
    总损失管理器
    
    管理检测损失和蒸馏损失
    """
    
    def __init__(
        self,
        model,
        lambda_det: float = 1.0,
        lambda_distill: float = 0.5,
        distill_loss_type: str = 'mse',
        distill_loss_scale: float = 10.0,
    ):
        """
        参数:
            model: 学生模型
            lambda_det: 检测损失权重
            lambda_distill: 蒸馏损失权重
            distill_loss_type: 蒸馏损失类型
            distill_loss_scale: 蒸馏损失缩放因子，用于调整蒸馏损失的量级
                                 默认为10.0，因为MSE Loss通常比检测损失小一个数量级
        """
        super().__init__()

        self.lambda_det = lambda_det
        self.lambda_distill = lambda_distill

        # 初始化各个损失
        self.det_loss = DetectionLoss(model)
        self.distill_loss = DistillationLoss(
            loss_type=distill_loss_type,
            loss_scale=distill_loss_scale,
        )
    
    def forward(
        self,
        predictions,
        targets,
        features_dict,
        teacher_rgb_features=None,
        teacher_ir_features=None,
    ) -> dict:
        """
        计算总损失

        参数:
            predictions: 模型预测
            targets: 目标标注
            features_dict: 特征字典，应包含fused_features和teacher特征
            teacher_rgb_features: Teacher RGB特征（可选，如果不在features_dict中）
            teacher_ir_features: Teacher IR特征（可选，如果不在features_dict中）

        返回:
            loss_dict: {
                'total': 总损失,
                'det': 检测损失,
                'distill': 蒸馏损失,
                ...
            }
        """
        global DEBUG_LOSS_COUNT
        
        loss_dict = {}

        # ======== DEBUG: 总损失计算 ========
        if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
            print(f"\n{'='*20} 🔍 总损失计算 {'='*20}")
            print(f"  lambda_det: {self.lambda_det}")
            print(f"  lambda_distill: {self.lambda_distill}")

        # 1. 检测损失
        det_loss_dict = self.det_loss(predictions, targets)
        loss_dict['det'] = det_loss_dict['total']
        
        if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
            print(f"  检测损失: {loss_dict['det'].item():.6f}")
        loss_dict.update({f'det_{k}': v for k, v in det_loss_dict.items() if k != 'total'})

        # 2. 蒸馏损失（当 lambda_distill > 0 时才计算）
        if self.lambda_distill > 0:
            # 🚨 从features_dict中提取Student基础和特征（无残差）
            if 'student_base_sum' not in features_dict:
                raise ValueError("features_dict中未找到'student_base_sum'！")

            student_base_sum = features_dict['student_base_sum']

            # 提取Teacher特征
            if 'teacher_rgb_features' in features_dict and 'teacher_ir_features' in features_dict:
                teacher_rgb_features = features_dict['teacher_rgb_features']
                teacher_ir_features = features_dict['teacher_ir_features']
            elif teacher_rgb_features is not None and teacher_ir_features is not None:
                # 使用传入的参数
                pass
            else:
                raise ValueError("未找到Teacher特征！")

            # 🚨 调用蒸馏损失（Student基础和特征 vs Teacher特征之和）
            distill_loss_dict = self.distill_loss(
                student_base_sum,  # Student的基础和特征（无残差）
                teacher_rgb_features,
                teacher_ir_features,
            )
            loss_dict['distill'] = distill_loss_dict['total']
            loss_dict.update({f'distill_{k}': v for k, v in distill_loss_dict.items() if k != 'total'})
        else:
            # 🚨 蒸馏损失已关闭
            loss_dict['distill'] = torch.tensor(0.0, device=loss_dict['det'].device)

        # 3. 总损失
        loss_dict['total'] = (
            self.lambda_det * loss_dict['det'] +
            self.lambda_distill * loss_dict['distill']
        )
        
        if DEBUG_MODE and DEBUG_LOSS_COUNT < 5:
            if self.lambda_distill > 0:
                print(f"  蒸馏损失: {loss_dict['distill'].item():.6f}")
                print(f"  加权检测损失: {(self.lambda_det * loss_dict['det']).item():.6f}")
                print(f"  加权蒸馏损失: {(self.lambda_distill * loss_dict['distill']).item():.6f}")
            else:
                print(f"  蒸馏损失: 已关闭 (lambda_distill=0)")
            print(f"  总损失: {loss_dict['total'].item():.6f}")
            print(f"{'='*60}")

        return loss_dict
    
    def update_weights(
        self,
        lambda_det: float = None,
        lambda_distill: float = None,
    ):
        """动态更新损失权重"""
        if lambda_det is not None:
            self.lambda_det = lambda_det
        if lambda_distill is not None:
            self.lambda_distill = lambda_distill
    
    def get_loss_weights(self):
        """获取当前损失权重"""
        return {
            'lambda_det': self.lambda_det,
            'lambda_distill': self.lambda_distill,
        }
