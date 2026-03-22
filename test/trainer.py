"""
对抗噪声生成 + 蒸馏训练逻辑

核心流程（回答你的问题）：
  Step1: 不加噪，前向传播学生模型，得到 loss → backward 得到梯度
  Step2: 用梯度（FGSM 风格）在 P3/P4/P5 特征上生成对抗噪声
         抛硬币决定污染 RGB 还是 IR（交替迫使两个模态都"发力"）
  Step3: 带噪再次前向传播，计算蒸馏 loss（对齐 teacher 融合特征）
         + 检测 loss（提高 mAP）
  残差头不受 distill loss 约束（只走检测 loss），不会补偿噪声

Teacher 选择（回答你的问题）：
  用双模态 baseline 作为 teacher，而非两个单模态 teacher 相加。
  理由：我们训练的是双模态模型，teacher 理应也是双模态的。
  单模态 teacher 的融合特征相加，本质上和 baseline 等价，但
  baseline 的融合 conv 层已经学过跨模态交互，语义更丰富。

对抗阶段截止（回答你的问题）：
  adv_epochs 控制对抗阶段的 epoch 数，之后进入冲刺阶段（纯检测 loss）。
  冲刺阶段停止 distill loss，让学生模型自由发挥学到的特征提取能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy


# ─────────────────────────────────────────────
# 1. FGSM 风格对抗噪声生成
# ─────────────────────────────────────────────

def generate_adversarial_noise(feats: dict, grad_feats: dict,
                                eps: float = 0.01) -> dict:
    """
    FGSM: delta = eps * sign(grad)
    feats      : {'p3': Tensor, 'p4': Tensor, 'p5': Tensor}
    grad_feats : 对应特征图关于 loss 的梯度（需要提前 retain_grad）
    返回噪声字典，shape 与 feats 相同
    """
    noise = {}
    for k in feats:
        if grad_feats[k] is not None:
            noise[k] = eps * grad_feats[k].sign().detach()
        else:
            noise[k] = torch.zeros_like(feats[k])
    return noise


def register_feat_hooks(backbone, storage: dict):
    """
    给 backbone 的 P3/P4/P5 输出注册 hook，
    使 tensor 保留 grad（用于后续提取梯度）。
    返回 handle 列表，用完后 remove。
    """
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            out.retain_grad()
            storage[name] = out
        return hook

    handles.append(backbone.proj_p3.register_forward_hook(make_hook('p3')))
    handles.append(backbone.proj_p4.register_forward_hook(make_hook('p4')))
    handles.append(backbone.proj_p5.register_forward_hook(make_hook('p5')))
    return handles


# ─────────────────────────────────────────────
# 2. 损失函数
# ─────────────────────────────────────────────

class DistillLoss(nn.Module):
    """
    特征蒸馏 loss：MSE(学生融合特征, teacher 融合特征)
    teacher 融合特征 = baseline 在相同输入下的 fused 输出
    """
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_fused: torch.Tensor,
                teacher_fused: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(student_fused / self.T, teacher_fused.detach() / self.T)


class ResidualErrorLoss(nn.Module):
    """
    残差头 loss：学习 teacher 检测误差（GT 与 teacher 预测的差）
    这里用 smooth L1 衡量残差头输出与 teacher 误差之间的差距。

    teacher_error_reg = gt_boxes - teacher_reg_pred  （回归误差）
    teacher_error_cls = gt_cls   - teacher_cls_pred  （分类误差，软标签形式）
    """
    def forward(self, student_cls, student_reg,
                teacher_cls_pred, teacher_reg_pred,
                gt_cls, gt_reg) -> torch.Tensor:
        # 教师误差 = GT - 教师预测
        teacher_reg_err = gt_reg.detach() - teacher_reg_pred.detach()
        teacher_cls_err = gt_cls.detach() - teacher_cls_pred.detach()

        # 学生残差头的目标：逼近教师的误差
        loss_reg = F.smooth_l1_loss(student_reg, teacher_reg_err)
        loss_cls = F.mse_loss(student_cls, teacher_cls_err)
        return loss_reg + loss_cls


class DetLoss(nn.Module):
    """
    标准检测 loss（简化版，实际可换 YOLOv8 loss）
    """
    def forward(self, cls_pred, reg_pred, gt_cls, gt_reg) -> torch.Tensor:
        loss_cls = F.binary_cross_entropy_with_logits(cls_pred, gt_cls)
        loss_reg = F.smooth_l1_loss(reg_pred, gt_reg)
        return loss_cls + loss_reg


# ─────────────────────────────────────────────
# 3. 训练器
# ─────────────────────────────────────────────

class AdversarialDistillTrainer:
    """
    三阶段训练器：
      Phase 1: 单模态预训练（此文件不负责，由 pretrain.py 完成）
      Phase 2: 对抗蒸馏（adv_epochs 个 epoch）
                - 抛硬币污染一个模态
                - distill_loss 对齐 teacher 融合特征
                - residual_loss 学习 teacher 检测误差
      Phase 3: 冲刺（finetune_epochs 个 epoch）
                - 停止对抗噪声和蒸馏 loss
                - 只用检测 loss 刷 mAP
    """

    def __init__(self,
                 student_model,
                 teacher_model,      # 双模态 baseline，frozen
                 optimizer,
                 device,
                 adv_eps: float = 0.01,
                 adv_epochs: int = 50,
                 finetune_epochs: int = 20,
                 lambda_distill: float = 1.0,
                 lambda_residual: float = 0.5):

        self.student   = student_model
        self.teacher   = teacher_model
        self.optimizer = optimizer
        self.device    = device
        self.adv_eps   = adv_eps
        self.adv_epochs      = adv_epochs
        self.finetune_epochs = finetune_epochs
        self.total_epochs    = adv_epochs + finetune_epochs

        self.lambda_d = lambda_distill
        self.lambda_r = lambda_residual

        self.distill_loss   = DistillLoss()
        self.residual_loss  = ResidualErrorLoss()
        self.det_loss       = DetLoss()

        # Teacher 固定，不更新梯度
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    # ── 阶段二：单步对抗蒸馏训练 ──────────────────
    def _adv_train_step(self, x_rgb, x_ir, gt_cls, gt_reg):
        """
        Step1: 干净前向，注册梯度 hook，backward 得梯度
        Step2: FGSM 生成噪声，抛硬币选污染模态
        Step3: 带噪前向，计算联合 loss，更新参数
        """
        self.student.train()

        # ── Step1: 干净前向，获取梯度 ──
        rgb_feat_store, ir_feat_store = {}, {}

        h_rgb = register_feat_hooks(self.student.rgb_backbone, rgb_feat_store)
        h_ir  = register_feat_hooks(self.student.ir_backbone,  ir_feat_store)

        cls_pred, reg_pred, fused, _, _ = self.student(x_rgb, x_ir)
        loss_clean = self.det_loss(cls_pred, reg_pred, gt_cls, gt_reg)
        self.optimizer.zero_grad()
        loss_clean.backward(retain_graph=False)
        # 此时 rgb_feat_store / ir_feat_store 中的 tensor 已有 .grad

        # remove hooks
        for h in h_rgb + h_ir:
            h.remove()

        # ── Step2: 生成对抗噪声，抛硬币选模态 ──
        corrupt_rgb = (random.random() < 0.5)

        if corrupt_rgb:
            grad_dict = {k: rgb_feat_store[k].grad for k in rgb_feat_store}
            noise_rgb = generate_adversarial_noise(rgb_feat_store, grad_dict, self.adv_eps)
            noise_ir  = None
        else:
            grad_dict = {k: ir_feat_store[k].grad for k in ir_feat_store}
            noise_ir  = generate_adversarial_noise(ir_feat_store, grad_dict, self.adv_eps)
            noise_rgb = None

        # ── Step3: 带噪前向 ──
        cls_pred_noisy, reg_pred_noisy, fused_noisy, _, _ = self.student(
            x_rgb, x_ir,
            adv_noise_rgb=noise_rgb,
            adv_noise_ir=noise_ir
        )

        # Teacher 融合特征（双模态 baseline，干净输入）
        with torch.no_grad():
            _, _, fused_teacher, _, _ = self.teacher(x_rgb, x_ir)
            cls_teacher, reg_teacher = self.teacher.det_head(fused_teacher)

        # distill loss：带噪学生融合特征 → 对齐 teacher 融合特征
        # （注意：fused_noisy 已经过 residual_head，teacher 也一样，一致对齐）
        loss_distill  = self.distill_loss(fused_noisy, fused_teacher)

        # residual loss：残差头学习 teacher 的检测误差
        loss_residual = self.residual_loss(
            cls_pred_noisy, reg_pred_noisy,
            cls_teacher, reg_teacher,
            gt_cls, gt_reg
        )

        # 检测 loss（保证基本的检测能力不退化）
        loss_det = self.det_loss(cls_pred_noisy, reg_pred_noisy, gt_cls, gt_reg)

        loss_total = (loss_det
                      + self.lambda_d * loss_distill
                      + self.lambda_r * loss_residual)

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return {
            'total':    loss_total.item(),
            'det':      loss_det.item(),
            'distill':  loss_distill.item(),
            'residual': loss_residual.item(),
            'corrupted': 'RGB' if corrupt_rgb else 'IR'
        }

    # ── 阶段三：冲刺步（只有检测 loss）──────────────
    def _finetune_step(self, x_rgb, x_ir, gt_cls, gt_reg):
        self.student.train()
        cls_pred, reg_pred, _, _, _ = self.student(x_rgb, x_ir)
        loss = self.det_loss(cls_pred, reg_pred, gt_cls, gt_reg)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'total': loss.item()}

    # ── 主训练循环 ────────────────────────────────
    def train(self, dataloader, num_classes=80):
        for epoch in range(self.total_epochs):
            is_adv_phase = epoch < self.adv_epochs

            phase_name = "对抗蒸馏" if is_adv_phase else "冲刺微调"
            print(f"\n[Epoch {epoch+1}/{self.total_epochs}] Phase: {phase_name}")

            epoch_losses = []
            for batch in dataloader:
                x_rgb  = batch['rgb'].to(self.device)
                x_ir   = batch['ir'].to(self.device)
                gt_cls = batch['gt_cls'].to(self.device)
                gt_reg = batch['gt_reg'].to(self.device)

                if is_adv_phase:
                    info = self._adv_train_step(x_rgb, x_ir, gt_cls, gt_reg)
                else:
                    info = self._finetune_step(x_rgb, x_ir, gt_cls, gt_reg)

                epoch_losses.append(info['total'])

            avg = sum(epoch_losses) / len(epoch_losses)
            print(f"  avg loss: {avg:.4f}")