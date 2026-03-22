"""
阶段一：单模态预训练 + 完整训练入口

数据假设：
  已有双模态 DataLoader，batch 包含：
    batch['rgb']    [B, 3, H, W]
    batch['ir']     [B, 3, H, W]
    batch['gt_cls'] [B, num_classes, H/8, W/8]   (anchor-free 风格)
    batch['gt_reg'] [B, 4, H/8, W/8]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from model   import (ModalityBackbone, FeatureAlignFusion,
                     ResidualHead, DetectionHead,
                     SingleModalDetector, BiModalStudentDetector)
from trainer import AdversarialDistillTrainer, DetLoss


# ─────────────────────────────────────────────
# 阶段一：单模态预训练
# ─────────────────────────────────────────────

def pretrain_single_modal(dataloader, device, num_classes=80,
                           lr=1e-4, epochs=30):
    """
    分别训练 RGB-only 和 IR-only 的单模态检测模型。
    目的：让 backbone 学会本模态的基本特征表达。
    """
    det_loss_fn = DetLoss()
    results = {}

    for modality in ['rgb', 'ir']:
        print(f"\n=== 单模态预训练: {modality.upper()} ===")
        model = SingleModalDetector(num_classes).to(device)
        opt   = torch.optim.AdamW(model.parameters(), lr=lr)
        sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

        for epoch in range(epochs):
            model.train()
            for batch in dataloader:
                x     = batch[modality].to(device)
                gt_cls = batch['gt_cls'].to(device)
                gt_reg = batch['gt_reg'].to(device)

                cls_p, reg_p = model(x)
                loss = det_loss_fn(cls_p, reg_p, gt_cls, gt_reg)
                opt.zero_grad()
                loss.backward()
                opt.step()

            sch.step()
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

        results[modality] = model
        torch.save(model.state_dict(), f'pretrained_{modality}.pth')
        print(f"  {modality.upper()} 预训练完成，已保存")

    return results


# ─────────────────────────────────────────────
# Baseline 双模态模型（直接拼接融合，无对抗）
# 作为 Teacher 使用
# ─────────────────────────────────────────────

def build_baseline_teacher(dataloader, device, num_classes=80,
                            lr=1e-4, epochs=30,
                            rgb_ckpt=None, ir_ckpt=None):
    """
    训练 baseline 双模态模型（简单拼接融合，无对抗蒸馏）。
    这个 baseline 将作为 teacher 指导学生模型。

    初始化：可从单模态预训练 ckpt 加载 backbone 权重。
    """
    print("\n=== 训练 Baseline Teacher（双模态拼接融合）===")
    model = BiModalStudentDetector(num_classes).to(device)

    # 如果有预训练 backbone，加载其 backbone 权重
    if rgb_ckpt:
        rgb_state = torch.load(rgb_ckpt, map_location=device)
        # 只加载 backbone 部分
        _load_backbone(model.rgb_backbone,
                       {k.replace('backbone.', ''): v
                        for k, v in rgb_state.items() if 'backbone' in k})
    if ir_ckpt:
        ir_state = torch.load(ir_ckpt, map_location=device)
        _load_backbone(model.ir_backbone,
                       {k.replace('backbone.', ''): v
                        for k, v in ir_state.items() if 'backbone' in k})

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    det_loss_fn = DetLoss()

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            x_rgb  = batch['rgb'].to(device)
            x_ir   = batch['ir'].to(device)
            gt_cls = batch['gt_cls'].to(device)
            gt_reg = batch['gt_reg'].to(device)

            cls_p, reg_p, _, _, _ = model(x_rgb, x_ir)
            loss = det_loss_fn(cls_p, reg_p, gt_cls, gt_reg)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    torch.save(model.state_dict(), 'baseline_teacher.pth')
    print("  Baseline Teacher 训练完成，已保存")
    return model


def _load_backbone(backbone_module, state_dict):
    try:
        backbone_module.load_state_dict(state_dict, strict=False)
        print("  backbone 权重加载成功（strict=False）")
    except Exception as e:
        print(f"  backbone 权重加载失败: {e}，使用随机初始化")


# ─────────────────────────────────────────────
# 主训练入口
# ─────────────────────────────────────────────

def main(dataloader, device='cuda', num_classes=80):

    # ── Phase 1：单模态预训练 ──────────────────────
    single_models = pretrain_single_modal(
        dataloader, device, num_classes=num_classes,
        lr=1e-4, epochs=30
    )

    # ── Phase 1.5：训练 Baseline Teacher ──────────
    teacher = build_baseline_teacher(
        dataloader, device, num_classes=num_classes,
        lr=1e-4, epochs=30,
        rgb_ckpt='pretrained_rgb.pth',
        ir_ckpt='pretrained_ir.pth'
    )

    # ── Phase 2 & 3：学生模型对抗蒸馏 + 冲刺 ──────
    student = BiModalStudentDetector(num_classes).to(device)

    # 用 baseline teacher 权重初始化学生（给一个好的起点）
    student.load_state_dict(
        torch.load('baseline_teacher.pth', map_location=device),
        strict=True
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5,
                                   weight_decay=1e-4)

    trainer = AdversarialDistillTrainer(
        student_model    = student,
        teacher_model    = teacher,
        optimizer        = optimizer,
        device           = device,
        adv_eps          = 0.01,    # FGSM 扰动幅度
        adv_epochs       = 50,      # 对抗蒸馏 epoch 数
        finetune_epochs  = 20,      # 冲刺 epoch 数
        lambda_distill   = 1.0,     # 蒸馏 loss 权重
        lambda_residual  = 0.5,     # 残差 loss 权重
    )

    trainer.train(dataloader, num_classes=num_classes)
    torch.save(student.state_dict(), 'student_final.pth')
    print("\n训练完成！模型已保存至 student_final.pth")


# ─────────────────────────────────────────────
# 真实数据加载
# ─────────────────────────────────────────────

def create_real_dataloader(data_yaml, mode='train', batch_size=16, imgsz=640, workers=8):
    """
    创建真实数据加载器
    
    Args:
        data_yaml: 数据集 YAML 配置路径
        mode: 'train' 或 'val'
        batch_size: 批次大小
        imgsz: 图像尺寸
        workers: 数据加载线程数
    
    Returns:
        DataLoader
    """
    from dataloader import create_bimodal_dataloader, collate_for_test
    
    loader = create_bimodal_dataloader(
        data_yaml=data_yaml,
        mode=mode,
        batch_size=batch_size,
        imgsz=imgsz,
        workers=workers,
    )
    
    # 包装 collate_fn
    class AdaptedLoader:
        def __init__(self, loader):
            self.loader = loader
            self.dataset = loader.dataset
        
        def __iter__(self):
            for batch in self.loader:
                yield collate_for_test(batch)
        
        def __len__(self):
            return len(self.loader)
    
    return AdaptedLoader(loader)


# ─────────────────────────────────────────────
# 伪 DataLoader（用于 smoke test）
# ─────────────────────────────────────────────

class FakeDataLoader:
    """仅用于 smoke test，实际用你自己的 DataLoader 替换"""
    def __init__(self, B=2, H=512, W=512, num_classes=80, n_batches=4):
        self.B = B; self.H = H; self.W = W
        self.nc = num_classes; self.n = n_batches

    def __iter__(self):
        for _ in range(self.n):
            # 特征图尺寸 = H/8（P3 对齐尺寸）
            fh, fw = self.H // 8, self.W // 8
            yield {
                'rgb':    torch.randn(self.B, 3, self.H, self.W),
                'ir':     torch.randn(self.B, 3, self.H, self.W),
                'gt_cls': torch.sigmoid(torch.randn(self.B, self.nc, fh, fw)),
                'gt_reg': torch.randn(self.B, 4, fh, fw),
            }

    def __len__(self):
        return self.n


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ADV 训练')
    parser.add_argument('--data', type=str, 
                       default='/root/autodl-tmp/ADV/M2D-LIF/data/FLIR.yaml',
                       help='数据集 YAML 配置路径')
    parser.add_argument('--batch', type=int, default=16, help='批次大小')
    parser.add_argument('--imgsz', type=int, default=640, help='图像尺寸')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--fake', action='store_true', help='使用假数据测试')
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    if args.fake:
        print("=== 使用假数据进行 smoke test ===")
        loader = FakeDataLoader(B=2, H=256, W=256, num_classes=3, n_batches=4)
    else:
        print(f"=== 使用真实数据: {args.data} ===")
        loader = create_real_dataloader(
            data_yaml=args.data,
            mode='train',
            batch_size=args.batch,
            imgsz=args.imgsz,
            workers=0,
        )
        print(f"数据集大小: {len(loader.dataset)}")
    
    print(f"使用设备: {device}")
    main(loader, device=device, num_classes=3)