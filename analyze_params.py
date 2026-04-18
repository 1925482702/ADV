import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from models.teacher import TeacherModel
from models.student import DualModalStudent

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_rgb = TeacherModel.load_from_checkpoint('/root/autodl-tmp/ADV/checkpoint/monomodal/FLIR_rgb.pt', device=str(device))
teacher_rgb.eval()
teacher_rgb.freeze()

teacher_ir = TeacherModel.load_from_checkpoint('/root/autodl-tmp/ADV/checkpoint/monomodal/FLIR_ir.pt', device=str(device))
teacher_ir.eval()
teacher_ir.freeze()

student = DualModalStudent(
    teacher_rgb=teacher_rgb,
    teacher_ir=teacher_ir,
    num_classes=3,
).to(device)

print('完整参数分析')
print('='*80)
print(f"{'模块':<30} {'参数量':>15} {'占比':>10}")
print('-'*80)

rgb_backbone = count_params(student.backbone_rgb.model)
ir_backbone = count_params(student.backbone_ir.model)
fusion = count_params(student.fusion)
neck = count_params(student.neck_layers)
detect_head = count_params(student.detect_head)
total = rgb_backbone + ir_backbone + fusion + neck + detect_head

modules = [
    ('Backbone RGB', rgb_backbone),
    ('Backbone IR', ir_backbone),
    ('Fusion', fusion),
    ('Neck', neck),
    ('Detect Head', detect_head),
]

for name, params in modules:
    pct = (params / total) * 100
    print(f'{name:<30} {params:>15,} {pct:>9.2f}%')

print('-'*80)
print(f'{"总计 (可训练)":<30} {total:>15,} {100.00:>9.2f}%')
print('='*80)

# 验证
actual_total = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(f'\n验证: 直接统计 = {actual_total:,}')
print(f'差异: {actual_total - total:,}')

# 对比Teacher参数量
print('\n' + '='*80)
print('Teacher模型参数量（冻结，不参与训练）')
print('='*80)
teacher_rgb_params = sum(p.numel() for p in teacher_rgb.parameters())
teacher_ir_params = sum(p.numel() for p in teacher_ir.parameters())
print(f'Teacher RGB: {teacher_rgb_params:,}')
print(f'Teacher IR:  {teacher_ir_params:,}')
print(f'Teacher总参数: {teacher_rgb_params + teacher_ir_params:,}')
print('='*80)

print('\n模型对比:')
print(f'Student可训练参数: {total:,} ({total/1e6:.2f}M)')
print(f'Teacher冻结参数:   {teacher_rgb_params + teacher_ir_params:,} ({(teacher_rgb_params + teacher_ir_params)/1e6:.2f}M)')
print(f'总参数量:          {total + teacher_rgb_params + teacher_ir_params:,} ({(total + teacher_rgb_params + teacher_ir_params)/1e6:.2f}M)')