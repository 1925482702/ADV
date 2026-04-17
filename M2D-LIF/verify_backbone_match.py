#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 Student 模型的两个 backbone 是否和 Teacher 模型匹配

用法：
    python verify_backbone_match.py \
        --student runs/shift_frozen/train/weights/best.pt \
        --rgb_teacher teacherTraining/runs/LLVIP_RGB_SHIFT_ADV/weights/best.pt \
        --ir_teacher teacherTraining/runs/LLVIP_IR_SHIFT_ADV/weights/best.pt
"""

import argparse
import torch
import torch.nn as nn


def load_model_weights(path):
    """加载模型权重"""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model' in ckpt:
        model = ckpt['model']
        if hasattr(model, 'state_dict'):
            return model.state_dict()
        return model
    return ckpt


def get_layer_mapping():
    """获取 Teacher → Student 的层映射"""
    # Teacher → Student RGB 分支
    teacher_to_rgb = {0: 3, 1: 5, 2: 7, 3: 9, 4: 11, 5: 14, 6: 16, 7: 19, 8: 21}
    # Teacher → Student IR 分支
    teacher_to_ir = {0: 4, 1: 6, 2: 8, 3: 10, 4: 12, 5: 15, 6: 17, 7: 20, 8: 22}
    return teacher_to_rgb, teacher_to_ir


def is_learnable_param(key):
    """判断是否是可学习参数（weight 或 bias）"""
    # 排除 BatchNorm 的统计量
    stat_names = ['running_mean', 'running_var', 'num_batches_tracked']
    for stat in stat_names:
        if stat in key:
            return False
    return True


def compare_weights(student_state, teacher_state, mapping, branch_name):
    """比较两个 backbone 的权重"""
    results = {
        'matched': [],
        'mismatched': [],
        'missing': [],
        'shape_mismatch': [],
        'stat_matched': [],
        'stat_mismatched': [],
    }
    
    for t_idx, s_idx in mapping.items():
        # 获取该层所有的参数
        teacher_layer_keys = [k for k in teacher_state.keys() 
                            if k.startswith(f'model.{t_idx}.') or 
                               (k.split('.')[0] == 'model' and len(k.split('.')) >= 2 and 
                                k.split('.')[1] == str(t_idx))]
        
        student_layer_keys = [k for k in student_state.keys()
                             if k.startswith(f'model.{s_idx}.') or
                                (k.split('.')[0] == 'model' and len(k.split('.')) >= 2 and
                                 k.split('.')[1] == str(s_idx))]
        
        for t_key in teacher_layer_keys:
            # 构建对应的 student key
            parts = t_key.split('.')
            if parts[0] == 'model':
                s_key = f'model.{s_idx}.' + '.'.join(parts[2:])
            else:
                s_key = f'model.{s_idx}.' + '.'.join(parts[1:])
            
            if s_key not in student_state:
                results['missing'].append((t_key, s_key))
                continue
            
            t_weight = teacher_state[t_key]
            s_weight = student_state[s_key]
            
            if t_weight.shape != s_weight.shape:
                results['shape_mismatch'].append((t_key, s_key, t_weight.shape, s_weight.shape))
                continue
            
            # 判断是否是可学习参数
            is_learnable = is_learnable_param(t_key)
            
            if torch.allclose(t_weight, s_weight, atol=1e-6):
                if is_learnable:
                    results['matched'].append((t_key, s_key))
                else:
                    results['stat_matched'].append((t_key, s_key))
            else:
                # 计算差异
                diff = (t_weight - s_weight).abs().max().item()
                if is_learnable:
                    results['mismatched'].append((t_key, s_key, diff))
                else:
                    results['stat_mismatched'].append((t_key, s_key, diff))
    
    return results


def print_results(results, branch_name):
    """打印比较结果"""
    print(f"\n{'='*60}")
    print(f"{branch_name} 分支验证结果")
    print(f"{'='*60}")
    
    # 可学习参数
    print(f"\n【可学习参数】(weight, bias)")
    print(f"  ✓ 匹配: {len(results['matched'])} 个")
    print(f"  ✗ 不匹配: {len(results['mismatched'])} 个")
    if results['mismatched']:
        for t_key, s_key, diff in results['mismatched'][:10]:
            print(f"      {t_key} → {s_key} (差异: {diff:.6f})")
    
    # 统计量
    print(f"\n【BatchNorm 统计量】(running_mean, running_var, num_batches_tracked)")
    print(f"  ✓ 匹配: {len(results['stat_matched'])} 个")
    print(f"  ✗ 不匹配: {len(results['stat_mismatched'])} 个 (训练后统计量会变化，这是正常的)")
    
    # 其他
    print(f"\n【其他】")
    print(f"  ✗ 形状不匹配: {len(results['shape_mismatch'])} 个")
    print(f"  ✗ 缺失: {len(results['missing'])} 个")


def main():
    parser = argparse.ArgumentParser(description='验证 backbone 权重匹配')
    parser.add_argument('--student', type=str, required=True, help='Student 模型路径')
    parser.add_argument('--rgb_teacher', type=str, required=True, help='RGB Teacher 模型路径')
    parser.add_argument('--ir_teacher', type=str, required=True, help='IR Teacher 模型路径')
    args = parser.parse_args()
    
    print("="*60)
    print("加载模型权重...")
    print("="*60)
    
    # 加载权重
    student_state = load_model_weights(args.student)
    rgb_teacher_state = load_model_weights(args.rgb_teacher)
    ir_teacher_state = load_model_weights(args.ir_teacher)
    
    print(f"Student 参数数量: {len(student_state)}")
    print(f"RGB Teacher 参数数量: {len(rgb_teacher_state)}")
    print(f"IR Teacher 参数数量: {len(ir_teacher_state)}")
    
    # 获取层映射
    teacher_to_rgb, teacher_to_ir = get_layer_mapping()
    
    print(f"\n层映射:")
    print(f"  Teacher → Student RGB: {teacher_to_rgb}")
    print(f"  Teacher → Student IR: {teacher_to_ir}")
    
    # 比较 RGB 分支
    rgb_results = compare_weights(student_state, rgb_teacher_state, teacher_to_rgb, "RGB")
    print_results(rgb_results, "RGB")
    
    # 比较 IR 分支
    ir_results = compare_weights(student_state, ir_teacher_state, teacher_to_ir, "IR")
    print_results(ir_results, "IR")
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    
    # 只统计可学习参数
    total_matched = len(rgb_results['matched']) + len(ir_results['matched'])
    total_mismatched = len(rgb_results['mismatched']) + len(ir_results['mismatched'])
    total_missing = len(rgb_results['missing']) + len(ir_results['missing'])
    total_shape_mismatch = len(rgb_results['shape_mismatch']) + len(ir_results['shape_mismatch'])
    
    print(f"【可学习参数】")
    print(f"  总匹配: {total_matched}")
    print(f"  总不匹配: {total_mismatched}")
    print(f"  总缺失: {total_missing}")
    print(f"  总形状不匹配: {total_shape_mismatch}")
    
    print(f"\n【BatchNorm 统计量】")
    stat_matched = len(rgb_results['stat_matched']) + len(ir_results['stat_matched'])
    stat_mismatched = len(rgb_results['stat_mismatched']) + len(ir_results['stat_mismatched'])
    print(f"  总匹配: {stat_matched}")
    print(f"  总不匹配: {stat_mismatched} (训练后统计量会变化，这是正常的)")
    
    if total_mismatched == 0 and total_missing == 0 and total_shape_mismatch == 0:
        print("\n✓ 所有可学习的 backbone 权重匹配！")
    else:
        print("\n✗ 存在不匹配的可学习 backbone 权重！")


if __name__ == "__main__":
    main()