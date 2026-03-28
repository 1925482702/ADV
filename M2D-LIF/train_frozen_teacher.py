"""
Train dual-branch model with frozen teacher backbones.

This script trains a dual-modal (RGB + IR) detection model where:
1. The backbone weights are loaded from pretrained RGB and IR teachers
2. The backbone parameters are frozen during training
3. Feature fusion uses learnable ConvFusion layers instead of simple Add
4. Only the neck and head are trained
"""

import argparse
import torch
import sys
import os

# Add ultralytics to path
sys.path.insert(0, '/root/autodl-tmp/ADV/M2D-LIF')

from ultralytics import YOLO


def load_frozen_backbones(model, rgb_weight, ir_weight):
    """
    Load backbone weights from teacher models and freeze them.
    
    Args:
        model: The student model (DetectionModel)
        rgb_weight: Path to RGB teacher checkpoint
        ir_weight: Path to IR teacher checkpoint
    """
    # Load teacher checkpoints
    rgb_ckpt = torch.load(rgb_weight, map_location='cpu')
    ir_ckpt = torch.load(ir_weight, map_location='cpu')
    
    rgb_state = rgb_ckpt['model'].state_dict() if 'model' in rgb_ckpt else rgb_ckpt
    ir_state = ir_ckpt['model'].state_dict() if 'model' in ir_ckpt else ir_ckpt
    
    model_state = model.model.state_dict()
    
    loaded_rgb = 0
    loaded_ir = 0
    frozen_count = 0
    
    # Teacher backbone layers: 0-9 (Conv, Conv, C2f, Conv, C2f, Conv, C2f, Conv, C2f, SPPF)
    # Student dual-branch backbone:
    #   RGB branch: layers 3, 5, 7, 9, 11 (before first fusion)
    #              layers 14, 16 (between P3-P4 fusion)
    #              layers 19, 21 (between P4-P5 fusion)
    #   IR branch: layers 4, 6, 8, 10, 12, 15, 17, 20, 22
    
    # Layer mapping: student_layer -> teacher_layer
    rgb_layer_map = {
        3: 0, 5: 1, 7: 2, 9: 3, 11: 4,   # P2-P3 stage
        14: 5, 16: 6,                     # P4 stage
        19: 7, 21: 8                      # P5 stage
    }
    
    ir_layer_map = {
        4: 0, 6: 1, 8: 2, 10: 3, 12: 4,   # P2-P3 stage
        15: 5, 17: 6,                      # P4 stage
        20: 7, 22: 8                       # P5 stage
    }
    
    for name, param in model.model.named_parameters():
        parts = name.split('.')
        if len(parts) >= 2 and parts[0] == 'model':
            try:
                layer_idx = int(parts[1])
            except ValueError:
                continue
            
            # Check if this is a backbone layer
            if layer_idx in rgb_layer_map:
                teacher_layer = rgb_layer_map[layer_idx]
                teacher_name = name.replace(f'model.{layer_idx}', f'model.{teacher_layer}')
                if teacher_name in rgb_state:
                    param.data.copy_(rgb_state[teacher_name])
                    loaded_rgb += 1
                param.requires_grad = False
                frozen_count += 1
                
            elif layer_idx in ir_layer_map:
                teacher_layer = ir_layer_map[layer_idx]
                teacher_name = name.replace(f'model.{layer_idx}', f'model.{teacher_layer}')
                if teacher_name in ir_state:
                    param.data.copy_(ir_state[teacher_name])
                    loaded_ir += 1
                param.requires_grad = False
                frozen_count += 1
    
    print(f"[INFO] Loaded {loaded_rgb} parameters from RGB teacher")
    print(f"[INFO] Loaded {loaded_ir} parameters from IR teacher")
    print(f"[INFO] Frozen {frozen_count} backbone parameters")
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.model.parameters())
    print(f"[INFO] Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description='Train dual-branch model with frozen teacher backbones')
    parser.add_argument('--scale', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model scale (must match teacher models)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='CUDA device')
    parser.add_argument('--rgb-weight', type=str, 
                        default='/root/autodl-tmp/ADV/checkpoint/monomodal/FLIR_rgb.pt',
                        help='Path to RGB teacher checkpoint')
    parser.add_argument('--ir-weight', type=str,
                        default='/root/autodl-tmp/ADV/checkpoint/monomodal/FLIR_ir.pt',
                        help='Path to IR teacher checkpoint')
    args = parser.parse_args()
    
    # Model config (use scale-specific yaml)
    model_yaml = f"/root/autodl-tmp/ADV/M2D-LIF/model_yaml/yolov8_dual_frozen_{args.scale}.yaml"
    
    # Check if scale-specific yaml exists
    import os
    if not os.path.exists(model_yaml):
        model_yaml = "/root/autodl-tmp/ADV/M2D-LIF/model_yaml/yolov8_dual_frozen.yaml"
        print(f"[WARNING] Scale-specific yaml not found, using default: {model_yaml}")
    
    print(f"[INFO] Model config: {model_yaml}")
    print(f"[INFO] Scale: {args.scale}")
    
    # Load model
    model = YOLO(model_yaml)
    
    # Load frozen backbone weights from teachers
    print(f"\n[INFO] Loading RGB teacher from: {args.rgb_weight}")
    print(f"[INFO] Loading IR teacher from: {args.ir_weight}")
    load_frozen_backbones(model, args.rgb_weight, args.ir_weight)
    
    # Start training
    print("\n[INFO] Starting training...")
    model.train(
        task='detect',
        data="/root/autodl-tmp/ADV/M2D-LIF/data/FLIR.yaml",
        epochs=args.epochs,
        imgsz=640,
        device=args.device,
        batch=args.batch,
        workers=8,
        project='./runs/frozen_teacher',
        name=f'yolov8{args.scale}_frozen_5layer',
        lr0=0.01,
        augment=True
    )


if __name__ == '__main__':
    main()
