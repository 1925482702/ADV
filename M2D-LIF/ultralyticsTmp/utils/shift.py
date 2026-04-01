# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Shift-based cross-modal alignment loss and utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftLoss(nn.Module):
    """Loss for shift prediction in cross-modal alignment."""

    def __init__(self, weight=0.1):
        """Initialize ShiftLoss.

        Args:
            weight (float): Weight of shift loss relative to detection loss
        """
        super().__init__()
        self.weight = weight

    def forward(self, pred_shift, gt_shift):
        """Calculate shift loss using Smooth L1 Loss.

        Args:
            pred_shift (torch.Tensor): Predicted shift (B, 2, H, W) or (B, 2)
            gt_shift (torch.Tensor): Ground truth shift (B, 2)

        Returns:
            torch.Tensor: Loss value
        """
        if pred_shift is None or gt_shift is None:
            return torch.tensor(0.0, device=pred_shift.device if pred_shift is not None else torch.device('cpu'))

        # Reshape if needed
        if len(pred_shift.shape) == 4:  # (B, 2, H, W)
            # Average pool to get global shift prediction
            pred_shift = F.adaptive_avg_pool2d(pred_shift, (1, 1))  # (B, 2, 1, 1)
            pred_shift = pred_shift.squeeze(-1).squeeze(-1)  # (B, 2)

        # Ensure gt_shift is 2D
        if len(gt_shift.shape) == 1:
            gt_shift = gt_shift.unsqueeze(0)

        # Calculate loss
        loss = F.smooth_l1_loss(pred_shift, gt_shift, reduction='mean')
        return loss * self.weight


def apply_shift_augmentation(batch, shift_augment=None, shift_prob=0.3, shift_range=(10, 50)):
    """Apply shift augmentation to a batch of images.

    Args:
        batch (dict): Batch dictionary with 'img' and other keys
        shift_augment (callable, optional): Shift augmentation callable
        shift_prob (float): Probability of applying shift
        shift_range (tuple): (min_shift, max_shift) in pixels

    Returns:
        dict: Updated batch with shift information
    """
    import numpy as np
    import random
    import cv2

    if shift_augment is not None:
        # Use the provided augmentation
        batch = shift_augment(batch)
    else:
        # Default behavior: mark as not applied
        if 'shift_dx' not in batch:
            batch['shift_dx'] = torch.zeros(batch['img'].shape[0], device=batch['img'].device)
        if 'shift_dy' not in batch:
            batch['shift_dy'] = torch.zeros(batch['img'].shape[0], device=batch['img'].device)
        if 'shift_applied' not in batch:
            batch['shift_applied'] = torch.zeros(batch['img'].shape[0], dtype=torch.bool, device=batch['img'].device)

    return batch


def get_shift_from_batch(batch):
    """Extract shift ground truth from batch.

    Args:
        batch (dict): Batch dictionary

    Returns:
        torch.Tensor: Shift ground truth (B, 2) or None
    """
    if 'shift_dx' not in batch or 'shift_dy' not in batch:
        return None

    shift_dx = batch['shift_dx']
    shift_dy = batch['shift_dy']

    # Convert to tensor if needed
    if not isinstance(shift_dx, torch.Tensor):
        shift_dx = torch.tensor(shift_dx, dtype=torch.float32)
    if not isinstance(shift_dy, torch.Tensor):
        shift_dy = torch.tensor(shift_dy, dtype=torch.float32)

    # Stack to (B, 2)
    shift_gt = torch.stack([shift_dx, shift_dy], dim=-1)

    return shift_gt
