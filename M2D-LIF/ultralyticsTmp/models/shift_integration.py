# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Cross-modal shift prediction integration utilities.

ShiftHead is managed OUTSIDE the YOLO model to avoid breaking
model.model[-1] == Detect assumption in v8DetectionLoss.
"""

import torch
import torch.nn as nn
from ultralytics.utils.shift import ShiftLoss, get_shift_from_batch
from ultralytics.data.augment import PairedShiftAugment


class ShiftAugmentationManager:
    """Manager for shift augmentation during training."""

    def __init__(self, shift_prob=0.3, shift_range=(10, 50), enabled=True):
        self.enabled = enabled
        if enabled:
            self.augment = PairedShiftAugment(shift_range=shift_range, shift_prob=shift_prob)
        else:
            self.augment = None

    def __call__(self, batch):
        if not self.enabled or self.augment is None:
            if 'shift_dx' not in batch:
                batch['shift_dx'] = torch.zeros(batch['img'].shape[0] if 'img' in batch else 1)
            if 'shift_dy' not in batch:
                batch['shift_dy'] = torch.zeros(batch['img'].shape[0] if 'img' in batch else 1)
            return batch
        return batch


class DetectionLossWithShift(nn.Module):
    """Detection loss with shift prediction component."""

    def __init__(self, base_loss_fn, shift_weight=0.1, enabled=True):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.enabled = enabled
        if enabled:
            self.shift_loss = ShiftLoss(weight=shift_weight)
        else:
            self.shift_loss = None

    def forward(self, preds, batch, shift_preds=None):
        det_loss, det_loss_items = self.base_loss_fn(preds, batch)

        shift_loss = 0.0
        if self.enabled and self.shift_loss is not None and shift_preds is not None:
            shift_gt = get_shift_from_batch(batch)
            if shift_gt is not None:
                shift_loss = self.shift_loss(shift_preds, shift_gt)

        total_loss = det_loss + shift_loss
        return total_loss, det_loss_items


def enable_shift_head_in_model(model, enabled=True):
    """No-op: ShiftHead is now managed externally to avoid breaking model structure.

    Use create_shift_head() instead to get a standalone ShiftHead module.
    """
    pass


def create_shift_head(model, device=None):
    """Create a standalone ShiftHead module based on the model's Detect layer channels.

    Args:
        model: YOLO model or DetectionModel
        device: Device to place the module on

    Returns:
        ShiftHead module (not attached to the model)
    """
    from ultralytics.nn.modules.shift import ShiftHead

    # Find Detect layer to get channel info
    for m in model.modules():
        if m.__class__.__name__ == 'Detect':
            ch = m.cv2[0][0].conv.in_channels
            head = ShiftHead(ch)
            if device is not None:
                head = head.to(device)
            return head

    raise RuntimeError("No Detect layer found in model")
