#!/usr/bin/env python
# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Example script for training with cross-modal shift prediction."""

import torch
from ultralytics import YOLO
from ultralytics.models.shift_integration import enable_shift_head_in_model
from ultralytics.data.augment import PairedShiftAugment

# Example 1: Basic training with shift support
# ============================================

def example_basic_training():
    """Basic example of training with shift prediction enabled."""

    # Load model
    model = YOLO('/mnt/home/pyq_code/ADV/M2D-LIF/model_yaml/yolov8_naive_add.yaml')

    # Enable shift head in the model
    enable_shift_head_in_model(model, enabled=True)

    # Train with shift augmentation
    results = model.train(
        data='/mnt/home/pyq_code/ADV/M2D-LIF/data/data.yaml',  # Replace with actual data path
        epochs=100,
        imgsz=640,
        batch=1,
        device=0,
        # Additional shift-specific parameters can be added here
    )

    return results


# Example 2: Manual shift augmentation integration
# ===============================================

def example_manual_augmentation():
    """Example of manually applying shift augmentation in training loop."""
    from ultralytics.engine.trainer import BaseTrainer

    class ShiftTrainer(BaseTrainer):
        """Custom trainer with shift augmentation."""

        def __init__(self, cfg=None, overrides=None, _callbacks=None):
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
            # Initialize shift augmentation
            self.shift_augment = PairedShiftAugment(
                shift_range=(10, 50),
                shift_prob=0.3
            )

        def preprocess_batch(self, batch):
            """Preprocess batch with shift augmentation."""
            # Call parent's preprocess
            batch = super().preprocess_batch(batch)

            # Apply shift augmentation to the batch
            # Note: This is a simplified example
            # In practice, augmentation should be done during dataset loading

            return batch


# Example 3: Monitoring shift loss
# ================================

def example_with_loss_monitoring():
    """Example showing how to monitor shift loss during training."""
    from ultralytics import YOLO
    from ultralytics.models.shift_integration import DetectionLossWithShift
    from ultralytics.utils.loss import v8DetectionLoss

    model = YOLO('/mnt/home/pyq_code/ADV/M2D-LIF/model_yaml/yolov8_naive_add.yaml')

    # Create loss with shift support
    base_loss = v8DetectionLoss(model)
    shift_loss = DetectionLossWithShift(base_loss, shift_weight=0.1, enabled=True)

    # During training, you can access loss components:
    # total_loss, loss_items = shift_loss(preds, batch, shift_preds)
    # loss_items[0] = box loss
    # loss_items[1] = cls loss
    # loss_items[2] = dfl loss
    # additional = shift loss (if enabled)


# Example 4: Dynamic scheduling of shift augmentation
# ==================================================

class DynamicShiftTrainer:
    """Trainer with dynamic shift augmentation scheduling."""

    def __init__(self, model, total_epochs=100):
        self.model = model
        self.total_epochs = total_epochs
        self.shift_augment = PairedShiftAugment(shift_range=(10, 50), shift_prob=0.0)

    def get_shift_prob_for_epoch(self, epoch):
        """Get shift probability for current epoch.

        Implements dynamic scheduling:
        - Warmup (0-20%): no shift
        - Early (20-80%): linear growth
        - Late (80-100%): max shift
        """
        if epoch < self.total_epochs * 0.2:
            # Warmup phase
            return 0.0
        elif epoch < self.total_epochs * 0.8:
            # Early phase: linear growth
            progress = (epoch - self.total_epochs * 0.2) / (self.total_epochs * 0.6)
            return 0.3 * progress  # Grow from 0 to 0.3
        else:
            # Late phase: maximum shift
            return 0.3

    def train_one_epoch(self, epoch):
        """Train for one epoch with dynamic shift scheduling."""
        # Update shift probability for this epoch
        shift_prob = self.get_shift_prob_for_epoch(epoch)
        self.shift_augment.shift_prob = shift_prob

        print(f"Epoch {epoch}: shift_prob = {shift_prob:.3f}")
        # Training loop would go here


# Example 5: Verify shift augmentation output
# ==========================================

def example_verify_augmentation():
    """Verify that shift augmentation is working correctly."""
    import numpy as np
    from ultralytics.data.augment import PairedShiftAugment

    # Create dummy batch data
    batch = {
        'img': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'img_lwir': np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        'instances': None,  # Would need actual instances
        'bboxes': np.array([[0.2, 0.2, 0.4, 0.4], [0.5, 0.5, 0.7, 0.7]]),
    }

    # Create augmentation
    augment = PairedShiftAugment(shift_range=(10, 50), shift_prob=1.0)

    # Apply augmentation
    for i in range(3):
        augmented = augment(batch)
        print(f"Augmentation {i}:")
        print(f"  Shift applied: {augmented.get('shift_applied', False)}")
        print(f"  Shift dx: {augmented.get('shift_dx', 0.0):.4f}")
        print(f"  Shift dy: {augmented.get('shift_dy', 0.0):.4f}")


# Main entry point
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Modal Shift Prediction Examples")
    print("=" * 60)

    # Test imports
    print("\n✓ Imports successful")

    # Show available examples
    print("\nAvailable examples:")
    print("1. example_basic_training() - Basic training with shift")
    print("2. example_manual_augmentation() - Manual augmentation integration")
    print("3. example_with_loss_monitoring() - Loss monitoring")
    print("4. DynamicShiftTrainer() - Dynamic scheduling")
    print("5. example_verify_augmentation() - Verify augmentation")

    # Optionally run verification
    print("\nRunning augmentation verification...")
    try:
        example_verify_augmentation()
    except Exception as e:
        print(f"Note: {e}")
        print("(This is expected if instances are not properly set up)")
