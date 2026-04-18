from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.models.yolo.obb import OBBTrainer
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.utils import DEFAULT_CFG
from ultralytics import YOLO
if __name__ == '__main__':


    args = dict(
        model=r"./ultralytics/cfg/models/v8/yolov8m.yaml",
        data=r"./FLIR_IR.yaml",  # 改为 FLIR_RGB.yaml 训练RGB
        amp=False,
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
        lr0=0.001,
        augment=True,
        workers=4,
        rect=False,
        
        # ========== 物体平移增强参数 ==========
        object_shift=0.5,       # 50%概率应用物体平移（0.0=禁用）
        max_shift_px=40,        # 最大平移40像素
        min_shift_px=8,         # 最小平移8像素
        shift_ratio=0.3,        # 30%物体被平移
        # ===================================
    )

    DEFAULT_CFG.save_dir = f"./runs/FLIR_IR_SHIFT"

    model_s = DetectionTrainer(overrides=args)
    model_s.train()
