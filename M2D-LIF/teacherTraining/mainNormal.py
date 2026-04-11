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
        data=r"./FLIR_IR.yaml",  # DroneVehicle
        amp=False,
        imgsz=640,
        epochs=100,
        batch=16,
        device=0,
        lr0=0.001,
        augment=True,
        workers=4,
        rect=False
    )

    DEFAULT_CFG.save_dir = f"./runs/FLIR_IR_SHIFT"

    model_s = DetectionTrainer(overrides=args)
    model_s.train()
