from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG
from datetime import datetime

from ultralytics.models.yolo.obb import OBBTrainer
if __name__ == '__main__':

    # 🔥 修复：使用正确的配置文件路径
    # yolov8-obb.yaml 已经包含所有scale定义（n/s/m/l/x）
    # YOLO会自动根据文件名中的 'm' 来选择对应的scale参数
    model_s = YOLO(r"./ultralytics/cfg/models/v8/yolov8-obb.yaml")  # 修改这里
    DEFAULT_CFG.save_dir = f""
    model_s.train(
        data=r"./Drone_IR.yaml",
        imgsz=640,
        epochs=100,
        batch=4,
        device=3,
        lr0=0.001,
        augment=True,
        workers=4,
        save=True,
        amp=False,
        rect=True
    )