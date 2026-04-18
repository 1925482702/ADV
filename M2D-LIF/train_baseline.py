from ultralytics import YOLO

# 加载baseline模型配置（naive add融合，不带M2D和LIF）
# 🚨 可以通过修改文件名后缀指定scale: yolov8_naive_add_s.yaml 或 yolov8_naive_add_m.yaml
# 或者直接在加载后调用 .to("cuda") 前设置 scale
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=str, default='s', choices=['n', 's', 'm', 'l', 'x'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch', type=int, default=16)
parser.add_argument('--device', type=int, default=0)
args = parser.parse_args()

# 🚨 Ultralytics 通过文件名后缀识别 scale
# 例如: yolov8_naive_add_s.yaml 使用 s scale
model_yaml = f"./model_yaml/yolov8_naive_add_{args.scale}.yaml"

# 检查是否存在对应scale的yaml，如果不存在则使用默认的
import os
if not os.path.exists(model_yaml):
    model_yaml = "/root/autodl-tmp/ADV/M2D-LIF/model_yaml/yolov8_naive_add.yaml"
    print(f"⚠️  未找到 {model_yaml}，使用默认配置")

print(f"使用模型配置: {model_yaml}")
print(f"Scale: {args.scale}")

model = YOLO(model_yaml)

# 训练参数
model.train(
    task='detect',
    data="/root/autodl-tmp/ADV/M2D-LIF/data/FLIR.yaml",
    epochs=args.epochs,
    imgsz=640,
    device=args.device,
    batch=args.batch,
    workers=8,
    project='./runs/baseline',
    name=f'yolov8{args.scale}_naive_add',
    lr0=0.01,
    augment=False
)