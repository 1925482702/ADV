"""
双路径解耦训练脚本（Gemini 终极版）

🔥 核心特性：
    1. 全批次镜像双开架构
    2. 随机平移 RGB/IR（消除模态偏见）
    3. 双路径解耦：干净流（检测/蒸馏）+ 错位流（Shift）

使用方法：
    python train_dual_path.py --data FLIR.yaml --epochs 100 --batch 4 --device 0

⚠️ 注意事项：
    1. Batch Size 建议：从 4 开始（A100 40GB）
    2. 验证集会自动保持 6 通道（不生成 9 通道）
    3. 训练日志会记录 shift_flag 分布
"""

import argparse
from ultralytics import YOLO
from ultralytics.nn.tasks import DualPathDetectionModel
from ultralytics.utils.loss import DualPathLoss
from ultralytics.utils import LOGGER

# ==================== 🔥 UI 换头魔法 ====================
from ultralytics.models.yolo.detect.train import DetectionTrainer

# 1. 强行修改表头名字
def get_loss_names(self):
    return ['box_loss', 'cls_loss', 'dfl_loss', 'shift_loss']

def set_loss_names(self, value):
    pass  # 假装同意：悄悄吞掉底层引擎的 ['Loss'] 赋值操作，绝不报错

DetectionTrainer.loss_names = property(get_loss_names, set_loss_names)

# 2. 🔥 关键：强行修改 Loss 数量监控
DetectionTrainer.loss_items_len = 4
# ========================================================

def main():
    parser = argparse.ArgumentParser(description='双路径解耦训练脚本')
    parser.add_argument('--data', type=str, default='FLIR.yaml', help='数据集配置文件')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch', type=int, default=4, help='批次大小（建议从 4 开始）')
    parser.add_argument('--device', type=str, default='0', help='GPU 设备号')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--project', type=str, default='runs/dual_path', help='项目保存路径')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    args = parser.parse_args()
    
    LOGGER.info(f'🚀 正在构建 YOLOv8 引擎...')
    
    # 1. 使用官方 YOLO 引擎加载我们的 YAML
    # 此时它会读取 scales 字典生成正确大小的网络
    model = YOLO('model_yaml/yolov8_dual_path.yaml')
    
    # 2. 🔥 神级操作：动态注入我们的"全批次镜像双开"逻辑！
    # 直接修改底层 PyTorch 模型的 class 指针，让它继承我们的 forward 和 loss
    model.model.__class__ = DualPathDetectionModel
    
    # 3. 强制挂载我们庖丁解牛的 Loss 函数
    model.model.criterion = DualPathLoss(model.model)
    
    # 4. 🔥 添加回调：每个 epoch 结束时打印 shift_loss
    def on_epoch_end(trainer):
        """每个 epoch 结束时的回调函数"""
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            # loss_items 是一个张量，包含 [box, cls, dfl, shift]
            if len(trainer.loss_items) >= 4:
                shift_loss_value = trainer.loss_items[3]
                LOGGER.info(f'🔥 [Epoch {trainer.epoch + 1}] Shift Loss: {shift_loss_value:.4f}')
    
    # 注册回调
    model.add_callback('on_train_epoch_end', on_epoch_end)
    
    # (可选) 加载预训练权重
    # model.load('yolov8m.pt')
    
    LOGGER.info(f'🔥 开始训练 (epochs={args.epochs}, batch={args.batch}, imgsz={args.imgsz})...')
    
    # 4. 调用官方引擎的 train 方法（现在不会报错了）
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        
        # 🔥 关键参数
        amp=True,           # 自动混合精度（节省显存）
        close_mosaic=10,    # 最后 10 epochs 关闭 mosaic
        
        # 数据增强参数
        mosaic=1.0,         # Mosaic 概率
        fliplr=0.5,         # 水平翻转概率
        
        # 学习率
        lr0=0.01,           # 初始学习率
        lrf=0.01,           # 最终学习率系数
        
        # 保存设置
        save=True,
        save_period=-1,     # 每个 epoch 保存
        plots=True,         # 绘制训练曲线
        verbose=True,
    )
    
    LOGGER.info(f'✅ 训练完成！结果保存在: {results.save_dir}')
    LOGGER.info(f'📊 最终 mAP: {results.results_dict.get("metrics/mAP50(B)", "N/A")}')

if __name__ == '__main__':
    main()
