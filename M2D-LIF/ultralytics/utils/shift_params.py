"""
跨进程共享的 Shift 参数管理器

使用 multiprocessing.Value 实现真正的进程间通信，
确保 DataLoader 的多个 worker 进程都能获取最新参数。
"""

import multiprocessing


class ShiftParameterManager:
    """
    跨进程共享的 Shift 参数管理器（完整版）

    支持 3 个关键参数的跨进程同步：
    - shift_ratio: 平移比例
    - shift_weight: shift loss 权重
    - shift_mask_weight: 未平移样本权重
    """

    def __init__(self, initial_ratio=0.3, initial_weight=0.5, initial_mask_weight=0.3):
        """
        初始化共享参数

        Args:
            initial_ratio: 初始平移比例（阶段 1 默认值）
            initial_weight: 初始 loss 权重
            initial_mask_weight: 初始未平移样本权重
        """
        # 'd' 代表 double (float) 类型，创建真正的共享内存
        self.shift_ratio = multiprocessing.Value('d', float(initial_ratio))
        self.shift_weight = multiprocessing.Value('d', float(initial_weight))
        self.shift_mask_weight = multiprocessing.Value('d', float(initial_mask_weight))

    def update_params(self, ratio=None, weight=None, mask_weight=None):
        """
        更新参数（在主进程/Trainer 中调用）

        Args:
            ratio: 新的平移比例
            weight: 新的 loss 权重
            mask_weight: 新的未平移样本权重
        """
        if ratio is not None:
            with self.shift_ratio.get_lock():
                self.shift_ratio.value = float(ratio)

        if weight is not None:
            with self.shift_weight.get_lock():
                self.shift_weight.value = float(weight)

        if mask_weight is not None:
            with self.shift_mask_weight.get_lock():
                self.shift_mask_weight.value = float(mask_weight)

    def get_ratio(self):
        """获取当前平移比例（在 DataLoader Worker 中调用）"""
        with self.shift_ratio.get_lock():
            return self.shift_ratio.value

    def get_weight(self):
        """获取当前 loss 权重（在 Loss 计算中调用）"""
        with self.shift_weight.get_lock():
            return self.shift_weight.value

    def get_mask_weight(self):
        """获取当前未平移样本权重（在 Loss 计算中调用）"""
        with self.shift_mask_weight.get_lock():
            return self.shift_mask_weight.value

    def get_params(self):
        """获取所有参数（批量获取，减少锁竞争）"""
        with self.shift_ratio.get_lock(), \
             self.shift_weight.get_lock(), \
             self.shift_mask_weight.get_lock():
            return {
                'shift_ratio': self.shift_ratio.value,
                'shift_weight': self.shift_weight.value,
                'shift_mask_weight': self.shift_mask_weight.value
            }


# 全局实例（必须在所有进程创建前实例化）
# 这里的初始值对应 U 型策略阶段 1 的配置
shift_param_manager = ShiftParameterManager(
    initial_ratio=0.3,
    initial_weight=0.5,
    initial_mask_weight=0.2
)