"""
U 型训练调度器

支持无状态的训练策略调度，天然支持断点续训。
"""

import yaml
from pathlib import Path


class UShapeShiftScheduler:
    """
    U 型训练调度器（无状态设计）

    特点：
    - 无状态：每次根据 epoch 计算配置，天然支持断点续训
    - 精美日志：阶段切换时输出详细信息
    - 灵活配置：支持 YAML 配置文件
    """

    def __init__(self, config_path=None, config_dict=None, logger=None):
        """
        初始化调度器

        Args:
            config_path: YAML 配置文件路径
            config_dict: 直接传入的配置字典
            logger: 日志记录器
        """
        if config_dict:
            self.config = config_dict
        elif config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            raise ValueError("必须提供 config_path 或 config_dict")

        self.logger = logger
        self.current_stage_idx = -1
        self.last_logged_epoch = -1
        self.stage_transitions = []

    def get_config_for_epoch(self, epoch):
        """
        根据 epoch 获取当前阶段的配置（无状态方法）

        Args:
            epoch: 当前 epoch (0-based)

        Returns:
            dict: 包含 shift_ratio, shift_weight, shift_mask_weight 的配置字典

        天然支持断点续训：
        - 每次都根据 epoch 重新计算
        - 不依赖内部状态
        - 恢复训练时自动进入正确阶段
        """
        schedule = self.config.get('stages', [])

        for i, stage in enumerate(schedule):
            if stage['start_epoch'] <= epoch <= stage['end_epoch']:
                # 检查是否切换了阶段
                if i != self.current_stage_idx and self.logger:
                    self.current_stage_idx = i
                    self.last_logged_epoch = epoch

                    # 记录阶段转换
                    self.stage_transitions.append({
                        'epoch': epoch,
                        'from_stage': schedule[i - 1]['name'] if i > 0 else "init",
                        'to_stage': stage['name']
                    })

                    # 精美的阶段切换日志
                    self._log_stage_transition(epoch, i, stage, len(schedule))

                return {
                    'shift_ratio': stage['shift_ratio'],
                    'shift_weight': stage['shift_weight'],
                    'shift_mask_weight': stage['shift_mask_weight'],
                    'stage_name': stage['name']
                }

        # 如果超出所有阶段，返回最后一个阶段的配置
        if schedule:
            last_stage = schedule[-1]
            if epoch > last_stage['end_epoch'] and self.logger:
                self.logger.warning(f"⚠️  Epoch {epoch} 超出配置范围，使用最后阶段配置")

            return {
                'shift_ratio': last_stage['shift_ratio'],
                'shift_weight': last_stage['shift_weight'],
                'shift_mask_weight': last_stage['shift_mask_weight'],
                'stage_name': last_stage['name']
            }

        # 默认配置
        return {
            'shift_ratio': 0.3,
            'shift_weight': 1.0,
            'shift_mask_weight': 0.5,
            'stage_name': 'default'
        }

    def _log_stage_transition(self, epoch, stage_idx, stage, total_stages):
        """输出精美的阶段切换日志"""
        stage_num = stage_idx + 1
        epochs = stage['end_epoch'] - stage['start_epoch'] + 1
        total_epochs = self.config.get('total_epochs', 120)
        percentage = epochs / total_epochs * 100

        self.logger.info("=" * 70)
        self.logger.info(f"📊 Epoch {epoch:3d}/{total_epochs:3d}: 切换到阶段 {stage_num}/{total_stages}")
        self.logger.info(f"   🎯 阶段名称: {stage['name']}")
        self.logger.info(f"   📋 阶段描述: {stage['description']}")
        self.logger.info(f"   ⏱️   Epoch 范围: {stage['start_epoch']:3d} - {stage['end_epoch']:3d} ({epochs:3d}轮, {percentage:.1f}%)")
        self.logger.info(f"   📊 参数配置:")
        self.logger.info(f"      - shift_ratio: {stage['shift_ratio']}")
        self.logger.info(f"      - shift_weight: {stage['shift_weight']}")
        self.logger.info(f"      - shift_mask_weight: {stage['shift_mask_weight']}")
        self.logger.info("=" * 70)

    def get_stage_summary(self):
        """返回阶段转换历史摘要"""
        return self.stage_transitions

    def print_schedule_summary(self):
        """打印完整的训练计划摘要"""
        schedule = self.config.get('stages', [])
        total_epochs = self.config.get('total_epochs', 0)

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info("📋 U型训练策略摘要")
        self.logger.info("=" * 70)
        self.logger.info(f"总训练轮数: {total_epochs} epochs")
        self.logger.info(f"阶段数量: {len(schedule)} 个")
        self.logger.info("")

        for i, stage in enumerate(schedule):
            epochs = stage['end_epoch'] - stage['start_epoch'] + 1
            percentage = epochs / total_epochs * 100
            self.logger.info(f"阶段 {i+1}: {stage['name']}")
            self.logger.info(f"  📅 Epoch 范围: {stage['start_epoch']:3d}-{stage['end_epoch']:3d} ({epochs:3d}轮, {percentage:.1f}%)")
            self.logger.info(f"  📊 shift_ratio: {stage['shift_ratio']}")
            self.logger.info(f"  ⚖️  shift_weight: {stage['shift_weight']}")
            self.logger.info(f"  🎯 描述: {stage['description']}")
            self.logger.info("")

        self.logger.info("=" * 70)