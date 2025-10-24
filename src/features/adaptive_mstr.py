# src/features/adaptive_mstr.py
"""
自适应多尺度时序表示（Adaptive MSTR）

集成 ADWIN++ 窗口调度器，实现动态特征提取
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.multi_scale_features import MultiScaleTemporalRepresentation
from src.features.adwin_scheduler_v2 import MultiScaleWindowScheduler


class AdaptiveMSTR(MultiScaleTemporalRepresentation):
    """
    自适应 MSTR

    功能:
    1. 根据 ADWIN 检测结果动态调整窗口长度
    2. 在稳定期使用短窗口（减少计算）
    3. 在漂移期使用长窗口（捕获更多上下文）

    使用示例:
        >>> amstr = AdaptiveMSTR()
        >>> for t in range(100):
        >>>     variance = compute_variance(em_history[:t])
        >>>     features = amstr.extract_features_adaptive(em_history[:t], variance)
    """

    def __init__(self,
                 base_scales: dict = None,
                 delta: float = 0.002,
                 enable_adaptive: bool = True):
        """
        Args:
            base_scales: 基准窗口长度
            delta: ADWIN 敏感度
            enable_adaptive: 是否启用自适应（False 则退化为固定窗口）
        """
        super().__init__(scales=base_scales)

        self.enable_adaptive = enable_adaptive

        if enable_adaptive:
            self.scheduler = MultiScaleWindowScheduler(
                base_scales=base_scales,
                delta=delta
            )
        else:
            self.scheduler = None

        # 记录窗口长度历史
        self.scale_history = []

    def extract_features_adaptive(self,
                                  em_history: np.ndarray,
                                  current_variance: float) -> np.ndarray:
        """
        自适应特征提取

        Args:
            em_history: [T, 6] EM 历史序列
            current_variance: 当前轮的 EM 方差

        Returns:
            features: [36] 多尺度特征向量
        """
        if not self.enable_adaptive or self.scheduler is None:
            # 固定窗口模式
            return self.extract_features(em_history)

        # 更新窗口长度（基于方差）
        updated_scales = self.scheduler.update(current_variance)

        # 动态调整窗口
        self.scales = updated_scales

        # 记录历史
        self.scale_history.append(updated_scales.copy())

        # 提取特征
        features = self.extract_features(em_history)

        return features

    def get_current_scales(self) -> dict:
        """返回当前窗口长度"""
        if self.scheduler is not None:
            return self.scheduler.get_scales()
        return self.scales

    def get_drift_statistics(self) -> dict:
        """返回漂移检测统计"""
        if self.scheduler is not None:
            return self.scheduler.get_all_statistics()
        return {}

    def get_scale_history(self) -> list:
        """返回窗口长度变化历史"""
        return self.scale_history.copy()
