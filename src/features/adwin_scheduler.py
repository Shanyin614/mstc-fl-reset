# src/features/adwin_scheduler.py
"""
ADWIN++ 自适应窗口调度器

基于 Hoeffding 不等式检测概念漂移（Concept Drift）
动态调整特征提取的窗口长度
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class ADWINWindowScheduler:
    """
    ADWIN (Adaptive Windowing) 概念漂移检测器

    核心思想:
    1. 维护一个滑动窗口存储最近的统计量（如方差）
    2. 使用 Hoeffding 不等式检测窗口内的显著变化
    3. 当检测到漂移时，截断窗口并重置

    数学原理:
        ε = sqrt((1/(2*n0) + 1/(2*n1)) * ln(2/δ))

        if |μ0 - μ1| > ε:
            drift_detected = True

    参数:
        - δ (delta): 置信度参数（越小越敏感，默认 0.002）
        - min_window: 最小窗口长度（默认 5）
        - max_window: 最大窗口长度（默认 100）
    """

    def __init__(self,
                 delta: float = 0.002,
                 min_window: int = 5,
                 max_window: int = 100):
        """
        Args:
            delta: 置信度参数（越小检测越敏感）
            min_window: 最小窗口长度
            max_window: 最大窗口长度
        """
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window

        # 使用 deque 实现滑动窗口（高效）
        self.window = deque(maxlen=max_window)

        # 统计信息
        self.drift_count = 0
        self.total_elements = 0
        self.drift_positions = []  # 记录漂移发生的位置

    def add_element(self, value: float) -> int:
        """
        添加新元素并检测漂移

        Args:
            value: 新的统计量（如 EM 方差）

        Returns:
            current_window_length: 当前窗口长度
        """
        self.window.append(value)
        self.total_elements += 1

        # 检测漂移
        if len(self.window) >= self.min_window * 2:
            if self._detect_drift():
                self._handle_drift()

        return len(self.window)

    # def _detect_drift(self) -> bool:
    #     """
    #     使用 Hoeffding 不等式检测漂移
    #
    #     Returns:
    #         drift_detected: 是否检测到漂移
    #     """
    #     window_list = list(self.window)
    #     n = len(window_list)
    #
    #     # 尝试不同的分割点（寻找最显著的变化）
    #     max_diff = 0
    #     best_split = n // 2
    #
    #     for split in range(self.min_window, n - self.min_window):
    #         w0 = window_list[:split]
    #         w1 = window_list[split:]
    #
    #         # 计算两个子窗口的均值
    #         mean0 = np.mean(w0)
    #         mean1 = np.mean(w1)
    #
    #         # Hoeffding bound
    #         n0, n1 = len(w0), len(w1)
    #         epsilon = self._compute_epsilon(n0, n1)
    #
    #         # 计算差异
    #         diff = abs(mean0 - mean1)
    #
    #         if diff > max_diff:
    #             max_diff = diff
    #             best_split = split
    #
    #         # 检测显著变化
    #         if diff > epsilon:
    #             return True
    #
    #     return False
    def _detect_drift(self) -> bool:
        """
        使用简化的 Z-Score 检测（更敏感）

        策略:
        1. 将窗口分为前半部分和后半部分
        2. 计算两部分的均值和方差
        3. 使用 Z-Score 判断差异是否显著

        Returns:
            drift_detected: 是否检测到漂移
        """
        window_list = list(self.window)
        n = len(window_list)

        if n < self.min_window * 2:
            return False

        # 分割点（中间位置）
        split = n // 2
        w0 = np.array(window_list[:split])
        w1 = np.array(window_list[split:])

        # 计算统计量
        mean0 = np.mean(w0)
        mean1 = np.mean(w1)
        std0 = np.std(w0, ddof=1) if len(w0) > 1 else 1e-8
        std1 = np.std(w1, ddof=1) if len(w1) > 1 else 1e-8

        # 合并标准误差
        pooled_std = np.sqrt((std0 ** 2 / len(w0)) + (std1 ** 2 / len(w1)))

        if pooled_std < 1e-8:
            pooled_std = 1e-8

        # 计算 Z-Score
        z_score = abs(mean0 - mean1) / pooled_std

        # 临界值（根据 delta 调整）
        # delta = 0.05 -> z_critical ≈ 1.96
        # delta = 0.01 -> z_critical ≈ 2.58
        z_critical = np.sqrt(-2 * np.log(self.delta))

        # 检测漂移
        return z_score > z_critical

    def _compute_epsilon(self, n0: int, n1: int) -> float:
        """
        计算 Hoeffding bound

        ε = sqrt((1/(2*n0) + 1/(2*n1)) * ln(2/δ))
        """
        if n0 == 0 or n1 == 0:
            return float('inf')

        epsilon = np.sqrt((1 / (2 * n0) + 1 / (2 * n1)) * np.log(2 / self.delta))
        return epsilon

    def _handle_drift(self):
        """
        处理检测到的漂移

        策略: 保留最近的 min_window 个元素，丢弃旧数据
        """
        self.drift_count += 1
        self.drift_positions.append(self.total_elements)

        # 截断窗口（只保留最近的数据）
        recent_data = list(self.window)[-self.min_window:]
        self.window.clear()
        self.window.extend(recent_data)

    def get_window_length(self) -> int:
        """返回当前窗口长度"""
        return len(self.window)

    def get_statistics(self) -> dict:
        """
        返回统计信息

        Returns:
            stats: {
                'total_elements': 总元素数,
                'drift_count': 漂移次数,
                'drift_rate': 漂移率,
                'current_window': 当前窗口长度,
                'drift_positions': 漂移位置列表
            }
        """
        return {
            'total_elements': self.total_elements,
            'drift_count': self.drift_count,
            'drift_rate': self.drift_count / max(1, self.total_elements),
            'current_window': len(self.window),
            'drift_positions': self.drift_positions.copy()
        }

    def reset(self):
        """重置调度器"""
        self.window.clear()
        self.drift_count = 0
        self.total_elements = 0
        self.drift_positions.clear()


class MultiScaleWindowScheduler:
    """
    多尺度窗口调度器

    为 Micro/Meso/Macro 三个尺度分别维护 ADWIN 检测器
    根据漂移情况动态调整窗口长度
    """

    def __init__(self,
                 base_scales: Optional[dict] = None,
                 delta: float = 0.002):
        """
        Args:
            base_scales: 基准窗口长度 {'micro': 5, 'meso': 15, 'macro': 40}
            delta: ADWIN 置信度参数
        """
        if base_scales is None:
            self.base_scales = {
                'micro': 5,
                'meso': 15,
                'macro': 40
            }
        else:
            self.base_scales = base_scales.copy()

        # 当前实际窗口长度
        self.current_scales = self.base_scales.copy()

        # 为每个尺度创建 ADWIN 检测器
        self.schedulers = {
            'micro': ADWINWindowScheduler(delta, min_window=3, max_window=15),
            'meso': ADWINWindowScheduler(delta, min_window=10, max_window=30),
            'macro': ADWINWindowScheduler(delta, min_window=20, max_window=80)
        }

    def update(self, em_variance: float) -> dict:
        """
        更新窗口长度（基于 EM 方差）

        Args:
            em_variance: 当前轮的 EM 方差

        Returns:
            current_scales: {'micro': L1, 'meso': L2, 'macro': L3}
        """
        # 为每个尺度添加方差
        for scale_name, scheduler in self.schedulers.items():
            L = scheduler.add_element(em_variance)
            self.current_scales[scale_name] = L

        return self.current_scales.copy()

    def get_scales(self) -> dict:
        """返回当前窗口长度"""
        return self.current_scales.copy()

    def get_all_statistics(self) -> dict:
        """返回所有尺度的统计信息"""
        stats = {}
        for scale_name, scheduler in self.schedulers.items():
            stats[scale_name] = scheduler.get_statistics()
        return stats
