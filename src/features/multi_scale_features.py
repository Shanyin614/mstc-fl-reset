# src/features/multi_scale_features.py
"""
Multi-Scale Temporal Representation (MSTR)

从 EM 历史序列中提取多尺度统计特征
- Micro Scale (L=5): 捕获短期波动
- Meso Scale (L=15): 捕获中期趋势
- Macro Scale (L=40): 捕获长期模式
"""

import numpy as np
from scipy import stats
from typing import List, Tuple, Optional


class MultiScaleTemporalRepresentation:
    """
    多尺度时序特征提取器

    输入: EM 历史序列 [T, 6]
        - T: 时间步数
        - 6: [Acc, Prec, Rec, F1, FPR, Loss]

    输出: 多尺度特征向量 [36]
        - 每个尺度: [均值(6) + 标准差(6)] = 12
        - 三个尺度: 12 × 3 = 36
    """

    def __init__(self,
                 scales: Optional[dict] = None,
                 feature_dim: int = 6):
        """
        Args:
            scales: 窗口长度配置 {'micro': 5, 'meso': 15, 'macro': 40}
            feature_dim: EM 指标维度（默认6维）
        """
        if scales is None:
            self.scales = {
                'micro': 5,  # 短期：最近 5 轮
                'meso': 15,  # 中期：最近 15 轮
                'macro': 40  # 长期：最近 40 轮
            }
        else:
            self.scales = scales

        self.feature_dim = feature_dim
        self.scale_names = ['micro', 'meso', 'macro']

        # 计算输出维度
        # 每个尺度: mean(6) + std(6) = 12
        self.output_dim = len(self.scales) * feature_dim * 2

    def extract_features(self, em_history: np.ndarray) -> np.ndarray:
        """
        从 EM 历史中提取多尺度特征

        Args:
            em_history: np.ndarray [T, 6]
                T >= 1 的历史序列

        Returns:
            features: np.ndarray [36]
                拼接后的多尺度特征向量

        示例:
            >>> em_history = np.random.rand(50, 6)
            >>> mstr = MultiScaleTemporalRepresentation()
            >>> features = mstr.extract_features(em_history)
            >>> print(features.shape)  # (36,)
        """
        if len(em_history) == 0:
            # 空历史：返回零向量
            return np.zeros(self.output_dim)

        feature_list = []

        for scale_name in self.scale_names:
            L = self.scales[scale_name]

            # 截取窗口 (如果历史不足，使用全部)
            if len(em_history) < L:
                window = em_history
            else:
                window = em_history[-L:]

            # 计算统计特征
            mean_feats = self._compute_mean(window)  # [6]
            std_feats = self._compute_std(window)  # [6]

            # 拼接: [mean(6), std(6)] = [12]
            scale_feats = np.concatenate([mean_feats, std_feats])
            feature_list.append(scale_feats)

        # 拼接所有尺度: [12, 12, 12] → [36]
        features = np.concatenate(feature_list)

        return features

    def extract_batch(self, em_history_list: List[np.ndarray]) -> np.ndarray:
        """
        批量提取多个 EN 的特征

        Args:
            em_history_list: List[np.ndarray]
                每个元素是一个 EN 的完整历史 [T_i, 6]

        Returns:
            features: np.ndarray [N, 36]
                N 个 EN 的特征矩阵

        示例:
            >>> histories = [np.random.rand(50, 6) for _ in range(20)]
            >>> features = mstr.extract_batch(histories)
            >>> print(features.shape)  # (20, 36)
        """
        features = []
        for em_hist in em_history_list:
            feat = self.extract_features(em_hist)
            features.append(feat)

        return np.array(features)

    def _compute_mean(self, window: np.ndarray) -> np.ndarray:
        """
        计算窗口内的均值

        Args:
            window: [T, D]

        Returns:
            mean: [D]
        """
        if len(window) == 0:
            return np.zeros(self.feature_dim)
        return np.mean(window, axis=0)

    def _compute_std(self, window: np.ndarray) -> np.ndarray:
        """
        计算窗口内的标准差

        Args:
            window: [T, D]

        Returns:
            std: [D]
        """
        if len(window) == 0:
            return np.zeros(self.feature_dim)
        if len(window) == 1:
            return np.zeros(self.feature_dim)
        return np.std(window, axis=0)

    def get_feature_names(self) -> List[str]:
        """
        返回特征名列表（用于可视化）

        Returns:
            names: List[str]
                ['micro_mean_acc', 'micro_mean_prec', ..., 'macro_std_loss']
        """
        metric_names = ['acc', 'prec', 'rec', 'f1', 'fpr', 'loss']
        names = []

        for scale_name in self.scale_names:
            # 均值特征
            for metric in metric_names:
                names.append(f"{scale_name}_mean_{metric}")
            # 标准差特征
            for metric in metric_names:
                names.append(f"{scale_name}_std_{metric}")

        return names

    def get_feature_dim(self) -> int:
        """
        计算特征维度

        Returns:
            总特征维度
        """
        # 每个尺度提取 4 个特征（均值、标准差、斜率、自相关）
        return len(self.scales) * 4

class EnhancedMSTR(MultiScaleTemporalRepresentation):
    """
    增强版 MSTR (可选)

    额外包含:
    - 趋势特征 (线性拟合斜率)
    - 自相关特征 (ACF lag-1)

    输出维度: 36 + 12 + 12 = 60
    """

    def extract_features(self, em_history: np.ndarray) -> np.ndarray:
        """
        提取增强特征

        Returns:
            features: np.ndarray [60]
        """
        if len(em_history) == 0:
            return np.zeros(60)

        feature_list = []

        for scale_name in self.scale_names:
            L = self.scales[scale_name]

            if len(em_history) < L:
                window = em_history
            else:
                window = em_history[-L:]

            # 基础特征
            mean_feats = self._compute_mean(window)
            std_feats = self._compute_std(window)

            # 增强特征
            trend_feats = self._compute_trend(window)
            acf_feats = self._compute_autocorr(window)

            # 拼接: [6+6+6+6] = [24]
            scale_feats = np.concatenate([
                mean_feats, std_feats, trend_feats, acf_feats
            ])
            feature_list.append(scale_feats)

        # 总计: 24 × 3 = 72
        features = np.concatenate(feature_list)
        return features

    def _compute_trend(self, window: np.ndarray) -> np.ndarray:
        """
        拟合线性趋势（斜率）

        y = a*t + b, 返回 a
        """
        if len(window) < 3:
            return np.zeros(self.feature_dim)

        t = np.arange(len(window))
        trends = []

        for d in range(window.shape[1]):
            y = window[:, d]
            # 线性回归
            slope, _ = np.polyfit(t, y, deg=1)
            trends.append(slope)

        return np.array(trends)

    def _compute_autocorr(self, window: np.ndarray, lag: int = 1) -> np.ndarray:
        """
        计算自相关系数 ACF(lag)
        """
        if len(window) < lag + 2:
            return np.zeros(self.feature_dim)

        acf_values = []

        for d in range(window.shape[1]):
            series = window[:, d]

            # 手动计算 ACF
            mean = series.mean()
            c0 = np.sum((series - mean) ** 2) / len(series)
            c_lag = np.sum((series[:-lag] - mean) * (series[lag:] - mean)) / len(series)

            acf = c_lag / (c0 + 1e-8)
            acf_values.append(acf)

        return np.array(acf_values)
