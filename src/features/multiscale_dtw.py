# src/features/multiscale_dtw.py
"""
多尺度 DTW 距离融合（支持时序聚合）
"""
import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import numpy as np
from fastdtw import fastdtw
from typing import List, Dict, Optional


class MultiScaleDTW:
    """
    多尺度 DTW 距离融合

    支持两种模式：
    1. 时序聚合：从客户端时间序列直接计算 DTW
    2. 特征聚合：从 MSTR 特征计算欧氏距离
    """

    def __init__(self,
                 scales: List[int] = [5, 10, 20],
                 weights: Optional[List[float]] = None,
                 mode: str = 'timeseries'):  # ✅ 新增参数
        """
        Args:
            scales: 时间尺度列表
            weights: 尺度权重
            mode: 'timeseries' 或 'features'
        """
        self.scales = scales
        self.mode = mode

        # 初始化权重
        if weights is None:
            self.weights = np.ones(len(scales)) / len(scales)
        else:
            if len(weights) != len(scales):
                raise ValueError(f"Weights length {len(weights)} != scales length {len(scales)}")
            self.weights = np.array(weights)
            self.weights = self.weights / self.weights.sum()

        print(f"✅ MultiScaleDTW initialized: mode={mode}, scales={scales}, weights={self.weights}")

    def compute_distance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        自动检测输入类型并计算距离矩阵

        Args:
            data:
                - 时间序列历史: (T, n_ensemble)
                - 特征矩阵: (n_clients, feature_dim)

        Returns:
            distance_matrix: (n_clients, n_clients)
        """
        n_samples, n_features = data.shape

        # 根据初始化模式选择计算方法
        if self.mode == 'timeseries':
            # 强制使用时序聚合
            return self._compute_from_timeseries_history(data)
        elif self.mode == 'features':
            # 强制使用特征聚合
            return self._compute_from_features(data)
        else:
            # 自动检测（兼容旧版本）
            if n_samples > n_features:
                return self._compute_from_timeseries_history(data)
            else:
                return self._compute_from_features(data)

    def compute_distance_matrix_from_client_timeseries(self,
                                                       client_timeseries: Dict[int, np.ndarray]) -> np.ndarray:
        """
        ✅ 新增方法：从客户端时间序列字典计算 DTW 距离

        Args:
            client_timeseries: {client_id: timeseries}
                timeseries shape: (T,) 或 (T, 1)

        Returns:
            distance_matrix: (n_clients, n_clients)
        """
        n_clients = len(client_timeseries)

        # 存储各尺度的距离矩阵
        scale_matrices = []

        for scale in self.scales:
            D_scale = np.zeros((n_clients, n_clients))

            for i in range(n_clients):
                for j in range(i + 1, n_clients):
                    # 获取两个客户端的时间序列
                    ts_i = client_timeseries[i]
                    ts_j = client_timeseries[j]

                    # 确保是 1D 数组
                    if ts_i.ndim > 1:
                        ts_i = ts_i.flatten()
                    if ts_j.ndim > 1:
                        ts_j = ts_j.flatten()

                    # 截取最近 scale 个时间步
                    T_i = len(ts_i)
                    T_j = len(ts_j)

                    if T_i < scale:
                        seq_i = ts_i
                    else:
                        seq_i = ts_i[-scale:]

                    if T_j < scale:
                        seq_j = ts_j
                    else:
                        seq_j = ts_j[-scale:]

                    # 转换为 float64
                    seq_i = seq_i.astype(np.float64)
                    seq_j = seq_j.astype(np.float64)

                    # 计算 DTW 距离
                    dist, _ = fastdtw(
                        seq_i,
                        seq_j,
                        dist=lambda x, y: np.abs(x - y)
                    )

                    D_scale[i, j] = dist
                    D_scale[j, i] = dist

            scale_matrices.append(D_scale)

        # 加权融合
        D_fused = np.zeros((n_clients, n_clients))
        for w, D_scale in zip(self.weights, scale_matrices):
            D_fused += w * D_scale

        return D_fused

    def _compute_from_timeseries_history(self, em_history: np.ndarray) -> np.ndarray:
        """
        从 EM 历史计算距离（每个 EM 对应一个虚拟客户端）

        Args:
            em_history: (T, n_ensemble)

        Returns:
            distance_matrix: (n_ensemble, n_ensemble)
        """
        T, n_ensemble = em_history.shape
        scale_matrices = []

        for scale in self.scales:
            if T < scale:
                recent_history = em_history
            else:
                recent_history = em_history[-scale:]

            D_scale = np.zeros((n_ensemble, n_ensemble))

            for i in range(n_ensemble):
                for j in range(i + 1, n_ensemble):
                    seq_i = recent_history[:, i].flatten().astype(np.float64)
                    seq_j = recent_history[:, j].flatten().astype(np.float64)

                    dist, _ = fastdtw(
                        seq_i,
                        seq_j,
                        dist=lambda x, y: np.abs(x - y)
                    )

                    D_scale[i, j] = dist
                    D_scale[j, i] = dist

            scale_matrices.append(D_scale)

        D_fused = np.zeros((n_ensemble, n_ensemble))
        for w, D_scale in zip(self.weights, scale_matrices):
            D_fused += w * D_scale

        return D_fused

    def _compute_from_features(self, features: np.ndarray) -> np.ndarray:
        """
        从特征矩阵计算欧氏距离

        Args:
            features: (n_clients, feature_dim)

        Returns:
            distance_matrix: (n_clients, n_clients)
        """
        n_clients = features.shape[0]
        D = np.zeros((n_clients, n_clients))

        for i in range(n_clients):
            for j in range(i + 1, n_clients):
                dist = np.linalg.norm(features[i] - features[j])
                D[i, j] = dist
                D[j, i] = dist

        return D

    def update_weights(self, clustering_quality: List[float]):
        """根据聚类质量更新尺度权重"""
        if len(clustering_quality) != len(self.scales):
            raise ValueError(f"Quality scores length != scales length")

        scores = np.array(clustering_quality)
        exp_scores = np.exp(scores - np.max(scores))
        self.weights = exp_scores / exp_scores.sum()

        print(f"  Updated DTW weights: {self.weights}")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("🧪 Testing MultiScaleDTW with timeseries mode...\n")

    # 测试 1: 客户端时间序列字典
    print("Test 1: Client timeseries dictionary")
    n_clients = 5
    T = 20

    # 模拟每个客户端的时间序列
    client_timeseries = {}
    for i in range(n_clients):
        # 每个客户端有不同的趋势
        trend = np.linspace(0.6 + i * 0.05, 0.8 + i * 0.05, T)
        noise = np.random.randn(T) * 0.02
        client_timeseries[i] = trend + noise

    dtw = MultiScaleDTW(scales=[5, 10, 20], mode='timeseries')
    D = dtw.compute_distance_matrix_from_client_timeseries(client_timeseries)

    print(f"  Distance matrix shape: {D.shape}")
    print(f"  Sample distances:")
    print(f"    Client 0 vs 1: {D[0, 1]:.4f}")
    print(f"    Client 0 vs 4: {D[0, 4]:.4f}")
    print(f"    Client 2 vs 3: {D[2, 3]:.4f}\n")

    # 测试 2: EM 历史输入
    print("Test 2: EM history input")
    em_history = np.random.randn(30, 6)
    D = dtw.compute_distance_matrix(em_history)
    print(f"  Distance matrix shape: {D.shape}\n")

    print("✅ All tests passed!")
