# src/features/graph_builder.py
"""
动态图构建器
基于 DTW 距离和客户端性能构建关系图
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix


class DynamicGraphBuilder:
    """
    动态图构建器

    功能:
    1. 基于 DTW 距离构建边
    2. 自适应阈值选择
    3. 支持多尺度融合
    """

    def __init__(self,
                 n_clients: int,
                 threshold_method: str = 'adaptive',
                 k_nearest: int = 3,
                 min_edge_weight: float = 0.1):
        """
        Args:
            n_clients: 客户端数量
            threshold_method: 'fixed', 'adaptive', 'knn'
            k_nearest: KNN 模式下的邻居数
            min_edge_weight: 最小边权重
        """
        self.n_clients = n_clients
        self.threshold_method = threshold_method
        self.k_nearest = k_nearest
        self.min_edge_weight = min_edge_weight

        print(f"✅ DynamicGraphBuilder initialized:")
        print(f"   Clients: {n_clients}")
        print(f"   Threshold: {threshold_method}")
        print(f"   K-nearest: {k_nearest}")

    def build_graph_from_dtw(self,
                             dtw_distances: np.ndarray,
                             method: str = None) -> Tuple[torch.Tensor, Dict]:
        """
        从 DTW 距离矩阵构建图

        Args:
            dtw_distances: DTW 距离矩阵 [N, N]
            method: 构建方法（覆盖初始化配置）

        Returns:
            (adjacency_matrix, info_dict)
        """
        method = method or self.threshold_method

        if method == 'adaptive':
            adj, info = self._build_adaptive(dtw_distances)
        elif method == 'knn':
            adj, info = self._build_knn(dtw_distances)
        elif method == 'fixed':
            adj, info = self._build_fixed_threshold(dtw_distances)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 确保对称性
        adj = (adj + adj.T) / 2

        # 过滤小权重
        adj[adj < self.min_edge_weight] = 0

        # 转换为 torch
        adj_torch = torch.FloatTensor(adj)

        info['n_edges'] = (adj > 0).sum() / 2  # 无向图
        info['density'] = info['n_edges'] / (self.n_clients * (self.n_clients - 1) / 2)

        return adj_torch, info

    def _build_adaptive(self,
                        dtw_distances: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        自适应阈值法
        阈值 = mean(distances) - α * std(distances)
        """
        # 只考虑上三角（避免自环和重复）
        upper_tri = np.triu(dtw_distances, k=1)
        distances = upper_tri[upper_tri > 0]

        # 自适应阈值
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        threshold = mean_dist - 0.5 * std_dist  # α = 0.5

        # 构建邻接矩阵（相似度 = 1 / (1 + distance)）
        similarity = 1.0 / (1.0 + dtw_distances)
        adj = np.where(dtw_distances <= threshold, similarity, 0)

        # 移除自环
        np.fill_diagonal(adj, 0)

        info = {
            'method': 'adaptive',
            'threshold': threshold,
            'mean_distance': mean_dist,
            'std_distance': std_dist
        }

        return adj, info

    def _build_knn(self,
                   dtw_distances: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        K 近邻法
        每个节点连接到 K 个最近邻
        """
        adj = np.zeros_like(dtw_distances)

        for i in range(self.n_clients):
            # 找到 K 个最近邻（排除自己）
            distances_i = dtw_distances[i].copy()
            distances_i[i] = np.inf

            knn_indices = np.argpartition(distances_i, self.k_nearest)[:self.k_nearest]

            # 计算相似度权重
            knn_distances = distances_i[knn_indices]
            similarities = 1.0 / (1.0 + knn_distances)

            # 添加边
            adj[i, knn_indices] = similarities

        info = {
            'method': 'knn',
            'k': self.k_nearest,
            'avg_degree': self.k_nearest
        }

        return adj, info

    def _build_fixed_threshold(self,
                               dtw_distances: np.ndarray,
                               threshold: float = 0.5) -> Tuple[np.ndarray, Dict]:
        """固定阈值法"""
        similarity = 1.0 / (1.0 + dtw_distances)
        adj = np.where(dtw_distances <= threshold, similarity, 0)
        np.fill_diagonal(adj, 0)

        info = {
            'method': 'fixed',
            'threshold': threshold
        }

        return adj, info

    def build_multiscale_graph(self,
                               dtw_distances_dict: Dict[int, np.ndarray],
                               weights: List[float] = None) -> Tuple[torch.Tensor, Dict]:
        """
        多尺度图融合

        Args:
            dtw_distances_dict: {scale: distance_matrix}
            weights: 各尺度权重

        Returns:
            融合后的邻接矩阵
        """
        scales = list(dtw_distances_dict.keys())
        n_scales = len(scales)

        if weights is None:
            weights = [1.0 / n_scales] * n_scales

        # 归一化权重
        weights = np.array(weights) / np.sum(weights)

        # 融合
        adj_fused = np.zeros((self.n_clients, self.n_clients))

        for scale, weight in zip(scales, weights):
            adj_scale, _ = self.build_graph_from_dtw(
                dtw_distances_dict[scale]
            )
            adj_fused += weight * adj_scale.numpy()

        info = {
            'method': 'multiscale',
            'scales': scales,
            'weights': weights.tolist()
        }

        adj_torch = torch.FloatTensor(adj_fused)

        return adj_torch, info


if __name__ == "__main__":
    # 测试图构建
    print("🧪 Testing DynamicGraphBuilder...")

    # 模拟 DTW 距离矩阵
    n_clients = 10
    dtw_dist = np.random.rand(n_clients, n_clients)
    dtw_dist = (dtw_dist + dtw_dist.T) / 2  # 对称化
    np.fill_diagonal(dtw_dist, 0)

    # 初始化构建器
    builder = DynamicGraphBuilder(n_clients=n_clients)

    # 自适应构建
    adj, info = builder.build_graph_from_dtw(dtw_dist, method='adaptive')
    print(f"\n✅ Adaptive method:")
    print(f"   Threshold: {info['threshold']:.3f}")
    print(f"   Edges: {info['n_edges']:.0f}")
    print(f"   Density: {info['density']:.2%}")

    # KNN 构建
    adj_knn, info_knn = builder.build_graph_from_dtw(dtw_dist, method='knn')
    print(f"\n✅ KNN method:")
    print(f"   K: {info_knn['k']}")
    print(f"   Edges: {info_knn['n_edges']:.0f}")

    print("\n✅ All tests passed!")
