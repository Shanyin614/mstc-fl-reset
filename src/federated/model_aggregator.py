"""
模型聚合模块
"""
import torch
import numpy as np
from typing import Dict, List


class ModelAggregator:
    """
    模型聚合器

    负责基于聚类结果进行模型聚合，包括：
    - 加权聚合（基于客户端准确率）
    - 集群级别聚合
    - 全局模型更新

    Args:
        n_clients: 客户端数量
        n_ensemble: 集成成员数量
        device: 计算设备

    Example:
        >>> aggregator = ModelAggregator(n_clients=10, n_ensemble=5)
        >>> aggregator.aggregate_models(
        ...     global_model=global_model,
        ...     ensemble=ensemble,
        ...     clustering_results=clustering_results,
        ...     client_accuracies=client_accuracies
        ... )
    """

    def __init__(
            self,
            n_clients: int,
            n_ensemble: int,
            device: torch.device = None
    ):
        self.n_clients = n_clients
        self.n_ensemble = n_ensemble
        self.device = device if device is not None else torch.device('cpu')

    def aggregate_models(
            self,
            global_model: torch.nn.Module,
            ensemble: List[torch.nn.Module],
            clustering_results: Dict,
            client_accuracies: List[float]
    ):
        """
        基于聚类的模型聚合

        策略：
        1. 获取微簇（micro clusters）
        2. 计算每个簇的权重（基于簇内客户端平均准确率 × 簇大小）
        3. 加权聚合簇内客户端的模型参数
        4. 更新全局模型

        Args:
            global_model: 全局模型
            ensemble: 集成模型列表
            clustering_results: 聚类结果（包含 'micro_clusters' 键）
            client_accuracies: 客户端准确率列表
        """
        micro_clusters = clustering_results['micro_clusters']

        # 1. 初始化全局模型参数
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])

        total_weight = 0.0

        # 2. 逐簇聚合
        for cluster_id, client_ids in micro_clusters.items():
            # 计算簇权重
            cluster_weight = self._compute_cluster_weight(client_ids, client_accuracies)

            # 聚合簇内客户端的模型
            for client_id in client_ids:
                em_idx = client_id % self.n_ensemble
                em_dict = ensemble[em_idx].state_dict()

                # 按权重累加参数
                for key in global_dict.keys():
                    global_dict[key] += em_dict[key] * cluster_weight / len(client_ids)

            total_weight += cluster_weight

        # 3. 归一化参数
        if total_weight > 0:
            for key in global_dict.keys():
                global_dict[key] /= total_weight

        # 4. 更新全局模型
        global_model.load_state_dict(global_dict)

        print(f"  Aggregated {len(micro_clusters)} clusters (total weight: {total_weight:.4f})")

    def _compute_cluster_weight(
            self,
            client_ids: List[int],
            client_accuracies: List[float]
    ) -> float:
        """
        计算簇权重

        公式：权重 = 簇内平均准确率 × 簇大小

        优势：
        - 准确率高的簇贡献更大
        - 规模大的簇影响更强

        Args:
            client_ids: 簇内客户端ID列表
            client_accuracies: 所有客户端准确率

        Returns:
            簇权重（浮点数）
        """
        cluster_accs = [client_accuracies[i] for i in client_ids]
        avg_acc = np.mean(cluster_accs)
        cluster_size = len(client_ids)

        return avg_acc * cluster_size
