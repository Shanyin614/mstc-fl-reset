# src/federated/ensemble_manager.py
# 完全替换 update_ensemble() 方法

"""
集成管理模块
"""
import torch
import numpy as np
from typing import List


class EnsembleManager:
    """集成成员管理器"""

    def __init__(self, n_clients: int, n_ensemble: int):
        self.n_clients = n_clients
        self.n_ensemble = n_ensemble
        self.em_history = []

    def update_ensemble(
            self,
            ensemble: List[torch.nn.Module],
            aggregated_models: List[torch.nn.Module],
            client_accuracies: List[float] = None
    ):
        """
        更新集成成员

        Args:
            ensemble: 集成模型列表 [n_ensemble]
            aggregated_models: 聚合后的簇代表模型 [n_clusters]
            client_accuracies: 客户端准确率（可选）
        """
        n_clusters = len(aggregated_models)
        n_ensemble = len(ensemble)

        print(f"  Updating {n_ensemble} ensemble members from {n_clusters} cluster models...")

        # ✅ 情况 1：聚合模型数量 >= 集成数量
        if n_clusters >= n_ensemble:
            for i in range(n_ensemble):
                ensemble[i].load_state_dict(aggregated_models[i].state_dict())
            print(f"  ✓ Updated all {n_ensemble} ensemble members")

        # ✅ 情况 2：聚合模型数量 < 集成数量（轮换更新）
        else:
            for i in range(n_clusters):
                em_idx = i % n_ensemble
                ensemble[em_idx].load_state_dict(aggregated_models[i].state_dict())
            print(f"  ✓ Updated {n_clusters} ensemble members (cyclic)")

        # ✅ 记录性能（如果提供了客户端准确率）
        if client_accuracies is not None:
            em_performances = self._compute_ensemble_performances(client_accuracies)
            self.em_history.append(em_performances)
            print(f"  EM performances: {[f'{acc:.3f}' for acc in em_performances]}")

    def _compute_ensemble_performances(self, client_accuracies: List[float]) -> List[float]:
        """计算每个集成成员的性能"""
        em_performances = []

        for em_idx in range(self.n_ensemble):
            client_ids = [i for i in range(self.n_clients) if i % self.n_ensemble == em_idx]
            em_acc = np.mean([client_accuracies[i] for i in client_ids])
            em_performances.append(em_acc)

        return em_performances

    def get_history(self) -> List[List[float]]:
        """获取集成成员历史性能"""
        return self.em_history
