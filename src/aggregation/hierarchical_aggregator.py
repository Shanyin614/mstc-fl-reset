# src/aggregation/hierarchical_aggregator.py
"""
三层层次聚合器
负责：Micro + Meso + Macro 三层权重计算和模型聚合
"""
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import copy


class HierarchicalAggregator:
    """
    三层层次聚合器

    职责：
    1. 计算 Macro 层稳定性权重
    2. 计算 Meso 层跨簇权重
    3. 基于三层权重聚合模型
    """

    def __init__(self, n_clients: int):
        self.n_clients = n_clients

        # 历史记录（用于 Macro 层）
        self.macro_history = []

        print(f"\n✅ HierarchicalAggregator initialized:")
        print(f"   Clients: {n_clients}")

    def aggregate(
            self,
            client_models: List[nn.Module],
            client_accuracies: List[float],
            micro_clusters: Dict[int, List[int]],
            meso_labels: np.ndarray,
            macro_labels: np.ndarray,
            round_id: int
    ) -> Dict[int, nn.Module]:
        """
        执行三层层次聚合

        Args:
            client_models: 客户端模型列表
            client_accuracies: 客户端准确率
            micro_clusters: Micro 簇 {cluster_id: [client_ids]}
            meso_labels: Meso 聚类标签
            macro_labels: Macro 聚类标签
            round_id: 当前轮次

        Returns:
            簇代表模型 {cluster_id: model}
        """
        # 1. 计算三层权重
        macro_weights = self._compute_macro_stability(macro_labels, round_id)
        meso_weights = self._compute_meso_weights(meso_labels, client_accuracies)

        # 2. 聚合每个 Micro 簇
        cluster_representatives = {}

        print(f"  Computing hierarchical weights:")

        for cluster_id, member_clients in micro_clusters.items():
            if len(member_clients) == 0:
                continue

            # 计算最终权重
            final_weight = self._compute_final_weight(
                member_clients,
                client_accuracies,
                meso_labels,
                macro_labels,
                meso_weights,
                macro_weights
            )

            # 加权聚合
            cluster_models = [client_models[i] for i in member_clients]
            cluster_weights = [final_weight / len(member_clients)] * len(member_clients)

            cluster_model = self._weighted_aggregate(cluster_models, cluster_weights)
            cluster_representatives[cluster_id] = cluster_model

        return cluster_representatives

    def _compute_final_weight(
            self,
            member_clients: List[int],
            client_accuracies: List[float],
            meso_labels: np.ndarray,
            macro_labels: np.ndarray,
            meso_weights: Dict[int, float],
            macro_weights: Dict[int, float]
    ) -> float:
        """计算簇的最终权重"""
        # Micro 层：基础权重
        cluster_accs = [client_accuracies[i] for i in member_clients]
        base_weight = np.mean(cluster_accs) * len(member_clients)

        # Meso 层：跨簇调节
        representative_client = member_clients[0]
        meso_id = int(meso_labels[representative_client])
        meso_adjustment = meso_weights.get(meso_id, 1.0)

        # Macro 层：稳定性调节
        macro_id = int(macro_labels[representative_client])
        macro_stability = macro_weights.get(macro_id, 1.0)

        # 最终权重
        final_weight = base_weight * meso_adjustment * macro_stability

        # 记录日志
        print(f"    Cluster (n={len(member_clients)}): "
              f"base={base_weight:.2f}, "
              f"meso={meso_adjustment:.2f}, "
              f"macro={macro_stability:.2f}, "
              f"final={final_weight:.2f}")

        return final_weight

    def _compute_macro_stability(
            self,
            macro_labels: np.ndarray,
            round_id: int
    ) -> Dict[int, float]:
        """
        计算 Macro 层稳定性权重
        基于时序一致性（Jaccard 相似度）
        """
        # 前 5 轮：积累历史
        if len(self.macro_history) < 5:
            self.macro_history.append(macro_labels.copy())
            return {int(i): 1.0 for i in np.unique(macro_labels)}

        # 计算稳定性
        stability = {}
        recent_history = self.macro_history[-5:]

        for macro_id in np.unique(macro_labels):
            current_members = set(np.where(macro_labels == macro_id)[0].tolist())

            # 与历史簇的 Jaccard 相似度
            jaccard_scores = []
            for hist_labels in recent_history:
                best_jaccard = 0.0
                for hist_id in np.unique(hist_labels):
                    hist_members = set(np.where(hist_labels == hist_id)[0].tolist())
                    intersection = len(current_members & hist_members)
                    union = len(current_members | hist_members)
                    jaccard = intersection / union if union > 0 else 0.0
                    best_jaccard = max(best_jaccard, jaccard)
                jaccard_scores.append(best_jaccard)

            # 稳定性权重 [0.5, 1.5]
            stability_score = np.mean(jaccard_scores)
            stability[int(macro_id)] = 0.5 + 1.0 * stability_score

        # 更新历史
        self.macro_history.append(macro_labels.copy())
        if len(self.macro_history) > 10:
            self.macro_history.pop(0)

        return stability

    def _compute_meso_weights(
            self,
            meso_labels: np.ndarray,
            client_accuracies: List[float]
    ) -> Dict[int, float]:
        """
        计算 Meso 层跨簇权重
        基于簇内方差（方差小 → 高权重）
        """
        weights = {}
        client_accs_array = np.array(client_accuracies)

        for meso_id in np.unique(meso_labels):
            cluster_mask = (meso_labels == meso_id)
            cluster_accs = client_accs_array[cluster_mask]
            cluster_variance = np.var(cluster_accs)

            # 指数衰减：exp(-α * variance)
            alpha = 5.0
            weight = np.exp(-alpha * cluster_variance)

            # 归一化到 [0.7, 1.3]
            weights[int(meso_id)] = 0.7 + 0.6 * weight

        return weights

    def _weighted_aggregate(
            self,
            models: List[nn.Module],
            weights: List[float]
    ) -> nn.Module:
        """加权聚合模型"""
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        aggregated_model = copy.deepcopy(models[0])
        state_dict = aggregated_model.state_dict()

        for key in state_dict.keys():
            state_dict[key] = torch.zeros_like(state_dict[key])
            for model, weight in zip(models, normalized_weights):
                state_dict[key] += weight * model.state_dict()[key].float()

        aggregated_model.load_state_dict(state_dict)
        return aggregated_model
