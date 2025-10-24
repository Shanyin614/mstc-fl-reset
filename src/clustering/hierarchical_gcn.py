# src/clustering/hierarchical_gcn.py
"""
层次 GCN 聚类系统
三级聚类架构：Micro → Meso → Macro
"""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from typing import Dict, List, Tuple, Optional

from src.models.gcn_encoder import GCNEncoder, normalize_adjacency
from src.features.graph_builder import DynamicGraphBuilder


class HierarchicalGCNClustering:
    """
    层次 GCN 聚类系统

    三级架构:
    - Micro Level:  客户端内部数据聚类（K-Means）
    - Meso Level:   客户端间关系聚类（GCN + Spectral）
    - Macro Level:  全局稳定性分析（时间序列聚类）
    """

    def __init__(self,
                 n_clients: int,
                 feature_dim: int,
                 embedding_dim: int = 64,
                 n_micro_clusters: int = 5,
                 n_meso_clusters: int = 3,
                 device: str = 'cpu'):
        """
        Args:
            n_clients: 客户端数量
            feature_dim: 输入特征维度
            embedding_dim: GCN 嵌入维度
            n_micro_clusters: Micro 级聚类数
            n_meso_clusters: Meso 级聚类数
        """
        self.n_clients = n_clients
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.n_micro_clusters = n_micro_clusters
        self.n_meso_clusters = n_meso_clusters
        self.device = device

        # GCN 编码器
        self.gcn_encoder = GCNEncoder(
            n_features=feature_dim,
            hidden_dim=128,
            embedding_dim=embedding_dim,
            n_layers=3
        ).to(device)

        # 图构建器
        self.graph_builder = DynamicGraphBuilder(
            n_clients=n_clients,
            threshold_method='adaptive',
            k_nearest=3
        )

        # 历史记录
        self.embedding_history = []
        self.cluster_history = {
            'micro': [],
            'meso': [],
            'macro': []
        }

        print(f"\n✅ HierarchicalGCNClustering initialized:")
        print(f"   Clients: {n_clients}")
        print(f"   Feature dim: {feature_dim}")
        print(f"   Embedding dim: {embedding_dim}")
        print(f"   Micro clusters: {n_micro_clusters}")
        print(f"   Meso clusters: {n_meso_clusters}")

    def cluster_hierarchical(self,
                             client_features: np.ndarray,
                             dtw_distances: np.ndarray,
                             round_idx: int) -> Dict:
        """
        执行三级层次聚类

        Args:
            client_features: 客户端特征 [N, D]
            dtw_distances: DTW 距离矩阵 [N, N]
            round_idx: 当前轮次

        Returns:
            聚类结果字典
        """
        results = {
            'round': round_idx,
            'micro': None,
            'meso': None,
            'macro': None
        }

        # ==================== Micro Level ====================
        print(f"\n  🔬 Micro-level clustering...")
        micro_labels = self._cluster_micro(client_features)
        results['micro'] = {
            'labels': micro_labels,
            'n_clusters': len(np.unique(micro_labels))
        }
        self.cluster_history['micro'].append(micro_labels)

        # ==================== Meso Level ====================
        print(f"  🔍 Meso-level clustering (GCN)...")
        meso_labels, embeddings = self._cluster_meso(
            client_features,
            dtw_distances
        )
        results['meso'] = {
            'labels': meso_labels,
            'n_clusters': len(np.unique(meso_labels)),
            'embeddings': embeddings
        }
        self.cluster_history['meso'].append(meso_labels)
        self.embedding_history.append(embeddings)

        # ==================== Macro Level ====================
        if len(self.embedding_history) >= 5:  # 至少 5 轮
            print(f"  🌍 Macro-level clustering...")
            macro_labels = self._cluster_macro()
            results['macro'] = {
                'labels': macro_labels,
                'n_clusters': len(np.unique(macro_labels))
            }
            self.cluster_history['macro'].append(macro_labels)
        else:
            results['macro'] = {'labels': meso_labels}  # 前期使用 Meso 结果

        # 打印聚类结果
        self._print_clustering_summary(results)

        return results

    def _cluster_micro(self, features: np.ndarray) -> np.ndarray:
        """
        Micro 级聚类（客户端内部）
        使用 K-Means
        """
        kmeans = KMeans(
            n_clusters=self.n_micro_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(features)
        return labels

    def _cluster_meso(self,
                      features: np.ndarray,
                      dtw_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Meso 级聚类（客户端间）
        使用 GCN + Spectral Clustering
        """
        # 1. 构建图
        adj, graph_info = self.graph_builder.build_graph_from_dtw(dtw_distances)
        adj = adj.to(self.device)

        # 2. 归一化邻接矩阵
        adj_norm = normalize_adjacency(adj)

        # 3. GCN 编码
        features_torch = torch.FloatTensor(features).to(self.device)
        embeddings = self.gcn_encoder.get_embeddings(features_torch, adj_norm)
        embeddings_np = embeddings.cpu().numpy()

        # 4. 谱聚类
        # spectral = SpectralClustering(
        #     n_clusters=self.n_meso_clusters,
        #     affinity='nearest_neighbors',
        #     n_neighbors=3,
        #     random_state=42
        # )
        n_neighbors = max(2, min(5, embeddings_np.shape[0] - 1))
        spectral = SpectralClustering(
            n_clusters=self.n_meso_clusters,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors,
            random_state=42
        )
        labels = spectral.fit_predict(embeddings_np)

        return labels, embeddings_np

    def _cluster_macro(self) -> np.ndarray:
        """
        Macro 级聚类（全局稳定性）
        基于嵌入历史的时间序列聚类
        """
        # 取最近 10 轮的嵌入
        recent_embeddings = self.embedding_history[-10:]

        # 计算时间维度的统计特征
        temporal_features = []
        for client_id in range(self.n_clients):
            client_trajectory = [emb[client_id] for emb in recent_embeddings]
            client_trajectory = np.array(client_trajectory)  # [T, D]

            # 统计特征
            mean_traj = np.mean(client_trajectory, axis=0)
            std_traj = np.std(client_trajectory, axis=0)
            trend = client_trajectory[-1] - client_trajectory[0]  # 趋势

            feature = np.concatenate([mean_traj, std_traj, trend])
            temporal_features.append(feature)

        temporal_features = np.array(temporal_features)

        # K-Means 聚类
        kmeans = KMeans(
            n_clusters=self.n_meso_clusters,
            random_state=42
        )
        labels = kmeans.fit_predict(temporal_features)

        return labels

    def get_cluster_weights(self, cluster_labels: np.ndarray) -> np.ndarray:
        """
        计算聚类权重（用于聚合）
        权重 ∝ 1 / cluster_size
        """
        weights = np.zeros(self.n_clients)

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = (cluster_labels == cluster_id)
            cluster_size = np.sum(cluster_mask)

            # 较小的簇获得更大的权重
            weights[cluster_mask] = 1.0 / cluster_size

        # 归一化
        weights = weights / np.sum(weights)

        return weights

    def aggregate_by_clusters(self,
                              client_models: List,
                              cluster_labels: np.ndarray,
                              global_model) -> None:
        """
        基于聚类的加权聚合

        Args:
            client_models: 客户端模型列表
            cluster_labels: 聚类标签
            global_model: 全局模型（待更新）
        """
        # 计算权重
        weights = self.get_cluster_weights(cluster_labels)

        # 加权聚合
        global_dict = global_model.state_dict()

        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])

            for i, model in enumerate(client_models):
                global_dict[key] += weights[i] * model.state_dict()[key].float()

        global_model.load_state_dict(global_dict)

    def _print_clustering_summary(self, results: Dict):
        """打印聚类摘要"""
        print(f"\n  📊 Clustering Summary (Round {results['round']}):")
        print(f"     Micro: {results['micro']['n_clusters']} clusters")
        print(f"     Meso:  {results['meso']['n_clusters']} clusters")

        # 打印 Meso 簇分布
        meso_labels = results['meso']['labels']
        for i in range(self.n_meso_clusters):
            clients_in_cluster = np.where(meso_labels == i)[0]
            print(f"       Cluster {i}: {len(clients_in_cluster)} clients {list(clients_in_cluster)}")

        if results['macro'] is not None and 'n_clusters' in results['macro']:
            print(f"     Macro: {results['macro']['n_clusters']} clusters")

    def save_embeddings(self, path: str):
        """保存嵌入历史"""
        np.savez(path,
                 embeddings=np.array(self.embedding_history),
                 cluster_history=self.cluster_history)
        print(f"✅ Embeddings saved to {path}")

    def load_embeddings(self, path: str):
        """加载嵌入历史"""
        data = np.load(path, allow_pickle=True)
        self.embedding_history = list(data['embeddings'])
        self.cluster_history = data['cluster_history'].item()
        print(f"✅ Embeddings loaded from {path}")

    # ====== 补丁 1：从距离矩阵构造临时特征（给 Micro & Meso 用） ======
    def _build_features_from_distance(self, dist: np.ndarray) -> np.ndarray:
        dist = dist.copy().astype(np.float32)
        np.fill_diagonal(dist, 0.0)
        pos = dist[dist > 0]
        gamma = 1.0 if pos.size == 0 else 1.0 / (float(np.median(pos)) ** 2 + 1e-8)

        # RBF 相似度签名
        S = np.exp(-(gamma * (dist ** 2))).astype(np.float32)  # [N, N]
        np.fill_diagonal(S, 1.0)

        # ——关键：输出维度对齐 GCN 的 in_features——
        N = S.shape[1]
        D = self.feature_dim
        if D <= N:
            X = S[:, :D]  # 裁剪到 D 维（你现在是 D==N==n_clients）
        else:
            # 需要补维就先拼上简单统计，再不够再补 0
            deg = S.sum(axis=1, keepdims=True)
            dens = ((S > 0.5).sum(axis=1, keepdims=True) / N)
            X = np.concatenate([S, deg, dens], axis=1)
            if X.shape[1] < D:
                X = np.pad(X, ((0, 0), (0, D - X.shape[1])), mode='constant')
        return X.astype(np.float32)

    # ====== 补丁 2：提供旧接口 cluster(...)，适配 run_mstc_fl.py ======
    def cluster(self,
                distance_matrix: np.ndarray,
                adjacency_matrix: np.ndarray) -> Dict:
        """
        兼容旧调用：输入距离/邻接，输出包含 micro_clusters 的字典
        """
        # 1) 用距离矩阵生成临时特征（不改你现有的数据流）
        client_features = self._build_features_from_distance(distance_matrix)  # [N, *]

        # 2) Micro：KMeans
        kmeans = KMeans(n_clusters=self.n_micro_clusters, random_state=42, n_init=10)
        micro_labels = kmeans.fit_predict(client_features)

        # 3) Meso：沿用你已有的 GCN+谱聚类
        meso_labels, embeddings = self._cluster_meso(client_features, distance_matrix)

        # 4) 组装为旧格式输出（run_mstc_fl.py 只用到了 micro_clusters）
        micro_clusters: Dict[int, List[int]] = {}
        for cid in np.unique(micro_labels):
            micro_clusters[int(cid)] = np.where(micro_labels == cid)[0].astype(int).tolist()

        return {
            "micro_clusters": micro_clusters,
            "meso_labels": meso_labels,
            "embeddings": embeddings
        }


if __name__ == "__main__":
    # 测试层次聚类
    print("🧪 Testing HierarchicalGCNClustering...")

    # 模拟数据
    n_clients = 10
    feature_dim = 64

    client_features = np.random.randn(n_clients, feature_dim)
    dtw_distances = np.random.rand(n_clients, n_clients)
    dtw_distances = (dtw_distances + dtw_distances.T) / 2
    np.fill_diagonal(dtw_distances, 0)

    # 初始化聚类系统
    clustering = HierarchicalGCNClustering(
        n_clients=n_clients,
        feature_dim=feature_dim,
        embedding_dim=32,
        n_micro_clusters=3,
        n_meso_clusters=2
    )

    # 执行聚类（模拟 10 轮）
    for round_idx in range(10):
        print(f"\n{'=' * 60}")
        print(f"Round {round_idx}")
        print(f"{'=' * 60}")

        results = clustering.cluster_hierarchical(
            client_features,
            dtw_distances,
            round_idx
        )

        # 模拟特征变化
        client_features += np.random.randn(n_clients, feature_dim) * 0.1

    print("\n✅ All tests passed!")
