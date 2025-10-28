"""
联邦学习工作流管理模块
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import torch
from typing import Dict, List, Tuple
from ..features.adaptive_mstr import AdaptiveMSTR
from ..features.multiscale_dtw import MultiScaleDTW
from ..features.adwin_scheduler_v2 import MultiClientADWIN
from ..clustering.hierarchical_gcn import HierarchicalGCNClustering


class WorkflowManager:
    """
    MSTC-FL 工作流管理器

    负责管理整个联邦学习的辅助流程，包括：
    - 漂移检测（ADWIN++）
    - 特征提取（MSTR）
    - 距离计算（Multi-Scale DTW）
    - 三层层次聚类（Micro + Meso + Macro）
    - 客户端训练管理
    - 集成模型更新
    - 全局模型评估

    Args:
        n_clients: 客户端数量
        use_timeseries_clustering: 是否使用时间序列聚类模式
        device: 计算设备

    Example:
        >>> workflow_mgr = WorkflowManager(n_clients=10, device=torch.device('cuda'))
        >>> workflow_mgr.set_components(
        ...     client_trainer=trainer,
        ...     model_aggregator=aggregator,
        ...     ensemble_manager=em_manager,
        ...     evaluator=evaluator
        ... )
    """

    def __init__(
            self,
            n_clients: int,
            use_timeseries_clustering: bool = True,
            device: torch.device = None
    ):
        self.n_clients = n_clients
        self.use_timeseries_clustering = use_timeseries_clustering
        self.device = device if device is not None else torch.device('cpu')

        # ✅ 依赖注入容器（延迟设置）
        self.client_trainer = None
        self.model_aggregator = None
        self.ensemble_manager = None
        self.evaluator = None

        # ========== 立即初始化的核心模块 ==========

        # 1️⃣ ADWIN++ 漂移检测器
        self.multi_client_adwin = MultiClientADWIN(
            n_clients=n_clients,
            delta=0.05,
            min_window=5,
            max_window=100,
            sensitivity=1.0
        )

        # 2️⃣ Multi-Scale DTW 距离计算器
        self.multiscale_dtw = MultiScaleDTW(
            scales=[5, 10, 20],
            weights=[0.5, 0.3, 0.2],
            mode='timeseries' if use_timeseries_clustering else 'features'
        )

        # 3️⃣ 层次 GCN 聚类器
        feature_dim = n_clients if use_timeseries_clustering else 16
        self.hierarchical_clustering = HierarchicalGCNClustering(
            n_clients=n_clients,
            feature_dim=feature_dim,
            embedding_dim=64,
            n_micro_clusters=5,
            n_meso_clusters=3,
            device=self.device
        )

        # 4️⃣ MSTR 特征提取器（仅特征模式需要）
        self.mstr_extractor = None
        if not use_timeseries_clustering:
            self.mstr_extractor = AdaptiveMSTR(
                base_scales={'micro': 5, 'meso': 15, 'macro': 40},
                delta=0.002,
                enable_adaptive=True
            )

    def set_components(
            self,
            client_trainer,
            model_aggregator,
            ensemble_manager,
            evaluator
    ):
        """
        设置依赖组件（依赖注入）

        Args:
            client_trainer: 客户端训练器
            model_aggregator: 模型聚合器
            ensemble_manager: 集成管理器
            evaluator: 模型评估器
        """
        self.client_trainer = client_trainer
        self.model_aggregator = model_aggregator
        self.ensemble_manager = ensemble_manager
        self.evaluator = evaluator

    def init_modules(self, feature_dim: int):
        """
        打印初始化信息

        Args:
            feature_dim: 特征维度（仅用于打印）
        """
        print(f"   ✓ ADWIN++ detectors: {self.n_clients} instances")
        print(f"   ✓ MultiScaleDTW: 3 scales")

        if not self.use_timeseries_clustering:
            print(f"   ✓ MSTR feature extractor: {feature_dim} dims")
        else:
            print(f"   ✓ MSTR feature extractor: Disabled (using timeseries)")

        print(f"   ✓ Hierarchical GCN clustering")

    # ========================================
    # 1️⃣ 漂移检测
    # ========================================

    def detect_drift(self, client_accuracies: List[float]) -> Dict:
        """
        ADWIN++ 漂移检测

        Args:
            client_accuracies: 客户端准确率列表 [N]

        Returns:
            漂移结果字典
            {
                'drift_clients': List[int],
                'n_drifts': int,
                'drift_dict': Dict[int, bool]
            }
        """
        # 调用 ADWIN++ 检测
        result = self.multi_client_adwin.update(client_accuracies)

        # 构建漂移字典
        drift_dict = {i: False for i in range(self.n_clients)}
        for client_id in result['drift_clients']:
            drift_dict[client_id] = True

        n_drifts = result['n_drifts']

        # 打印检测结果
        print(f"  Detected {n_drifts}/{self.n_clients} clients with drift")
        if n_drifts > 0:
            print(f"  Drifted clients: {result['drift_clients']}")

        return {
            'drift_clients': result['drift_clients'],
            'n_drifts': n_drifts,
            'drift_dict': drift_dict
        }

    # ========================================
    # 2️⃣ DTW 距离计算
    # ========================================

    def compute_dtw_distances(
            self,
            client_timeseries: Dict[int, List[float]] = None,
            em_history: List[List[float]] = None
    ) -> np.ndarray:
        """
        计算 DTW 距离矩阵

        Args:
            client_timeseries: 客户端时间序列 {client_id: [acc_1, acc_2, ...]}
            em_history: 集成成员历史 [n_rounds, n_ensemble]

        Returns:
            DTW 距离矩阵 [n_clients, n_clients]
        """
        if self.use_timeseries_clustering:
            # 时间序列模式：直接计算时序 DTW
            client_timeseries_np = {
                i: np.array(ts) for i, ts in client_timeseries.items()
            }
            dtw_distances = self.multiscale_dtw.compute_distance_matrix_from_client_timeseries(
                client_timeseries_np
            )

            print(f"  DTW matrix: {dtw_distances.shape}")
            if dtw_distances[dtw_distances > 0].size > 0:
                print(f"  Distance range: [{dtw_distances[dtw_distances > 0].min():.4f}, "
                      f"{dtw_distances.max():.4f}]")

        else:
            # 特征模式：先提取 MSTR 特征，再计算 DTW
            mstr_features = self._extract_mstr_features(em_history)
            dtw_distances = self.multiscale_dtw.compute_distance_matrix(mstr_features)
            print(f"  DTW matrix: {dtw_distances.shape}")

        return dtw_distances

    def _extract_mstr_features(self, em_history: List[List[float]]) -> np.ndarray:
        """
        MSTR 特征提取（内部方法）

        Args:
            em_history: 集成成员历史 [n_rounds, n_ensemble]

        Returns:
            MSTR 特征矩阵 [n_clients, feature_dim]
        """
        em_history_array = np.array(em_history)

        if len(em_history_array) == 0:
            print(f"  MSTR features: ({self.n_clients}, 16) (random initialization)")
            return np.random.randn(self.n_clients, 16)

        features_list = []
        n_ensemble = em_history_array.shape[1]

        for i in range(self.n_clients):
            em_idx = i % n_ensemble
            client_em_history = em_history_array[:, em_idx].reshape(-1, 1)

            if len(client_em_history) < 5:
                client_features = np.random.randn(16)
            else:
                current_variance = np.var(client_em_history[-5:])
                client_features = self.mstr_extractor.extract_features_adaptive(
                    client_em_history,
                    current_variance
                )

            features_list.append(client_features)

        features = np.array(features_list)
        print(f"  MSTR features: {features.shape}")

        return features

    # ========================================
    # 3️⃣ 三层层次聚类
    # ========================================

    def perform_clustering(
            self,
            dtw_distances: np.ndarray,
            round_idx: int = 0
    ) -> Dict:
        """
        执行三层层次聚类

        Args:
            dtw_distances: DTW 距离矩阵 [N, N]
            round_idx: 当前轮次（用于 Macro 层时序分析）

        Returns:
            包含 micro, meso, macro 三层结果的字典
            {
                'round': int,
                'micro': {'labels': np.ndarray, 'n_clusters': int},
                'meso': {'labels': np.ndarray, 'n_clusters': int, 'embeddings': np.ndarray},
                'macro': {'labels': np.ndarray, 'n_clusters': int} or None
            }
        """
        # 从距离矩阵构建临时特征
        client_features = self.hierarchical_clustering._build_features_from_distance(
            dtw_distances
        )

        # 调用三层聚类方法
        clustering_results = self.hierarchical_clustering.cluster_hierarchical(
            client_features=client_features,
            dtw_distances=dtw_distances,
            round_idx=round_idx
        )

        # 打印三层结果
        print(f"\n  📊 Three-Level Clustering (Round {round_idx}):")
        print(f"     Micro:  {clustering_results['micro']['n_clusters']} clusters")
        print(f"     Meso:   {clustering_results['meso']['n_clusters']} clusters")

        if clustering_results['macro'] is not None and 'n_clusters' in clustering_results['macro']:
            print(f"     Macro:  {clustering_results['macro']['n_clusters']} clusters")
        else:
            print(f"     Macro:  Not computed yet (need ≥5 rounds)")

        # 打印 Micro 簇分布
        micro_labels = clustering_results['micro']['labels']
        for cluster_id in range(clustering_results['micro']['n_clusters']):
            members = np.where(micro_labels == cluster_id)[0]
            print(f"       Micro Cluster {cluster_id}: {len(members)} clients {list(members)}")

        return clustering_results

    # ========================================
    # 4️⃣ 客户端训练
    # ========================================

    def train_clients(
            self,
            fed_data,
            ensemble: List,
            optimizers: List,
            schedulers: List,
            client_timeseries: Dict
    ) -> Tuple[List, List[float]]:
        """
        训练所有客户端

        Args:
            fed_data: 联邦数据对象
            ensemble: 集成模型列表
            optimizers: 优化器列表
            client_timeseries: 客户端时间序列历史

        Returns:
            (client_models, client_accuracies)
        """
        # 委托给 ClientTrainer
        client_results = self.client_trainer.train_clients(
            fed_data=fed_data,
            ensemble=ensemble,
            optimizers=optimizers,
            schedulers = schedulers
        )

        client_models = client_results['client_models']
        client_accuracies = client_results['client_accuracies']

        # 更新时间序列历史
        for client_id, acc in enumerate(client_accuracies):
            client_timeseries[client_id].append(acc)

        # 打印客户端性能
        print(f"  Client accuracies:")
        for i, acc in enumerate(client_accuracies):
            print(f"    Client {i}: {acc:.4f}")

        return client_models, client_accuracies

    # ========================================
    # 5️⃣ 集成模型更新
    # ========================================

    def update_ensemble(
            self,
            cluster_representatives: Dict,
            client_accuracies: List[float],
            micro_clusters: Dict,
            ensemble: List
    ):
        """
        更新集成模型

        Args:
            cluster_representatives: 簇代表模型 {cluster_id: model}
            client_accuracies: 客户端准确率 [N]
            micro_clusters: Micro 簇字典 {cluster_id: [client_ids]}
            ensemble: 集成模型列表
        """
        # 聚合簇代表
        aggregated_models = list(cluster_representatives.values())

        print(f"  Updating ensemble from {len(aggregated_models)} cluster representatives...")

        # 更新集成
        self.ensemble_manager.update_ensemble(
            ensemble=ensemble,
            aggregated_models=aggregated_models,
            client_accuracies=client_accuracies
        )

        print(f"  Updated {len(ensemble)} ensemble members")

    # ========================================
    # 6️⃣ 全局模型评估
    # ========================================

    # src/federated/workflow_manager.py
    # 找到 global_evaluation() 方法（约 Line 410-470），完全替换为：

    def global_evaluation(
            self,
            fed_data,
            ensemble: List,
            n_clients: int,
            global_history: List,
            round_idx: int,
            drift_results: Dict,
            clustering_results: Dict
    ) -> Dict:
        """
        全局评估（集成投票版本）

        Args:
            fed_data: 联邦数据对象
            ensemble: 集成模型列表
            n_clients: 客户端数量
            global_history: 全局历史记录
            round_idx: 当前轮次
            drift_results: 漂移检测结果
            clustering_results: 聚类结果

        Returns:
            评估指标字典
            {
                'round': int,
                'accuracy': float,
                'f1_score': float,
                'precision': float,
                'recall': float,
                'n_drifts': int,
                'n_clusters': int
            }
        """
        # ✅ 使用整个测试集进行评估
        test_loader = fed_data.get_test_loader(batch_size=512)

        y_true_all = []
        y_pred_all = []

        # 遍历测试集批次
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            # 集成投票
            predictions = []
            for model in ensemble:
                model.eval()
                with torch.no_grad():
                    logits = model(batch_X)
                    pred = torch.argmax(logits, dim=1)
                    predictions.append(pred)

            # 多数投票
            stacked_preds = torch.stack(predictions)  # (n_ensemble, batch_size)
            final_pred = torch.mode(stacked_preds, dim=0)[0]  # 多数投票

            y_true_all.append(batch_y.cpu().numpy())
            y_pred_all.append(final_pred.cpu().numpy())

        # 合并所有批次
        y_true_all = np.concatenate(y_true_all)
        y_pred_all = np.concatenate(y_pred_all)

        # ✅ 使用现有的 evaluator.evaluate_binary()
        metrics = self.evaluator.evaluate_binary(
            y_true=y_true_all,
            y_pred=y_pred_all
        )

        # ✅ 补充轮次信息
        metrics['round'] = round_idx
        metrics['n_drifts'] = drift_results.get('n_drifts', 0)

        # ✅ 修复聚类结果提取
        if 'micro' in clustering_results and 'labels' in clustering_results['micro']:
            metrics['n_clusters'] = len(np.unique(clustering_results['micro']['labels']))
        else:
            metrics['n_clusters'] = 0

        # 记录历史
        global_history.append(metrics['f1_score'])

        # 打印评估结果
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")

        return metrics
