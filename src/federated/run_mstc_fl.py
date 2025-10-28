# experiments/run_mstc_fl.py
"""
MSTC-FL 主实验脚本（方案 A：原生 DataLoader 实现）
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import sys
import os
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))

from src.models.simple_ids import SimpleIDS
from src.features.adwin_scheduler_v2 import MultiClientADWIN
from src.features.adaptive_mstr import AdaptiveMSTR
from src.features.multiscale_dtw import MultiScaleDTW
from src.clustering.hierarchical_gcn import HierarchicalGCNClustering
from src.utils.unsw_nb15_loader import FederatedUNSWNB15
from src.core.early_stopping import EarlyStopping
from src.core.losses import FocalLoss
from src.federated.client_trainer import ClientTrainer
from src.federated.evaluator import ModelEvaluator
from src.federated.model_aggregator import ModelAggregator
from src.federated.ensemble_manager import EnsembleManager
from src.federated.workflow_manager import WorkflowManager
from src.aggregation.hierarchical_aggregator import HierarchicalAggregator
class MSTCFL:
    """
    MSTC-FL 系统（方案 A：完整 DataLoader 实现）

    特点：
    - 使用 PyTorch DataLoader 进行批量训练
    - 支持时序聚合和特征聚合
    - 完整的训练/评估流程
    - 三层层次聚类（Micro + Meso + Macro）
    """

    def __init__(self,
                 n_clients: int = 10,
                 n_ensemble: int = 6,
                 use_timeseries_clustering: bool = True,
                 learning_rate: float = 0.001,
                 local_epochs: int = 1,
                 batch_size: int = 64):
        """
        Args:
            n_clients: 客户端数量
            n_ensemble: 集成成员数量
            use_timeseries_clustering: 是否使用时序聚合
            learning_rate: 学习率
            local_epochs: 本地训练轮数
            batch_size: 批量大小
        """
        self.n_clients = n_clients
        self.n_ensemble = n_ensemble
        self.use_timeseries_clustering = use_timeseries_clustering
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # ========== 模型组件 ==========
        self.ensemble = None
        self.global_model = None
        self.optimizers = None
        self.schedulers = None

        # ========== 存储 ==========
        self.client_timeseries = {i: [] for i in range(n_clients)}
        self.em_history = []
        self.global_history = []
        self.round_stats = []

        # ========== 设备 ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ========== 核心组件初始化 ==========

        # 1️⃣ 客户端训练器
        self.client_trainer = ClientTrainer(
            n_clients=n_clients,
            n_ensemble=n_ensemble,
            local_epochs=local_epochs,
            batch_size=batch_size,
            device=self.device
        )

        # 2️⃣ 集成管理器
        self.ensemble_manager = EnsembleManager(
            n_clients=n_clients,
            n_ensemble=n_ensemble
        )

        # 3️⃣ 模型聚合器
        self.model_aggregator = ModelAggregator(
            n_clients=n_clients,
            n_ensemble=n_ensemble,
            device=self.device
        )

        # 4️⃣ 工作流管理器
        self.workflow_manager = WorkflowManager(
            n_clients=n_clients,
            use_timeseries_clustering=use_timeseries_clustering,
            device=self.device
        )

        # 5️⃣ 模型评估器
        self.evaluator = ModelEvaluator(device=self.device)

        # 6️⃣ 三层聚合器
        self.hierarchical_aggregator = HierarchicalAggregator(
            n_clients=n_clients
        )

        # ✅ 依赖注入：将组件注入到 workflow_manager
        self.workflow_manager.set_components(
            client_trainer=self.client_trainer,
            model_aggregator=self.model_aggregator,
            ensemble_manager=self.ensemble_manager,
            evaluator=self.evaluator
        )

        # ========== 历史记录 ==========
        self.history = {
            'micro_clusters': [],
            'meso_labels': [],
            'macro_labels': [],
            'client_accuracies': [],
            'round_stats': [],
            'drift_history': []
        }

        # 打印初始化信息
        print(f"\n✅ MSTC-FL initialized:")
        print(f"   - Clients: {self.n_clients}")
        print(f"   - Ensemble: {self.n_ensemble}")
        print(f"   - Clustering mode: {'Timeseries' if self.use_timeseries_clustering else 'Features'}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Local epochs: {self.local_epochs}")
        print(f"   - Batch size: {self.batch_size}")
        print(f"   - Device: {self.device}")

    # ========================================
    # 模型初始化
    # ========================================

    def _init_models(self, fed_data: FederatedUNSWNB15):
        """
        延迟初始化模型（仅在第一轮训练时调用）

        Args:
            input_dim: 输入特征维度
        """
        if self.ensemble is None:
            input_dim = fed_data.input_dim

            print(f"\n🔧 Initializing models...")

            # 1️⃣ IDS 模型
            self.ensemble = [
                SimpleIDS(input_dim).to(self.device)
                for _ in range(self.n_ensemble)
            ]
            self.global_model = SimpleIDS(input_dim).to(self.device)

            # 2️⃣ 优化器
            self.optimizers = [
                optim.Adam(model.parameters(), lr=self.learning_rate)
                for model in self.ensemble
            ]

            # 3️⃣ 学习率调度器
            self.schedulers = [
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=50, eta_min=1e-5
                )
                for optimizer in self.optimizers
            ]

            print(f"   ✓ IDS models: {self.n_ensemble} ensemble + 1 global")
            print(f"   ✓ Optimizers: {self.n_ensemble} Adam")
            print(f"   ✓ Schedulers: {self.n_ensemble} CosineAnnealing")

            # 4️⃣ 打印 workflow_manager 已初始化的模块
            feature_dim = self.n_clients if self.use_timeseries_clustering else 16
            self.workflow_manager.init_modules(feature_dim)

            print(f"\n✅ All modules initialized (input_dim={input_dim})")

    # ========================================
    # 辅助方法
    # ========================================

    def _extract_micro_clusters(self, micro_labels: np.ndarray) -> Dict[int, List[int]]:
        """
        将 Micro 标签数组转换为簇字典

        Args:
            micro_labels: Micro 聚类标签 [N]

        Returns:
            {cluster_id: [client_ids]}
        """
        micro_clusters = {}
        for cluster_id in np.unique(micro_labels):
            members = np.where(micro_labels == cluster_id)[0].tolist()
            micro_clusters[int(cluster_id)] = members
        return micro_clusters

    # ========================================
    # 训练流程
    # ========================================

    def train_round(self, fed_data: FederatedUNSWNB15, round_id: int) -> Dict:
        """
        训练一轮

        Args:
            fed_data: 联邦数据对象
            round_id: 当前轮次（从 0 开始）

        Returns:
            评估指标字典
        """
        print(f"\n{'=' * 70}")
        print(f"Round {round_id + 1}")
        print(f"{'=' * 70}")

        # ✅ 延迟初始化模型
        if self.ensemble is None:
            self._init_models(fed_data)

        # 1️⃣ 客户端训练（✅ 委托给 workflow_manager）
        print("\n1️⃣ Training clients...")
        client_models, client_accuracies = self.workflow_manager.train_clients(
            fed_data=fed_data,
            ensemble=self.ensemble,
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            client_timeseries=self.client_timeseries
        )

        # 2️⃣ 漂移检测
        print("\n2️⃣ Drift detection (ADWIN++)...")
        drift_results = self.workflow_manager.detect_drift(client_accuracies)

        # 3️⃣ DTW 距离计算
        print("\n3️⃣ Computing DTW distances...")
        dtw_distances = self.workflow_manager.compute_dtw_distances(
            client_timeseries=self.client_timeseries
        )

        # 4️⃣ 三层聚类
        print("\n4️⃣ Hierarchical clustering...")
        clustering_results = self.workflow_manager.perform_clustering(
            dtw_distances,
            round_idx=round_id
        )

        # ✅ 提取三层结果
        micro_labels = clustering_results['micro']['labels']
        meso_labels = clustering_results['meso']['labels']
        macro_labels = (
            clustering_results['macro']['labels']
            if clustering_results['macro'] is not None
            else meso_labels
        )

        # ✅ 转换为簇字典
        micro_clusters = self._extract_micro_clusters(micro_labels)

        # 5️⃣ 三层聚合
        print("\n5️⃣ Hierarchical aggregation...")
        cluster_representatives = self.hierarchical_aggregator.aggregate(
            client_models=client_models,
            client_accuracies=client_accuracies,
            micro_clusters=micro_clusters,
            meso_labels=meso_labels,
            macro_labels=macro_labels,
            round_id=round_id
        )

        # 6️⃣ 更新集成（✅ 委托给 workflow_manager）
        print("\n6️⃣ Updating ensemble...")
        self.workflow_manager.update_ensemble(
            cluster_representatives=cluster_representatives,
            client_accuracies=client_accuracies,
            micro_clusters=micro_clusters,
            ensemble=self.ensemble
        )

        # 7️⃣ 全局评估（✅ 委托给 workflow_manager）
        print("\n7️⃣ Global evaluation...")
        metrics = self.workflow_manager.global_evaluation(
            fed_data=fed_data,
            ensemble=self.ensemble,
            n_clients=self.n_clients,
            global_history=self.global_history,
            round_idx=round_id,
            drift_results=drift_results,
            clustering_results=clustering_results
        )

        # ✅ 记录历史
        self.history['micro_clusters'].append(micro_clusters)
        self.history['meso_labels'].append(meso_labels)
        self.history['macro_labels'].append(macro_labels)
        self.history['client_accuracies'].append(client_accuracies)
        self.history['drift_history'].append(drift_results)
        self.history['round_stats'].append({
            'round': round_id + 1,
            'global_acc': metrics['accuracy'],
            'global_f1': metrics['f1_score'],
            'global_precision': metrics['precision'],
            'global_recall': metrics['recall'],
            'n_micro_clusters': len(micro_clusters),
            'n_meso_clusters': len(np.unique(meso_labels)),
            'n_macro_clusters': len(np.unique(macro_labels)),
            'n_drifts': drift_results['n_drifts']
        })

        # ✅ 同步到原有的 round_stats（保持兼容）
        self.round_stats.append(self.history['round_stats'][-1])

        # ✅ 打印三层信息
        print(f"\n{'=' * 70}")
        print(f"Round {round_id + 1} Summary:")
        print(f"  Global Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Global F1 Score:  {metrics['f1_score']:.4f}")
        print(f"  Micro Clusters:   {len(micro_clusters)}")
        print(f"  Meso Clusters:    {len(np.unique(meso_labels))}")
        print(f"  Macro Clusters:   {len(np.unique(macro_labels))}")
        print(f"  Detected Drifts:  {drift_results['n_drifts']}")
        print(f"{'=' * 70}\n")

        return metrics

    # ========================================
    # 实验运行
    # ========================================

    def run_experiment(self, fed_data: FederatedUNSWNB15, n_rounds: int = 50) -> Dict:
        """
        运行完整实验

        Args:
            fed_data: 联邦数据对象
            n_rounds: 训练轮数

        Returns:
            实验历史字典
        """
        print(f"\n🚀 Starting MSTC-FL experiment: {n_rounds} rounds\n")

        early_stopping = EarlyStopping(patience=5, min_delta=0.01)

        for round_idx in range(n_rounds):
            # ✅ 修正：参数顺序为 (fed_data, round_id)
            self.train_round(fed_data, round_idx)

            # 早停检查
            if len(self.global_history) > 0:
                if early_stopping(self.global_history[-1]):
                    print(f"\n⏹️  Early stopping at round {round_idx + 1}")
                    print(f"   Best score: {early_stopping.best_score:.4f}")
                    break

        print(f"\n✅ Experiment completed!")

        return {
            'round_stats': self.round_stats,
            'global_history': self.global_history,
            'em_history': self.ensemble_manager.get_history(),
            'history': self.history  # ✅ 新增：返回完整历史
        }

# ==================== 主程序 ====================
if __name__ == "__main__":
    print("🚀 Starting MSTC-FL Experiment (Method A: Native DataLoader)...\n")
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]  # 项目根（…/MSTC-FL）
    DATA = ROOT / "data" / "unsw"
    train_path = str(DATA / "UNSW_NB15_training-set.csv")
    test_path = str(DATA / "UNSW_NB15_testing-set.csv")
    # 加载数据集
    fed_data = FederatedUNSWNB15(
        train_path=train_path,
        test_path=test_path,
        n_clients=10,
        alpha=0.5  # Non-IID 程度
    )

    # 创建 MSTC-FL 实例
    mstc_fl = MSTCFL(
        n_clients=10,
        n_ensemble=5,
        use_timeseries_clustering=True,  # 使用时序聚合
        learning_rate=0.001,
        local_epochs=3,
        batch_size=128
    )

    # 快速测试：3 轮
    # print(f"\n🧪 Quick test: 3 rounds...\n")
    # for round_idx in range(3):
    #     mstc_fl.train_round(fed_data, round_idx)
    #
    # print("\n✅ Quick test completed!")

    #完整训练（取消注释运行 50 轮）
    history = mstc_fl.run_experiment(fed_data, n_rounds=25)

    # 保存结果
    import pickle
    from pathlib import Path

    # ✅ 确保 results/ 目录存在
    ROOT = Path(__file__).resolve().parents[1]
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)  # ✅ 创建目录

    with open(results_dir / 'mstc_fl_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f"\n✅ Results saved to {results_dir / 'mstc_fl_history.pkl'}")
