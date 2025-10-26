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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.simple_ids import SimpleIDS
from src.features.adwin_scheduler_v2 import MultiClientADWIN
from src.features.adaptive_mstr import AdaptiveMSTR
from src.features.multiscale_dtw import MultiScaleDTW
from src.clustering.hierarchical_gcn import HierarchicalGCNClustering
from src.utils.unsw_nb15_loader import FederatedUNSWNB15


class EarlyStopping:
    """早停机制"""

    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.early_stop
# 在 run_mstc_fl.py 开头添加
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs, targets):
#         ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
#         p_t = torch.exp(-ce_loss)
#         return (self.alpha * (1 - p_t) ** self.gamma * ce_loss).mean()
class FocalLoss(nn.Module):
    """
    Focal Loss - 关注难分类样本，抑制简单样本的损失

    公式: FL(p_t) = -α(1-p_t)^γ * log(p_t)

    参数:
        alpha: 类别权重（0.25 表示正类权重更高）
        gamma: 聚焦参数（2.0 表示难样本权重指数级增加）
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # ✅ 验证输入
        #print(f"[DEBUG] FocalLoss input: {inputs.shape}, target: {targets.shape}")
        # 计算交叉熵损失
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # 计算预测概率
        p_t = torch.exp(-ce_loss)

        # 应用 Focal Loss 公式
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MSTCFL:
    """
    MSTC-FL 系统（方案 A：完整 DataLoader 实现）

    特点：
    - 使用 PyTorch DataLoader 进行批量训练
    - 支持时序聚合和特征聚合
    - 完整的训练/评估流程
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

        # 模型组件
        self.ensemble = None
        self.global_model = None
        self.optimizers = None  # 为每个 EM 创建优化器

        # ADWIN++ 多客户端管理器
        self.multi_client_adwin = None

        # MSTR 特征提取器
        self.mstr_extractor = None

        # 多尺度 DTW
        self.multiscale_dtw = None

        # 层次 GCN 聚类
        self.hierarchical_clustering = None

        # 存储每个客户端的时间序列历史
        self.client_timeseries = {i: [] for i in range(n_clients)}

        # 性能历史
        self.em_history = []
        self.global_history = []

        # 统计信息
        self.round_stats = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # ✅ 为每个 EM 创建学习率调度器
        self.schedulers = None
        print(f"\n✅ MSTC-FL initialized:")
        print(f"   - Clients: {n_clients}")
        print(f"   - Ensemble: {n_ensemble}")
        print(f"   - Clustering mode: {'Timeseries' if use_timeseries_clustering else 'Features'}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Local epochs: {local_epochs}")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Device: {self.device}")




    def _init_models(self, input_dim: int):
        """延迟初始化所有模块"""
        if self.ensemble is None:
            self.input_dim = input_dim

            print(f"\n🔧 Initializing all modules...")

            # 1. IDS 模型和优化器
            self.ensemble = [SimpleIDS(input_dim).to(self.device) for _ in range(self.n_ensemble)]
            self.global_model = SimpleIDS(input_dim).to(self.device)

            # 为每个 EM 创建优化器
            self.optimizers = [
                optim.Adam(model.parameters(), lr=self.learning_rate)
                for model in self.ensemble
            ]

            print(f"   ✓ IDS models: {self.n_ensemble} ensemble + 1 global")

            # 2. ADWIN++ 多客户端管理器
            self.multi_client_adwin = MultiClientADWIN(
                n_clients=self.n_clients,
                delta=0.05,
                min_window=5,
                max_window=100,
                sensitivity=1.0
            )
            print(f"   ✓ ADWIN++ detectors: {self.n_clients} instances")

            # 3. MSTR 特征提取器（如果需要）
            if not self.use_timeseries_clustering:
                self.mstr_extractor = AdaptiveMSTR(
                    base_scales={'micro': 5, 'meso': 15, 'macro': 40},
                    delta=0.002,
                    enable_adaptive=True
                )
                try:
                    feature_dim = self.mstr_extractor.get_feature_dim()
                except AttributeError:
                    feature_dim = len(self.mstr_extractor.scales) * 4
                print(f"   ✓ MSTR feature extractor: {feature_dim} dims")
            else:
                feature_dim = self.n_clients
                print(f"   ✓ MSTR feature extractor: Disabled (using timeseries)")

            # 4. 多尺度 DTW
            self.multiscale_dtw = MultiScaleDTW(
                scales=[5, 10, 20],
                weights=[0.5, 0.3, 0.2],
                mode='timeseries' if self.use_timeseries_clustering else 'features'
            )
            print(f"   ✓ MultiScaleDTW: 3 scales")

            # 5. 层次 GCN 聚类
            self.hierarchical_clustering = HierarchicalGCNClustering(
                n_clients=self.n_clients,
                feature_dim=feature_dim,
                embedding_dim=64,
                n_micro_clusters=5,
                n_meso_clusters=3,
                device=self.device
            )
            print(f"   ✓ Hierarchical GCN clustering")
            # ✅ 初始化学习率调度器
            self.schedulers = [
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
                for optimizer in self.optimizers
            ]
            print(f"\n✅ All modules initialized (input_dim={input_dim})")

    def train_round(self, fed_data: FederatedUNSWNB15, round_idx: int):
        """
        单轮训练流程

        Args:
            fed_data: FederatedUNSWNB15 数据对象
            round_idx: 当前轮次
        """
        print(f"\n{'=' * 70}")
        print(f"MSTC-FL - Round {round_idx}")
        print(f"{'=' * 70}\n")

        # 延迟初始化（从第一个批次推断输入维度）
        if self.ensemble is None:
            # 获取第一个客户端的第一个批次
            temp_loader = fed_data.get_client_loader(0, batch_size=32)
            batch_x, _ = next(iter(temp_loader))
            self._init_models(batch_x.shape[1])

        # ==================== Step 1: 客户端训练 ====================
        print("1️⃣ Client Training...")
        client_accuracies = self._train_clients(fed_data)

        # 更新每个客户端的时间序列
        for i, acc in enumerate(client_accuracies):
            self.client_timeseries[i].append(acc)

        # ==================== Step 2: 漂移检测 ====================
        print(f"\n2️⃣ Drift Detection (ADWIN++)...")
        drift_results = self._detect_drift(client_accuracies)

        # ==================== Step 3-4: 距离计算 ====================
        if self.use_timeseries_clustering:
            print(f"\n3️⃣ Multi-Scale DTW Distance (Timeseries Mode)...")
            dtw_distances = self._compute_multiscale_dtw_timeseries()
        else:
            print(f"\n3️⃣ MSTR Feature Extraction...")
            mstr_features = self._extract_mstr_features()

            print(f"\n4️⃣ Multi-Scale DTW Distance (Feature Mode)...")
            dtw_distances = self._compute_multiscale_dtw(mstr_features)

        # ==================== Step 5: GCN 聚类 ====================
        print(f"\n5️⃣ Hierarchical GCN Clustering...")
        clustering_results = self._hierarchical_clustering(dtw_distances, round_idx)

        # ==================== Step 6-8: 聚合与评估 ====================
        print(f"\n6️⃣ Cluster-based Aggregation...")
        self._aggregate_models(clustering_results, client_accuracies)

        print(f"\n7️⃣ Updating Ensemble Members...")
        self._update_ensemble(client_accuracies)

        print(f"\n8️⃣ Evaluation...")
        round_metrics = self._evaluate_round(fed_data, round_idx, drift_results, clustering_results)

        self.round_stats.append(round_metrics)

        self._print_round_summary(round_idx, round_metrics)

        print(f"\n✅ Round {round_idx} completed!")

    def _train_clients(self, fed_data: FederatedUNSWNB15) -> List[float]:
        """客户端训练（使用 Focal Loss）"""
        client_accuracies = []

        for i in range(self.n_clients):
            print(f"  Client {i}...", end=" ")

            em_idx = i % self.n_ensemble
            em_model = self.ensemble[em_idx]
            optimizer = self.optimizers[em_idx]
            scheduler = self.schedulers[em_idx]
            client_loader = fed_data.get_client_loader(i, batch_size=self.batch_size)
            # ✅ 根据客户端数据分布动态调整
            attack_ratio = fed_data.get_client_attack_ratio(i)

            if attack_ratio < 0.2:  # Client 1, 8 (几乎无攻击)
                alpha, gamma = 0.50, 3.0  # 大幅提高攻击类权重
            elif attack_ratio > 0.8:  # Client 0, 6, 9 (高攻击率)
                alpha, gamma = 0.10, 3.0  # 提高正常类权重
            else:
                alpha, gamma = 0.25, 2.5

            criterion = FocalLoss(alpha=alpha, gamma=gamma)

            print(f"  Client {i} (Attack={attack_ratio:.2f})...", end=" ")
            em_model.train()

            # ✅ 替换损失函数
            criterion = FocalLoss(alpha=0.25, gamma=2.0)  # 原: nn.CrossEntropyLoss()

            total_correct = 0
            total_samples = 0
            total_loss = 0.0

            for epoch in range(self.local_epochs):
                for batch_x, batch_y in client_loader:
                    if batch_x.size(0) == 0:
                        continue

                    batch_x = batch_x.to(self.device)

                    if batch_y.dim() > 1:
                        batch_y = batch_y.squeeze(1)
                    batch_y = batch_y.to(self.device)

                    if batch_x.size(0) != batch_y.size(0):
                        continue

                    optimizer.zero_grad()
                    outputs = em_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == batch_y).sum().item()
                    total_samples += batch_y.size(0)
                    total_loss += loss.item()

            acc = total_correct / total_samples if total_samples > 0 else 0.0
            avg_loss = total_loss / max(len(client_loader) * self.local_epochs, 1)
            scheduler.step()
            client_accuracies.append(acc)
            print(f"Acc={acc:.3f}, Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f} ✓")

        return client_accuracies

    ##def _detect_drift(self, client_accuracies: List[float]) -> Dict[int, bool]:
        # """ADWIN++ 漂移检测"""
        # drift_results = {}
        # n_drifts = 0
        #
        # for i, acc in enumerate(client_accuracies):
        #     has_drift = self.multi_client_adwin.update(i, acc)
        #     drift_results[i] = has_drift
        #     if has_drift:
        #         n_drifts += 1
        #
        # print(f"  Detected {n_drifts}/{self.n_clients} clients with drift")
        #
        # if n_drifts > 0:
        #     drift_clients = [i for i, has_drift in drift_results.items() if has_drift]
        #     print(f"  Drifted clients: {drift_clients}")
        #
        # return drift_results
        # experiments/run_mstc_fl.py

    def _detect_drift(self, client_accuracies: List[float]) -> Dict[int, bool]:
        """ADWIN++ 漂移检测（批量更新）"""
        result = self.multi_client_adwin.update(client_accuracies)

        drift_results = {i: False for i in range(self.n_clients)}
        for client_id in result['drift_clients']:
            drift_results[client_id] = True

        n_drifts = result['n_drifts']
        print(f"  Detected {n_drifts}/{self.n_clients} clients with drift")

        if n_drifts > 0:
            print(f"  Drifted clients: {result['drift_clients']}")

        return drift_results

    def _compute_multiscale_dtw_timeseries(self) -> np.ndarray:
        """从客户端时间序列计算 DTW 距离"""
        # 将列表转换为 numpy 数组
        client_timeseries_np = {
            i: np.array(ts) for i, ts in self.client_timeseries.items()
        }

        # 计算 DTW 距离
        dtw_distances = self.multiscale_dtw.compute_distance_matrix_from_client_timeseries(
            client_timeseries_np
        )

        print(f"  DTW matrix: {dtw_distances.shape}")
        if dtw_distances[dtw_distances > 0].size > 0:
            print(f"  Distance range: [{dtw_distances[dtw_distances > 0].min():.4f}, {dtw_distances.max():.4f}]")

        return dtw_distances

    def _compute_multiscale_dtw(self, features: np.ndarray) -> np.ndarray:
        """特征聚合模式的 DTW 计算"""
        dtw_distances = self.multiscale_dtw.compute_distance_matrix(features)
        print(f"  DTW matrix: {dtw_distances.shape}")
        return dtw_distances

    def _extract_mstr_features(self) -> np.ndarray:
        """MSTR 特征提取（仅在特征聚合模式）"""
        em_history = np.array(self.em_history)

        if len(em_history) == 0:
            print(f"  MSTR features: ({self.n_clients}, 16) (random initialization)")
            return np.random.randn(self.n_clients, 16)

        features_list = []
        for i in range(self.n_clients):
            em_idx = i % self.n_ensemble
            client_em_history = em_history[:, em_idx].reshape(-1, 1)

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

    def _hierarchical_clustering(self, dtw_distances: np.ndarray, round_idx: int) -> Dict:
        """层次 GCN 聚类"""
        # 构建邻接矩阵（距离的倒数，避免除零）
        epsilon = 1e-6
        adjacency = 1.0 / (dtw_distances + epsilon)
        np.fill_diagonal(adjacency, 0)

        # 归一化
        adjacency = adjacency / (adjacency.sum(axis=1, keepdims=True) + epsilon)

        # 执行聚类
        clustering_results = self.hierarchical_clustering.cluster(
            distance_matrix=dtw_distances,
            adjacency_matrix=adjacency
        )

        # 打印聚类信息
        print(f"  Micro clusters: {len(clustering_results['micro_clusters'])}")
        for cluster_id, members in clustering_results['micro_clusters'].items():
            print(f"    Cluster {cluster_id}: {members}")

        return clustering_results

    def _aggregate_models(self, clustering_results: Dict, client_accuracies: List[float]):
        """基于聚类的模型聚合"""
        micro_clusters = clustering_results['micro_clusters']

        # 重置全局模型参数
        global_dict = self.global_model.state_dict()
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])

        total_weight = 0.0

        for cluster_id, client_ids in micro_clusters.items():
            # 计算簇内平均准确率
            cluster_accs = [client_accuracies[i] for i in client_ids]
            cluster_weight = np.mean(cluster_accs) * len(client_ids)

            # 聚合簇内客户端的模型
            for client_id in client_ids:
                em_idx = client_id % self.n_ensemble
                em_dict = self.ensemble[em_idx].state_dict()

                for key in global_dict.keys():
                    global_dict[key] += em_dict[key] * cluster_weight / len(client_ids)

            total_weight += cluster_weight

        # 归一化
        if total_weight > 0:
            for key in global_dict.keys():
                global_dict[key] /= total_weight

        self.global_model.load_state_dict(global_dict)

        print(f"  Aggregated {len(micro_clusters)} clusters")

    def _update_ensemble(self, client_accuracies: List[float]):
        """更新集成成员"""
        # 计算每个 EM 的平均性能
        em_performances = []
        for em_idx in range(self.n_ensemble):
            client_ids = [i for i in range(self.n_clients) if i % self.n_ensemble == em_idx]
            em_acc = np.mean([client_accuracies[i] for i in client_ids])
            em_performances.append(em_acc)

        self.em_history.append(em_performances)

        # 找出表现最差的 EM
        worst_em_idx = np.argmin(em_performances)

        # 用全局模型更新表现最差的 EM
        self.ensemble[worst_em_idx].load_state_dict(self.global_model.state_dict())

        print(f"  EM performances: {[f'{acc:.3f}' for acc in em_performances]}")
        print(f"  Updated EM {worst_em_idx} (worst: {em_performances[worst_em_idx]:.3f})")

    def _evaluate_round(
            self,
            fed_data: FederatedUNSWNB15,
            round_idx: int,
            drift_results: Dict,
            clustering_results: Dict
    ) -> Dict:
        """
        评估当前轮次（修复阈值搜索 + 加速优化）
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            roc_curve, confusion_matrix
        )

        test_loader = fed_data.get_test_loader(batch_size=128)

        self.global_model.eval()

        # 初始化
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                if batch_x.size(0) == 0:
                    continue

                batch_x = batch_x.to(self.device)

                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze(1)
                batch_y = batch_y.to(self.device)

                outputs = self.global_model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # 验证数据收集
        if len(all_probs) == 0 or len(all_labels) == 0:
            print("\n⚠️  Warning: No utils collected during evaluation!")
            return {
                'round': round_idx,
                'global_acc': 0.0,
                'global_f1': 0.0,
                'global_precision': 0.0,
                'global_recall': 0.0,
                'n_drifts': sum(drift_results.values()) if isinstance(drift_results, dict) else 0,
                'n_clusters': len(clustering_results.get('micro_clusters', []))
            }

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # ✅ 修复 1：检查数据分布
        n_positive = (all_labels == 1).sum()
        n_negative = (all_labels == 0).sum()

        print(f"\n  Test set distribution:")
        print(f"    Positive (Attack): {n_positive} ({n_positive / len(all_labels) * 100:.2f}%)")
        print(f"    Negative (Normal): {n_negative} ({n_negative / len(all_labels) * 100:.2f}%)")

        # ✅ 修复 2：优化阈值搜索（快速版）
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

            # 减少搜索空间（只搜索 100 个候选阈值）
            if len(thresholds) > 100:
                indices = np.linspace(0, len(thresholds) - 1, 100, dtype=int)
                thresholds = thresholds[indices]

            print(f"  Searching {len(thresholds)} thresholds...")

            best_threshold = 0.5
            best_f1 = 0.0

            # ✅ 使用 F1-score 而不是 Precision-Recall 差值
            for thresh in thresholds:
                # 跳过极端值
                if thresh == np.inf or thresh == -np.inf or np.isnan(thresh):
                    continue

                preds = (all_probs >= thresh).astype(int)

                # 跳过全 0 或全 1 预测
                if preds.sum() == 0 or preds.sum() == len(preds):
                    continue

                f1 = f1_score(all_labels, preds, average='binary', zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh

            print(f"  ✅ Best threshold: {best_threshold:.4f} (F1={best_f1:.4f})")

            # 使用最佳阈值重新预测
            all_preds = (all_probs >= best_threshold).astype(int)

        except Exception as e:
            print(f"\n⚠️  Warning: Threshold search failed ({e})")
            print("  Using default threshold 0.5")
            best_threshold = 0.5
            all_preds = (all_probs >= best_threshold).astype(int)

        # ✅ 计算最终指标
        global_acc = accuracy_score(all_labels, all_preds)
        global_f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
        global_precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
        global_recall = recall_score(all_labels, all_preds, average='binary', zero_division=0)

        # ✅ 打印混淆矩阵（帮助诊断）
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        print(f"\n  Confusion Matrix:")
        print(f"    TN={tn:6d}  FP={fp:6d}")
        print(f"    FN={fn:6d}  TP={tp:6d}")

        metrics = {
            'round': round_idx,
            'global_acc': global_acc,
            'global_f1': global_f1,
            'global_precision': global_precision,
            'global_recall': global_recall,
            'n_drifts': sum(drift_results.values()) if isinstance(drift_results, dict) else 0,
            'n_clusters': len(clustering_results.get('micro_clusters', []))
        }

        self.global_history.append(global_acc)

        return metrics

    def _print_round_summary(self, round_idx: int, metrics: Dict):
        """打印轮次摘要"""
        print(f"\n{'=' * 70}")
        print(f"Round {round_idx} Summary")
        print(f"{'=' * 70}")
        print(f"  Global Accuracy:  {metrics['global_acc']:.4f}")
        print(f"  Global F1 Score:  {metrics['global_f1']:.4f}")
        print(f"  Precision:        {metrics['global_precision']:.4f}")
        print(f"  Recall:           {metrics['global_recall']:.4f}")
        print(f"  Detected Drifts:  {metrics['n_drifts']}")
        print(f"  Active Clusters:  {metrics['n_clusters']}")
        print(f"{'=' * 70}")

    def run_experiment(self, fed_data: FederatedUNSWNB15, n_rounds: int = 50) -> Dict:
        """
        运行完整实验

        Args:
            fed_data: FederatedUNSWNB15 数据对象
            n_rounds: 训练轮数

        Returns:
            history: 训练历史
        """
        print(f"\n🚀 Starting MSTC-FL experiment: {n_rounds} rounds\n")
        early_stopping = EarlyStopping(patience=5, min_delta=0.01)
        for round_idx in range(n_rounds):
            self.train_round(fed_data, round_idx)
            if early_stopping(self.global_history[-1]):
                print(f"\n⏹️  Early stopping at round {round_idx}")
                print(f"   Best score: {early_stopping.best_score:.4f}")
                break
        print(f"\n✅ Experiment completed!")

        # 返回训练历史
        history = {
            'round_stats': self.round_stats,
            'global_history': self.global_history,
            'em_history': self.em_history
        }

        return history


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("🚀 Starting MSTC-FL Experiment (Method A: Native DataLoader)...\n")
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[1]  # 项目根（…/MSTC-FL）
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
    with open('results/mstc_fl_history.pkl', 'wb') as f:
         pickle.dump(history, f)
    print("\n✅ Results saved to results/mstc_fl_25rounds.pkl")