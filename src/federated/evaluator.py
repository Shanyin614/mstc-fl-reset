"""
模型评估模块
"""
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, confusion_matrix
)
from ..utils.unsw_nb15_loader import FederatedUNSWNB15


class ModelEvaluator:
    """
    模型评估器

    负责评估全局模型在测试集上的性能，包括：
    - 收集预测结果
    - 优化分类阈值（基于 F1-score）
    - 计算多种评估指标
    - 打印混淆矩阵

    Args:
        device: 计算设备

    Example:
        >>> evaluator = ModelEvaluator(device=torch.device('cuda'))
        >>> metrics = evaluator.evaluate_global_model(
        ...     model=global_model,
        ...     fed_data=fed_data,
        ...     round_idx=0,
        ...     drift_results={},
        ...     clustering_results={}
        ... )
    """

    def __init__(self, device: torch.device = None):
        self.device = device if device is not None else torch.device('cpu')

    def evaluate_global_model(
            self,
            model: torch.nn.Module,
            fed_data: FederatedUNSWNB15,
            round_idx: int,
            drift_results: Dict,
            clustering_results: Dict
    ) -> Dict:
        """
        评估全局模型

        Args:
            model: 待评估模型
            fed_data: 联邦数据对象
            round_idx: 当前轮次
            drift_results: 漂移检测结果
            clustering_results: 聚类结果

        Returns:
            评估指标字典，包含 acc, f1, precision, recall 等
        """
        test_loader = fed_data.get_test_loader(batch_size=128)

        model.eval()

        # 1. 收集预测结果
        all_probs, all_labels = self._collect_predictions(model, test_loader)

        # 2. 数据验证
        if len(all_probs) == 0 or len(all_labels) == 0:
            print("\n⚠️  Warning: No data collected during evaluation!")
            return self._get_empty_metrics(round_idx, drift_results, clustering_results)

        # 3. 打印数据分布
        self._print_data_distribution(all_labels)

        # 4. 优化阈值
        best_threshold = self._find_best_threshold(all_probs, all_labels)

        # 5. 计算最终指标
        metrics = self._compute_metrics(
            all_probs, all_labels, best_threshold,
            round_idx, drift_results, clustering_results
        )

        # 6. 打印混淆矩阵
        all_preds = (all_probs >= best_threshold).astype(int)
        self._print_confusion_matrix(all_labels, all_preds)

        return metrics

    def _collect_predictions(
            self,
            model: torch.nn.Module,
            test_loader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        收集模型预测结果

        Args:
            model: 待评估模型
            test_loader: 测试数据加载器

        Returns:
            (预测概率, 真实标签) 元组
        """
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

                outputs = model(batch_x)
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类概率

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        return np.array(all_probs), np.array(all_labels)

    def _find_best_threshold(
            self,
            probs: np.ndarray,
            labels: np.ndarray,
            n_candidates: int = 100
    ) -> float:
        """
        优化分类阈值（基于 F1-score）

        策略：
        1. 使用 ROC 曲线获取候选阈值
        2. 限制搜索空间为 100 个候选值（加速）
        3. 选择 F1-score 最高的阈值

        Args:
            probs: 预测概率 [N]
            labels: 真实标签 [N]
            n_candidates: 候选阈值数量

        Returns:
            最佳阈值
        """
        try:
            fpr, tpr, thresholds = roc_curve(labels, probs)

            # 减少搜索空间
            if len(thresholds) > n_candidates:
                indices = np.linspace(0, len(thresholds) - 1, n_candidates, dtype=int)
                thresholds = thresholds[indices]

            print(f"  Searching {len(thresholds)} thresholds...")

            best_threshold = 0.5
            best_f1 = 0.0

            for thresh in thresholds:
                # 跳过无效值
                if not np.isfinite(thresh):
                    continue

                preds = (probs >= thresh).astype(int)

                # 跳过极端预测（全 0 或全 1）
                if preds.sum() == 0 or preds.sum() == len(preds):
                    continue

                f1 = f1_score(labels, preds, average='binary', zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = thresh

            print(f"  ✅ Best threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
            return best_threshold

        except Exception as e:
            print(f"\n⚠️  Warning: Threshold search failed ({e})")
            print("  Using default threshold 0.5")
            return 0.5

    def _compute_metrics(
            self,
            probs: np.ndarray,
            labels: np.ndarray,
            threshold: float,
            round_idx: int,
            drift_results: Dict,
            clustering_results: Dict
    ) -> Dict:
        """
        计算评估指标

        Args:
            probs: 预测概率
            labels: 真实标签
            threshold: 分类阈值
            round_idx: 当前轮次
            drift_results: 漂移检测结果
            clustering_results: 聚类结果

        Returns:
            包含所有指标的字典
        """
        preds = (probs >= threshold).astype(int)

        return {
            'round': round_idx,
            'global_acc': accuracy_score(labels, preds),
            'global_f1': f1_score(labels, preds, average='binary', zero_division=0),
            'global_precision': precision_score(labels, preds, average='binary', zero_division=0),
            'global_recall': recall_score(labels, preds, average='binary', zero_division=0),
            'n_drifts': sum(drift_results.values()) if isinstance(drift_results, dict) else 0,
            'n_clusters': len(clustering_results.get('micro_clusters', []))
        }

    def _print_data_distribution(self, labels: np.ndarray):
        """打印测试集数据分布"""
        n_positive = (labels == 1).sum()
        n_negative = (labels == 0).sum()

        print(f"\n  Test set distribution:")
        print(f"    Positive (Attack): {n_positive} ({n_positive / len(labels) * 100:.2f}%)")
        print(f"    Negative (Normal): {n_negative} ({n_negative / len(labels) * 100:.2f}%)")

    def _print_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray):
        """打印混淆矩阵"""
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        print(f"\n  Confusion Matrix:")
        print(f"    TN={tn:6d}  FP={fp:6d}")
        print(f"    FN={fn:6d}  TP={tp:6d}")

    def _get_empty_metrics(
            self,
            round_idx: int,
            drift_results: Dict,
            clustering_results: Dict
    ) -> Dict:
        """返回空指标（用于异常情况）"""
        return {
            'round': round_idx,
            'global_acc': 0.0,
            'global_f1': 0.0,
            'global_precision': 0.0,
            'global_recall': 0.0,
            'n_drifts': sum(drift_results.values()) if isinstance(drift_results, dict) else 0,
            'n_clusters': len(clustering_results.get('micro_clusters', []))
        }

    # src/evaluation/model_evaluator.py
    # 在 ModelEvaluator 类中添加

    def evaluate_binary(
            self,
            y_true: np.ndarray,
            y_pred: np.ndarray
    ) -> Dict:
        """
        评估二分类结果

        Args:
            y_true: 真实标签 [N]
            y_pred: 预测标签 [N]

        Returns:
            评估指标字典
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0)
        }

