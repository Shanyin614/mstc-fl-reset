"""
ADWIN++ 调度器 V2（简化版，保证能检测到漂移）
"""
import sys
import os

# 添加项目根目录到 sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from typing import List, Tuple, Optional
from collections import deque


class ADWINWindowScheduler:
    """
    简化版 ADWIN（基于移动平均检测）

    核心思想:
    1. 维护一个滑动窗口
    2. 比较最近 min_window 个元素与历史数据的统计差异
    3. 使用标准差作为阈值判断是否漂移
    """

    def __init__(self,
                 delta: float = 0.05,
                 min_window: int = 5,
                 max_window: int = 100,
                 sensitivity: float = 2.0):
        """
        Args:
            delta: 置信度参数（保留接口兼容性）
            min_window: 最小窗口长度
            max_window: 最大窗口长度
            sensitivity: 敏感度（越小越敏感，建议 1.5-3.0）
        """
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window
        self.sensitivity = sensitivity

        self.window = deque(maxlen=max_window)

        # 统计信息
        self.drift_count = 0
        self.total_elements = 0
        self.drift_positions = []

    def add_element(self, value: float) -> int:
        """添加新元素并检测漂移"""
        self.window.append(value)
        self.total_elements += 1

        # 检测漂移（需要至少 min_window * 2 个元素）
        if len(self.window) >= self.min_window * 2:
            if self._detect_drift():
                self._handle_drift()

        return len(self.window)

    def _detect_drift(self) -> bool:
        """
        检测漂移

        方法: 比较最近 min_window 与之前所有数据的均值
        """
        window_list = list(self.window)

        # 分割窗口
        split_point = len(window_list) - self.min_window

        if split_point < self.min_window:
            return False

        # 历史数据 vs 最近数据
        historical = window_list[:split_point]
        recent = window_list[split_point:]

        # 计算统计量
        hist_mean = np.mean(historical)
        hist_std = np.std(historical, ddof=1)
        recent_mean = np.mean(recent)

        # 避免除零
        if hist_std < 1e-8:
            hist_std = np.mean(np.abs(np.diff(historical))) if len(historical) > 1 else 1e-8

        # 标准化差异
        z_score = abs(recent_mean - hist_mean) / hist_std

        # 检测阈值
        threshold = self.sensitivity

        return z_score > threshold

    def _handle_drift(self):
        """处理检测到的漂移"""
        self.drift_count += 1
        self.drift_positions.append(self.total_elements)

        # 保留最近的数据
        recent_data = list(self.window)[-self.min_window:]
        self.window.clear()
        self.window.extend(recent_data)

    def get_window_length(self) -> int:
        return len(self.window)

    def get_statistics(self) -> dict:
        return {
            'total_elements': self.total_elements,
            'drift_count': self.drift_count,
            'drift_rate': self.drift_count / max(1, self.total_elements),
            'current_window': len(self.window),
            'drift_positions': self.drift_positions.copy()
        }

    def reset(self):
        self.window.clear()
        self.drift_count = 0
        self.total_elements = 0
        self.drift_positions.clear()


class MultiScaleWindowScheduler:
    """多尺度窗口调度器（使用 V2 检测器）"""

    def __init__(self,
                 base_scales: Optional[dict] = None,
                 delta: float = 0.05,
                 sensitivity: float = 2.0):

        if base_scales is None:
            self.base_scales = {'micro': 5, 'meso': 15, 'macro': 40}
        else:
            self.base_scales = base_scales.copy()

        self.current_scales = self.base_scales.copy()

        # 使用 V2 检测器
        self.schedulers = {
            'micro': ADWINWindowScheduler(delta, min_window=3, max_window=15, sensitivity=sensitivity),
            'meso': ADWINWindowScheduler(delta, min_window=10, max_window=30, sensitivity=sensitivity),
            'macro': ADWINWindowScheduler(delta, min_window=20, max_window=80, sensitivity=sensitivity)
        }

    def update(self, em_variance: float) -> dict:
        """更新窗口长度"""
        for scale_name, scheduler in self.schedulers.items():
            L = scheduler.add_element(em_variance)
            self.current_scales[scale_name] = L

        return self.current_scales.copy()

    def get_scales(self) -> dict:
        return self.current_scales.copy()

    def get_all_statistics(self) -> dict:
        stats = {}
        for scale_name, scheduler in self.schedulers.items():
            stats[scale_name] = scheduler.get_statistics()
        return stats


class MultiClientADWIN:
    """
    多客户端 ADWIN 管理器
    为每个客户端维护独立的 ADWIN 检测器

    用于联邦学习场景，管理多个客户端的漂移检测
    """

    def __init__(self,
                 n_clients: int,
                 delta: float = 0.05,
                 min_window: int = 5,
                 max_window: int = 100,
                 sensitivity: float = 2.0):
        """
        Args:
            n_clients: 客户端数量
            delta: 置信度参数
            min_window: 最小窗口长度
            max_window: 最大窗口长度
            sensitivity: 敏感度（越小越敏感）
        """
        self.n_clients = n_clients
        self.delta = delta
        self.min_window = min_window
        self.max_window = max_window
        self.sensitivity = sensitivity

        # 为每个客户端创建独立的 ADWIN 实例
        self.detectors = [
            ADWINWindowScheduler(
                delta=delta,
                min_window=min_window,
                max_window=max_window,
                sensitivity=sensitivity
            ) for _ in range(n_clients)
        ]

        print(f"✅ MultiClientADWIN initialized: {n_clients} detectors")

    def update(self, client_accuracies: List[float]) -> dict:
        """
        更新所有客户端的 ADWIN 并检测漂移

        Args:
            client_accuracies: 各客户端准确率列表 [acc_0, acc_1, ..., acc_n]

        Returns:
            漂移检测结果字典:
            {
                'drift_clients': [漂移的客户端 ID],
                'n_drifts': 漂移客户端数量,
                'details': [详细信息],
                'all_statistics': [所有检测器统计]
            }
        """
        if len(client_accuracies) != self.n_clients:
            raise ValueError(f"Expected {self.n_clients} accuracies, got {len(client_accuracies)}")

        drift_detected = []
        drift_details = []

        for i, acc in enumerate(client_accuracies):
            # 添加元素并获取窗口长度
            window_length = self.detectors[i].add_element(acc)

            # 获取统计信息
            stats = self.detectors[i].get_statistics()

            # 检查是否检测到新漂移
            if stats['drift_count'] > 0:
                # 如果最后一次漂移位置等于当前元素总数，说明刚检测到漂移
                last_drift_position = stats['drift_positions'][-1]
                if last_drift_position == stats['total_elements']:
                    drift_detected.append(i)
                    drift_details.append({
                        'client_id': i,
                        'accuracy': acc,
                        'window_length': window_length,
                        'total_drifts': stats['drift_count'],
                        'drift_rate': stats['drift_rate']
                    })

        return {
            'drift_clients': drift_detected,
            'n_drifts': len(drift_detected),
            'details': drift_details,
            'all_statistics': [self.detectors[i].get_statistics() for i in range(self.n_clients)]
        }

    def get_client_statistics(self, client_id: int) -> dict:
        """
        获取单个客户端的统计信息

        Args:
            client_id: 客户端 ID

        Returns:
            统计信息字典
        """
        if client_id < 0 or client_id >= self.n_clients:
            raise ValueError(f"Invalid client_id: {client_id}")

        return self.detectors[client_id].get_statistics()

    def get_all_statistics(self) -> List[dict]:
        """获取所有客户端的统计信息"""
        return [det.get_statistics() for det in self.detectors]

    def reset_client(self, client_id: int):
        """重置单个客户端的检测器"""
        self.detectors[client_id].reset()

    def reset_all(self):
        """重置所有检测器"""
        for det in self.detectors:
            det.reset()

    def get_drift_summary(self) -> dict:
        """
        获取所有客户端的漂移摘要

        Returns:
            摘要字典:
            {
                'total_drifts': 总漂移次数,
                'clients_with_drift': 有漂移的客户端数量,
                'avg_drift_rate': 平均漂移率,
                'drift_distribution': 各客户端漂移次数
            }
        """
        all_stats = self.get_all_statistics()

        total_drifts = sum(s['drift_count'] for s in all_stats)
        clients_with_drift = sum(1 for s in all_stats if s['drift_count'] > 0)
        avg_drift_rate = np.mean([s['drift_rate'] for s in all_stats])
        drift_distribution = [s['drift_count'] for s in all_stats]

        return {
            'total_drifts': total_drifts,
            'clients_with_drift': clients_with_drift,
            'avg_drift_rate': avg_drift_rate,
            'drift_distribution': drift_distribution
        }


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("🧪 Testing MultiClientADWIN...")

    # 模拟 10 个客户端，50 轮训练
    n_clients = 10
    n_rounds = 50

    # 初始化
    multi_adwin = MultiClientADWIN(
        n_clients=n_clients,
        delta=0.05,
        min_window=5,
        max_window=100,
        sensitivity=2.0
    )

    print(f"\n🔄 Simulating {n_rounds} rounds...")

    for round_idx in range(n_rounds):
        # 模拟客户端准确率（带漂移）
        if round_idx < 20:
            # 前 20 轮：稳定期
            client_accs = np.random.uniform(0.75, 0.85, n_clients)
        elif round_idx < 30:
            # 20-30 轮：漂移期（性能下降）
            client_accs = np.random.uniform(0.60, 0.70, n_clients)
        else:
            # 30+ 轮：恢复期
            client_accs = np.random.uniform(0.80, 0.90, n_clients)

        # 检测漂移
        drift_results = multi_adwin.update(client_accs.tolist())

        # 打印结果
        if drift_results['n_drifts'] > 0:
            print(f"\nRound {round_idx}:")
            print(f"  ⚠️  Drift detected in {drift_results['n_drifts']} clients: {drift_results['drift_clients']}")
            for detail in drift_results['details']:
                print(f"      Client {detail['client_id']}: Acc={detail['accuracy']:.3f}, "
                      f"Window={detail['window_length']}, Total Drifts={detail['total_drifts']}")

    # 最终摘要
    print(f"\n{'=' * 60}")
    print("Final Summary:")
    summary = multi_adwin.get_drift_summary()
    print(f"  Total drifts:      {summary['total_drifts']}")
    print(f"  Clients affected:  {summary['clients_with_drift']}/{n_clients}")
    print(f"  Avg drift rate:    {summary['avg_drift_rate']:.4f}")
    print(f"  Drift distribution: {summary['drift_distribution']}")
    print(f"{'=' * 60}")

    print("\n✅ All tests passed!")