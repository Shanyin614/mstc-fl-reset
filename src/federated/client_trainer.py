"""
客户端训练模块
"""
import torch
import torch.nn as nn
from typing import List, Dict
from ..core.losses import FocalLoss
from ..utils.unsw_nb15_loader import FederatedUNSWNB15


class ClientTrainer:
    """
    客户端训练器

    负责管理所有客户端的本地训练流程，包括：
    - 数据加载
    - 损失函数动态调整
    - 模型训练
    - 学习率调度

    Args:
        n_clients: 客户端数量
        n_ensemble: 集成成员数量
        local_epochs: 本地训练轮数
        batch_size: 批量大小
        device: 训练设备
    """

    def __init__(
            self,
            n_clients: int,
            n_ensemble: int,
            local_epochs: int = 1,
            batch_size: int = 64,
            device: torch.device = None
    ):
        self.n_clients = n_clients
        self.n_ensemble = n_ensemble
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device('cpu')

    def train_clients(
            self,
            fed_data: FederatedUNSWNB15,
            ensemble: List[nn.Module],
            optimizers: List,
            schedulers: List = None
    ) -> Dict:
        """
        训练所有客户端

        Args:
            fed_data: 联邦数据对象
            ensemble: 集成模型列表
            optimizers: 优化器列表
            schedulers: 学习率调度器列表（可选）

        Returns:
            {
                'client_models': List[nn.Module],
                'client_accuracies': List[float]
            }
        """
        client_accuracies = []
        client_models = []

        # 处理可选的 schedulers
        if schedulers is None:
            schedulers = [None] * len(ensemble)

        for i in range(self.n_clients):
            acc = self._train_single_client(
                client_id=i,
                fed_data=fed_data,
                ensemble=ensemble,
                optimizers=optimizers,
                schedulers=schedulers
            )
            client_accuracies.append(acc)

            # 保存对应的集成成员模型
            em_idx = i % self.n_ensemble
            client_models.append(ensemble[em_idx])

        return {
            'client_models': client_models,
            'client_accuracies': client_accuracies
        }

    def _train_single_client(
            self,
            client_id: int,
            fed_data: FederatedUNSWNB15,
            ensemble: List[nn.Module],
            optimizers: List,
            schedulers: List
    ) -> float:
        """
        训练单个客户端

        Args:
            client_id: 客户端ID
            fed_data: 联邦数据对象
            ensemble: 集成模型列表
            optimizers: 优化器列表
            schedulers: 学习率调度器列表（可选）

        Returns:
            客户端准确率
        """
        # 1. 选择对应的集成成员
        em_idx = client_id % self.n_ensemble
        em_model = ensemble[em_idx]
        optimizer = optimizers[em_idx]
        scheduler = schedulers[em_idx] if schedulers else None

        # 2. 获取客户端数据
        client_loader = fed_data.get_client_loader(client_id, batch_size=self.batch_size)

        # 3. 动态调整损失函数参数
        attack_ratio = fed_data.get_client_attack_ratio(client_id)
        alpha, gamma = FocalLoss.get_adaptive_params(attack_ratio)
        criterion = FocalLoss(alpha=alpha, gamma=gamma)

        print(f"  Client {client_id} (Attack={attack_ratio:.2f})...", end=" ")

        # 4. 训练
        em_model.train()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0

        for epoch in range(self.local_epochs):
            for batch_x, batch_y in client_loader:
                if batch_x.size(0) == 0:
                    continue

                # 数据预处理
                batch_x = batch_x.to(self.device)
                if batch_y.dim() > 1:
                    batch_y = batch_y.squeeze(1)
                batch_y = batch_y.to(self.device)

                if batch_x.size(0) != batch_y.size(0):
                    continue

                # 前向传播
                optimizer.zero_grad()
                outputs = em_model(batch_x)
                loss = criterion(outputs, batch_y)

                # 反向传播
                loss.backward()
                optimizer.step()

                # 统计
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == batch_y).sum().item()
                total_samples += batch_y.size(0)
                total_loss += loss.item()

        # 5. 计算指标
        acc = total_correct / total_samples if total_samples > 0 else 0.0
        avg_loss = total_loss / max(len(client_loader) * self.local_epochs, 1)

        # 6. 更新学习率（如果有调度器）
        if scheduler is not None:
            scheduler.step()
            print(f"Acc={acc:.3f}, Loss={avg_loss:.4f}, LR={scheduler.get_last_lr()[0]:.6f} ✓")
        else:
            print(f"Acc={acc:.3f}, Loss={avg_loss:.4f} ✓")

        return acc
