# src/models/gcn_encoder.py
"""
Graph Convolutional Network (GCN) Encoder
用于将客户端特征映射到低维嵌入空间，支持层次聚类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class GCNLayer(nn.Module):
    """
    单层图卷积层

    公式: H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))
    其中:
    - A: 邻接矩阵（带自环）
    - D: 度矩阵
    - H: 节点特征
    - W: 可学习权重
    - σ: 激活函数
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 dropout: float = 0.3):
        super(GCNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 可学习参数
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """Xavier 初始化"""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [N, in_features]
            adj: 归一化邻接矩阵 [N, N]

        Returns:
            更新后的节点特征 [N, out_features]
        """
        # 1. 特征变换: X @ W
        x = self.dropout(x)
        support = torch.mm(x, self.weight)  # [N, out_features]

        # 2. 图卷积: A @ (X @ W)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)

        # 3. 加偏置
        if self.bias is not None:
            output = output + self.bias

        return output


class GCNEncoder(nn.Module):
    """
    多层 GCN 编码器

    架构:
    Input (n_features)
      → GCN Layer 1 (hidden_dim) + ReLU + Dropout
      → GCN Layer 2 (hidden_dim) + ReLU + Dropout
      → GCN Layer 3 (embedding_dim)
      → Output (低维嵌入)
    """

    def __init__(self,
                 n_features: int,
                 hidden_dim: int = 128,
                 embedding_dim: int = 64,
                 n_layers: int = 3,
                 dropout: float = 0.3):
        super(GCNEncoder, self).__init__()

        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # 构建多层 GCN
        self.layers = nn.ModuleList()

        # 输入层
        self.layers.append(GCNLayer(n_features, hidden_dim, dropout=dropout))

        # 隐藏层
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim, dropout=dropout))

        # 输出层（嵌入层）
        self.layers.append(GCNLayer(hidden_dim, embedding_dim, dropout=0.0))

        print(f"✅ GCNEncoder initialized:")
        print(f"   Input: {n_features} features")
        print(f"   Hidden: {n_layers - 2} layers × {hidden_dim} dims")
        print(f"   Output: {embedding_dim} dims")

    def forward(self,
                x: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [N, n_features]
            adj: 归一化邻接矩阵 [N, N]

        Returns:
            节点嵌入 [N, embedding_dim]
        """
        # 逐层传播
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)

        # 最后一层（不加激活）
        x = self.layers[-1](x, adj)

        # L2 归一化（用于后续聚类）
        x = F.normalize(x, p=2, dim=1)

        return x

    def get_embeddings(self,
                       features: torch.Tensor,
                       adj: torch.Tensor) -> torch.Tensor:
        """获取嵌入（推理模式）"""
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(features, adj)
        return embeddings


def normalize_adjacency(adj: torch.Tensor,
                        add_self_loop: bool = True) -> torch.Tensor:
    """
    对称归一化邻接矩阵

    公式: A_norm = D^(-1/2) @ A @ D^(-1/2)

    Args:
        adj: 原始邻接矩阵 [N, N]
        add_self_loop: 是否添加自环

    Returns:
        归一化后的邻接矩阵
    """
    if add_self_loop:
        # 添加自环
        adj = adj + torch.eye(adj.size(0), device=adj.device)

    # 计算度矩阵
    degree = adj.sum(dim=1)

    # 处理孤立节点（度为 0）
    degree[degree == 0] = 1.0

    # D^(-1/2)
    degree_inv_sqrt = torch.pow(degree, -0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0

    # 构建对角矩阵
    degree_matrix = torch.diag(degree_inv_sqrt)

    # 对称归一化: D^(-1/2) @ A @ D^(-1/2)
    adj_normalized = torch.mm(torch.mm(degree_matrix, adj), degree_matrix)

    return adj_normalized


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将 scipy 稀疏矩阵转换为 PyTorch 稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GCNTrainer:
    """GCN 训练器"""

    def __init__(self,
                 model: GCNEncoder,
                 lr: float = 0.01,
                 weight_decay: float = 5e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_epoch(self,
                    features: torch.Tensor,
                    adj: torch.Tensor,
                    labels: torch.Tensor,
                    train_mask: torch.Tensor) -> float:
        """
        训练一个 epoch（节点分类任务）

        Args:
            features: 节点特征 [N, D]
            adj: 邻接矩阵 [N, N]
            labels: 节点标签 [N]
            train_mask: 训练集掩码 [N]

        Returns:
            训练损失
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 前向传播
        embeddings = self.model(features, adj)

        # 计算损失（只在训练集上）
        loss = F.cross_entropy(
            embeddings[train_mask],
            labels[train_mask]
        )

        # 反向传播
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self,
                 features: torch.Tensor,
                 adj: torch.Tensor,
                 labels: torch.Tensor,
                 test_mask: torch.Tensor) -> Tuple[float, float]:
        """
        评估模型

        Returns:
            (loss, accuracy)
        """
        self.model.eval()

        with torch.no_grad():
            embeddings = self.model(features, adj)

            # 损失
            loss = F.cross_entropy(
                embeddings[test_mask],
                labels[test_mask]
            ).item()

            # 准确率
            pred = embeddings[test_mask].argmax(dim=1)
            acc = (pred == labels[test_mask]).float().mean().item()

        return loss, acc


if __name__ == "__main__":
    # 测试 GCN
    print("🧪 Testing GCN Encoder...")

    # 模拟数据
    n_nodes = 10  # 10 个客户端
    n_features = 64  # 特征维度

    # 随机特征
    features = torch.randn(n_nodes, n_features)

    # 随机邻接矩阵（稀疏）
    adj = torch.rand(n_nodes, n_nodes)
    adj = (adj > 0.7).float()  # 30% 连接
    adj = (adj + adj.T) / 2  # 对称化

    # 归一化
    adj_norm = normalize_adjacency(adj)

    # 初始化编码器
    encoder = GCNEncoder(
        n_features=n_features,
        hidden_dim=32,
        embedding_dim=16,
        n_layers=3
    )

    # 前向传播
    embeddings = encoder(features, adj_norm)

    print(f"\n✅ Test passed!")
    print(f"   Input shape:  {features.shape}")
    print(f"   Output shape: {embeddings.shape}")
    print(f"   Embeddings normalized: {torch.allclose(embeddings.norm(dim=1), torch.ones(n_nodes), atol=1e-5)}")
