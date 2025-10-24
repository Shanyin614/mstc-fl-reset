# src/utils/unsw_nb15_loader.py
"""
UNSW-NB15 数据集加载器（完整修复版）
修复问题：
1. 移除未实现的 UNSWNB15Dataset 类依赖
2. 修复 Non-IID 数据分割的 NumPy 数组转换
3. 统一数据接口（直接使用 NumPy 数组）
4. 添加完整的错误检查和类型验证
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import torch
from typing import Tuple, List


class FederatedUNSWNB15:
    """
    联邦学习场景的 UNSW-NB15 数据加载器（完整重构版）

    核心功能：
    - 加载并预处理 UNSW-NB15 数据集
    - 使用 Dirichlet 分布创建 Non-IID 客户端分割
    - 提供训练/测试 DataLoader
    - 支持攻击率查询

    使用场景：
    - 10 个客户端代表不同网络区域
    - 高度异构的攻击模式（Non-IID）
    - 不平衡数据分布
    """

    def __init__(
            self,
            train_path: str,
            test_path: str,
            n_clients: int = 10,
            alpha: float = 0.5,
            seed: int = 42
    ):
        """
        初始化联邦数据加载器

        Args:
            train_path: 训练集 CSV 路径
            test_path: 测试集 CSV 路径
            n_clients: 客户端数量
            alpha: Dirichlet 分布参数（越小越不平衡）
            seed: 随机种子
        """
        print(f"\n🔄 Initializing FederatedUNSWNB15...")
        print(f"   Clients: {n_clients}")
        print(f"   Alpha: {alpha}")
        print(f"   Seed: {seed}")

        self.n_clients = n_clients
        self.alpha = alpha
        self.seed = seed

        np.random.seed(seed)
        torch.manual_seed(seed)

        # 加载并预处理数据
        print("\n🔄 Loading and preprocessing utils...")
        self.train_data, self.test_data, self.scaler = self._load_and_preprocess(
            train_path, test_path
        )

        # 分割数据到各客户端
        print("\n🔄 Splitting utils into Non-IID clients...")
        self.client_data = self._split_data_noniid(self.train_data)

        # ✅ 验证初始化成功
        assert hasattr(self, 'client_data'), "client_data not created"
        assert isinstance(self.client_data, list), "client_data should be a list"
        assert len(self.client_data) == n_clients, f"Expected {n_clients} clients, got {len(self.client_data)}"

        # 验证每个客户端数据是 NumPy 数组
        for i, cd in enumerate(self.client_data):
            assert isinstance(cd, np.ndarray), f"Client {i} utils is {type(cd)}, expected ndarray"
            assert cd.ndim == 2, f"Client {i} utils should be 2D, got shape {cd.shape}"

        # 缓存攻击率
        self._attack_ratios = None

        # 打印数据分布
        self._print_distribution()

        print("\n✅ FederatedUNSWNB15 initialized successfully!")

    def _load_and_preprocess(
            self,
            train_path: str,
            test_path: str
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """
        加载并预处理数据

        返回:
            (train_data, test_data, scaler)
            - train_data: (n_samples, n_features+1) 最后一列是标签
            - test_data: (n_samples, n_features+1)
            - scaler: 拟合好的 StandardScaler
        """

        # 读取 CSV
        print("   Loading CSV files...")
        train_df = pd.read_csv(train_path, low_memory=False)
        test_df = pd.read_csv(test_path, low_memory=False)

        # 删除无用列
        cols_to_drop = ['id', 'attack_cat']
        train_df = train_df.drop(cols_to_drop, axis=1, errors='ignore')
        test_df = test_df.drop(cols_to_drop, axis=1, errors='ignore')

        # 处理分类特征（编码为数值）
        categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        if 'label' in categorical_cols:
            categorical_cols.remove('label')

        print(f"   Encoding {len(categorical_cols)} categorical features...")
        for col in categorical_cols:
            le = LabelEncoder()
            # 合并训练+测试集的类别（确保测试集编码一致）
            all_values = pd.concat([train_df[col], test_df[col]]).astype(str)
            le.fit(all_values)

            train_df[col] = le.transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))

        # 分离特征和标签
        X_train = train_df.drop('label', axis=1).values
        y_train = train_df['label'].values

        X_test = test_df.drop('label', axis=1).values
        y_test = test_df['label'].values

        # 转换为二分类（0=正常, 1=攻击）
        y_train = (y_train != 0).astype(int)
        y_test = (y_test != 0).astype(int)

        print("\nTrain dataset:")
        print(f"  Samples: {len(X_train)}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: 2 (Binary)")
        print(f"  Attack ratio: {y_train.mean():.2%}")

        print("Test dataset:")
        print(f"  Samples: {len(X_test)}")
        print(f"  Features: {X_test.shape[1]}")
        print(f"  Attack ratio: {y_test.mean():.2%}")

        # 标准化
        print("\n   Standardizing features...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 合并特征和标签
        train_data = np.hstack([X_train, y_train.reshape(-1, 1)])
        test_data = np.hstack([X_test, y_test.reshape(-1, 1)])

        return train_data, test_data, scaler

    def _split_data_noniid(self, data: np.ndarray) -> List[np.ndarray]:
        """
        使用 Dirichlet 分布创建 Non-IID 数据分割

        Args:
            data: 训练数据 (n_samples, n_features+1)

        Returns:
            客户端数据列表，每个元素是 NumPy 数组
        """
        labels = data[:, -1].astype(int)
        n_classes = len(np.unique(labels))

        print(f"   Classes: {n_classes}")
        print(f"   Using Dirichlet(α={self.alpha}) for Non-IID split...")

        # 收集每个客户端的样本索引
        client_indices = [[] for _ in range(self.n_clients)]

        for k in range(n_classes):
            # 获取当前类别的所有样本索引
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)

            print(f"   Class {k}: {len(idx_k)} samples")

            # Dirichlet 分布生成每个客户端的样本比例
            proportions = np.random.dirichlet(np.repeat(self.alpha, self.n_clients))

            # 根据比例分配样本
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)

            # 将索引添加到对应客户端
            for i, idx in enumerate(idx_k_split):
                if i < self.n_clients:  # 防止越界
                    client_indices[i].extend(idx.tolist())

        # ✅ 转换为 NumPy 数组
        print("\n   Converting to NumPy arrays...")
        client_data = []
        for i, indices in enumerate(client_indices):
            if len(indices) == 0:
                print(f"   ⚠️  Warning: Client {i} has no utils!")
                # 创建空数组（保持维度一致）
                client_data.append(np.empty((0, data.shape[1])))
            else:
                # 使用索引提取数据
                client_data.append(data[np.array(indices)])

        # ✅ 验证转换成功
        print(f"\n✅ Created {len(client_data)} client datasets:")
        for i, cd in enumerate(client_data):
            assert isinstance(cd, np.ndarray), f"Client {i} utils is {type(cd)}, expected ndarray"
            print(f"   Client {i}: {len(cd):6d} samples, shape={cd.shape}")

        return client_data

    def _print_distribution(self):
        """打印数据分布"""
        print("\n" + "=" * 70)
        print("Federated Data Distribution (Non-IID)")
        print("=" * 70)

        for i, client_data in enumerate(self.client_data):
            n_samples = len(client_data)
            if n_samples == 0:
                print(f"Client {i}:     0 samples, Attack ratio: N/A")
                continue

            n_attacks = (client_data[:, -1] == 1).sum()
            attack_ratio = n_attacks / n_samples * 100
            print(f"Client {i}: {n_samples:6d} samples, Attack ratio: {attack_ratio:5.2f}%")

    def get_client_loader(
            self,
            client_id: int,
            batch_size: int = 64,
            shuffle: bool = True
    ) -> DataLoader:
        """
        获取指定客户端的 DataLoader

        Args:
            client_id: 客户端 ID (0 到 n_clients-1)
            batch_size: 批量大小
            shuffle: 是否打乱数据

        Returns:
            PyTorch DataLoader
        """
        if client_id < 0 or client_id >= self.n_clients:
            raise ValueError(f"client_id must be in [0, {self.n_clients - 1}], got {client_id}")

        client_data = self.client_data[client_id]

        # ✅ 验证数据类型
        assert isinstance(client_data, np.ndarray), f"Client {client_id} utils is {type(client_data)}"

        if len(client_data) == 0:
            raise ValueError(f"Client {client_id} has no utils")

        # 分离特征和标签
        X = torch.FloatTensor(client_data[:, :-1])
        y = torch.LongTensor(client_data[:, -1].astype(int))

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_loader(self, batch_size: int = 128) -> DataLoader:
        """
        获取测试集 DataLoader

        Args:
            batch_size: 批量大小

        Returns:
            PyTorch DataLoader
        """
        X = torch.FloatTensor(self.test_data[:, :-1])
        y = torch.LongTensor(self.test_data[:, -1].astype(int))

        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def get_client_attack_ratio(self, client_id: int) -> float:
        """
        获取指定客户端的攻击样本比例

        Args:
            client_id: 客户端 ID (0-based)

        Returns:
            攻击样本占比 (0.0-1.0)

        Raises:
            ValueError: 如果 client_id 超出范围
            TypeError: 如果数据类型错误
        """
        # 防御性检查
        if not hasattr(self, 'client_data'):
            raise AttributeError("client_data not initialized. Call __init__() first.")

        if client_id < 0 or client_id >= self.n_clients:
            raise ValueError(f"client_id must be in [0, {self.n_clients - 1}], got {client_id}")

        client_data = self.client_data[client_id]

        # ✅ 验证数据类型
        if not isinstance(client_data, np.ndarray):
            raise TypeError(f"Client {client_id} utils is {type(client_data)}, expected ndarray")

        if len(client_data) == 0:
            return 0.0

        # ✅ 现在可以安全使用 NumPy 索引
        attack_count = (client_data[:, -1] == 1).sum()
        return float(attack_count / len(client_data))

    @property
    def input_dim(self) -> int:
        """获取输入特征维度（不含标签）"""
        return self.train_data.shape[1] - 1

    def get_all_attack_ratios(self) -> List[float]:
        """
        获取所有客户端的攻击率（带缓存）

        Returns:
            攻击率列表 [client_0_ratio, client_1_ratio, ...]
        """
        if self._attack_ratios is None:
            print("\n🔄 Computing attack ratios for all clients...")
            self._attack_ratios = [
                self.get_client_attack_ratio(i)
                for i in range(self.n_clients)
            ]
            print("✅ Attack ratios cached")

        return self._attack_ratios


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import os

    print("=" * 70)
    print("UNSW-NB15 Federated Data Loader - Test")
    print("=" * 70)

    # 打印当前工作目录
    print(f"\nCurrent Working Directory: {os.getcwd()}")

    # 计算项目根目录（假设当前文件在 src/utils/unsw_nb15_loader.py）
    ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = ROOT / "utils" / "unsw"

    train_path = DATA_DIR / "UNSW_NB15_training-set.csv"
    test_path = DATA_DIR / "UNSW_NB15_testing-set.csv"

    print(f"\nData paths:")
    print(f"  Train: {train_path}")
    print(f"  Test: {test_path}")

    # 检查文件是否存在
    if not train_path.exists():
        print(f"\n❌ Error: Training file not found at {train_path}")
        print("Please download UNSW-NB15 dataset and place it in utils/raw/")
        exit(1)

    if not test_path.exists():
        print(f"\n❌ Error: Test file not found at {test_path}")
        exit(1)

    print("\n✅ Data files found")

    # ========================================
    # 测试 1: 初始化
    # ========================================
    print("\n" + "=" * 70)
    print("Test 1: Initialization")
    print("=" * 70)

    fed_data = FederatedUNSWNB15(
        train_path=str(train_path),
        test_path=str(test_path),
        n_clients=10,
        alpha=0.5,
        seed=42
    )

    # ========================================
    # 测试 2: 客户端 DataLoader
    # ========================================
    print("\n" + "=" * 70)
    print("Test 2: Client DataLoader")
    print("=" * 70)

    client_id = 0
    loader = fed_data.get_client_loader(client_id, batch_size=32, shuffle=True)

    print(f"\nClient {client_id} DataLoader:")
    print(f"  Total samples: {len(fed_data.client_data[client_id])}")
    print(f"  Batch size: 32")
    print(f"  Batches: {len(loader)}")

    # 测试第一个 batch
    for batch_x, batch_y in loader:
        print(f"\n✅ Batch loaded successfully:")
        print(f"   Features shape: {batch_x.shape}")
        print(f"   Labels shape: {batch_y.shape}")
        print(f"   Labels: {batch_y.unique().tolist()}")
        break

    # ========================================
    # 测试 3: 测试集 DataLoader
    # ========================================
    print("\n" + "=" * 70)
    print("Test 3: Test DataLoader")
    print("=" * 70)

    test_loader = fed_data.get_test_loader(batch_size=64)

    print(f"\nTest DataLoader:")
    print(f"  Total samples: {len(fed_data.test_data)}")
    print(f"  Batch size: 64")
    print(f"  Batches: {len(test_loader)}")

    # 测试第一个 batch
    for batch_x, batch_y in test_loader:
        print(f"\n✅ Test batch loaded successfully:")
        print(f"   Features shape: {batch_x.shape}")
        print(f"   Labels shape: {batch_y.shape}")
        break

    # ========================================
    # 测试 4: 攻击率查询
    # ========================================
    print("\n" + "=" * 70)
    print("Test 4: Attack Ratio Query")
    print("=" * 70)

    print("\nAttack ratios for all clients:")
    for i in range(fed_data.n_clients):
        ratio = fed_data.get_client_attack_ratio(i)
        print(f"  Client {i}: {ratio:.2%}")

    # ========================================
    # 测试 5: 缓存攻击率
    # ========================================
    print("\n" + "=" * 70)
    print("Test 5: Cached Attack Ratios")
    print("=" * 70)

    ratios = fed_data.get_all_attack_ratios()
    print(f"\n✅ All attack ratios (cached): {[f'{r:.2%}' for r in ratios]}")

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 70)
    print("All Tests PASSED! ✅")
    print("=" * 70)

    print("\nDataset Summary:")
    print(f"  Train samples: {len(fed_data.train_data)}")
    print(f"  Test samples: {len(fed_data.test_data)}")
    print(f"  Features: {fed_data.input_dim}")
    print(f"  Clients: {fed_data.n_clients}")
    print(f"  Non-IID alpha: {fed_data.alpha}")

    print("\n✅ FederatedUNSWNB15 is ready for federated learning experiments!")
