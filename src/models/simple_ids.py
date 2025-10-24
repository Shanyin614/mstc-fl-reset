# src/models/simple_ids.py
import torch.nn as nn


class SimpleIDS(nn.Module):
    """简单的 IDS 模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(SimpleIDS, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
