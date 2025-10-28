"""
自定义损失函数模块
"""
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss - 关注难分类样本，抑制简单样本的损失

    公式: FL(p_t) = -α(1-p_t)^γ * log(p_t)

    解决问题：
        - 类别不平衡（攻击样本 vs 正常样本）
        - 难分类样本权重不足

    Args:
        alpha: 类别权重（0.25 表示正类权重更高）
        gamma: 聚焦参数（2.0 表示难样本权重指数级增加）
        reduction: 损失归约方式 ('mean', 'sum', 'none')

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> outputs = model(inputs)  # [batch_size, 2]
        >>> loss = criterion(outputs, targets)  # [batch_size]
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算 Focal Loss

        Args:
            inputs: 模型输出 logits [batch_size, num_classes]
            targets: 真实标签 [batch_size]

        Returns:
            损失值 (scalar)
        """
        # 计算交叉熵损失（不进行归约）
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # 计算预测概率 p_t
        p_t = torch.exp(-ce_loss)

        # 应用 Focal Loss 公式
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        # 根据 reduction 参数返回
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

    @staticmethod
    def get_adaptive_params(attack_ratio: float) -> tuple:
        """
        根据攻击比例动态调整 Focal Loss 参数

        策略：
            - 攻击率低（< 0.2）：提高攻击类权重（alpha=0.5）
            - 攻击率高（> 0.8）：提高正常类权重（alpha=0.1）
            - 中等攻击率：平衡权重（alpha=0.25）

        Args:
            attack_ratio: 攻击样本比例 [0, 1]

        Returns:
            (alpha, gamma) 元组

        Example:
            >>> alpha, gamma = FocalLoss.get_adaptive_params(0.15)
            >>> criterion = FocalLoss(alpha=alpha, gamma=gamma)
        """
        if attack_ratio < 0.2:  # 几乎无攻击（Client 1, 8）
            return 0.50, 3.0  # 大幅提高攻击类权重
        elif attack_ratio > 0.8:  # 高攻击率（Client 0, 6, 9）
            return 0.10, 3.0  # 提高正常类权重
        else:
            return 0.25, 2.5  # 平衡权重
