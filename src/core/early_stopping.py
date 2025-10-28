"""
早停机制模块
"""


class EarlyStopping:
    """
    早停机制

    Args:
        patience: 容忍轮数（当验证分数连续 patience 轮未改善时触发早停）
        min_delta: 最小改进量（小于此值视为未改善）

    Example:
        >>> early_stopping = EarlyStopping(patience=5, min_delta=0.001)
        >>> for epoch in range(100):
        ...     val_score = train_epoch()
        ...     if early_stopping(val_score):
        ...         print("Early stopping triggered!")
        ...         break
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score: float) -> bool:
        """
        检查是否应该早停

        Args:
            val_score: 当前验证分数（越高越好）

        Returns:
            是否触发早停（True 表示应该停止训练）
        """
        if self.best_score is None:
            # 第一次调用，记录初始分数
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            # 分数未改善
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 分数改善，重置计数器
            self.best_score = val_score
            self.counter = 0

        return self.early_stop

    def reset(self):
        """重置早停状态（用于多轮实验）"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
