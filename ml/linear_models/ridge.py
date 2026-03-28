"""Ridge回帰 (L2正則化付き線形回帰)

損失関数: L = (1/2n) * ||y - Xw||^2 + α * ||w||^2

正則化項 α*||w||^2 により、重みの大きさにペナルティを課す。
多重共線性の問題を緩和し、過学習を抑制する。

解析解: w = (X^T X + αI)^{-1} X^T y
勾配: dw = (1/n) * X^T(Xw - y) + 2αw
"""

import numpy as np


class Ridge:
    """解析解によるRidge回帰"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        X_b = np.c_[np.ones(X.shape[0]), X]
        n_features = X_b.shape[1]
        # バイアス項には正則化をかけない
        reg_matrix = self.alpha * np.eye(n_features)
        reg_matrix[0, 0] = 0
        self.weights = np.linalg.inv(X_b.T @ X_b + reg_matrix) @ X_b.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights
