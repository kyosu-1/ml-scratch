"""サポートベクターマシン (SVM) - ヒンジ損失 + L2正則化

線形SVMを勾配降下法で学習する。

損失関数:
    L = (1/n) Σ max(0, 1 - y_i * (w・x_i + b)) + λ * ||w||^2

勾配:
    if y_i * (w・x_i + b) >= 1:
        dw = 2λw
        db = 0
    else:
        dw = 2λw - y_i * x_i
        db = -y_i
"""

import numpy as np


class SVM:

    def __init__(self, lr: float = 0.001, lambda_param: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        # ラベルを {-1, 1} に変換
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            for idx in range(n_samples):
                condition = y_[idx] * (X[idx] @ self.weights + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.lr * (
                        2 * self.lambda_param * self.weights - y_[idx] * X[idx]
                    )
                    self.bias -= self.lr * (-y_[idx])

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(X @ self.weights + self.bias).astype(int)
