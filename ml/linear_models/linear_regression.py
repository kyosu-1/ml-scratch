"""線形回帰 (Linear Regression)

モデル: y = Xw + b
損失関数: L = (1/2n) * ||y - (Xw + b)||^2  (MSE)
勾配降下法でパラメータを更新する。

正規方程式による解析解: w = (X^T X)^{-1} X^T y も実装。
"""

import numpy as np


class LinearRegression:
    """勾配降下法による線形回帰"""

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = X @ self.weights + self.bias

            # 勾配計算
            dw = (1 / n_samples) * (X.T @ (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias


class LinearRegressionNormal:
    """正規方程式による線形回帰 (解析解)

    w = (X^T X)^{-1} X^T y
    """

    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionNormal":
        # バイアス項を追加 (1の列を先頭に追加)
        X_b = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights
