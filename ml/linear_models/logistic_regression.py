"""ロジスティック回帰 (Logistic Regression)

モデル: p(y=1|x) = sigmoid(Xw + b)
損失関数: Binary Cross-Entropy
    L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]

勾配:
    dw = (1/n) * X^T (p - y)
    db = (1/n) * Σ (p - y)
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:

    def __init__(self, lr: float = 0.01, n_iters: int = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            z = X @ self.weights + self.bias
            p = sigmoid(z)

            dw = (1 / n_samples) * (X.T @ (p - y))
            db = (1 / n_samples) * np.sum(p - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.weights + self.bias
        return sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
