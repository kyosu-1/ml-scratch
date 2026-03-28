"""Lasso回帰 (L1正則化付き線形回帰)

損失関数: L = (1/2n) * ||y - Xw||^2 + α * ||w||_1

L1正則化はスパース解を生む（不要な特徴量の重みを0にする）。
特徴量選択の効果がある。

座標降下法 (Coordinate Descent) で最適化:
    各特徴量 j について、他の特徴量を固定して最適な w_j を求める。
    w_j = soft_threshold(ρ_j, α) / z_j

    ρ_j = X_j^T (y - Xw + w_j * X_j)  (j番目を除いた残差)
    z_j = X_j^T X_j
    soft_threshold(ρ, α) = sign(ρ) * max(|ρ| - α, 0)
"""

import numpy as np


def _soft_threshold(rho: float, alpha: float) -> float:
    if rho > alpha:
        return rho - alpha
    elif rho < -alpha:
        return rho + alpha
    return 0.0


class Lasso:

    def __init__(self, alpha: float = 1.0, n_iters: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.n_iters = n_iters
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Lasso":
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = np.mean(y)

        for _ in range(self.n_iters):
            weights_old = self.weights.copy()

            for j in range(n_features):
                # j番目の特徴量を除いた予測
                residual = y - self.bias - X @ self.weights + self.weights[j] * X[:, j]
                rho_j = X[:, j] @ residual
                z_j = X[:, j] @ X[:, j]

                if z_j == 0:
                    self.weights[j] = 0
                else:
                    self.weights[j] = _soft_threshold(rho_j, n_samples * self.alpha) / z_j

            self.bias = np.mean(y - X @ self.weights)

            if np.max(np.abs(self.weights - weights_old)) < self.tol:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
