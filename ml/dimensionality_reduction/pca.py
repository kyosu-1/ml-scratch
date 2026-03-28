"""主成分分析 (PCA: Principal Component Analysis)

データの分散が最大となる方向（主成分）を見つけ、次元削減を行う。

アルゴリズム:
1. データを中心化 (平均を引く)
2. 共分散行列を計算: C = (1/n) * X^T X
3. 共分散行列の固有値分解を行う
4. 固有値が大きい順にk個の固有ベクトルを選択
5. 選択した固有ベクトルで射影: Z = X @ W
"""

import numpy as np


class PCA:

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components = None  # 主成分 (固有ベクトル)
        self.mean = None
        self.explained_variance_ = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 共分散行列
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 固有値分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 固有値が大きい順にソート
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # 上位 n_components 個を選択
        self.components = eigenvectors[:, : self.n_components]
        self.explained_variance_ = eigenvalues[: self.n_components]

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        return Z @ self.components.T + self.mean
