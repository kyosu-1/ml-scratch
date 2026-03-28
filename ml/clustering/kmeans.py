"""K-Means クラスタリング

アルゴリズム:
1. K個のセントロイドをランダムに初期化
2. 各データ点を最も近いセントロイドに割り当て
3. 各クラスタの平均をセントロイドとして更新
4. 収束するまで 2-3 を繰り返す

収束条件: セントロイドの変化量がtol以下、またはmax_itersに到達。
"""

import numpy as np


class KMeans:

    def __init__(self, n_clusters: int = 3, max_iters: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X: np.ndarray) -> "KMeans":
        n_samples = X.shape[0]
        # ランダムにセントロイドを初期化
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[indices].copy()

        for _ in range(self.max_iters):
            # 各点を最も近いセントロイドに割り当て
            labels = self._assign_clusters(X)

            # セントロイドを更新
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if (labels == k).sum() > 0 else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            # 収束判定
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break

            self.centroids = new_centroids

        self.labels_ = self._assign_clusters(X)
        return self

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        # 各データ点と各セントロイドのユークリッド距離を計算
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_clusters(X)
