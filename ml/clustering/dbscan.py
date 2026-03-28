"""DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

密度ベースのクラスタリング手法。K-Meansと異なりクラスタ数を事前に指定する必要がない。

パラメータ:
- eps: 近傍の半径
- min_samples: コア点となるために必要な近傍点の最小数

アルゴリズム:
1. 各点について eps 半径内の近傍点を探索
2. min_samples 以上の近傍を持つ点をコア点とする
3. コア点から到達可能な全ての点を同一クラスタに割り当て
4. どのクラスタにも属さない点をノイズ (-1) とする
"""

import numpy as np


class DBSCAN:

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None

    def fit(self, X: np.ndarray) -> "DBSCAN":
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # -1 = ノイズ
        cluster_id = 0

        for i in range(n_samples):
            if self.labels_[i] != -1:
                continue

            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                continue  # ノイズのまま

            # 新しいクラスタを拡張
            self._expand_cluster(X, i, neighbors, cluster_id)
            cluster_id += 1

        return self

    def _region_query(self, X: np.ndarray, idx: int) -> list:
        distances = np.linalg.norm(X - X[idx], axis=1)
        return list(np.where(distances <= self.eps)[0])

    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id
        queue = list(neighbors)
        i = 0
        while i < len(queue):
            q = queue[i]
            if self.labels_[q] == -1:
                self.labels_[q] = cluster_id
            elif self.labels_[q] != -1:
                i += 1
                continue
            # まだ未処理の場合（上の -1 チェックでカバー済み）
            self.labels_[q] = cluster_id
            q_neighbors = self._region_query(X, q)
            if len(q_neighbors) >= self.min_samples:
                queue.extend(q_neighbors)
            i += 1
