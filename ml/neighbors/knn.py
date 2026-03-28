"""k近傍法 (k-Nearest Neighbors)

怠惰学習 (lazy learning) の代表的アルゴリズム。
学習フェーズではデータを保存するだけで、予測時にk個の近傍点を探索する。

分類: k個の近傍点の多数決
回帰: k個の近傍点の平均値

距離関数はユークリッド距離を使用。
計算量: 予測時 O(n * d) (n: データ数, d: 次元数)
"""

import numpy as np


class KNNClassifier:

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> int:
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        k_labels = self.y_train[k_indices]
        counts = np.bincount(k_labels.astype(int))
        return int(np.argmax(counts))


class KNNRegressor:

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x: np.ndarray) -> float:
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_indices = np.argsort(distances)[: self.k]
        return float(np.mean(self.y_train[k_indices]))
