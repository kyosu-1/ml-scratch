"""決定木 (Decision Tree) - CART アルゴリズム

分類・回帰の両方に対応。
- 分類: ジニ不純度 (Gini Impurity) で分割
    Gini(S) = 1 - Σ p_k^2
- 回帰: 分散 (Variance) の削減で分割

各ノードで全特徴量・全閾値を走査し、最適な分割を貪欲に選択する。
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Node:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    value: Optional[float] = None  # 葉ノードの予測値


class DecisionTreeClassifier:

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        self.n_classes = len(np.unique(y))
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _gini(self, y: np.ndarray) -> float:
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        best_gain = -1
        best_feat, best_thresh = None, None

        parent_gini = self._gini(y)
        n = len(y)

        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                right_mask = ~left_mask

                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                # 情報利得 = 親のジニ - 子のジニの加重平均
                gain = parent_gini - (
                    left_mask.sum() / n * self._gini(y[left_mask])
                    + right_mask.sum() / n * self._gini(y[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        # 終了条件: 純粋なノード, 最大深度, 最小サンプル数
        if (
            len(np.unique(y)) == 1
            or depth >= self.max_depth
            or len(y) < self.min_samples_split
        ):
            # 最頻クラスを予測値とする
            values, counts = np.unique(y, return_counts=True)
            return Node(value=values[np.argmax(counts)])

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            values, counts = np.unique(y, return_counts=True)
            return Node(value=values[np.argmax(counts)])

        left_mask = X[:, feat_idx] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)

        return Node(feature_idx=feat_idx, threshold=threshold, left=left, right=right)

    def _predict_one(self, x: np.ndarray, node: Node) -> float:
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(x, self.root) for x in X])


class DecisionTreeRegressor:

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def _variance_reduction(self, y, left_mask):
        n = len(y)
        if left_mask.sum() == 0 or (~left_mask).sum() == 0:
            return 0
        return np.var(y) - (
            left_mask.sum() / n * np.var(y[left_mask])
            + (~left_mask).sum() / n * np.var(y[~left_mask])
        )

    def _best_split(self, X, y):
        best_gain = -1
        best_feat, best_thresh = None, None

        for feat_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feat_idx])
            for thresh in thresholds:
                left_mask = X[:, feat_idx] <= thresh
                gain = self._variance_reduction(y, left_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _grow_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return Node(value=np.mean(y))

        feat_idx, threshold = self._best_split(X, y)
        if feat_idx is None:
            return Node(value=np.mean(y))

        left_mask = X[:, feat_idx] <= threshold
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature_idx=feat_idx, threshold=threshold, left=left, right=right)

    def _predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])
