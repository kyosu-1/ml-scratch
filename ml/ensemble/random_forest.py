"""ランダムフォレスト (Random Forest)

バギング + 特徴量のランダムサンプリングによるアンサンブル学習。

アルゴリズム:
1. ブートストラップサンプリングでデータのサブセットを作成
2. 各サブセットで決定木を学習（特徴量もランダムに選択）
3. 予測時は全ての木の多数決（分類）/ 平均（回帰）を取る

分散を低減し、過学習を抑制する効果がある。
"""

import numpy as np
from ml.tree.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: str = "sqrt",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def _get_n_features(self, n_features: int) -> int:
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        return n_features

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        self.trees = []
        n_samples, n_features = X.shape
        n_sub_features = self._get_n_features(n_features)

        for _ in range(self.n_estimators):
            # ブートストラップサンプリング
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # 特徴量のランダム選択
            feat_indices = np.random.choice(n_features, n_sub_features, replace=False)
            X_sub = X_boot[:, feat_indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sub, y_boot)
            self.trees.append((tree, feat_indices))

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # 各木の予測を集める
        predictions = np.array([
            tree.predict(X[:, feat_idx]) for tree, feat_idx in self.trees
        ])
        # 多数決
        return np.array([
            np.bincount(predictions[:, i].astype(int)).argmax()
            for i in range(X.shape[0])
        ])
