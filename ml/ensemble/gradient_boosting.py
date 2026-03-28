"""勾配ブースティング (Gradient Boosting)

弱学習器（決定木）を逐次的に追加し、前のモデルの残差を学習する。

回帰の場合:
    F_0(x) = mean(y)
    F_m(x) = F_{m-1}(x) + lr * h_m(x)

    ここで h_m は残差 r_m = y - F_{m-1}(x) を学習した決定木。

損失関数にMSEを使う場合、負の勾配 = 残差 となる。
"""

import numpy as np
from ml.tree.decision_tree import DecisionTreeRegressor


class GradientBoostingRegressor:

    def __init__(
        self,
        n_estimators: int = 100,
        lr: float = 0.1,
        max_depth: int = 3,
    ):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        self.initial_prediction = np.mean(y)
        current_pred = np.full(len(y), self.initial_prediction)

        self.trees = []
        for _ in range(self.n_estimators):
            # 残差 (負の勾配) を計算
            residuals = y - current_pred

            # 残差を学習する決定木を作成
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)

            # 予測を更新
            current_pred += self.lr * tree.predict(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            pred += self.lr * tree.predict(X)
        return pred
