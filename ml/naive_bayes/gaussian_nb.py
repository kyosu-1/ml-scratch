"""ガウシアンナイーブベイズ (Gaussian Naive Bayes)

ベイズの定理と条件付き独立性の仮定に基づく分類器。

P(y|x) ∝ P(y) * Π P(x_i|y)

各特徴量がクラスごとにガウス分布に従うと仮定:
    P(x_i|y) = (1/√(2πσ²)) * exp(-(x_i - μ)² / (2σ²))

対数を取って計算を安定化:
    log P(y|x) = log P(y) + Σ log P(x_i|y)
"""

import numpy as np


class GaussianNB:

    def __init__(self):
        self.classes_ = None
        self.mean_ = None      # 各クラスの各特徴量の平均
        self.var_ = None       # 各クラスの各特徴量の分散
        self.priors_ = None    # 各クラスの事前確率

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNB":
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.mean_ = np.zeros((n_classes, n_features))
        self.var_ = np.zeros((n_classes, n_features))
        self.priors_ = np.zeros(n_classes)

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[idx] = X_c.mean(axis=0)
            self.var_[idx] = X_c.var(axis=0) + 1e-9  # ゼロ除算防止
            self.priors_[idx] = len(X_c) / len(X)

        return self

    def _log_likelihood(self, X: np.ndarray) -> np.ndarray:
        """各クラスの対数尤度を計算"""
        n_classes = len(self.classes_)
        log_likelihoods = np.zeros((X.shape[0], n_classes))

        for idx in range(n_classes):
            log_prior = np.log(self.priors_[idx])
            # ガウス分布の対数確率密度
            log_prob = -0.5 * np.sum(
                np.log(2 * np.pi * self.var_[idx])
                + (X - self.mean_[idx]) ** 2 / self.var_[idx],
                axis=1,
            )
            log_likelihoods[:, idx] = log_prior + log_prob

        return log_likelihoods

    def predict(self, X: np.ndarray) -> np.ndarray:
        log_likelihoods = self._log_likelihood(X)
        return self.classes_[np.argmax(log_likelihoods, axis=1)]
