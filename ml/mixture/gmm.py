"""ガウス混合モデル (GMM) + EMアルゴリズム

K-Meansの確率的な拡張。各クラスタがガウス分布で表現され、
データ点が各クラスタに属する「確率」（soft assignment）を計算する。

モデル:
    p(x) = Σ_k π_k N(x | μ_k, Σ_k)

    π_k: 混合係数（各クラスタの重み、Σπ_k = 1）
    μ_k: 各クラスタの平均
    Σ_k: 各クラスタの共分散行列

EMアルゴリズム:
    E-step: 各データ点の各クラスタへの所属確率（責任度）を計算
        γ(z_nk) = π_k N(x_n|μ_k,Σ_k) / Σ_j π_j N(x_n|μ_j,Σ_j)

    M-step: 責任度を使ってパラメータを更新
        N_k = Σ_n γ(z_nk)
        μ_k = (1/N_k) Σ_n γ(z_nk) x_n
        Σ_k = (1/N_k) Σ_n γ(z_nk) (x_n - μ_k)(x_n - μ_k)^T
        π_k = N_k / N
"""

import numpy as np


class GaussianMixture:

    def __init__(self, n_components: int = 3, n_iters: int = 100, tol: float = 1e-6):
        self.n_components = n_components
        self.n_iters = n_iters
        self.tol = tol

    def _gaussian_pdf(self, X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """多変量ガウス分布の確率密度関数"""
        n_features = X.shape[1]
        diff = X - mean
        cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(n_features))
        cov_det = np.linalg.det(cov + 1e-6 * np.eye(n_features))
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** n_features * np.abs(cov_det)) + 1e-10)
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        return norm_const * np.exp(exponent)

    def fit(self, X: np.ndarray) -> "GaussianMixture":
        n_samples, n_features = X.shape
        K = self.n_components

        # 初期化: K-Means的にランダムに選択
        indices = np.random.choice(n_samples, K, replace=False)
        self.means_ = X[indices].copy()
        self.covariances_ = np.array([np.eye(n_features) for _ in range(K)])
        self.weights_ = np.ones(K) / K

        log_likelihood_old = -np.inf

        for _ in range(self.n_iters):
            # E-step: 責任度の計算
            responsibilities = self._e_step(X)

            # M-step: パラメータの更新
            self._m_step(X, responsibilities)

            # 対数尤度の計算（収束判定）
            log_likelihood = self._log_likelihood(X)
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood

        self.responsibilities_ = self._e_step(X)
        self.labels_ = np.argmax(self.responsibilities_, axis=1)
        return self

    def _e_step(self, X: np.ndarray) -> np.ndarray:
        """各データ点の各クラスタへの責任度を計算"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )

        # 正規化
        total = responsibilities.sum(axis=1, keepdims=True) + 1e-10
        return responsibilities / total

    def _m_step(self, X: np.ndarray, responsibilities: np.ndarray):
        """責任度に基づいてパラメータを更新"""
        n_samples, n_features = X.shape

        for k in range(self.n_components):
            resp_k = responsibilities[:, k]
            N_k = resp_k.sum() + 1e-10

            # 平均の更新
            self.means_[k] = (resp_k[:, np.newaxis] * X).sum(axis=0) / N_k

            # 共分散の更新
            diff = X - self.means_[k]
            self.covariances_[k] = (resp_k[:, np.newaxis, np.newaxis] *
                                     (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / N_k

            # 混合係数の更新
            self.weights_[k] = N_k / n_samples

    def _log_likelihood(self, X: np.ndarray) -> float:
        likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            likelihood += self.weights_[k] * self._gaussian_pdf(
                X, self.means_[k], self.covariances_[k]
            )
        return np.sum(np.log(likelihood + 1e-10))

    def predict(self, X: np.ndarray) -> np.ndarray:
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._e_step(X)
