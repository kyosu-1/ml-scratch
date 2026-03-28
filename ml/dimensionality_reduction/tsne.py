"""t-SNE (t-distributed Stochastic Neighbor Embedding)

高次元データを低次元（通常2D）に可視化するための非線形次元削減手法。

アルゴリズム:
1. 高次元空間での条件付き確率 p_{j|i} を計算（ガウス分布）
    p_{j|i} = exp(-||x_i - x_j||^2 / 2σ_i^2) / Σ_{k≠i} exp(-||x_i - x_k||^2 / 2σ_i^2)
    対称化: p_{ij} = (p_{j|i} + p_{i|j}) / 2n

2. 低次元空間での類似度 q_{ij} をStudent-t分布で計算
    q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / Σ_{k≠l} (1 + ||y_k - y_l||^2)^{-1}

3. KLダイバージェンスを最小化するように低次元座標を勾配降下法で更新
    dC/dy_i = 4 * Σ_j (p_{ij} - q_{ij}) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^{-1}

perplexityパラメータでσ_iを制御（二分探索）。
"""

import numpy as np


class TSNE:

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        lr: float = 200.0,
        n_iters: int = 1000,
        momentum: float = 0.8,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.lr = lr
        self.n_iters = n_iters
        self.momentum = momentum

    def _compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        sum_sq = np.sum(X ** 2, axis=1)
        distances = sum_sq[:, np.newaxis] + sum_sq[np.newaxis, :] - 2 * X @ X.T
        return np.maximum(distances, 0)

    def _binary_search_sigma(self, distances_i: np.ndarray, target_perplexity: float):
        """二分探索でσ_iを求め、条件付き確率を返す"""
        lo, hi = 1e-20, 1e4
        target_entropy = np.log(target_perplexity)

        for _ in range(50):
            sigma = (lo + hi) / 2
            p = np.exp(-distances_i / (2 * sigma ** 2))
            p_sum = max(np.sum(p), 1e-10)
            p_normalized = p / p_sum

            entropy = -np.sum(p_normalized * np.log(np.maximum(p_normalized, 1e-10)))

            if entropy > target_entropy:
                hi = sigma
            else:
                lo = sigma

            if abs(entropy - target_entropy) < 1e-5:
                break

        return p_normalized

    def _compute_joint_probabilities(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        distances = self._compute_pairwise_distances(X)
        P = np.zeros((n, n))

        for i in range(n):
            # i番目を除いた距離
            dist_i = distances[i].copy()
            dist_i[i] = np.inf
            P[i] = self._binary_search_sigma(dist_i, self.perplexity)
            P[i, i] = 0

        # 対称化
        P = (P + P.T) / (2 * n)
        P = np.maximum(P, 1e-12)
        return P

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]

        # 高次元の同時確率
        P = self._compute_joint_probabilities(X)
        # Early exaggeration
        P *= 4.0

        # 低次元座標をランダム初期化
        Y = np.random.randn(n, self.n_components) * 0.01
        velocity = np.zeros_like(Y)

        for iteration in range(self.n_iters):
            # Early exaggerationを解除
            if iteration == 100:
                P /= 4.0

            # 低次元での距離と類似度 q_{ij}
            dist_Y = self._compute_pairwise_distances(Y)
            inv_dist = 1.0 / (1.0 + dist_Y)
            np.fill_diagonal(inv_dist, 0)
            Q = inv_dist / max(np.sum(inv_dist), 1e-10)
            Q = np.maximum(Q, 1e-12)

            # 勾配計算
            PQ_diff = P - Q
            grad = np.zeros_like(Y)
            for i in range(n):
                diff = Y[i] - Y
                grad[i] = 4 * np.sum(
                    (PQ_diff[i] * inv_dist[i])[:, np.newaxis] * diff, axis=0
                )

            # モメンタム付き更新
            velocity = self.momentum * velocity - self.lr * grad
            Y += velocity

            # 中心化
            Y -= np.mean(Y, axis=0)

        return Y
