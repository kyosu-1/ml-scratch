"""バッチ正規化 (Batch Normalization)

ミニバッチ単位で入力を正規化し、学習を安定化・高速化する。

forward (学習時):
    μ = mean(x, axis=0)
    σ² = var(x, axis=0)
    x_hat = (x - μ) / √(σ² + ε)
    y = γ * x_hat + β

forward (推論時):
    移動平均の μ, σ² を使用

backward:
    dγ = Σ (dL/dy * x_hat)
    dβ = Σ dL/dy
    dx_hat = dL/dy * γ
    dx = (1/N) * (1/√(σ²+ε)) * (N*dx_hat - Σdx_hat - x_hat*Σ(dx_hat*x_hat))
"""

import numpy as np


class BatchNorm1D:

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.eps = eps
        self.momentum = momentum

        # 推論時の移動平均
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)

        self.grad_gamma = None
        self.grad_beta = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.batch_mean = np.mean(x, axis=0)
            self.batch_var = np.var(x, axis=0)

            self.x_hat = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            out = self.gamma * self.x_hat + self.beta

            # 移動平均を更新
            self.running_mean = (
                (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            )
            self.running_var = (
                (1 - self.momentum) * self.running_var + self.momentum * self.batch_var
            )

            self.input = x
            return out
        else:
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_hat + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        n = grad.shape[0]

        self.grad_gamma = np.sum(grad * self.x_hat, axis=0)
        self.grad_beta = np.sum(grad, axis=0)

        dx_hat = grad * self.gamma
        inv_std = 1.0 / np.sqrt(self.batch_var + self.eps)

        dx = (1.0 / n) * inv_std * (
            n * dx_hat
            - np.sum(dx_hat, axis=0)
            - self.x_hat * np.sum(dx_hat * self.x_hat, axis=0)
        )
        return dx

    def params_and_grads(self):
        return [
            (self.gamma, self.grad_gamma, "gamma"),
            (self.beta, self.grad_beta, "beta"),
        ]
