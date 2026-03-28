"""オプティマイザ (Optimizers)

モデルのパラメータを勾配に基づいて更新する。

各オプティマイザは layers のリストを受け取り、
layer.params_and_grads() で取得したパラメータと勾配を更新する。
"""

import numpy as np


class SGD:
    """確率的勾配降下法 (Stochastic Gradient Descent) + モメンタム

    v_t = μ * v_{t-1} + lr * grad
    w_t = w_{t-1} - v_t
    """

    def __init__(self, lr: float = 0.01, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}

    def step(self, layers):
        for layer in layers:
            if not hasattr(layer, "params_and_grads"):
                continue
            for param, grad, name in layer.params_and_grads():
                if grad is None:
                    continue
                key = id(param)
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(param)
                v = self.velocities[key]
                v[:] = self.momentum * v + self.lr * grad
                param -= v


class Adam:
    """Adam (Adaptive Moment Estimation)

    m_t = β1 * m_{t-1} + (1-β1) * g_t          (1次モーメント)
    v_t = β2 * v_{t-1} + (1-β2) * g_t^2         (2次モーメント)
    m_hat = m_t / (1 - β1^t)                     (バイアス補正)
    v_hat = v_t / (1 - β2^t)
    w_t = w_{t-1} - lr * m_hat / (√v_hat + ε)
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, "params_and_grads"):
                continue
            for param, grad, name in layer.params_and_grads():
                if grad is None:
                    continue
                key = id(param)
                if key not in self.m:
                    self.m[key] = np.zeros_like(param)
                    self.v[key] = np.zeros_like(param)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
