"""全結合層 (Linear / Fully Connected Layer)

y = Xw + b

forward: 入力xを保存し、Xw + bを計算
backward: 勾配を計算して返す
    dL/dw = X^T @ dL/dy
    dL/db = sum(dL/dy, axis=0)
    dL/dX = dL/dy @ w^T

重みの初期化: He初期化 (ReLU向け)
    w ~ N(0, sqrt(2/fan_in))
"""

import numpy as np


class Linear:

    def __init__(self, in_features: int, out_features: int):
        # He初期化
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(
            2.0 / in_features
        )
        self.bias = np.zeros(out_features)

        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_weights = self.input.T @ grad
        self.grad_bias = np.sum(grad, axis=0)
        return grad @ self.weights.T

    def params_and_grads(self):
        return [
            (self.weights, self.grad_weights, "weights"),
            (self.bias, self.grad_bias, "bias"),
        ]
