"""活性化関数 (Activation Functions)

各活性化関数はforwardとbackwardの両方を提供する。
backwardは勾配の逆伝播に使用。
"""

import numpy as np


class ReLU:
    """ReLU: f(x) = max(0, x), f'(x) = 1 if x > 0 else 0"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (self.input > 0)


class Sigmoid:
    """Sigmoid: f(x) = 1/(1+e^-x), f'(x) = f(x)(1-f(x))"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.output * (1 - self.output)


class Tanh:
    """Tanh: f(x) = tanh(x), f'(x) = 1 - tanh(x)^2"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - self.output ** 2)


class Softmax:
    """Softmax: f(x_i) = e^{x_i} / Σ e^{x_j}

    数値安定性のため、maxを引いてからexpを計算する。
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # CrossEntropyLossと組み合わせる場合は直接 (softmax - y) が勾配となるため、
        # 単独で使う場合のヤコビアンベースの実装
        n = self.output.shape[-1]
        jacobian = -self.output[..., :, np.newaxis] * self.output[..., np.newaxis, :]
        diag_idx = np.arange(n)
        jacobian[..., diag_idx, diag_idx] += self.output
        return np.einsum("...ij,...j->...i", jacobian, grad)
