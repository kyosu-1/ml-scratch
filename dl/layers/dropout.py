"""ドロップアウト (Dropout)

学習時にランダムにニューロンを無効化して過学習を抑制する。

forward (学習時):
    mask ~ Bernoulli(1 - p)
    y = x * mask / (1 - p)   ← inverted dropout

forward (推論時):
    y = x  (スケーリング済みのため何もしない)
"""

import numpy as np


class Dropout:

    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
            return x * self.mask
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.mask

    def params_and_grads(self):
        return []
