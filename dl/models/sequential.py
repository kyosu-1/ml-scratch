"""Sequential モデル

層を順番に積み重ねてニューラルネットワークを構築する。

使い方:
    model = Sequential([
        Linear(784, 128),
        ReLU(),
        Linear(128, 10),
        Softmax(),
    ])
    out = model.forward(x)
    model.backward(grad)
"""

import numpy as np


class Sequential:

    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, "training"):
                layer.training = False
