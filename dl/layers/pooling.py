"""プーリング層 (Pooling Layers)

空間方向のダウンサンプリングを行い、特徴マップのサイズを縮小する。

MaxPool2D:
    各ウィンドウ内の最大値を取る。
    backward: 最大値の位置にのみ勾配を伝播。

Flatten:
    多次元テンソルを1次元に展開する。
"""

import numpy as np


class MaxPool2D:
    """2次元最大プーリング

    入力: (batch, channels, H, W)
    出力: (batch, channels, H//pool_size, W//pool_size)
    """

    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch, c, h, w = x.shape
        p = self.pool_size
        h_out = h // p
        w_out = w // p

        # reshape して max を取る
        x_reshaped = x.reshape(batch, c, h_out, p, w_out, p)
        out = x_reshaped.max(axis=(3, 5))

        # backward 用に最大値の位置を保存
        self.max_mask = (x_reshaped == out[:, :, :, np.newaxis, :, np.newaxis])
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch, c, h_out, w_out = grad.shape
        p = self.pool_size

        # 勾配を最大値の位置にのみ伝播
        grad_expanded = grad[:, :, :, np.newaxis, :, np.newaxis]
        dx = self.max_mask * grad_expanded

        # 複数の最大値がある場合に備えて正規化
        mask_sum = self.max_mask.sum(axis=(3, 5), keepdims=True)
        mask_sum = np.maximum(mask_sum, 1)
        dx = dx / mask_sum

        return dx.reshape(self.input.shape)

    def params_and_grads(self):
        return []


class Flatten:
    """多次元テンソルを (batch, -1) に展開"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(self.input_shape)

    def params_and_grads(self):
        return []
