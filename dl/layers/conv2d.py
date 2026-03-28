"""2次元畳み込み層 (Conv2D)

入力: (batch, in_channels, H, W)
出力: (batch, out_channels, H_out, W_out)

H_out = (H + 2*padding - kernel_size) / stride + 1

forward: im2col で行列演算に変換して畳み込みを計算
backward: col2im で勾配を戻す

im2col: 畳み込みの窓を列ベクトルとして並べ、行列積で一括計算する高速化手法。
"""

import numpy as np


class Conv2D:

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He初期化
        fan_in = in_channels * kernel_size * kernel_size
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / fan_in)
        self.bias = np.zeros(out_channels)

        self.grad_weights = None
        self.grad_bias = None

    def _im2col(self, x: np.ndarray) -> np.ndarray:
        batch, c, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        h_out = (h + 2 * self.padding - k) // s + 1
        w_out = (w + 2 * self.padding - k) // s + 1

        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            )

        cols = np.zeros((batch, c, k, k, h_out, w_out))
        for i in range(k):
            i_max = i + s * h_out
            for j in range(k):
                j_max = j + s * w_out
                cols[:, :, i, j, :, :] = x[:, :, i:i_max:s, j:j_max:s]

        # (batch, c*k*k, h_out*w_out)
        cols = cols.reshape(batch, c * k * k, h_out * w_out)
        return cols

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        batch, c, h, w = x.shape
        k = self.kernel_size
        s = self.stride
        h_out = (h + 2 * self.padding - k) // s + 1
        w_out = (w + 2 * self.padding - k) // s + 1

        self.cols = self._im2col(x)

        # weights を (out_channels, in_channels*k*k) に変形
        w_reshaped = self.weights.reshape(self.out_channels, -1)

        # 畳み込み = 行列積
        out = w_reshaped @ self.cols + self.bias.reshape(-1, 1, 1)
        out = out.reshape(batch, self.out_channels, h_out, w_out)

        self.input = x
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch, _, h_out, w_out = grad.shape

        # grad を (batch, out_channels, h_out*w_out) に変形
        grad_reshaped = grad.reshape(batch, self.out_channels, -1)

        w_reshaped = self.weights.reshape(self.out_channels, -1)

        # 勾配計算
        self.grad_weights = np.sum(
            grad_reshaped @ self.cols.transpose(0, 2, 1), axis=0
        ).reshape(self.weights.shape)
        self.grad_bias = np.sum(grad_reshaped, axis=(0, 2))

        # 入力に対する勾配
        d_cols = w_reshaped.T @ grad_reshaped
        # col2im (簡易版: im2colの逆操作)
        dx = self._col2im(d_cols, self.input_shape)
        return dx

    def _col2im(self, cols, input_shape):
        batch, c, h, w = input_shape
        k = self.kernel_size
        s = self.stride
        h_out = (h + 2 * self.padding - k) // s + 1
        w_out = (w + 2 * self.padding - k) // s + 1

        h_padded = h + 2 * self.padding
        w_padded = w + 2 * self.padding
        dx_padded = np.zeros((batch, c, h_padded, w_padded))

        cols = cols.reshape(batch, c, k, k, h_out, w_out)
        for i in range(k):
            i_max = i + s * h_out
            for j in range(k):
                j_max = j + s * w_out
                dx_padded[:, :, i:i_max:s, j:j_max:s] += cols[:, :, i, j, :, :]

        if self.padding > 0:
            return dx_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return dx_padded

    def params_and_grads(self):
        return [
            (self.weights, self.grad_weights, "weights"),
            (self.bias, self.grad_bias, "bias"),
        ]
