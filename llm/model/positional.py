"""位置エンコーディング (Positional Encoding)

Transformerは系列の順序情報を持たない（Self-Attentionは位置に依存しない集合演算）。
位置エンコーディングを加算することで系列の順序を表現する。

正弦波位置エンコーディング (Sinusoidal):
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    各次元が異なる周波数のsin/cosを使い、位置をユニークに表現する。
    相対位置の情報も線形変換で取り出せるという理論的な利点がある。

学習可能な位置エンコーディング (Learnable):
    位置ごとの埋め込みベクトルを学習する。GPT系で一般的。
"""

import numpy as np


class SinusoidalPositionalEncoding:
    """正弦波位置エンコーディング（固定、学習不要）"""

    def __init__(self, d_model: int, max_len: int = 512):
        self.encoding = np.zeros((max_len, d_model))

        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
        )

        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.shape[1]
        return x + self.encoding[:seq_len]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad  # 加算なのでそのまま

    def params_and_grads(self):
        return []


class LearnablePositionalEncoding:
    """学習可能な位置エンコーディング（GPTスタイル）"""

    def __init__(self, max_len: int, d_model: int):
        self.weights = np.random.randn(max_len, d_model) * 0.01
        self.grad_weights = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, d_model)"""
        self.seq_len = x.shape[1]
        return x + self.weights[: self.seq_len]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_weights[: self.seq_len] = np.sum(grad, axis=0)
        return grad

    def params_and_grads(self):
        return [(self.weights, self.grad_weights, "pos_weights")]
