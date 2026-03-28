"""埋め込み層 (Embedding Layer)

離散的なトークンID（整数）を固定長の密ベクトルに変換する。
実質的にはルックアップテーブル。

forward: indices → embedding_matrix[indices]
backward: 該当インデックスに勾配を加算
"""

import numpy as np


class Embedding:

    def __init__(self, vocab_size: int, embed_dim: int):
        self.weights = np.random.randn(vocab_size, embed_dim) * 0.01
        self.grad_weights = None

    def forward(self, indices: np.ndarray) -> np.ndarray:
        """indices: (batch, seq_len) の整数配列"""
        self.indices = indices
        return self.weights[indices]

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_weights = np.zeros_like(self.weights)
        np.add.at(self.grad_weights, self.indices, grad)
        return grad  # 入力が離散値なので勾配は形式的

    def params_and_grads(self):
        return [(self.weights, self.grad_weights, "weights")]
