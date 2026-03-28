"""Attention メカニズム と Transformer

Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(QK^T / √d_k) V

Multi-Head Attention:
    各ヘッドで独立に Q, K, V を線形変換してアテンションを計算し、
    結果を連結して再度線形変換する。

Transformer ブロック:
    Multi-Head Attention → Add & LayerNorm → FFN → Add & LayerNorm
"""

import numpy as np


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class LayerNorm:
    """Layer Normalization

    各サンプルの特徴量方向で正規化する (BatchNormは各バッチ方向)。
    """

    def __init__(self, n_features: int, eps: float = 1e-5):
        self.gamma = np.ones(n_features)
        self.beta = np.zeros(n_features)
        self.eps = eps
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.grad_gamma = np.sum(grad * self.x_hat, axis=tuple(range(grad.ndim - 1)))
        self.grad_beta = np.sum(grad, axis=tuple(range(grad.ndim - 1)))

        n = grad.shape[-1]
        dx_hat = grad * self.gamma
        inv_std = 1.0 / np.sqrt(self.var + self.eps)

        dx = (1.0 / n) * inv_std * (
            n * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - self.x_hat * np.sum(dx_hat * self.x_hat, axis=-1, keepdims=True)
        )
        return dx

    def params_and_grads(self):
        return [
            (self.gamma, self.grad_gamma, "gamma"),
            (self.beta, self.grad_beta, "beta"),
        ]


class ScaledDotProductAttention:
    """Scaled Dot-Product Attention

    入力: Q (batch, seq_len, d_k), K (batch, seq_len, d_k), V (batch, seq_len, d_v)
    出力: (batch, seq_len, d_v)
    """

    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None):
        self.Q, self.K, self.V = Q, K, V
        d_k = Q.shape[-1]

        # QK^T / √d_k
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        self.attn_weights = _softmax(scores, axis=-1)
        return self.attn_weights @ V

    def backward(self, grad: np.ndarray):
        d_k = self.Q.shape[-1]

        # V に関する勾配
        dV = self.attn_weights.transpose(0, 2, 1) @ grad

        # attention weights に関する勾配
        d_attn = grad @ self.V.transpose(0, 2, 1)

        # softmax の勾配
        s = self.attn_weights
        d_scores = s * (d_attn - np.sum(d_attn * s, axis=-1, keepdims=True))
        d_scores /= np.sqrt(d_k)

        dQ = d_scores @ self.K
        dK = d_scores.transpose(0, 2, 1) @ self.Q

        return dQ, dK, dV


class MultiHeadAttention:
    """Multi-Head Attention

    入力: (batch, seq_len, d_model)
    出力: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(1.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        self.grad_W_q = self.grad_W_k = self.grad_W_v = self.grad_W_o = None
        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = x.shape
        return x.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3).reshape(
            batch * self.n_heads, seq_len, self.d_k
        )

    def _merge_heads(self, x: np.ndarray, batch: int) -> np.ndarray:
        seq_len = x.shape[1]
        return x.reshape(batch, self.n_heads, seq_len, self.d_k).transpose(0, 2, 1, 3).reshape(
            batch, seq_len, self.d_model
        )

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        self.input = x
        batch = x.shape[0]

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        self.Q_proj, self.K_proj, self.V_proj = Q, K, V

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attn_out = self.attention.forward(Q, K, V, mask)
        attn_out = self._merge_heads(attn_out, batch)

        self.attn_out = attn_out
        return attn_out @ self.W_o

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch = grad.shape[0]

        self.grad_W_o = self.attn_out.reshape(-1, self.d_model).T @ grad.reshape(-1, self.d_model)
        d_attn_out = grad @ self.W_o.T

        d_attn_out = self._split_heads(d_attn_out)
        dQ, dK, dV = self.attention.backward(d_attn_out)

        dQ = self._merge_heads(dQ, batch)
        dK = self._merge_heads(dK, batch)
        dV = self._merge_heads(dV, batch)

        x_flat = self.input.reshape(-1, self.d_model)
        self.grad_W_q = x_flat.T @ dQ.reshape(-1, self.d_model)
        self.grad_W_k = x_flat.T @ dK.reshape(-1, self.d_model)
        self.grad_W_v = x_flat.T @ dV.reshape(-1, self.d_model)

        dx = dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T
        return dx

    def params_and_grads(self):
        return [
            (self.W_q, self.grad_W_q, "W_q"),
            (self.W_k, self.grad_W_k, "W_k"),
            (self.W_v, self.grad_W_v, "W_v"),
            (self.W_o, self.grad_W_o, "W_o"),
        ]


class TransformerBlock:
    """Transformer Encoder ブロック

    Multi-Head Attention → Add & LayerNorm → FFN → Add & LayerNorm
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)

        scale = np.sqrt(1.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(1.0 / d_ff)
        self.b2 = np.zeros(d_model)

        self.grad_W1 = self.grad_b1 = None
        self.grad_W2 = self.grad_b2 = None

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Multi-Head Attention + 残差接続 + LayerNorm
        attn_out = self.mha.forward(x, mask)
        self.residual1_input = x
        x = self.ln1.forward(x + attn_out)
        self.after_ln1 = x

        # Feed-Forward Network + 残差接続 + LayerNorm
        self.ffn_input = x
        self.hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        ffn_out = self.hidden @ self.W2 + self.b2

        self.residual2_input = x
        x = self.ln2.forward(x + ffn_out)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        # LayerNorm2 の逆伝播
        d = self.ln2.backward(grad)

        # 残差接続
        d_ffn = d
        d_residual2 = d

        # FFN の逆伝播
        self.grad_W2 = self.hidden.reshape(-1, self.hidden.shape[-1]).T @ d_ffn.reshape(-1, d_ffn.shape[-1])
        self.grad_b2 = np.sum(d_ffn, axis=tuple(range(d_ffn.ndim - 1)))

        d_hidden = d_ffn @ self.W2.T
        d_hidden = d_hidden * (self.hidden > 0)  # ReLU backward

        self.grad_W1 = self.ffn_input.reshape(-1, self.ffn_input.shape[-1]).T @ d_hidden.reshape(-1, d_hidden.shape[-1])
        self.grad_b1 = np.sum(d_hidden, axis=tuple(range(d_hidden.ndim - 1)))

        d = d_hidden @ self.W1.T + d_residual2

        # LayerNorm1 の逆伝播
        d = self.ln1.backward(d)

        # 残差接続
        d_attn = d
        d_residual1 = d

        # Multi-Head Attention の逆伝播
        dx = self.mha.backward(d_attn) + d_residual1
        return dx

    def params_and_grads(self):
        return (
            self.mha.params_and_grads()
            + self.ln1.params_and_grads()
            + self.ln2.params_and_grads()
            + [
                (self.W1, self.grad_W1, "W1"),
                (self.b1, self.grad_b1, "b1"),
                (self.W2, self.grad_W2, "W2"),
                (self.b2, self.grad_b2, "b2"),
            ]
        )
