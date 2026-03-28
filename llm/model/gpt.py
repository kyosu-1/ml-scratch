"""GPT (Generative Pre-trained Transformer)

Transformer Decoderのみで構成される自己回帰言語モデル。

構造:
    Token Embedding + Position Embedding
    → N × [Masked Multi-Head Attention → Add & LN → FFN → Add & LN]
    → Layer Norm
    → Linear (語彙サイズ)

因果的マスク (Causal Mask):
    各トークンは自分より前のトークンしか参照できない。
    未来の情報を見せないことで、自己回帰的な生成を可能にする。
"""

import numpy as np


def _softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _gelu(x):
    """GELU活性化関数: x * Φ(x) の近似"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _gelu_backward(x, grad):
    """GELUの勾配"""
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    t = np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)
    sech2 = 1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)) ** 2
    return grad * (cdf + x * 0.5 * t * sech2)


class CausalSelfAttention:
    """因果的自己注意機構

    未来のトークンへのAttentionをマスクする。
    """

    def __init__(self, d_model: int, n_heads: int):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        scale = np.sqrt(1.0 / d_model)
        self.W_qkv = np.random.randn(d_model, 3 * d_model) * scale
        self.b_qkv = np.zeros(3 * d_model)
        self.W_o = np.random.randn(d_model, d_model) * scale
        self.b_o = np.zeros(d_model)

        self.grad_W_qkv = self.grad_b_qkv = None
        self.grad_W_o = self.grad_b_o = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (batch, seq_len, d_model)"""
        self.input = x
        batch, seq_len, _ = x.shape
        D = self.d_model

        # Q, K, V を一括計算
        qkv = x @ self.W_qkv + self.b_qkv  # (batch, seq, 3*D)
        Q, K, V = qkv[:, :, :D], qkv[:, :, D:2*D], qkv[:, :, 2*D:]

        # ヘッド分割: (batch, n_heads, seq, d_k)
        Q = Q.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention + 因果マスク
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        # 因果マスク: 上三角を -inf にして未来を隠す
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        scores[:, :, causal_mask] = -1e9

        self.attn_weights = _softmax(scores, axis=-1)
        attn_out = self.attn_weights @ V  # (batch, n_heads, seq, d_k)

        # ヘッド結合: (batch, seq, d_model)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, D)
        self.attn_out = attn_out

        return attn_out @ self.W_o + self.b_o

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch, seq_len, D = grad.shape

        # W_o の勾配
        self.grad_W_o = self.attn_out.reshape(-1, D).T @ grad.reshape(-1, D)
        self.grad_b_o = np.sum(grad, axis=(0, 1))
        d_attn_out = grad @ self.W_o.T

        # ヘッド分割に戻す
        d_attn_out = d_attn_out.reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention backward
        qkv = self.input @ self.W_qkv + self.b_qkv
        Q = qkv[:, :, :D].reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = qkv[:, :, D:2*D].reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = qkv[:, :, 2*D:].reshape(batch, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        dV = self.attn_weights.transpose(0, 1, 3, 2) @ d_attn_out
        d_attn_w = d_attn_out @ V.transpose(0, 1, 3, 2)

        # softmax backward
        s = self.attn_weights
        d_scores = s * (d_attn_w - np.sum(d_attn_w * s, axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.d_k)

        dQ = d_scores @ K
        dK = d_scores.transpose(0, 1, 3, 2) @ Q

        # ヘッド結合して戻す
        dQ = dQ.transpose(0, 2, 1, 3).reshape(batch, seq_len, D)
        dK = dK.transpose(0, 2, 1, 3).reshape(batch, seq_len, D)
        dV = dV.transpose(0, 2, 1, 3).reshape(batch, seq_len, D)

        d_qkv = np.concatenate([dQ, dK, dV], axis=-1)

        x_flat = self.input.reshape(-1, D)
        self.grad_W_qkv = x_flat.T @ d_qkv.reshape(-1, 3 * D)
        self.grad_b_qkv = np.sum(d_qkv, axis=(0, 1))

        dx = d_qkv @ self.W_qkv.T
        return dx

    def params_and_grads(self):
        return [
            (self.W_qkv, self.grad_W_qkv, "W_qkv"),
            (self.b_qkv, self.grad_b_qkv, "b_qkv"),
            (self.W_o, self.grad_W_o, "W_o"),
            (self.b_o, self.grad_b_o, "b_o"),
        ]


class LayerNorm:
    """Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        self.grad_gamma = self.grad_beta = None

    def forward(self, x):
        self.input = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, grad):
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


class TransformerDecoderBlock:
    """GPTの1ブロック: Causal Self-Attention → FFN（Pre-LN構成）"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)

        scale_1 = np.sqrt(1.0 / d_model)
        scale_2 = np.sqrt(1.0 / d_ff)
        self.W1 = np.random.randn(d_model, d_ff) * scale_1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale_2
        self.b2 = np.zeros(d_model)

        self.grad_W1 = self.grad_b1 = None
        self.grad_W2 = self.grad_b2 = None

    def forward(self, x):
        # Pre-LN: LNを先に適用 (GPT-2スタイル)
        # Attention block
        self.residual1 = x
        h = self.ln1.forward(x)
        h = self.attn.forward(h)
        x = x + h  # 残差接続

        # FFN block
        self.residual2 = x
        h = self.ln2.forward(x)
        self.ffn_input = h
        self.hidden_pre = h @ self.W1 + self.b1
        self.hidden = _gelu(self.hidden_pre)
        h = self.hidden @ self.W2 + self.b2
        x = x + h  # 残差接続

        return x

    def backward(self, grad):
        # FFN backward
        d_ffn = grad  # 残差接続
        d_residual2 = grad

        self.grad_W2 = self.hidden.reshape(-1, self.hidden.shape[-1]).T @ d_ffn.reshape(-1, d_ffn.shape[-1])
        self.grad_b2 = np.sum(d_ffn, axis=tuple(range(d_ffn.ndim - 1)))

        d_hidden = d_ffn @ self.W2.T
        d_hidden = _gelu_backward(self.hidden_pre, d_hidden)

        self.grad_W1 = self.ffn_input.reshape(-1, self.ffn_input.shape[-1]).T @ d_hidden.reshape(-1, d_hidden.shape[-1])
        self.grad_b1 = np.sum(d_hidden, axis=tuple(range(d_hidden.ndim - 1)))

        d = d_hidden @ self.W1.T
        d = self.ln2.backward(d) + d_residual2

        # Attention backward
        d_attn = d  # 残差接続
        d_residual1 = d

        d = self.attn.backward(d_attn)
        d = self.ln1.backward(d) + d_residual1

        return d

    def params_and_grads(self):
        return (
            self.ln1.params_and_grads()
            + self.attn.params_and_grads()
            + self.ln2.params_and_grads()
            + [
                (self.W1, self.grad_W1, "W1"),
                (self.b1, self.grad_b1, "b1"),
                (self.W2, self.grad_W2, "W2"),
                (self.b2, self.grad_b2, "b2"),
            ]
        )


class GPT:
    """GPTモデル

    Args:
        vocab_size: 語彙サイズ
        d_model: 埋め込み次元数
        n_heads: Attentionヘッド数
        n_layers: Transformerブロック数
        d_ff: FFN中間層の次元数
        max_seq_len: 最大系列長
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 128,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token Embedding
        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.grad_token_emb = None

        # Position Embedding (学習可能)
        self.pos_emb = np.random.randn(max_seq_len, d_model) * 0.02
        self.grad_pos_emb = None

        # Transformer Decoder Blocks
        self.blocks = [
            TransformerDecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ]

        # 最終 Layer Norm
        self.final_ln = LayerNorm(d_model)

        # 出力射影（token embeddingと重み共有も可能だが、ここでは別に持つ）
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        self.grad_lm_head = None

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """
        input_ids: (batch, seq_len) 整数配列
        returns: (batch, seq_len, vocab_size) logits
        """
        self.input_ids = input_ids
        batch, seq_len = input_ids.shape

        # 埋め込み
        x = self.token_emb[input_ids] + self.pos_emb[:seq_len]
        self.emb_output = x

        # Transformerブロック
        for block in self.blocks:
            x = block.forward(x)

        # 最終LayerNorm
        x = self.final_ln.forward(x)
        self.final_output = x

        # logits
        logits = x @ self.lm_head
        return logits

    def backward(self, grad_logits: np.ndarray) -> None:
        """
        grad_logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len, _ = grad_logits.shape

        # lm_head の勾配
        self.grad_lm_head = self.final_output.reshape(-1, self.d_model).T @ grad_logits.reshape(-1, self.vocab_size)
        grad = grad_logits @ self.lm_head.T

        # final LayerNorm
        grad = self.final_ln.backward(grad)

        # Transformerブロック（逆順）
        for block in reversed(self.blocks):
            grad = block.backward(grad)

        # 位置エンコーディングの勾配
        self.grad_pos_emb = np.zeros_like(self.pos_emb)
        self.grad_pos_emb[:seq_len] = np.sum(grad, axis=0)

        # トークン埋め込みの勾配
        self.grad_token_emb = np.zeros_like(self.token_emb)
        np.add.at(self.grad_token_emb, self.input_ids, grad)

    def get_all_layers(self) -> list:
        """オプティマイザに渡すための全レイヤーリスト"""
        layers = []
        layers.append(self)  # token_emb, pos_emb, lm_head
        for block in self.blocks:
            layers.append(block)
        layers.append(self.final_ln)
        return layers

    def params_and_grads(self):
        """GPT本体のパラメータ（埋め込みとlm_head）"""
        return [
            (self.token_emb, self.grad_token_emb, "token_emb"),
            (self.pos_emb, self.grad_pos_emb, "pos_emb"),
            (self.lm_head, self.grad_lm_head, "lm_head"),
        ]

    def count_parameters(self) -> int:
        total = 0
        for layer in self.get_all_layers():
            for param, _, _ in layer.params_and_grads():
                total += param.size
        return total
