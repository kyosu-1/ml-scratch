"""拡散モデル (Denoising Diffusion Probabilistic Model, DDPM)

ノイズの付加（拡散過程）と除去（逆過程）を繰り返すことでデータを生成する。

拡散過程 (Forward):
    q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

    任意の時刻tに直接ジャンプ可能:
    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t) I)
    ᾱ_t = Π_{s=1}^{t} (1 - β_s)

逆過程 (Reverse):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)

    ニューラルネットワークがノイズ ε_θ(x_t, t) を予測し、
    そこから μ を計算:
    μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) × ε_θ(x_t, t))

損失関数:
    L = E[‖ε - ε_θ(x_t, t)‖²]
    実際に加えたノイズと、モデルが予測したノイズのMSE。
"""

import numpy as np


class NoisePredictor:
    """ノイズ予測ネットワーク ε_θ(x_t, t)

    時刻情報tを正弦波エンコーディングで埋め込み、
    ノイズ入りデータ x_t と結合して処理する。
    """

    def __init__(self, data_dim: int, hidden_dim: int = 64, time_embed_dim: int = 16):
        self.data_dim = data_dim
        self.time_embed_dim = time_embed_dim
        input_dim = data_dim + time_embed_dim

        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, data_dim) * scale2
        self.b3 = np.zeros(data_dim)

    def _time_embedding(self, t: np.ndarray) -> np.ndarray:
        """時刻の正弦波エンコーディング"""
        half = self.time_embed_dim // 2
        freqs = np.exp(-np.log(1000) * np.arange(half) / half)
        args = t[:, np.newaxis] * freqs[np.newaxis, :]
        return np.concatenate([np.sin(args), np.cos(args)], axis=-1)

    def forward(self, x_t: np.ndarray, t: np.ndarray) -> np.ndarray:
        self.x_t = x_t
        self.t_emb = self._time_embedding(t)
        self.input = np.concatenate([x_t, self.t_emb], axis=-1)

        self.h1_pre = self.input @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.h1_pre)
        self.h2_pre = self.h1 @ self.W2 + self.b2
        self.h2 = np.maximum(0, self.h2_pre)
        self.output = self.h2 @ self.W3 + self.b3
        return self.output

    def backward(self, grad: np.ndarray):
        self.grad_W3 = self.h2.T @ grad
        self.grad_b3 = np.sum(grad, axis=0)
        grad = grad @ self.W3.T

        grad = grad * (self.h2_pre > 0)
        self.grad_W2 = self.h1.T @ grad
        self.grad_b2 = np.sum(grad, axis=0)
        grad = grad @ self.W2.T

        grad = grad * (self.h1_pre > 0)
        self.grad_W1 = self.input.T @ grad
        self.grad_b1 = np.sum(grad, axis=0)

    def params_and_grads(self):
        return [
            (self.W1, self.grad_W1, "W1"), (self.b1, self.grad_b1, "b1"),
            (self.W2, self.grad_W2, "W2"), (self.b2, self.grad_b2, "b2"),
            (self.W3, self.grad_W3, "W3"), (self.b3, self.grad_b3, "b3"),
        ]


class DDPM:
    """Denoising Diffusion Probabilistic Model"""

    def __init__(
        self,
        data_dim: int,
        n_timesteps: int = 100,
        hidden_dim: int = 64,
        lr: float = 1e-3,
    ):
        self.data_dim = data_dim
        self.n_timesteps = n_timesteps
        self.lr = lr

        # ノイズスケジュール (線形)
        self.betas = np.linspace(1e-4, 0.02, n_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)  # ᾱ_t

        # ノイズ予測ネットワーク
        self.model = NoisePredictor(data_dim, hidden_dim)

    def _add_noise(self, x_0: np.ndarray, t: np.ndarray):
        """拡散過程: x_0 にノイズを加えて x_t を作る"""
        alpha_bar = self.alpha_bars[t][:, np.newaxis]
        noise = np.random.randn(*x_0.shape)
        x_t = np.sqrt(alpha_bar) * x_0 + np.sqrt(1 - alpha_bar) * noise
        return x_t, noise

    def train_step(self, x_0: np.ndarray):
        """1ステップの学習"""
        batch_size = x_0.shape[0]

        # ランダムな時刻を選択
        t = np.random.randint(0, self.n_timesteps, batch_size)

        # ノイズを加える
        x_t, noise = self._add_noise(x_0, t)

        # ノイズを予測
        noise_pred = self.model.forward(x_t, t.astype(np.float64))

        # MSE損失
        loss = np.mean((noise - noise_pred) ** 2)

        # 逆伝播
        grad = 2 * (noise_pred - noise) / batch_size
        self.model.backward(grad)

        # パラメータ更新
        for param, g, _ in self.model.params_and_grads():
            param -= self.lr * np.clip(g, -1, 1)

        return loss

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """逆過程: ノイズからデータを生成"""
        x = np.random.randn(n_samples, self.data_dim)

        for t in reversed(range(self.n_timesteps)):
            t_batch = np.full(n_samples, t, dtype=np.float64)
            noise_pred = self.model.forward(x, t_batch)

            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]

            # μ_θ = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) × ε_θ)
            mu = (1 / np.sqrt(alpha)) * (x - (beta / np.sqrt(1 - alpha_bar)) * noise_pred)

            if t > 0:
                sigma = np.sqrt(beta)
                x = mu + sigma * np.random.randn(*x.shape)
            else:
                x = mu

        return x
