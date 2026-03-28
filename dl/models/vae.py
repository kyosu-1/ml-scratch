"""変分オートエンコーダ (VAE: Variational Autoencoder)

データの潜在表現を学習し、新しいデータを生成できる生成モデル。

構造:
    Encoder: x → μ, log σ² (潜在空間のパラメータ)
    サンプリング: z = μ + σ × ε, ε ~ N(0, I)  (再パラメータ化トリック)
    Decoder: z → x̂ (再構成)

損失関数 (ELBO):
    L = 再構成誤差 + KLダイバージェンス
    L = E[‖x - x̂‖²] + KL(q(z|x) ‖ p(z))

    再構成誤差: データをどれだけ正確に復元できるか
    KL項: 潜在分布 q(z|x) を事前分布 p(z)=N(0,I) に近づける

    KL(N(μ,σ²) ‖ N(0,1)) = -0.5 × Σ(1 + log σ² - μ² - σ²)
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class VAE:

    def __init__(self, input_dim: int, hidden_dim: int = 64, latent_dim: int = 2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        scale_enc = np.sqrt(2.0 / input_dim)
        scale_hid = np.sqrt(2.0 / hidden_dim)
        scale_lat = np.sqrt(2.0 / latent_dim)

        # Encoder: input → hidden → (μ, log_var)
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * scale_enc
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * scale_hid
        self.b_mu = np.zeros(latent_dim)
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * scale_hid
        self.b_logvar = np.zeros(latent_dim)

        # Decoder: latent → hidden → output
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * scale_lat
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = np.random.randn(hidden_dim, input_dim) * scale_hid
        self.b_dec2 = np.zeros(input_dim)

    def encode(self, x: np.ndarray):
        """Encoder: x → μ, log_var"""
        self.enc_input = x
        self.enc_hidden_pre = x @ self.W_enc1 + self.b_enc1
        self.enc_hidden = np.maximum(0, self.enc_hidden_pre)  # ReLU

        mu = self.enc_hidden @ self.W_mu + self.b_mu
        log_var = self.enc_hidden @ self.W_logvar + self.b_logvar
        return mu, log_var

    def reparameterize(self, mu: np.ndarray, log_var: np.ndarray):
        """再パラメータ化トリック: z = μ + σ × ε"""
        self.epsilon = np.random.randn(*mu.shape)
        std = np.exp(0.5 * log_var)
        z = mu + std * self.epsilon
        return z

    def decode(self, z: np.ndarray):
        """Decoder: z → x̂"""
        self.dec_input = z
        self.dec_hidden_pre = z @ self.W_dec1 + self.b_dec1
        self.dec_hidden = np.maximum(0, self.dec_hidden_pre)  # ReLU

        logits = self.dec_hidden @ self.W_dec2 + self.b_dec2
        x_recon = _sigmoid(logits)  # [0, 1] に正規化
        return x_recon

    def forward(self, x: np.ndarray):
        """順伝播: x → μ, log_var, z, x̂"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z

    def loss(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray):
        """ELBO損失 = 再構成誤差 + KLダイバージェンス"""
        batch_size = x.shape[0]

        # 再構成誤差 (Binary Cross-Entropy)
        recon_loss = -np.sum(
            x * np.log(x_recon + 1e-10) + (1 - x) * np.log(1 - x_recon + 1e-10)
        ) / batch_size

        # KLダイバージェンス
        kl_loss = -0.5 * np.sum(1 + log_var - mu ** 2 - np.exp(log_var)) / batch_size

        return recon_loss + kl_loss, recon_loss, kl_loss

    def backward(self, x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray):
        """逆伝播"""
        batch_size = x.shape[0]
        std = np.exp(0.5 * log_var)

        # Decoder逆伝播
        # sigmoid CE勾配: x_recon - x
        d_logits = (x_recon - x) / batch_size

        self.grad_W_dec2 = self.dec_hidden.T @ d_logits
        self.grad_b_dec2 = np.sum(d_logits, axis=0)

        d_dec_hidden = d_logits @ self.W_dec2.T
        d_dec_hidden *= (self.dec_hidden_pre > 0)  # ReLU

        self.grad_W_dec1 = self.dec_input.T @ d_dec_hidden
        self.grad_b_dec1 = np.sum(d_dec_hidden, axis=0)

        dz = d_dec_hidden @ self.W_dec1.T

        # 再パラメータ化の勾配
        d_mu = dz + mu / batch_size  # KL項の勾配
        d_log_var = dz * 0.5 * std * self.epsilon + 0.5 * (-1 + np.exp(log_var)) / batch_size

        # Encoder逆伝播
        d_enc_hidden = (d_mu @ self.W_mu.T + d_log_var @ self.W_logvar.T)
        d_enc_hidden *= (self.enc_hidden_pre > 0)  # ReLU

        self.grad_W_mu = self.enc_hidden.T @ d_mu
        self.grad_b_mu = np.sum(d_mu, axis=0)
        self.grad_W_logvar = self.enc_hidden.T @ d_log_var
        self.grad_b_logvar = np.sum(d_log_var, axis=0)

        self.grad_W_enc1 = self.enc_input.T @ d_enc_hidden
        self.grad_b_enc1 = np.sum(d_enc_hidden, axis=0)

    def params_and_grads(self):
        return [
            (self.W_enc1, self.grad_W_enc1, "W_enc1"),
            (self.b_enc1, self.grad_b_enc1, "b_enc1"),
            (self.W_mu, self.grad_W_mu, "W_mu"),
            (self.b_mu, self.grad_b_mu, "b_mu"),
            (self.W_logvar, self.grad_W_logvar, "W_logvar"),
            (self.b_logvar, self.grad_b_logvar, "b_logvar"),
            (self.W_dec1, self.grad_W_dec1, "W_dec1"),
            (self.b_dec1, self.grad_b_dec1, "b_dec1"),
            (self.W_dec2, self.grad_W_dec2, "W_dec2"),
            (self.b_dec2, self.grad_b_dec2, "b_dec2"),
        ]

    def generate(self, n_samples: int = 1) -> np.ndarray:
        """事前分布からサンプリングして生成"""
        z = np.random.randn(n_samples, self.latent_dim)
        return self.decode(z)
