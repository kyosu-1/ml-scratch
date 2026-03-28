"""敵対的生成ネットワーク (GAN: Generative Adversarial Network)

2つのネットワークが敵対的に学習する生成モデル。

Generator (G): ノイズ z ~ N(0,I) → 偽データ
Discriminator (D): データ → 本物(1) / 偽物(0)

min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]

D: 本物を1、偽物を0と判定するよう学習
G: Dを騙す（D(G(z))を1に近づける）よう学習

実装上は G の損失に -log(D(G(z))) を使う（勾配消失を防ぐため）。
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)


def _leaky_relu_backward(x, grad, alpha=0.2):
    return grad * np.where(x > 0, 1, alpha)


class Generator:
    """ノイズ → 偽データ"""

    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        scale1 = np.sqrt(2.0 / latent_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(latent_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * scale2
        self.b3 = np.zeros(output_dim)

    def forward(self, z: np.ndarray) -> np.ndarray:
        self.z = z
        self.h1_pre = z @ self.W1 + self.b1
        self.h1 = _leaky_relu(self.h1_pre)
        self.h2_pre = self.h1 @ self.W2 + self.b2
        self.h2 = _leaky_relu(self.h2_pre)
        logits = self.h2 @ self.W3 + self.b3
        self.output = np.tanh(logits)  # [-1, 1]
        return self.output

    def backward(self, grad: np.ndarray):
        # tanh backward
        grad = grad * (1 - self.output ** 2)

        self.grad_W3 = self.h2.T @ grad
        self.grad_b3 = np.sum(grad, axis=0)
        grad = grad @ self.W3.T

        grad = _leaky_relu_backward(self.h2_pre, grad)
        self.grad_W2 = self.h1.T @ grad
        self.grad_b2 = np.sum(grad, axis=0)
        grad = grad @ self.W2.T

        grad = _leaky_relu_backward(self.h1_pre, grad)
        self.grad_W1 = self.z.T @ grad
        self.grad_b1 = np.sum(grad, axis=0)

    def params_and_grads(self):
        return [
            (self.W1, self.grad_W1, "W1"), (self.b1, self.grad_b1, "b1"),
            (self.W2, self.grad_W2, "W2"), (self.b2, self.grad_b2, "b2"),
            (self.W3, self.grad_W3, "W3"), (self.b3, self.grad_b3, "b3"),
        ]


class Discriminator:
    """データ → 本物(1) / 偽物(0)"""

    def __init__(self, input_dim: int, hidden_dim: int):
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, 1) * scale2
        self.b3 = np.zeros(1)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        self.h1_pre = x @ self.W1 + self.b1
        self.h1 = _leaky_relu(self.h1_pre)
        self.h2_pre = self.h1 @ self.W2 + self.b2
        self.h2 = _leaky_relu(self.h2_pre)
        logits = self.h2 @ self.W3 + self.b3
        self.output = _sigmoid(logits)
        return self.output

    def backward(self, grad: np.ndarray):
        # sigmoid backward
        grad = grad * self.output * (1 - self.output)

        self.grad_W3 = self.h2.T @ grad
        self.grad_b3 = np.sum(grad, axis=0)
        grad = grad @ self.W3.T

        grad = _leaky_relu_backward(self.h2_pre, grad)
        self.grad_W2 = self.h1.T @ grad
        self.grad_b2 = np.sum(grad, axis=0)
        grad = grad @ self.W2.T

        grad = _leaky_relu_backward(self.h1_pre, grad)
        self.grad_W1 = self.input.T @ grad
        self.grad_b1 = np.sum(grad, axis=0)
        return grad @ self.W1.T

    def params_and_grads(self):
        return [
            (self.W1, self.grad_W1, "W1"), (self.b1, self.grad_b1, "b1"),
            (self.W2, self.grad_W2, "W2"), (self.b2, self.grad_b2, "b2"),
            (self.W3, self.grad_W3, "W3"), (self.b3, self.grad_b3, "b3"),
        ]


class GAN:
    """GANの学習ループを管理するクラス"""

    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
    ):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, hidden_dim, data_dim)
        self.discriminator = Discriminator(data_dim, hidden_dim)
        self.lr_g = lr_g
        self.lr_d = lr_d

    def train_step(self, real_data: np.ndarray):
        """1ステップの学習"""
        batch_size = real_data.shape[0]

        # --- Discriminatorの学習 ---
        # 本物データ
        d_real = self.discriminator.forward(real_data)
        d_loss_real = -np.mean(np.log(d_real + 1e-10))
        grad_real = -1.0 / (d_real + 1e-10) / batch_size
        self.discriminator.backward(grad_real)
        grads_d_real = [(p.copy(), g.copy()) for p, g, _ in self.discriminator.params_and_grads()]

        # 偽データ
        z = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(z)
        d_fake = self.discriminator.forward(fake_data)
        d_loss_fake = -np.mean(np.log(1 - d_fake + 1e-10))
        grad_fake = 1.0 / (1 - d_fake + 1e-10) / batch_size
        self.discriminator.backward(grad_fake)

        # D のパラメータ更新（両方の勾配を合算）
        for i, (param, grad, _) in enumerate(self.discriminator.params_and_grads()):
            total_grad = grads_d_real[i][1] + grad
            param -= self.lr_d * np.clip(total_grad, -1, 1)

        d_loss = d_loss_real + d_loss_fake

        # --- Generatorの学習 ---
        z = np.random.randn(batch_size, self.latent_dim)
        fake_data = self.generator.forward(z)
        d_fake = self.discriminator.forward(fake_data)

        # -log(D(G(z))) を最小化
        g_loss = -np.mean(np.log(d_fake + 1e-10))

        grad_g = -1.0 / (d_fake + 1e-10) / batch_size
        # Dを通してGに勾配を伝播
        grad_g = grad_g * d_fake * (1 - d_fake)  # sigmoid backward
        # D の逆伝播（Gへの勾配を取得）
        grad_to_g = self.discriminator.backward(grad_g)
        self.generator.backward(grad_to_g)

        # G のパラメータ更新
        for param, grad, _ in self.generator.params_and_grads():
            param -= self.lr_g * np.clip(grad, -1, 1)

        return d_loss, g_loss

    def generate(self, n_samples: int = 1) -> np.ndarray:
        z = np.random.randn(n_samples, self.latent_dim)
        return self.generator.forward(z)
