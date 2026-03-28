"""深層学習フレームワークのテスト

勾配チェック (numerical gradient) でbackwardの正しさを検証する。
"""

import numpy as np
import pytest


def numerical_gradient(f, x, eps=1e-5):
    """数値微分による勾配計算"""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old + eps
        f_plus = f()
        x[idx] = old - eps
        f_minus = f()
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        x[idx] = old
        it.iternext()
    return grad


class TestLinearLayer:
    def test_forward_shape(self):
        from dl.layers.linear import Linear
        layer = Linear(4, 3)
        x = np.random.randn(2, 4)
        out = layer.forward(x)
        assert out.shape == (2, 3)

    def test_gradient_check(self):
        from dl.layers.linear import Linear
        layer = Linear(3, 2)
        x = np.random.randn(2, 3)
        out = layer.forward(x)
        grad_out = np.random.randn(*out.shape)
        layer.backward(grad_out)

        def f():
            return np.sum(layer.forward(x) * grad_out)

        num_grad = numerical_gradient(f, x)
        dx = layer.backward(grad_out)
        np.testing.assert_allclose(dx, num_grad, atol=1e-5)


class TestActivations:
    @pytest.mark.parametrize("ActClass", ["ReLU", "Sigmoid", "Tanh"])
    def test_gradient_check(self, ActClass):
        from dl.activations import activations
        cls = getattr(activations, ActClass)
        act = cls()
        x = np.random.randn(3, 4) * 0.5
        out = act.forward(x)
        grad_out = np.random.randn(*out.shape)

        def f():
            return np.sum(act.forward(x) * grad_out)

        num_grad = numerical_gradient(f, x)
        dx = act.backward(grad_out)
        np.testing.assert_allclose(dx, num_grad, atol=1e-5)


class TestLosses:
    def test_mse_gradient(self):
        from dl.losses.losses import MSELoss
        loss_fn = MSELoss()
        y_pred = np.random.randn(5)
        y_true = np.random.randn(5)

        def f():
            return loss_fn.forward(y_pred, y_true)

        loss_fn.forward(y_pred, y_true)
        grad = loss_fn.backward()
        num_grad = numerical_gradient(f, y_pred)
        np.testing.assert_allclose(grad, num_grad, atol=1e-5)

    def test_cross_entropy_gradient(self):
        from dl.losses.losses import CrossEntropyLoss
        from dl.activations.activations import Softmax
        softmax = Softmax()
        loss_fn = CrossEntropyLoss()

        logits = np.random.randn(3, 4)
        y_true = np.array([0, 2, 1])
        probs = softmax.forward(logits)

        loss_fn.forward(probs, y_true)
        grad = loss_fn.backward()
        assert grad.shape == (3, 4)


class TestOptimizers:
    def test_sgd_step(self):
        from dl.layers.linear import Linear
        from dl.optimizers.optimizers import SGD
        layer = Linear(3, 2)
        w_before = layer.weights.copy()
        x = np.random.randn(2, 3)
        layer.forward(x)
        layer.backward(np.ones((2, 2)))
        opt = SGD(lr=0.01)
        opt.step([layer])
        assert not np.allclose(layer.weights, w_before)

    def test_adam_step(self):
        from dl.layers.linear import Linear
        from dl.optimizers.optimizers import Adam
        layer = Linear(3, 2)
        w_before = layer.weights.copy()
        x = np.random.randn(2, 3)
        layer.forward(x)
        layer.backward(np.ones((2, 2)))
        opt = Adam(lr=0.01)
        opt.step([layer])
        assert not np.allclose(layer.weights, w_before)


class TestRNN:
    def test_forward_shape(self):
        from dl.layers.rnn import RNN
        rnn = RNN(input_size=4, hidden_size=8)
        x = np.random.randn(2, 5, 4)  # batch=2, seq=5, features=4
        out = rnn.forward(x)
        assert out.shape == (2, 5, 8)

    def test_backward_shape(self):
        from dl.layers.rnn import RNN
        rnn = RNN(input_size=4, hidden_size=8)
        x = np.random.randn(2, 5, 4)
        out = rnn.forward(x)
        grad = np.random.randn(*out.shape)
        dx = rnn.backward(grad)
        assert dx.shape == x.shape


class TestLSTM:
    def test_forward_shape(self):
        from dl.layers.rnn import LSTM
        lstm = LSTM(input_size=4, hidden_size=8)
        x = np.random.randn(2, 5, 4)
        out = lstm.forward(x)
        assert out.shape == (2, 5, 8)

    def test_backward_shape(self):
        from dl.layers.rnn import LSTM
        lstm = LSTM(input_size=4, hidden_size=8)
        x = np.random.randn(2, 5, 4)
        out = lstm.forward(x)
        grad = np.random.randn(*out.shape)
        dx = lstm.backward(grad)
        assert dx.shape == x.shape


class TestGRU:
    def test_forward_shape(self):
        from dl.layers.rnn import GRU
        gru = GRU(input_size=4, hidden_size=8)
        x = np.random.randn(2, 5, 4)
        out = gru.forward(x)
        assert out.shape == (2, 5, 8)


class TestAttention:
    def test_scaled_dot_product(self):
        from dl.layers.attention import ScaledDotProductAttention
        attn = ScaledDotProductAttention()
        Q = np.random.randn(2, 3, 4)
        K = np.random.randn(2, 3, 4)
        V = np.random.randn(2, 3, 4)
        out = attn.forward(Q, K, V)
        assert out.shape == (2, 3, 4)

    def test_multi_head(self):
        from dl.layers.attention import MultiHeadAttention
        mha = MultiHeadAttention(d_model=8, n_heads=2)
        x = np.random.randn(2, 5, 8)
        out = mha.forward(x)
        assert out.shape == (2, 5, 8)

    def test_transformer_block(self):
        from dl.layers.attention import TransformerBlock
        block = TransformerBlock(d_model=8, n_heads=2, d_ff=16)
        x = np.random.randn(2, 5, 8)
        out = block.forward(x)
        assert out.shape == (2, 5, 8)

    def test_transformer_backward(self):
        from dl.layers.attention import TransformerBlock
        block = TransformerBlock(d_model=8, n_heads=2, d_ff=16)
        x = np.random.randn(2, 5, 8)
        out = block.forward(x)
        grad = np.random.randn(*out.shape)
        dx = block.backward(grad)
        assert dx.shape == x.shape


class TestPooling:
    def test_maxpool2d(self):
        from dl.layers.pooling import MaxPool2D
        pool = MaxPool2D(pool_size=2)
        x = np.random.randn(1, 1, 4, 4)
        out = pool.forward(x)
        assert out.shape == (1, 1, 2, 2)

    def test_flatten(self):
        from dl.layers.pooling import Flatten
        flat = Flatten()
        x = np.random.randn(2, 3, 4, 4)
        out = flat.forward(x)
        assert out.shape == (2, 48)
        dx = flat.backward(out)
        assert dx.shape == (2, 3, 4, 4)


class TestEmbedding:
    def test_forward(self):
        from dl.layers.embedding import Embedding
        emb = Embedding(vocab_size=100, embed_dim=16)
        indices = np.array([[1, 5, 3], [2, 0, 9]])
        out = emb.forward(indices)
        assert out.shape == (2, 3, 16)

    def test_backward(self):
        from dl.layers.embedding import Embedding
        emb = Embedding(vocab_size=10, embed_dim=4)
        indices = np.array([[1, 2], [3, 4]])
        out = emb.forward(indices)
        grad = np.random.randn(*out.shape)
        emb.backward(grad)
        assert emb.grad_weights.shape == (10, 4)


class TestSequentialModel:
    def test_train_loop(self):
        from dl.layers.linear import Linear
        from dl.activations.activations import ReLU, Softmax
        from dl.losses.losses import CrossEntropyLoss
        from dl.optimizers.optimizers import Adam
        from dl.models.sequential import Sequential

        model = Sequential([
            Linear(4, 8),
            ReLU(),
            Linear(8, 3),
            Softmax(),
        ])
        loss_fn = CrossEntropyLoss()
        opt = Adam(lr=0.01)

        X = np.random.randn(20, 4)
        y = np.random.randint(0, 3, 20)

        # 学習が進むことを確認
        model.train()
        out = model.forward(X)
        initial_loss = loss_fn.forward(out, y)

        for _ in range(50):
            out = model.forward(X)
            loss = loss_fn.forward(out, y)
            grad = loss_fn.backward()
            model.backward(grad)
            opt.step(model.layers)

        final_loss = loss
        assert final_loss < initial_loss
