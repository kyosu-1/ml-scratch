"""リカレントニューラルネットワーク (RNN / LSTM / GRU)

系列データを処理するためのレイヤー。

RNN:
    h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
    y_t = W_hy @ h_t + b_y

LSTM:
    f_t = σ(W_f @ [h_{t-1}, x_t] + b_f)   (忘却ゲート)
    i_t = σ(W_i @ [h_{t-1}, x_t] + b_i)   (入力ゲート)
    c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c) (候補セル)
    c_t = f_t * c_{t-1} + i_t * c̃_t        (セル状態)
    o_t = σ(W_o @ [h_{t-1}, x_t] + b_o)   (出力ゲート)
    h_t = o_t * tanh(c_t)                   (隠れ状態)

GRU:
    z_t = σ(W_z @ [h_{t-1}, x_t] + b_z)   (更新ゲート)
    r_t = σ(W_r @ [h_{t-1}, x_t] + b_r)   (リセットゲート)
    h̃_t = tanh(W_h @ [r_t * h_{t-1}, x_t] + b_h)
    h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
"""

import numpy as np


def _sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class RNN:
    """Vanilla RNN

    入力: (batch, seq_len, input_size)
    出力: (batch, seq_len, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        scale = np.sqrt(1.0 / hidden_size)

        self.W_xh = np.random.randn(input_size, hidden_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)

        self.grad_W_xh = None
        self.grad_W_hh = None
        self.grad_b_h = None

    def forward(self, x: np.ndarray, h0: np.ndarray = None) -> np.ndarray:
        batch, seq_len, _ = x.shape
        if h0 is None:
            h0 = np.zeros((batch, self.hidden_size))

        self.inputs = x
        self.hiddens = [h0]

        h = h0
        for t in range(seq_len):
            h = np.tanh(x[:, t] @ self.W_xh + h @ self.W_hh + self.b_h)
            self.hiddens.append(h)

        self.output = np.stack(self.hiddens[1:], axis=1)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = grad.shape
        self.grad_W_xh = np.zeros_like(self.W_xh)
        self.grad_W_hh = np.zeros_like(self.W_hh)
        self.grad_b_h = np.zeros_like(self.b_h)

        dx = np.zeros_like(self.inputs)
        dh_next = np.zeros((batch, self.hidden_size))

        for t in reversed(range(seq_len)):
            dh = grad[:, t] + dh_next
            # tanh の勾配: 1 - tanh^2
            dtanh = dh * (1 - self.hiddens[t + 1] ** 2)

            self.grad_W_xh += self.inputs[:, t].T @ dtanh
            self.grad_W_hh += self.hiddens[t].T @ dtanh
            self.grad_b_h += np.sum(dtanh, axis=0)

            dx[:, t] = dtanh @ self.W_xh.T
            dh_next = dtanh @ self.W_hh.T

        return dx

    def params_and_grads(self):
        return [
            (self.W_xh, self.grad_W_xh, "W_xh"),
            (self.W_hh, self.grad_W_hh, "W_hh"),
            (self.b_h, self.grad_b_h, "b_h"),
        ]


class LSTM:
    """Long Short-Term Memory

    入力: (batch, seq_len, input_size)
    出力: (batch, seq_len, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        scale = np.sqrt(1.0 / hidden_size)

        # 4つのゲートの重みを一括で管理 [f, i, c̃, o]
        self.W = np.random.randn(concat_size, 4 * hidden_size) * scale
        self.b = np.zeros(4 * hidden_size)

        self.grad_W = None
        self.grad_b = None

    def forward(self, x: np.ndarray, h0: np.ndarray = None, c0: np.ndarray = None) -> np.ndarray:
        batch, seq_len, _ = x.shape
        H = self.hidden_size

        if h0 is None:
            h0 = np.zeros((batch, H))
        if c0 is None:
            c0 = np.zeros((batch, H))

        self.inputs = x
        self.cache = []
        h, c = h0, c0
        outputs = []

        for t in range(seq_len):
            concat = np.concatenate([h, x[:, t]], axis=1)
            gates = concat @ self.W + self.b

            f = _sigmoid(gates[:, :H])
            i = _sigmoid(gates[:, H:2*H])
            c_tilde = np.tanh(gates[:, 2*H:3*H])
            o = _sigmoid(gates[:, 3*H:])

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

            self.cache.append((concat, f, i, c_tilde, o, c, h))
            outputs.append(h)

        self.h0 = h0
        self.c0 = c0
        self.output = np.stack(outputs, axis=1)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = grad.shape
        H = self.hidden_size

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)
        dx = np.zeros_like(self.inputs)

        dh_next = np.zeros((batch, H))
        dc_next = np.zeros((batch, H))

        for t in reversed(range(seq_len)):
            concat, f, i, c_tilde, o, c, h = self.cache[t]
            c_prev = self.cache[t - 1][5] if t > 0 else self.c0

            dh = grad[:, t] + dh_next

            # 出力ゲートの勾配
            tanh_c = np.tanh(c)
            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c ** 2) + dc_next

            # 各ゲートの勾配
            df = dc * c_prev
            di = dc * c_tilde
            dc_tilde = dc * i
            dc_next = dc * f

            # ゲート活性化関数の勾配
            df_raw = df * f * (1 - f)
            di_raw = di * i * (1 - i)
            dc_tilde_raw = dc_tilde * (1 - c_tilde ** 2)
            do_raw = do * o * (1 - o)

            dgates = np.concatenate([df_raw, di_raw, dc_tilde_raw, do_raw], axis=1)

            self.grad_W += concat.T @ dgates
            self.grad_b += np.sum(dgates, axis=0)

            d_concat = dgates @ self.W.T
            dh_next = d_concat[:, :H]
            dx[:, t] = d_concat[:, H:]

        return dx

    def params_and_grads(self):
        return [
            (self.W, self.grad_W, "W"),
            (self.b, self.grad_b, "b"),
        ]


class GRU:
    """Gated Recurrent Unit

    入力: (batch, seq_len, input_size)
    出力: (batch, seq_len, hidden_size)
    """

    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        concat_size = input_size + hidden_size
        scale = np.sqrt(1.0 / hidden_size)

        # 更新ゲート・リセットゲート
        self.W_z = np.random.randn(concat_size, hidden_size) * scale
        self.b_z = np.zeros(hidden_size)
        self.W_r = np.random.randn(concat_size, hidden_size) * scale
        self.b_r = np.zeros(hidden_size)
        # 候補隠れ状態
        self.W_h = np.random.randn(concat_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)

        self.grad_W_z = self.grad_b_z = None
        self.grad_W_r = self.grad_b_r = None
        self.grad_W_h = self.grad_b_h = None

    def forward(self, x: np.ndarray, h0: np.ndarray = None) -> np.ndarray:
        batch, seq_len, _ = x.shape
        H = self.hidden_size

        if h0 is None:
            h0 = np.zeros((batch, H))

        self.inputs = x
        self.cache = []
        h = h0
        outputs = []

        for t in range(seq_len):
            concat = np.concatenate([h, x[:, t]], axis=1)

            z = _sigmoid(concat @ self.W_z + self.b_z)
            r = _sigmoid(concat @ self.W_r + self.b_r)

            concat_r = np.concatenate([r * h, x[:, t]], axis=1)
            h_tilde = np.tanh(concat_r @ self.W_h + self.b_h)

            h_new = (1 - z) * h + z * h_tilde

            self.cache.append((concat, concat_r, z, r, h_tilde, h, h_new))
            h = h_new
            outputs.append(h)

        self.h0 = h0
        self.output = np.stack(outputs, axis=1)
        return self.output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch, seq_len, _ = grad.shape
        H = self.hidden_size

        self.grad_W_z = np.zeros_like(self.W_z)
        self.grad_b_z = np.zeros_like(self.b_z)
        self.grad_W_r = np.zeros_like(self.W_r)
        self.grad_b_r = np.zeros_like(self.b_r)
        self.grad_W_h = np.zeros_like(self.W_h)
        self.grad_b_h = np.zeros_like(self.b_h)

        dx = np.zeros_like(self.inputs)
        dh_next = np.zeros((batch, H))

        for t in reversed(range(seq_len)):
            concat, concat_r, z, r, h_tilde, h_prev, h_new = self.cache[t]
            dh = grad[:, t] + dh_next

            # GRUの勾配
            dz = dh * (h_tilde - h_prev)
            dh_tilde = dh * z
            dh_prev = dh * (1 - z)

            # h_tilde の勾配 (tanh)
            dh_tilde_raw = dh_tilde * (1 - h_tilde ** 2)
            self.grad_W_h += concat_r.T @ dh_tilde_raw
            self.grad_b_h += np.sum(dh_tilde_raw, axis=0)

            d_concat_r = dh_tilde_raw @ self.W_h.T
            dr_h = d_concat_r[:, :H]
            dx[:, t] += d_concat_r[:, H:]

            # リセットゲートの勾配
            dr = dr_h * h_prev
            dh_prev += dr_h * r
            dr_raw = dr * r * (1 - r)
            self.grad_W_r += concat.T @ dr_raw
            self.grad_b_r += np.sum(dr_raw, axis=0)

            # 更新ゲートの勾配
            dz_raw = dz * z * (1 - z)
            self.grad_W_z += concat.T @ dz_raw
            self.grad_b_z += np.sum(dz_raw, axis=0)

            d_concat = dz_raw @ self.W_z.T + dr_raw @ self.W_r.T
            dh_next = dh_prev + d_concat[:, :H]
            dx[:, t] += d_concat[:, H:]

        return dx

    def params_and_grads(self):
        return [
            (self.W_z, self.grad_W_z, "W_z"),
            (self.b_z, self.grad_b_z, "b_z"),
            (self.W_r, self.grad_W_r, "W_r"),
            (self.b_r, self.grad_b_r, "b_r"),
            (self.W_h, self.grad_W_h, "W_h"),
            (self.b_h, self.grad_b_h, "b_h"),
        ]
