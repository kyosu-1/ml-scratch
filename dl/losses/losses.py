"""損失関数 (Loss Functions)

各損失関数はforward (損失値の計算) とbackward (勾配の計算) を提供する。
"""

import numpy as np


class MSELoss:
    """平均二乗誤差: L = (1/n) * Σ (y - y_hat)^2"""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self) -> np.ndarray:
        n = self.y_pred.shape[0]
        return 2 * (self.y_pred - self.y_true) / n


class CrossEntropyLoss:
    """交差エントロピー損失 (Softmax + Cross Entropy)

    Softmaxと組み合わせて使う。

    L = -(1/n) * Σ Σ y_k * log(p_k)

    y_true: one-hotまたはクラスインデックス
    y_pred: softmax出力 (確率)

    勾配: dL/dy_pred = (p - y) / n
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        n = y_pred.shape[0]

        if y_true.ndim == 1:
            # クラスインデックスの場合
            self.y_true_onehot = np.zeros_like(y_pred)
            self.y_true_onehot[np.arange(n), y_true.astype(int)] = 1
        else:
            self.y_true_onehot = y_true

        loss = -np.sum(self.y_true_onehot * np.log(self.y_pred)) / n
        return loss

    def backward(self) -> np.ndarray:
        n = self.y_pred.shape[0]
        return (self.y_pred - self.y_true_onehot) / n


class BinaryCrossEntropyLoss:
    """二値交差エントロピー: L = -(1/n) * Σ [y*log(p) + (1-y)*log(1-p)]"""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        self.y_true = y_true
        n = len(y_true)
        loss = -(1 / n) * np.sum(
            y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred)
        )
        return loss

    def backward(self) -> np.ndarray:
        n = len(self.y_true)
        return (
            -(self.y_true / self.y_pred - (1 - self.y_true) / (1 - self.y_pred)) / n
        )
