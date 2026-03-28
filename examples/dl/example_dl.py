"""深層学習フレームワークの使用例

2層ニューラルネットで分類タスクを学習する。
"""

import numpy as np
from dl.layers.linear import Linear
from dl.layers.dropout import Dropout
from dl.activations.activations import ReLU, Softmax
from dl.losses.losses import CrossEntropyLoss
from dl.optimizers.optimizers import Adam
from dl.models.sequential import Sequential
from utils.data import make_classification, train_test_split, standardize
from utils.metrics import accuracy


def main():
    print("=== ニューラルネットワーク (2層MLP) ===")

    # データ準備
    X, y = make_classification(n_samples=300, n_features=4, n_classes=3)
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # モデル構築
    model = Sequential([
        Linear(4, 32),
        ReLU(),
        Dropout(p=0.2),
        Linear(32, 16),
        ReLU(),
        Linear(16, 3),
        Softmax(),
    ])

    loss_fn = CrossEntropyLoss()
    optimizer = Adam(lr=0.01)

    # 学習
    n_epochs = 100
    batch_size = 32

    for epoch in range(n_epochs):
        model.train()

        # ミニバッチ学習
        indices = np.random.permutation(len(X_train))
        total_loss = 0
        n_batches = 0

        for start in range(0, len(X_train), batch_size):
            batch_idx = indices[start : start + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            # Forward
            out = model.forward(X_batch)
            loss = loss_fn.forward(out, y_batch)
            total_loss += loss
            n_batches += 1

            # Backward
            grad = loss_fn.backward()
            model.backward(grad)

            # パラメータ更新
            optimizer.step(model.layers)

        if (epoch + 1) % 20 == 0:
            model.eval()
            pred = np.argmax(model.forward(X_test), axis=1)
            acc = accuracy(y_test, pred)
            print(f"  Epoch {epoch+1:3d} | Loss: {total_loss/n_batches:.4f} | Accuracy: {acc:.4f}")

    # 最終評価
    model.eval()
    pred = np.argmax(model.forward(X_test), axis=1)
    print(f"\n  最終 Accuracy: {accuracy(y_test, pred):.4f}")


if __name__ == "__main__":
    main()
