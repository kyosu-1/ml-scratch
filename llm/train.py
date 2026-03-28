"""GPTの学習ループ

次トークン予測タスクで自己回帰的に学習する。

入力: [t_0, t_1, t_2, ..., t_{n-1}]
目標: [t_1, t_2, t_3, ..., t_n]

各位置で次のトークンを予測するCross-Entropy損失を最小化する。
"""

import numpy as np
from llm.model.gpt import GPT
from dl.optimizers.optimizers import Adam


def _softmax(x, axis=-1):
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray):
    """
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) 整数
    returns: (loss, grad_logits)
    """
    batch, seq_len, vocab_size = logits.shape

    # Softmax
    probs = _softmax(logits, axis=-1)

    # Cross-Entropy損失
    # targets の位置の確率を取得
    loss = 0
    for b in range(batch):
        for t in range(seq_len):
            loss -= np.log(max(probs[b, t, targets[b, t]], 1e-10))
    loss /= batch * seq_len

    # 勾配: softmax - one_hot
    grad = probs.copy()
    for b in range(batch):
        for t in range(seq_len):
            grad[b, t, targets[b, t]] -= 1
    grad /= batch * seq_len

    return loss, grad


def create_sequences(token_ids: list[int], seq_len: int) -> tuple:
    """トークン列をシーケンスに分割"""
    inputs = []
    targets = []
    for i in range(0, len(token_ids) - seq_len, seq_len // 2):
        if i + seq_len + 1 > len(token_ids):
            break
        inputs.append(token_ids[i : i + seq_len])
        targets.append(token_ids[i + 1 : i + seq_len + 1])
    return np.array(inputs), np.array(targets)


def train(
    model: GPT,
    train_text: str,
    tokenizer,
    n_epochs: int = 10,
    seq_len: int = 32,
    batch_size: int = 4,
    lr: float = 3e-4,
    print_every: int = 1,
) -> list[float]:
    """GPTモデルの学習

    returns: エポックごとの平均損失のリスト
    """
    # テキストをトークン化
    token_ids = tokenizer.encode(train_text)

    # シーケンスに分割
    X, Y = create_sequences(token_ids, seq_len)
    n_samples = len(X)

    if n_samples == 0:
        raise ValueError("テキストが短すぎます。seq_lenを小さくしてください。")

    optimizer = Adam(lr=lr)
    layers = model.get_all_layers()
    losses = []

    for epoch in range(n_epochs):
        # シャッフル
        perm = np.random.permutation(n_samples)
        total_loss = 0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            batch_idx = perm[start : start + batch_size]
            x_batch = X[batch_idx]
            y_batch = Y[batch_idx]

            # Forward
            logits = model.forward(x_batch)

            # 損失計算
            loss, grad = cross_entropy_loss(logits, y_batch)
            total_loss += loss
            n_batches += 1

            # Backward
            model.backward(grad)

            # 勾配クリッピング
            for layer in layers:
                for param, g, _ in layer.params_and_grads():
                    if g is not None:
                        np.clip(g, -1.0, 1.0, out=g)

            # パラメータ更新
            optimizer.step(layers)

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)

        if (epoch + 1) % print_every == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {avg_loss:.4f}")

    return losses
