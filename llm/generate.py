"""テキスト生成ユーティリティ

自己回帰的にトークンを1つずつ生成する。

サンプリング戦略:
- Greedy: 常に最も確率の高いトークンを選択
- Temperature: logitsをTで割ってからsoftmax。T>1で多様に、T<1で決定的に
- Top-k: 上位k個のトークンからのみサンプリング
- Top-p (Nucleus): 累積確率がpに達するまでのトークンからサンプリング
"""

import numpy as np


def _softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sample_token(
    logits: np.ndarray,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> int:
    """logitsから1トークンをサンプリングする

    logits: (vocab_size,) の1次元配列
    """
    # Temperature
    if temperature == 0:
        return int(np.argmax(logits))
    logits = logits / temperature

    # Top-k フィルタリング
    if top_k > 0:
        top_k = min(top_k, len(logits))
        indices = np.argsort(logits)[::-1][:top_k]
        mask = np.full_like(logits, -1e9)
        mask[indices] = logits[indices]
        logits = mask

    # Top-p (Nucleus) フィルタリング
    if top_p > 0:
        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]
        probs = _softmax(sorted_logits)
        cumsum = np.cumsum(probs)

        # 累積確率がpを超える位置を見つけ、それ以降をマスク
        cutoff_idx = np.searchsorted(cumsum, top_p) + 1
        mask_indices = sorted_indices[cutoff_idx:]
        logits[mask_indices] = -1e9

    probs = _softmax(logits)
    return int(np.random.choice(len(probs), p=probs))


def generate(
    model,
    input_ids: np.ndarray,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    eos_token_id: int = None,
) -> np.ndarray:
    """自己回帰的にテキストを生成する

    model: GPTモデル
    input_ids: (1, seq_len) の初期トークン列
    """
    generated = list(input_ids[0])

    for _ in range(max_new_tokens):
        # 現在の系列でforward
        current_ids = np.array([generated])
        logits = model.forward(current_ids)

        # 最後のトークンのlogitsを使う
        next_logits = logits[0, -1]

        # サンプリング
        next_token = sample_token(
            next_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        generated.append(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            break

    return np.array([generated])
