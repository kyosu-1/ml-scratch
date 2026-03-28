"""Word2Vec (Skip-gram with Negative Sampling)

分布仮説「似た文脈に出現する単語は似た意味を持つ」に基づく単語埋め込み学習。

Skip-gram:
    中心語から周辺語を予測する。
    目的: max Σ log P(w_context | w_center)

Negative Sampling:
    softmax（語彙全体の正規化）は計算コストが高い。
    代わりに二値分類に帰着させる:
    - 正例: (center, context) の実際のペア → 1に近づける
    - 負例: (center, random_word) のランダムペア → 0に近づける

    L = log σ(v_c · v_w) + Σ_neg log σ(-v_c · v_n)

    σ: シグモイド関数
    v_c: 中心語の埋め込み
    v_w: 正例コンテキスト語の埋め込み
    v_n: 負例の埋め込み
"""

import numpy as np
from collections import Counter


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class Word2Vec:

    def __init__(
        self,
        embed_dim: int = 50,
        window_size: int = 2,
        n_negatives: int = 5,
        lr: float = 0.025,
        n_epochs: int = 5,
        min_count: int = 1,
    ):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.n_negatives = n_negatives
        self.lr = lr
        self.n_epochs = n_epochs
        self.min_count = min_count

    def _build_vocab(self, corpus: list[list[str]]):
        """語彙を構築"""
        word_counts = Counter()
        for sentence in corpus:
            word_counts.update(sentence)

        self.vocab = {
            word: idx
            for idx, (word, count) in enumerate(word_counts.items())
            if count >= self.min_count
        }
        self.idx_to_word = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 負例サンプリング用の確率分布（頻度の3/4乗）
        counts = np.array([word_counts[self.idx_to_word[i]] for i in range(self.vocab_size)],
                          dtype=np.float64)
        powered = counts ** 0.75
        self.neg_sampling_probs = powered / powered.sum()

    def _generate_training_pairs(self, corpus: list[list[str]]):
        """(center, context) ペアを生成"""
        pairs = []
        for sentence in corpus:
            indices = [self.vocab[w] for w in sentence if w in self.vocab]
            for i, center in enumerate(indices):
                window_start = max(0, i - self.window_size)
                window_end = min(len(indices), i + self.window_size + 1)
                for j in range(window_start, window_end):
                    if i != j:
                        pairs.append((center, indices[j]))
        return pairs

    def fit(self, corpus: list[list[str]]) -> "Word2Vec":
        """
        corpus: 文のリスト。各文は単語のリスト。
        例: [["the", "cat", "sat"], ["the", "dog", "ran"]]
        """
        self._build_vocab(corpus)

        # 埋め込み行列の初期化
        self.W_center = np.random.randn(self.vocab_size, self.embed_dim) * 0.01
        self.W_context = np.random.randn(self.vocab_size, self.embed_dim) * 0.01

        pairs = self._generate_training_pairs(corpus)

        for epoch in range(self.n_epochs):
            total_loss = 0
            np.random.shuffle(pairs)

            for center_idx, context_idx in pairs:
                # 正例
                v_center = self.W_center[center_idx]
                v_context = self.W_context[context_idx]

                score = _sigmoid(v_center @ v_context)
                loss = -np.log(score + 1e-10)

                # 正例の勾配
                grad = (score - 1)  # dL/d(dot)
                grad_center = grad * v_context
                grad_context = grad * v_center

                self.W_center[center_idx] -= self.lr * grad_center
                self.W_context[context_idx] -= self.lr * grad_context

                # 負例
                neg_indices = np.random.choice(
                    self.vocab_size, self.n_negatives, p=self.neg_sampling_probs
                )
                for neg_idx in neg_indices:
                    v_neg = self.W_context[neg_idx]
                    score_neg = _sigmoid(v_center @ v_neg)
                    loss += -np.log(1 - score_neg + 1e-10)

                    grad_neg = score_neg
                    self.W_center[center_idx] -= self.lr * grad_neg * v_neg
                    self.W_context[neg_idx] -= self.lr * grad_neg * v_center

                total_loss += loss

        return self

    def get_embedding(self) -> np.ndarray:
        """単語埋め込み行列を返す"""
        return self.W_center

    def word_vector(self, word: str) -> np.ndarray:
        """単語のベクトルを返す"""
        return self.W_center[self.vocab[word]]

    def most_similar(self, word: str, top_k: int = 5) -> list[tuple[str, float]]:
        """コサイン類似度で最も近い単語を返す"""
        if word not in self.vocab:
            return []
        vec = self.word_vector(word)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-10)

        similarities = self.W_center @ vec_norm
        norms = np.linalg.norm(self.W_center, axis=1) + 1e-10
        similarities = similarities / norms

        top_indices = np.argsort(similarities)[::-1][1:top_k + 1]
        return [(self.idx_to_word[i], similarities[i]) for i in top_indices]
