"""Byte Pair Encoding (BPE) トークナイザー

GPT系モデルで使われるサブワード分割アルゴリズム。

アルゴリズム（学習）:
1. テキストを文字（バイト）単位に分割 → 初期語彙
2. 隣接するトークンペアの出現頻度を数える
3. 最頻ペアを新しいトークンとしてマージ
4. 語彙サイズが目標に達するまで 2-3 を繰り返す

エンコード:
    学習したマージルールを優先度順に適用してトークン列に変換

デコード:
    トークンIDを対応する文字列に変換して結合
"""

import re
from collections import Counter


class BPETokenizer:

    def __init__(self, vocab_size: int = 256 + 256):
        self.vocab_size = vocab_size
        self.merges = {}          # (token_a, token_b) → merged_token_id
        self.vocab = {}           # token_id → bytes
        self.inverse_vocab = {}   # bytes → token_id

    def _get_pair_counts(self, token_ids_list: list[list[int]]) -> Counter:
        """全テキスト中の隣接ペアの出現回数を数える"""
        counts = Counter()
        for ids in token_ids_list:
            for i in range(len(ids) - 1):
                counts[(ids[i], ids[i + 1])] += 1
        return counts

    def _merge_pair(self, token_ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """トークン列中のペアを新しいIDにマージする"""
        merged = []
        i = 0
        while i < len(token_ids):
            if i < len(token_ids) - 1 and (token_ids[i], token_ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(token_ids[i])
                i += 1
        return merged

    def train(self, text: str):
        """BPEの語彙を学習する"""
        # 初期語彙: 個々のバイト (0-255)
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 特殊トークン
        self.special_tokens = {"<pad>": 256, "<eos>": 257}
        next_id = 258
        for token, tid in self.special_tokens.items():
            self.vocab[tid] = token.encode("utf-8")

        # テキストをバイト列に変換
        text_bytes = text.encode("utf-8")
        token_ids = list(text_bytes)

        # テキストを文（行）単位で分割して処理
        lines = text.split("\n")
        token_ids_list = [list(line.encode("utf-8")) for line in lines if line]

        num_merges = self.vocab_size - next_id

        for _ in range(num_merges):
            counts = self._get_pair_counts(token_ids_list)
            if not counts:
                break

            # 最頻ペアを選択
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                break

            new_id = next_id
            next_id += 1

            # 語彙に追加
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.merges[best_pair] = new_id

            # 全テキストでマージを適用
            token_ids_list = [
                self._merge_pair(ids, best_pair, new_id) for ids in token_ids_list
            ]

        # 逆引き辞書を構築
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._vocab_size = next_id

    def encode(self, text: str) -> list[int]:
        """テキストをトークンIDのリストに変換"""
        token_ids = list(text.encode("utf-8"))

        # 学習したマージルールを順番に適用
        for pair, new_id in self.merges.items():
            token_ids = self._merge_pair(token_ids, pair, new_id)

        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """トークンIDのリストをテキストに変換"""
        byte_sequences = []
        for tid in token_ids:
            if tid in self.vocab:
                byte_sequences.append(self.vocab[tid])
        return b"".join(byte_sequences).decode("utf-8", errors="replace")

    @property
    def actual_vocab_size(self) -> int:
        return self._vocab_size if hasattr(self, "_vocab_size") else len(self.vocab)
