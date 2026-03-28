"""LLM (GPT) のフルスクラッチ実装デモ

小さなテキストデータでBPEトークナイザーの学習 → GPTモデルの学習 → テキスト生成を行う。
"""

import numpy as np
from llm.tokenizer.bpe import BPETokenizer
from llm.model.gpt import GPT
from llm.train import train
from llm.generate import generate


def main():
    # --- データ準備 ---
    text = """The cat sat on the mat.
The dog sat on the log.
The cat and the dog played in the park.
The bird flew over the tree.
The fish swam in the sea.
The cat chased the bird.
The dog chased the cat.
The bird sang a song.
The cat sat on the mat again.
The dog played with the ball.
The cat and the dog are friends.
The bird flew to the tree.
The fish jumped out of the sea.
The cat slept on the mat.
The dog ran in the park.
The bird sang in the morning.
The cat played with the ball.
The dog sat on the mat.
The cat and the bird are friends.
The dog and the fish are friends."""

    # --- トークナイザー ---
    print("=== BPEトークナイザーの学習 ===")
    tokenizer = BPETokenizer(vocab_size=300)
    tokenizer.train(text)
    print(f"  語彙サイズ: {tokenizer.actual_vocab_size}")

    sample = "The cat sat"
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded)
    print(f"  エンコード: '{sample}' → {encoded}")
    print(f"  デコード: {encoded} → '{decoded}'")

    # --- GPTモデル ---
    print("\n=== GPTモデルの構築 ===")
    model = GPT(
        vocab_size=tokenizer.actual_vocab_size,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=64,
    )
    print(f"  パラメータ数: {model.count_parameters():,}")

    # --- 学習 ---
    print("\n=== 学習 ===")
    losses = train(
        model=model,
        train_text=text,
        tokenizer=tokenizer,
        n_epochs=50,
        seq_len=16,
        batch_size=4,
        lr=1e-3,
        print_every=10,
    )

    # --- テキスト生成 ---
    print("\n=== テキスト生成 ===")
    prompt = "The cat"
    input_ids = np.array([tokenizer.encode(prompt)])
    print(f"  プロンプト: '{prompt}'")

    for name, kwargs in [
        ("Greedy", {"temperature": 0}),
        ("Sampling (T=0.8)", {"temperature": 0.8}),
        ("Top-k (k=5)", {"temperature": 0.8, "top_k": 5}),
        ("Top-p (p=0.9)", {"temperature": 0.8, "top_p": 0.9}),
    ]:
        output_ids = generate(model, input_ids, max_new_tokens=30, **kwargs)
        output_text = tokenizer.decode(output_ids[0].tolist())
        print(f"  {name}: '{output_text}'")


if __name__ == "__main__":
    main()
