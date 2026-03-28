"""LLM (GPT) のテスト"""

import numpy as np
import pytest


class TestBPETokenizer:
    def test_encode_decode_roundtrip(self):
        from llm.tokenizer.bpe import BPETokenizer
        text = "hello world. hello there."
        tokenizer = BPETokenizer(vocab_size=280)
        tokenizer.train(text)

        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        assert decoded == "hello"

    def test_compression(self):
        from llm.tokenizer.bpe import BPETokenizer
        text = "abababababababab"
        tokenizer = BPETokenizer(vocab_size=260)
        tokenizer.train(text)

        encoded = tokenizer.encode(text)
        # BPEによりバイト列より短くなっているはず
        assert len(encoded) < len(text)

    def test_unknown_text(self):
        from llm.tokenizer.bpe import BPETokenizer
        tokenizer = BPETokenizer(vocab_size=260)
        tokenizer.train("hello world")

        # 学習に含まれない文字でもバイトレベルで処理可能
        encoded = tokenizer.encode("xyz")
        decoded = tokenizer.decode(encoded)
        assert decoded == "xyz"


class TestGPTModel:
    def test_forward_shape(self):
        from llm.model.gpt import GPT
        model = GPT(vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32)
        input_ids = np.array([[1, 5, 3, 10]])
        logits = model.forward(input_ids)
        assert logits.shape == (1, 4, 50)

    def test_backward_runs(self):
        from llm.model.gpt import GPT
        model = GPT(vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32)
        input_ids = np.array([[1, 5, 3]])
        logits = model.forward(input_ids)
        grad = np.random.randn(*logits.shape) * 0.01
        model.backward(grad)

        # 勾配が計算されていること
        assert model.grad_token_emb is not None
        assert model.grad_lm_head is not None

    def test_causal_masking(self):
        """因果マスクが正しく機能するか: 同じプレフィックスのlogitsが一致する"""
        from llm.model.gpt import GPT
        model = GPT(vocab_size=30, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32)

        # [1, 2, 3] と [1, 2, 3, 4] の先頭3トークンのlogitsは一致すべき
        ids_short = np.array([[1, 2, 3]])
        ids_long = np.array([[1, 2, 3, 4]])

        logits_short = model.forward(ids_short)
        logits_long = model.forward(ids_long)

        np.testing.assert_allclose(
            logits_short[0], logits_long[0, :3], atol=1e-6,
            err_msg="因果マスクが正しくない: 同じプレフィックスで異なるlogitsが出力された"
        )

    def test_parameter_count(self):
        from llm.model.gpt import GPT
        model = GPT(vocab_size=50, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32)
        n_params = model.count_parameters()
        assert n_params > 0


class TestTraining:
    def test_loss_decreases(self):
        from llm.tokenizer.bpe import BPETokenizer
        from llm.model.gpt import GPT
        from llm.train import train

        text = ("the cat sat on the mat. the dog ran in the park. "
                "the bird flew over the tree. the fish swam in the sea. ") * 10
        tokenizer = BPETokenizer(vocab_size=270)
        tokenizer.train(text)

        model = GPT(
            vocab_size=tokenizer.actual_vocab_size,
            d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32,
        )

        losses = train(
            model, text, tokenizer,
            n_epochs=10, seq_len=4, batch_size=4, lr=1e-3, print_every=100,
        )

        assert losses[-1] < losses[0], "損失が減少していない"

    def test_create_sequences(self):
        from llm.train import create_sequences
        token_ids = list(range(100))
        X, Y = create_sequences(token_ids, seq_len=10)
        assert X.shape[1] == 10
        assert Y.shape[1] == 10
        # ターゲットは入力の1つ右シフト
        np.testing.assert_array_equal(X[0][1:], Y[0][:-1])


class TestGeneration:
    def test_greedy_generation(self):
        from llm.model.gpt import GPT
        from llm.generate import generate

        model = GPT(vocab_size=30, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=32)
        input_ids = np.array([[1, 2]])

        output = generate(model, input_ids, max_new_tokens=5, temperature=0)
        assert output.shape[1] == 2 + 5  # 入力2 + 生成5

    def test_sampling_strategies(self):
        from llm.generate import sample_token

        logits = np.array([1.0, 2.0, 3.0, 0.5, 0.1])

        # Greedy: 最大値のインデックス
        assert sample_token(logits, temperature=0) == 2

        # Top-k: 上位k個の中からサンプリング
        for _ in range(10):
            token = sample_token(logits, temperature=1.0, top_k=2)
            assert token in [1, 2]  # 上位2個

    def test_eos_stops_generation(self):
        from llm.model.gpt import GPT
        from llm.generate import generate

        model = GPT(vocab_size=30, d_model=16, n_heads=2, n_layers=1, d_ff=32, max_seq_len=64)
        input_ids = np.array([[1]])

        output = generate(model, input_ids, max_new_tokens=50, temperature=0.5, eos_token_id=None)
        assert output.shape[1] == 51  # 1 + 50


class TestPositionalEncoding:
    def test_sinusoidal(self):
        from llm.model.positional import SinusoidalPositionalEncoding
        pe = SinusoidalPositionalEncoding(d_model=16, max_len=100)
        x = np.zeros((2, 10, 16))
        out = pe.forward(x)
        # 位置エンコーディングが加算されていること
        assert not np.allclose(out, 0)
        assert out.shape == (2, 10, 16)

    def test_learnable(self):
        from llm.model.positional import LearnablePositionalEncoding
        pe = LearnablePositionalEncoding(max_len=100, d_model=16)
        x = np.zeros((2, 10, 16))
        out = pe.forward(x)
        grad = np.ones_like(out)
        pe.backward(grad)
        assert pe.grad_weights is not None
