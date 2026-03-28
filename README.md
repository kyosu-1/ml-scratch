# ml-scratch

機械学習・深層学習・強化学習のアルゴリズムをフルスクラッチ（NumPyのみ）で実装し、理解を深めるためのリポジトリ。

## 構成

```
ml/                 # 機械学習
  linear_models/    # 線形回帰, ロジスティック回帰, Ridge, Lasso
  tree/             # 決定木 (CART, 分類・回帰)
  svm/              # サポートベクターマシン
  neighbors/        # KNN (分類・回帰)
  clustering/       # K-Means, DBSCAN
  dimensionality_reduction/  # PCA, t-SNE
  ensemble/         # Random Forest, Gradient Boosting
  naive_bayes/      # ガウシアンナイーブベイズ

dl/                 # 深層学習
  layers/           # Linear, Conv2D, BatchNorm, Dropout, RNN, LSTM, GRU,
                    # Embedding, MaxPool2D, Flatten,
                    # Multi-Head Attention, Transformer Block
  activations/      # ReLU, Sigmoid, Tanh, Softmax
  losses/           # MSE, CrossEntropy, BinaryCrossEntropy
  optimizers/       # SGD (momentum), Adam
  models/           # Sequential モデル

rl/                 # 強化学習
  bandits/          # 多腕バンディット (Epsilon-Greedy, UCB)
  dynamic_programming/  # 価値反復, 方策反復
  td_learning/      # Q-Learning, SARSA
  policy_gradient/  # REINFORCE
  dqn/              # DQN (Experience Replay, Target Network)

llm/                # LLM (GPT フルスクラッチ)
  tokenizer/        # BPEトークナイザー
  model/            # GPT (Causal Self-Attention, Positional Encoding)
  train.py          # 次トークン予測学習ループ
  generate.py       # テキスト生成 (Greedy, Temperature, Top-k, Top-p)

utils/              # データ生成, 評価指標
tests/              # テスト (64 tests)
examples/           # 使用例
```

## セットアップ

```bash
uv sync
```

## 使い方

```python
from ml.linear_models.linear_regression import LinearRegression

model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

サンプル実行:
```bash
PYTHONPATH=. uv run python examples/ml/example_ml.py
PYTHONPATH=. uv run python examples/dl/example_dl.py
PYTHONPATH=. uv run python examples/rl/example_rl.py
PYTHONPATH=. uv run python examples/llm/example_llm.py
```

テスト:
```bash
PYTHONPATH=. uv run pytest tests/ -v
```

## 方針

- **外部ライブラリは NumPy のみ**（可視化用に matplotlib は許容）
- 各アルゴリズムは独立したモジュールとして実装
- 数式やアルゴリズムの解説はコード内のdocstringに記載
