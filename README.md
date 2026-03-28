# ml-scratch

機械学習・深層学習・強化学習のアルゴリズムをフルスクラッチ（NumPyのみ）で実装し、理解を深めるためのリポジトリ。

## 構成

```
ml/                 # 機械学習
  linear_models/    # 線形回帰, ロジスティック回帰
  tree/             # 決定木, CART
  svm/              # サポートベクターマシン
  clustering/       # K-Means, DBSCAN
  dimensionality_reduction/  # PCA, t-SNE
  ensemble/         # Random Forest, Gradient Boosting
  naive_bayes/      # ナイーブベイズ

dl/                 # 深層学習
  layers/           # Linear, Conv2D, BatchNorm, etc.
  activations/      # ReLU, Sigmoid, Softmax, etc.
  losses/           # CrossEntropy, MSE
  optimizers/       # SGD, Adam
  models/           # Sequential モデル

rl/                 # 強化学習
  bandits/          # 多腕バンディット (Epsilon-Greedy, UCB)
  dynamic_programming/  # 価値反復, 方策反復
  td_learning/      # Q-Learning, SARSA

utils/              # ユーティリティ（データ生成, 評価指標, 可視化）
examples/           # 使用例
```

## セットアップ

```bash
pip install -e .
```

## 使い方

```python
from ml.linear_models.linear_regression import LinearRegression

model = LinearRegression(lr=0.01, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 方針

- **外部ライブラリは NumPy のみ**（可視化用に matplotlib は許容）
- 各アルゴリズムは独立したモジュールとして実装
- 数式やアルゴリズムの解説はコード内のdocstringに記載
