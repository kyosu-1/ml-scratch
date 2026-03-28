"""データ生成・前処理ユーティリティ"""

import numpy as np


def make_classification(
    n_samples: int = 100,
    n_features: int = 2,
    n_classes: int = 2,
    random_state: int = 42,
):
    """分類用のダミーデータを生成"""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    y = np.zeros(n_samples, dtype=int)

    samples_per_class = n_samples // n_classes
    for c in range(n_classes):
        start = c * samples_per_class
        end = start + samples_per_class
        X[start:end] += c * 2  # クラス間にオフセット
        y[start:end] = c

    # シャッフル
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: int = 42,
):
    """回帰用のダミーデータを生成"""
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features)
    w = rng.randn(n_features)
    y = X @ w + noise * rng.randn(n_samples)
    return X, y


def train_test_split(X, y, test_size=0.2, random_state=42):
    """データをトレーニングセットとテストセットに分割"""
    rng = np.random.RandomState(random_state)
    n = len(y)
    indices = rng.permutation(n)
    test_n = int(n * test_size)

    test_idx = indices[:test_n]
    train_idx = indices[test_n:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def normalize(X: np.ndarray) -> np.ndarray:
    """Min-Max正規化 [0, 1]"""
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)
    return (X - min_val) / (max_val - min_val + 1e-8)


def standardize(X: np.ndarray) -> np.ndarray:
    """標準化 (Z-score)"""
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
