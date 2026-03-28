"""機械学習アルゴリズムの使用例"""

import numpy as np
from utils.data import make_classification, make_regression, train_test_split, standardize
from utils.metrics import accuracy, mse, r2_score


def example_linear_regression():
    print("=== 線形回帰 ===")
    from ml.linear_models.linear_regression import LinearRegression, LinearRegressionNormal

    X, y = make_regression(n_samples=200, n_features=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LinearRegression(lr=0.01, n_iters=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  勾配降下法 - MSE: {mse(y_test, pred):.4f}, R²: {r2_score(y_test, pred):.4f}")

    model_n = LinearRegressionNormal()
    model_n.fit(X_train, y_train)
    pred_n = model_n.predict(X_test)
    print(f"  正規方程式 - MSE: {mse(y_test, pred_n):.4f}, R²: {r2_score(y_test, pred_n):.4f}")


def example_logistic_regression():
    print("\n=== ロジスティック回帰 ===")
    from ml.linear_models.logistic_regression import LogisticRegression

    X, y = make_classification(n_samples=200)
    X = standardize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy(y_test, pred):.4f}")


def example_decision_tree():
    print("\n=== 決定木 ===")
    from ml.tree.decision_tree import DecisionTreeClassifier

    X, y = make_classification(n_samples=200, n_features=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy(y_test, pred):.4f}")


def example_kmeans():
    print("\n=== K-Means ===")
    from ml.clustering.kmeans import KMeans

    X, y = make_classification(n_samples=150, n_classes=3)
    model = KMeans(n_clusters=3)
    model.fit(X)
    print(f"  クラスタ割り当て (先頭10個): {model.labels_[:10]}")


def example_pca():
    print("\n=== PCA ===")
    from ml.dimensionality_reduction.pca import PCA

    X, _ = make_classification(n_samples=100, n_features=5)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    print(f"  元の次元: {X.shape[1]} → 削減後: {X_reduced.shape[1]}")
    print(f"  寄与率: {pca.explained_variance_ / pca.explained_variance_.sum()}")


def example_naive_bayes():
    print("\n=== ナイーブベイズ ===")
    from ml.naive_bayes.gaussian_nb import GaussianNB

    X, y = make_classification(n_samples=200)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"  Accuracy: {accuracy(y_test, pred):.4f}")


if __name__ == "__main__":
    example_linear_regression()
    example_logistic_regression()
    example_decision_tree()
    example_kmeans()
    example_pca()
    example_naive_bayes()
