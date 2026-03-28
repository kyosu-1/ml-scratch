"""機械学習アルゴリズムのテスト"""

import numpy as np
import pytest
from utils.data import make_classification, make_regression, train_test_split, standardize


class TestLinearRegression:
    def test_gradient_descent(self):
        from ml.linear_models.linear_regression import LinearRegression
        X, y = make_regression(n_samples=200, n_features=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = LinearRegression(lr=0.01, n_iters=1000).fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        assert r2 > 0.9

    def test_normal_equation(self):
        from ml.linear_models.linear_regression import LinearRegressionNormal
        X, y = make_regression(n_samples=200, n_features=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = LinearRegressionNormal().fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = 1 - np.sum((y_test - pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)
        assert r2 > 0.95


class TestLogisticRegression:
    def test_binary_classification(self):
        from ml.linear_models.logistic_regression import LogisticRegression
        X, y = make_classification(n_samples=200)
        X = standardize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = LogisticRegression(lr=0.1, n_iters=1000).fit(X_train, y_train)
        pred = model.predict(X_test)
        assert np.mean(pred == y_test) > 0.8


class TestRidge:
    def test_ridge(self):
        from ml.linear_models.ridge import Ridge
        X, y = make_regression(n_samples=200, n_features=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = Ridge(alpha=1.0).fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = np.mean((y_test - pred) ** 2)
        assert mse < 1.0


class TestLasso:
    def test_lasso(self):
        from ml.linear_models.lasso import Lasso
        X, y = make_regression(n_samples=200, n_features=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = Lasso(alpha=0.01, n_iters=1000).fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = np.mean((y_test - pred) ** 2)
        assert mse < 1.0


class TestDecisionTree:
    def test_classifier(self):
        from ml.tree.decision_tree import DecisionTreeClassifier
        X, y = make_classification(n_samples=200, n_features=4)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
        pred = model.predict(X_test)
        assert np.mean(pred == y_test) > 0.8

    def test_regressor(self):
        from ml.tree.decision_tree import DecisionTreeRegressor
        X, y = make_regression(n_samples=200, n_features=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = DecisionTreeRegressor(max_depth=5).fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = np.mean((y_test - pred) ** 2)
        assert mse < 1.0


class TestSVM:
    def test_classification(self):
        from ml.svm.svm import SVM
        X, y = make_classification(n_samples=100)
        X = standardize(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # SVM は {-1, 1} ラベル
        model = SVM(lr=0.001, n_iters=500).fit(X_train, y_train)
        pred = model.predict(X_test)
        # sign出力なので -1/1 と 0/1 の一致を見る
        pred_binary = np.where(pred >= 0, 1, 0)
        assert np.mean(pred_binary == y_test) > 0.7


class TestKMeans:
    def test_clustering(self):
        from ml.clustering.kmeans import KMeans
        X, _ = make_classification(n_samples=150, n_classes=3)
        model = KMeans(n_clusters=3).fit(X)
        assert len(np.unique(model.labels_)) == 3
        assert model.labels_.shape == (150,)


class TestDBSCAN:
    def test_clustering(self):
        from ml.clustering.dbscan import DBSCAN
        # 明確に分離された2クラスタ
        rng = np.random.RandomState(42)
        cluster1 = rng.randn(30, 2) + np.array([0, 0])
        cluster2 = rng.randn(30, 2) + np.array([10, 10])
        X = np.vstack([cluster1, cluster2])
        model = DBSCAN(eps=1.5, min_samples=3).fit(X)
        n_clusters = len(set(model.labels_) - {-1})
        assert n_clusters == 2


class TestPCA:
    def test_dimensionality_reduction(self):
        from ml.dimensionality_reduction.pca import PCA
        X, _ = make_classification(n_samples=100, n_features=5)
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        assert X_reduced.shape == (100, 2)

    def test_inverse_transform(self):
        from ml.dimensionality_reduction.pca import PCA
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        pca = PCA(n_components=3)
        X_t = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_t)
        np.testing.assert_allclose(X, X_reconstructed, atol=1e-10)


class TestKNN:
    def test_classifier(self):
        from ml.neighbors.knn import KNNClassifier
        X, y = make_classification(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = KNNClassifier(k=5).fit(X_train, y_train)
        pred = model.predict(X_test)
        assert np.mean(pred == y_test) > 0.8

    def test_regressor(self):
        from ml.neighbors.knn import KNNRegressor
        X, y = make_regression(n_samples=200, n_features=2)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = KNNRegressor(k=5).fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = np.mean((y_test - pred) ** 2)
        assert mse < 2.0


class TestRandomForest:
    def test_classifier(self):
        from ml.ensemble.random_forest import RandomForestClassifier
        X, y = make_classification(n_samples=200, n_features=4)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = RandomForestClassifier(n_estimators=10, max_depth=5).fit(X_train, y_train)
        pred = model.predict(X_test)
        assert np.mean(pred == y_test) > 0.7


class TestGradientBoosting:
    def test_regressor(self):
        from ml.ensemble.gradient_boosting import GradientBoostingRegressor
        X, y = make_regression(n_samples=200, n_features=3)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = GradientBoostingRegressor(n_estimators=50, lr=0.1, max_depth=3).fit(X_train, y_train)
        pred = model.predict(X_test)
        mse = np.mean((y_test - pred) ** 2)
        assert mse < 1.0


class TestNaiveBayes:
    def test_gaussian_nb(self):
        from ml.naive_bayes.gaussian_nb import GaussianNB
        X, y = make_classification(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = GaussianNB().fit(X_train, y_train)
        pred = model.predict(X_test)
        assert np.mean(pred == y_test) > 0.8
