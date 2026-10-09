from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class LinearRegressionGD:
    def __init__(
        self,
        learning_rate=0.01,
        l2=0.0,
        max_iter=10000,
        tol=1e-8,
    ) -> None:
        self.learning_rate = learning_rate
        self.l2 = l2
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, k = X.shape
        self.w = np.zeros(k)
        self.b = 0.0
        prev_loss = np.inf
        for _ in range(self.max_iter):
            error = X @ self.w + self.b - y
            loss = np.mean(error ** 2) + self.l2 * np.sum(self.w ** 2)
            if prev_loss - loss < self.tol:
                break
            prev_loss = loss
            grad_w = 2 / m * X.T @ error + 2 * self.l2 * self.w
            grad_b = 2 / m * np.sum(error)
            self.w -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b


class Model:
    def __init__(
        self,
        learning_rate=0.1,
        l2=1e-5,
        max_iter=20000,
        tol=1e-10,
    ) -> None:
        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
        self.scaler = StandardScaler()
        self.regressor = LinearRegressionGD(learning_rate, l2, max_iter, tol)

    def _features(self, X: pd.DataFrame) -> np.ndarray:
        onehot = self.encoder.transform(X[["algorithm"]])
        is_counting = (X["algorithm"] == "counting_sort").to_numpy()
        size = np.where(is_counting, X["n"] + X["value_range"], X["n"])
        columns = [
            np.log(size),
            X["unique_ratio"].to_numpy(),
            X["max_frequency_ratio"].to_numpy(),
            X["unique_count"].to_numpy(),
        ]
        features = [onehot]
        for column in columns:
            features.append(onehot * column[:, None])
        return np.hstack(features)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.encoder.fit(X[["algorithm"]])
        features = self.scaler.fit_transform(self._features(X))
        self.regressor.fit(features, np.log(y))

    def predict(self, X: pd.DataFrame):
        features = self.scaler.transform(self._features(X))
        return np.exp(self.regressor.predict(features))

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "Model":
        return joblib.load(path)
