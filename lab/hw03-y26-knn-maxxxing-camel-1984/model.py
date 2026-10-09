import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import eda

K = 5
N_OBJECTS = 300
N_COMPONENTS = 1

class Model:
    def __init__(self) -> None:
        self.knn = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
        self.columns = None
        self.mean_ = None
        self.std_ = None
        self.W_ = None
        self.X_train = None
        self.y_train = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.columns = list(X.columns[~eda.feature_strange_mask(X)])
        X = X[self.columns].to_numpy()
        y = y.to_numpy()

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X = (X - self.mean_) / self.std_

        lda = LinearDiscriminantAnalysis().fit(X, y)
        self.W_ = lda.scalings_[:, :N_COMPONENTS]
        X = X @ self.W_

        X, _, y, _ = train_test_split(X, y,
                                      train_size=N_OBJECTS,
                                      stratify=y,
                                      random_state=0)
        self.X_train = X
        self.y_train = y
        self.knn.fit(X, y)

    def predict(self, X: pd.DataFrame):
        X = X[self.columns].to_numpy()
        X = (X - self.mean_) / self.std_
        X = X @ self.W_
        _, indices = self.knn.kneighbors(X)
        return np.median(self.y_train[indices], axis=1).astype(int)

    def save(self, path: Path) -> None:
        state = {
            "X": self.X_train.astype(np.float16),
            "y": self.y_train.astype(np.int8),
            "columns": self.columns,
            "mean": self.mean_.astype(np.float32),
            "std": self.std_.astype(np.float32),
            "W": self.W_.astype(np.float32),
        }
        with path.open("wb") as model_file:
            pickle.dump(state, model_file)

    @classmethod
    def load(cls, path: Path) -> "Model":
        with path.open("rb") as model_file:
            state = pickle.load(model_file)

        model = cls()
        model.columns = state["columns"]
        model.mean_ = state["mean"]
        model.std_ = state["std"]
        model.W_ = state["W"]
        model.X_train = state["X"].astype(np.float32)
        model.y_train = state["y"].astype(np.int64)
        model.knn.fit(model.X_train, model.y_train)
        return model
