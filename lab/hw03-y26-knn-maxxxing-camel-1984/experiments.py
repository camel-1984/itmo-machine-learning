import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import eda

class Model:
    def __init__(self,
                 scale: bool = True,
                 noize: bool = False,
                 metric: str = "euclidean",
                 agg: str = "median",
                 k: int = 21,
                 n_features: int | None = 20,
                 n_objects: int | None = 1000,
                 reduce: str = "lda",
                 n_components: int = 2,
) -> None:
        self.scale = scale
        self.noize = noize
        self.metric = metric
        self.agg = agg
        self.k = k
        self.n_features = n_features
        self.n_objects = n_objects
        self.reduce = reduce
        self.n_components = n_components

        self.knn = KNeighborsClassifier(n_neighbors=k, metric=metric)

        self.columns = None
        self.y_train = None

        self.X_train = None
        self.mean_ = None
        self.std_ = None
        self.W_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.columns = X.columns
        if not self.noize:
            self.columns = X.columns[~eda.feature_strange_mask(X)]
        X = X[self.columns]
        if self.reduce == "top" and self.n_features is not None:
            spearman = (eda.feature_target_spearman(X,y)
                        .sort_values(by="spearman",
                        key=abs,
                        ascending=False))
            need = spearman["feature"].iloc[:self.n_features]
            self.columns = need
            X = X[self.columns]

        X = X.to_numpy()
        y = y.to_numpy()

        if self.scale:
            self.mean_ = X.mean(axis=0)
            self.std_ = X.std(axis=0)
            X = (X - self.mean_) / self.std_
        if self.reduce == "lda":
            lda = LinearDiscriminantAnalysis(n_components=self.n_components).fit(X, y)
            self.W_ = lda.scalings_[:, :self.n_components]
            X = X @ self.W_

        if self.n_objects is not None and self.n_objects < len(X):
            X, _, y, _ = train_test_split(X, y,
                             train_size=self.n_objects,
                             stratify=y,
                             random_state=0)

        self.X_train = X
        self.y_train = y

        self.knn.fit(X, y)

    def predict(self, X: pd.DataFrame):
        X = X[self.columns].to_numpy()
        if self.scale:
            X = (X - self.mean_) / self.std_
        if self.W_ is not None:
            X = X @ self.W_
        if self.agg == "mode":
            return self.knn.predict(X)
        _, indices = self.knn.kneighbors(X)
        labels = self.y_train[indices]
        return np.median(labels, axis=1).astype(int)


    def save(self, path: Path) -> None:
        state = {
            "X": np.asarray(self.X_train).astype(np.float16),
            "y": self.y_train.astype(np.int8),
            "columns": list(self.columns),
            "k": self.k,
            "metric": self.metric,
            "agg": self.agg,
            "scale": self.scale,
            "mean": None if self.mean_ is None else self.mean_.astype(np.float32),
            "std": None if self.std_ is None else self.std_.astype(np.float32),
            "W": None if self.W_ is None else self.W_.astype(np.float32),
        }
        with path.open("wb") as model_file:
            pickle.dump(state, model_file)


    @classmethod
    def load(cls, path: Path) -> "Model":
        with path.open("rb") as model_file:
            state = pickle.load(model_file)

        model = cls(k=state["k"], metric=state["metric"],
                    agg=state["agg"], scale=state["scale"])
        model.columns = state["columns"]
        model.mean_ = state["mean"]
        model.std_ = state["std"]
        model.W_ = state["W"]
        model.X_train = state["X"].astype(np.float32)
        model.y_train = state["y"].astype(np.int64)

        model.knn.fit(model.X_train, model.y_train)
        return model
