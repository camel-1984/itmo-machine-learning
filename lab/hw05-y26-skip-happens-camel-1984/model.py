"""Реализуйте одно дерево классификации самостоятельно."""

import pickle
import zlib
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd

GENRE_SHARES = [f"playlist_genre_{i}_share" for i in range(4)]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if all(col in X.columns for col in ["track_genre", *GENRE_SHARES]):
        shares = X[GENRE_SHARES].to_numpy(dtype=float)
        genre = X["track_genre"].to_numpy(dtype=int)
        X["genre_share"] = shares[np.arange(len(X)), genre]
    if "track_plays_7d" in X.columns and "playlist_repeat_share" in X.columns:
        X["repeat_fatigue"] = X["track_plays_7d"] * (1 - X["playlist_repeat_share"])
    if "listening_context" not in X.columns:
        return X
    ctx = X["listening_context"].to_numpy()
    if "energy" in X.columns:
        energy = X["energy"].to_numpy(dtype=float)
        X["energy_work"] = np.where(ctx == 1, energy, 0.0)
        X["energy_workout"] = np.where(ctx == 2, 1 - energy, 0.0)
        X["energy_sleep"] = np.where(ctx == 4, energy, 0.0)
    if "instrumentalness" in X.columns:
        instr = X["instrumentalness"].to_numpy(dtype=float)
        X["instr_work"] = np.where(ctx == 1, 1 - instr, 0.0)
        X["instr_workout"] = np.where(ctx == 2, instr, 0.0)
    if "track_genre" in X.columns:
        genre = X["track_genre"].to_numpy(dtype=float)
        for c in (1, 2, 4):
            X[f"genre_ctx{c}"] = np.where(ctx == c, genre, -1.0)
    return X


def _children(feature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.full(len(feature), -1)
    right = np.full(len(feature), -1)
    stack = []
    for k in range(len(feature)):
        if k > 0:
            if feature[k - 1] >= 0:
                left[k - 1] = k
            else:
                right[stack.pop()] = k
        if feature[k] >= 0:
            stack.append(k)
    return left, right


class Model:
    def __init__(self, case: str, features: list[str], max_depth: int | None = 6,
                 min_samples_leaf: int = 30, threshold: float = 0.5,
                 smoothing: float = 1.0):
        self.case = case
        self.features = list(features)
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.threshold = threshold
        self.smoothing = smoothing

        self._X = None
        self._y = None
        self.feature = None
        self.thresholds = None
        self.left = None
        self.right = None
        self.value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self._X = add_features(X[self.features]).to_numpy(dtype=float)
        self._y = np.asarray(y, dtype=float)
        self.feature, self.thresholds, self.left, self.right, self.value = [], [], [], [], []
        self._build(np.argsort(self._X, axis=0), 0)
        self._X, self._y = None, None
        self._to_arrays()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = add_features(X[self.features]).to_numpy(dtype=float)
        node = np.zeros(len(X), dtype=int)
        while True:
            f = self.feature[node]
            active = np.flatnonzero(f >= 0)
            if active.size == 0:
                break
            cur = node[active]
            go_left = X[active, f[active]] <= self.thresholds[cur]
            node[active] = np.where(go_left, self.left[cur], self.right[cur])
        return self.value[node]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> None:
        internal = self.feature >= 0
        state = {
            "case": self.case,
            "features": self.features,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "threshold": self.threshold,
            "smoothing": self.smoothing,
            "feature": self.feature.astype(np.int16).tobytes(),
            "thresholds": self.thresholds[internal].tobytes(),
            "value": self.value[~internal].tobytes(),
        }
        with open(path, "wb") as f:
            f.write(zlib.compress(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), 9))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with open(path, "rb") as f:
            state = pickle.loads(zlib.decompress(f.read()))
        feature = np.frombuffer(state.pop("feature"), dtype=np.int16).astype(int)
        thresholds = np.frombuffer(state.pop("thresholds"), dtype=float)
        value = np.frombuffer(state.pop("value"), dtype=float)
        model = cls(**state)
        internal = feature >= 0
        model.feature = feature
        model.thresholds = np.zeros(len(feature))
        model.thresholds[internal] = thresholds
        model.value = np.zeros(len(feature))
        model.value[~internal] = value
        model.left, model.right = _children(feature)
        return model

    def _to_arrays(self) -> None:
        self.feature = np.asarray(self.feature, dtype=int)
        self.thresholds = np.asarray(self.thresholds, dtype=float)
        self.left = np.asarray(self.left, dtype=int)
        self.right = np.asarray(self.right, dtype=int)
        self.value = np.asarray(self.value, dtype=float)

    def _build(self, order: np.ndarray, depth: int) -> int:
        y = self._y[order[:, 0]]
        n = len(y)
        n_1 = y.sum()

        k = len(self.value)
        self.feature.append(-1)
        self.thresholds.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.value.append(float((n_1 + self.smoothing) / (n + 2 * self.smoothing)))

        if ((self.max_depth is not None and depth >= self.max_depth) or
                n_1 == 0 or n_1 == n or
                n < 2 * self.min_samples_leaf):
            return k

        cols = np.arange(order.shape[1])
        xs = self._X[order, cols]
        ys = self._y[order]
        p = n_1 / n
        best_q, best_j, best_t = self._best_split(xs, ys, self.min_samples_leaf)
        if best_j is None or best_q >= 2 * p * (1 - p):
            return k

        go_left = (self._X[order, best_j] <= best_t).T
        self.feature[k] = best_j
        self.thresholds[k] = best_t
        self.left[k] = self._build(order.T[go_left].reshape(len(cols), -1).T, depth + 1)
        self.right[k] = self._build(order.T[~go_left].reshape(len(cols), -1).T, depth + 1)
        return k

    @staticmethod
    def _best_split(xs: np.ndarray,
                    ys: np.ndarray,
                    min_samples_leaf: int
                    ) -> tuple[float, int, float] | tuple[float, None, None]:
        n = len(ys)
        left_n = np.arange(1, n)[:, None]
        right_n = n - left_n

        left_1 = np.cumsum(ys, axis=0)[:-1]
        right_1 = ys[:, 0].sum() - left_1

        p_l = left_1 / left_n
        p_r = right_1 / right_n
        g_l = 2 * p_l * (1 - p_l)
        g_r = 2 * p_r * (1 - p_r)
        q = (left_n * g_l + right_n * g_r) / n

        valid = (xs[:-1] != xs[1:]) & (left_n >= min_samples_leaf) & (right_n >= min_samples_leaf)
        if not valid.any():
            return np.inf, None, None

        q = np.where(valid, q, np.inf)
        j, i = divmod(int(np.argmin(q.T)), n - 1)
        threshold = (xs[i, j] + xs[i + 1, j]) / 2

        return float(q[i, j]), j, float(threshold)
