import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from threadpoolctl import threadpool_limits

def mae(
        model_cls: type,
        X: pd.DataFrame,
        y: pd.Series,
        **params
) -> tuple[float, float]:

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    for train_index, valid_index in skf.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        model = model_cls(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_valid)
        maes.append(mean_absolute_error(y_valid, pred))

    return float(np.mean(maes)), float(np.std(maes))


def size_and_time(
        model_cls: type,
        X_train: pd.DataFrame,
        y: pd.Series,
        X_test: pd.DataFrame,
        **params
) -> tuple[float, float]:

    model = model_cls(**params)
    model.fit(X_train, y)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model"
        model.save(path)
        size = path.stat().st_size / 1024
        model = model_cls.load(path)

    X_test = pd.concat([X_test, X_test], axis=0, ignore_index=True)

    with threadpool_limits(limits=1):
        model.predict(X_test.iloc[:len(X_test) // 10])

        times = []
        for _ in range(5):
            before = time.perf_counter()
            model.predict(X_test)
            times.append(time.perf_counter() - before)

    return size, float(np.median(times))
