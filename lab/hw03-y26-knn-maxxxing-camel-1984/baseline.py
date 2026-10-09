import numpy as np
import pandas as pd


class BaselineModel:
    def __init__(self) -> None:
        self.value = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.value = int(y.median())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)
