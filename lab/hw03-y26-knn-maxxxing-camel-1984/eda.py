import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif

def target_distribution(
        df: pd.DataFrame
) -> pd.DataFrame:
    counts = df["target"].value_counts().sort_index()
    shares = df["target"].value_counts(normalize=True).sort_index()
    frame = pd.concat([counts, shares], axis=1).reset_index()
    frame = frame.rename(columns={"proportion": "percent"})
    frame["percent"] = (frame["percent"] * 100).round(1)
    return frame

def feature_na(
        df: pd.DataFrame
) -> int:
    return int(df.isna().values.sum())

def feature_nunique(
        df: pd.DataFrame
) -> pd.DataFrame:
    return (df.drop(columns="target").nunique(axis=0).reset_index()
            .rename(columns={"index": "feature", 0: "nunique"}))

def feature_std(
        df: pd.DataFrame
) -> pd.DataFrame:
    frame = df.drop(columns="target").describe().T[["std"]].reset_index()
    return frame.rename(columns={"index": "feature"})

def feature_mean(
        df: pd.DataFrame
) -> pd.DataFrame:
    frame = df.drop(columns="target").describe().T[["mean"]].reset_index()
    return frame.rename(columns={"index": "feature"})

def feature_strange_mask(
        df: pd.DataFrame
) -> np.ndarray:
    features = df.drop(columns="target", errors="ignore")
    std = features.std(axis=0)
    mean = features.mean(axis=0)
    std_mask = (np.isclose(std, 0.1, rtol=0.03, atol=0)
            | np.isclose(std, 1, rtol=0.03, atol=0)
            | np.isclose(std, 10, rtol=0.03, atol=0))
    mean_mask = (mean.abs() / std) < 0.03
    return std_mask & mean_mask

def feature_strange(
        df: pd.DataFrame
) -> pd.Series:
    mask = feature_strange_mask(df)
    return pd.Series(df.drop(columns="target").columns[mask])

def feature_target_spearman(
        df: pd.DataFrame,
        s: pd.Series
) -> pd.DataFrame:
    return (df.drop(columns="target", errors="ignore").corrwith(s, method="spearman")
            .reset_index().rename(columns={"index": "feature", 0: "spearman"}))

def feature_target_mi(
        df: pd.DataFrame
) -> pd.DataFrame:
    features = df.drop(columns="target")
    mi = mutual_info_classif(features, df["target"], random_state=0)
    return pd.DataFrame({"feature": features.columns, "mi": mi})

