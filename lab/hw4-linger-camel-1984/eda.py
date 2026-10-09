import numpy as np
import pandas as pd

def df_na(
        df: pd.DataFrame
) -> int:
    return int(df.isna().values.sum())

def target_zero(
        df: pd.DataFrame
) -> int:
    return int((df["median_runtime_us"] == 0).sum())

def target_negative(
        df: pd.DataFrame
) -> int:
    return int((df["median_runtime_us"] < 0).sum())


def residuals(
        df: pd.DataFrame
) -> pd.Series:
    resid = pd.Series(index=df.index, dtype=float)
    for algorithm, group in df.groupby("algorithm"):
        if algorithm == "counting_sort":
            x = np.log(group["n"] + group["value_range"])
        else:
            x = np.log(group["n"])
        y = np.log(group["median_runtime_us"])
        k, b = np.polyfit(x, y, 1)
        resid[group.index] = y - (k * x + b)
    return resid

def residual_corr(
        df: pd.DataFrame
) -> pd.DataFrame:
    frame = df.copy()
    frame["resid"] = residuals(df)
    features = df.columns.drop(["algorithm", "n", "interquartile_range", "median_runtime_us"])
    rows = {}
    for algorithm, group in frame.groupby("algorithm"):
        rows[algorithm] = group[features].corrwith(group["resid"], method="spearman")
    return pd.DataFrame(rows).T
