import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

def target_hist(
        s: pd.Series,
        ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(x=s, bins=50, ax=ax)
    return ax

def target_hist_log(
        s: pd.Series,
        ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    sns.histplot(x=s, bins=50, log_scale=True, ax=ax)
    return ax

def target_n_algo_scatter_log(
        df: pd.DataFrame,
        ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    sns.scatterplot(data=df, x="n", y="median_runtime_us", hue="algorithm", ax=ax)
    ax.set(xscale="log", yscale="log")
    return ax

def target_counting_sort_scatter_log(
        df: pd.DataFrame,
        ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots()
    cs = df[df["algorithm"] == "counting_sort"]
    sns.scatterplot(data=cs, x="value_range", y="median_runtime_us", hue="n", ax=ax)
    ax.set(xscale="log", yscale="log")
    return ax

def residual_corr_heatmap(
        corr: pd.DataFrame,
        ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(corr, annot=True, center=0, vmin=-1, vmax=1, ax=ax)
    return ax
