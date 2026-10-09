import matplotlib.pyplot as plt
import pandas as pd

COLOR = "#4C72B0"

def _style(
        ax: plt.Axes,
        title: str,
        xlabel: str,
        ylabel: str
) -> None:
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def _feature_bars(
        df: pd.DataFrame,
        column: str,
        title: str,
        ylabel: str
) -> tuple[plt.Figure, plt.Axes]:
    names = df["feature"]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(df)), df[column], color=COLOR)

    step = 10
    ax.set_xticks(range(0, len(names), step))
    ax.set_xticklabels(names[::step], rotation=45, ha="right")
    ax.set_xlim(-1, len(names))

    _style(ax, title, "признак", ylabel)
    fig.tight_layout()
    return fig, ax


def target_distribution(
        df: pd.DataFrame
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(df["target"], df["count"], color=COLOR)
    ax.bar_label(bars, labels=[f"{p}%" for p in df["percent"]], padding=3)
    ax.set_xticks(df["target"])

    _style(ax, "Распределение классов", "класс", "число объектов")
    fig.tight_layout()
    return fig, ax


def feature_nunique(
        df: pd.DataFrame
) -> tuple[plt.Figure, plt.Axes]:
    return _feature_bars(df, "nunique", "Число уникальных значений признака", "уникальных значений")

def feature_medians(
        df: pd.DataFrame
) -> tuple[plt.Figure, plt.Axes]:
    return _feature_bars(df, "median", "Медиана каждого признака", "медиана")

def feature_std(
    df: pd.DataFrame
    ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _feature_bars(df, "std", "std каждого признака", "std")
    return fig, ax

def feature_std_close_up(
    df: pd.DataFrame
    ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _feature_bars(df.iloc[124:], "std", "std каждого признака", "std")
    return fig, ax

def feature_mean(
    df: pd.DataFrame
    ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _feature_bars(df, "mean", "mean каждого признака", "mean")
    return fig, ax

def feature_mean_close_up(
    df: pd.DataFrame
    ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _feature_bars(df.iloc[124:], "mean", "mean каждого признака", "mean")
    return fig, ax

def feature_target_spearman(
    df: pd.DataFrame
    ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _feature_bars(df, "spearman", "spearman каждого признака", "spearman")
    return fig, ax

def feature_target_mi(
    df: pd.DataFrame
    ) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = _feature_bars(df, "mi", "mutual information каждого признака с target", "mi")
    return fig, ax