from pathlib import Path

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

OUT = Path(__file__).resolve().parent.parent / "figures"
COLORS = {"Lunch": "#2a6fdb", "Dinner": "#e8833a"}


def load_data():
    return px.data.tips()


def plot_matplotlib(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["total_bill"], df["tip"], c=df["time"].map(COLORS), alpha=0.7)
    ax.set_xlabel("Сумма счёта, $")
    ax.set_ylabel("Чаевые, $")
    ax.set_title("Чаевые vs сумма счёта")
    return fig


def plot_seaborn(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="time", palette=COLORS, alpha=0.7, ax=ax)
    ax.set_xlabel("Сумма счёта, $")
    ax.set_ylabel("Чаевые, $")
    ax.set_title("Чаевые vs сумма счёта")
    return fig


def plot_plotly(df):
    return px.scatter(
        df, x="total_bill", y="tip", color="time", color_discrete_map=COLORS,
        hover_data=["day", "size"], title="Чаевые vs сумма счёта", template="plotly_white",
    )


def main():
    OUT.mkdir(exist_ok=True)
    df = load_data()
    plot_matplotlib(df).savefig(OUT / "matplotlib.png", dpi=150)
    plot_matplotlib(df).savefig(OUT / "matplotlib.pdf")
    plot_seaborn(df).savefig(OUT / "seaborn.png", dpi=150)
    plot_plotly(df).write_html(OUT / "plotly.html")
    print(f"saved to {OUT}")


if __name__ == "__main__":
    main()
