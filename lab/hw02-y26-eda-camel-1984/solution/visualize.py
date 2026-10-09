import textwrap

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

WEEK_DAYS = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]

BLUE = "#3579B8"
RED = "#B33440"
GREY = "#8D939C"


def accidents_by_month_line(
    counts: pd.DataFrame,
    x: str = "year_month",
    y: str = "count",
    rolling: str = "rolling",
    title: str = "Число ДТП по месяцам",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=150)
    ax.plot(counts[x], counts[y], linewidth=1.2, color=BLUE, label="Месяц")
    ax.plot(counts[x], counts[rolling], linewidth=2.2, color=RED, label="Скользящее среднее, 12 мес.")

    ax.set_title(title)
    ax.set_ylabel("Число ДТП")
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.grid(alpha=0.2)
    ax.margins(x=0.01)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", ncols=2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return fig, ax


def accidents_by_category_bars_draft(
    counts: pd.DataFrame,
    x: str = "category",
    y: str = "count",
    title: str = "Категории ДТП",
    ylabel: str = "Число ДТП",
) -> tuple[plt.Figure, plt.Axes]:

    frame = counts.sort_values(x)
    fig, ax = plt.subplots(dpi=200)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.bar(frame[x], frame[y])

    return fig, ax

def accidents_by_category_bars(
    counts: pd.DataFrame,
    x: str = "category",
    y: str = "count",
    share: str = "share",
    aggregate: str = "aggregate",
    title: str = "ДТП по типам",
) -> tuple[plt.Figure, plt.Axes]:
    plotted = counts.iloc[::-1]
    labels = plotted[x].map(lambda label: textwrap.fill(label, 30))
    colors = [GREY if flag else BLUE for flag in plotted[aggregate]]

    fig, ax = plt.subplots(
        figsize=(11, max(3.0, 0.42 * len(plotted) + 1.4)), dpi=150, layout="constrained",
    )
    bars = ax.barh(labels, plotted[y], color=colors, height=0.75)

    offset = counts[y].max() * 0.01
    for bar, value, fraction in zip(bars, plotted[y], plotted[share]):
        ax.text(
            bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
            f"{value:,}".replace(",", " ") + f" · {fraction:.1%}",
            va="center", fontsize=8,
        )

    ax.set_xlim(right=counts[y].max() * 1.2)
    ax.set_title(title)
    ax.set_xlabel("Число ДТП")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    return fig, ax

def driving_experience_bars(
    counts: pd.DataFrame,
    x: str = "years_of_driving_experience",
    y: str = "count",
    title: str = "Стаж вождения: значения в источнике",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(11, 4), dpi=150, layout="constrained")
    ax.bar(counts[x], counts[y], color=BLUE, width=0.8)

    ax.set_title(f"{title}, N = {int(counts[y].sum()):,}".replace(",", " "))
    ax.set_xlabel("Стаж вождения, лет")
    ax.set_ylabel("Число водителей")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.margins(x=0.01)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return fig, ax


def driving_experience_ecdf(
    counts: pd.DataFrame,
    x: str = "years_of_driving_experience",
    y: str = "count",
    title: str = "Стаж вождения водителей",
    xlabel: str = "Стаж вождения, лет",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, layout="constrained")
    ax.ecdf(
        counts[x].to_numpy(dtype=float), weights=counts[y].to_numpy(dtype=float),
        color=BLUE, linewidth=1.8,
    )

    ax.set_title(f"{title}, N = {int(counts[y].sum()):,}".replace(",", " "))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Доля не выше значения")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return fig, ax

def driving_experience_gender_ecdf(
    counts: pd.DataFrame,
    x: str = "years_of_driving_experience",
    y: str = "count",
    group: str = "gender",
    colors: dict[str, str] | None = None,
    title: str = "Стаж вождения: мужчины и женщины",
    xlabel: str = "Стаж вождения, лет",
) -> tuple[plt.Figure, plt.Axes]:
    palette = colors or {"Мужской": BLUE, "Женский": RED}

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, layout="constrained")
    for name, part in counts.groupby(group, sort=False):
        size = int(part[y].sum())
        ax.ecdf(
            part[x].to_numpy(dtype=float), weights=part[y].to_numpy(dtype=float),
            color=palette.get(name, GREY), linewidth=1.8,
            label=f"{name}, N = {size:,}".replace(",", " "),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Доля не выше значения")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return fig, ax


def accidents_week_day_hour_heatmap(
    matrix: pd.DataFrame,
    cmap: str = "YlOrRd",
    title: str = "ДТП по дням недели и часам",
) -> tuple[plt.Figure, plt.Axes]:
    values = matrix.to_numpy(dtype=float)
    hours = list(matrix.columns)
    days = [WEEK_DAYS[int(day)] for day in matrix.index]

    fig, ax = plt.subplots(figsize=(13, 4.4), dpi=150, layout="constrained")
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0)

    ax.set_title(f"{title}, N = {int(np.nansum(values)):,}".replace(",", " "))
    ax.set_xlabel("Час суток")

    ticks = [position for position in range(len(hours)) if position % 2 == 0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([hours[position] for position in ticks])
    ax.set_yticks(range(len(days)))
    ax.set_yticklabels(days)

    ax.set_xticks(np.arange(-0.5, len(hours), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(days), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)

    bar = fig.colorbar(image, ax=ax, pad=0.015)
    bar.set_label("Число ДТП")
    bar.outline.set_visible(False)

    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)

    return fig, ax

def driver_gender_pie(
    counts: pd.DataFrame,
    x: str = "gender",
    y: str = "count",
    colors: dict[str, str] | None = None,
    title: str = "Пол водителей",
) -> tuple[plt.Figure, plt.Axes]:
    palette = colors or {"Мужской": BLUE, "Женский": RED, "Не определен": GREY}
    plotted = counts.sort_values(y, ascending=False)
    total = int(plotted[y].sum())

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150, layout="constrained")
    wedges, _, labels = ax.pie(
        plotted[y], colors=[palette.get(name, GREY) for name in plotted[x]],
        startangle=90, counterclock=False,
        autopct=lambda share: f"{share:.1f}%", pctdistance=0.72,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        textprops={"fontsize": 10},
    )
    shares = plotted[y] / total
    for wedge, label, share in zip(wedges, labels, shares):
        label.set_fontweight("bold")
        if share >= 0.05:
            label.set_color("white")
            continue
        angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        label.set_position((1.15 * np.cos(angle), 1.15 * np.sin(angle)))
        label.set_color("#3B3F45")
        label.set_fontsize(9)

    ax.legend(
        wedges,
        [f"{name} — {value:,}".replace(",", " ") for name, value in zip(plotted[x], plotted[y])],
        frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5),
    )
    ax.set_title(f"{title}, N = {total:,}".replace(",", " "))
    ax.set_aspect("equal")

    return fig, ax


def pca_scatter(
    frame: pd.DataFrame,
    variance: np.ndarray,
    x: str = "pc1",
    y: str = "pc2",
    size: str = "count",
    title: str = "Профили территорий в координатах главных компонент",
) -> tuple[plt.Figure, plt.Axes]:
    largest = frame[size].max()
    sizes = 20 + 180 * frame[size] / largest

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150, layout="constrained")
    ax.axhline(0, color=GREY, linewidth=0.8, alpha=0.5)
    ax.axvline(0, color=GREY, linewidth=0.8, alpha=0.5)
    points = ax.scatter(
        frame[x], frame[y], s=sizes, color=BLUE, alpha=0.6,
        edgecolor="white", linewidth=0.5,
    )

    ax.set_title(f"{title}, ячеек: {len(frame)}")
    ax.set_xlabel(f"PC1 ({variance[0]:.1%} дисперсии)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1%} дисперсии)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    handles, labels = points.legend_elements(
        prop="sizes", num=3, alpha=0.6,
        func=lambda area: (area - 20) * largest / 180,
    )

    return fig, ax


def tsne_compare(
    frames: list[tuple[pd.DataFrame, str]],
    color: pd.Series | None = None,
    color_label: str = "",
    size: str = "count",
    title: str = "t-SNE по профилям территорий",
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        1, len(frames), figsize=(6.2 * len(frames), 6), dpi=150, layout="constrained",
    )
    axes = np.atleast_1d(axes)

    drawn = None
    for ax, (frame, label) in zip(axes, frames):
        largest = frame[size].max()
        sizes = 15 + 140 * frame[size] / largest
        values = None if color is None else color.reindex(frame.index)
        drawn = ax.scatter(
            frame["tsne1"], frame["tsne2"], s=sizes,
            c=BLUE if values is None else values,
            cmap=None if values is None else "YlOrRd",
            alpha=0.75, edgecolor="white", linewidth=0.5,
        )
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)

    if color is not None:
        bar = fig.colorbar(drawn, ax=axes.tolist(), pad=0.015)
        bar.set_label(color_label)
        bar.outline.set_visible(False)

    fig.suptitle(f"{title}, ячеек: {len(frames[0][0])}")

    return fig, axes


def cell_difference_strip(
    cells: pd.DataFrame,
    observed: float,
    ci: tuple[float, float],
    column: str = "difference",
    title: str = "Разница смертности ночных ДТП внутри ячеек",
) -> tuple[plt.Figure, plt.Axes]:
    values = cells[column].sort_values().to_numpy()
    positions = np.arange(len(values))
    colors = [RED if value > 0 else BLUE for value in values]

    fig, ax = plt.subplots(figsize=(11, 5), dpi=150, layout="constrained")
    ax.axvspan(ci[0], ci[1], color=GREY, alpha=0.18, zorder=0)
    ax.axvline(0, color="#3B3F45", linewidth=1.0, zorder=1)
    ax.axvline(observed, color="#3B3F45", linewidth=1.6, linestyle="--", zorder=2)
    ax.hlines(positions, 0, values, colors=colors, linewidth=1.6, zorder=3)

    above = int((values > 0).sum())
    ax.set_title(
        f"{title}\nбез освещения смертельнее в {above} ячейках из {len(values)}"
    )
    ax.set_xlabel("Доля смертельных без освещения минус с освещением, п.п.")
    ax.set_ylabel("Ячейки, отсортированы по разнице")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_yticks([])
    ax.margins(y=0.01)
    ax.annotate(
        f"среднее {observed:+.1%}\n95% ДИ [{ci[0]:+.1%}; {ci[1]:+.1%}]",
        xy=(observed, len(values) * 0.06), xytext=(8, 0), textcoords="offset points",
        fontsize=9, va="bottom",
    )
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    return fig, ax


def permutation_null(
    null: np.ndarray,
    observed: float,
    p_value: float,
    xlabel: str = "Средняя по ячейкам разница долей, п.п.",
    title: str = "Перестановочный тест: распределение при H0",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150, layout="constrained")
    ax.hist(null, bins=40, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.axvline(observed, color=RED, linewidth=2.0)

    ax.set_title(f"{title}, перестановок: {len(null):,}".replace(",", " "))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Число перестановок")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    limit = 1 / len(null)
    p_text = f"p < {limit:.4f}" if p_value == 0 else f"p = {p_value:.4f}"
    ax.annotate(
        f"наблюдение {observed:+.1%}\n{p_text}",
        xy=(observed, ax.get_ylim()[1] * 0.85), xytext=(10, 0), textcoords="offset points",
        color=RED, fontsize=10, va="top",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return fig, ax


def effect_comparison(
    estimates: list[tuple[str, float, tuple[float, float]]],
    title: str = "Эффект освещения: до и после контроля территории",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=150, layout="constrained")

    for i, (label, value, ci) in enumerate(reversed(estimates)):
        ax.plot([ci[0], ci[1]], [i, i], color=GREY, linewidth=2.5, solid_capstyle="round")
        ax.plot(value, i, "o", color=BLUE, markersize=9, zorder=3)
        ax.annotate(
            f"{value:+.1%}", xy=(value, i), xytext=(0, 12), textcoords="offset points",
            ha="center", fontsize=9,
        )

    ax.axvline(0, color="#3B3F45", linewidth=1.0)
    ax.set_yticks(range(len(estimates)))
    ax.set_yticklabels([label for label, _, _ in reversed(estimates)])
    ax.set_title(title)
    ax.set_xlabel("Разница долей смертельных ДТП, п.п.")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(-0.6, len(estimates) - 0.2)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    return fig, ax


def brand_mortality_bars(
    frame: pd.DataFrame,
    highlight: str = "ВАЗ",
    title: str = "Смертность среди ехавших в ТС, по маркам",
) -> tuple[plt.Figure, plt.Axes]:
    plotted = frame.sort_values("mortality")
    positions = np.arange(len(plotted))
    colors = [RED if brand == highlight else BLUE for brand in plotted["brand"]]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150, layout="constrained")
    ax.hlines(
        positions, plotted["ci_lower"], plotted["ci_upper"],
        colors=GREY, linewidth=2.5, zorder=1,
    )
    ax.scatter(plotted["mortality"], positions, color=colors, s=70, zorder=2)

    for y, (_, row) in zip(positions, plotted.iterrows()):
        ax.annotate(
            f"{row['died']:.0f} из {row['occupants']:.0f}",
            xy=(row["ci_upper"], y), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=8, color="#5B6068",
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(plotted["brand"])
    ax.set_title(f"{title}\nточка — доля погибших, отрезок — 95% интервал (бутстреп)")
    ax.set_xlabel("Доля погибших среди ехавших")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(left=0)
    ax.margins(x=0.12, y=0.04)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    return fig, ax


def experience_outcome_line(
    table: pd.DataFrame,
    share: str = "share",
    size: str = "drivers",
    title: str = "Доля пострадавших водителей по группам стажа",
) -> tuple[plt.Figure, plt.Axes]:
    positions = np.arange(len(table))
    values = table[share].to_numpy()
    lower = values - table["ci_lower"].to_numpy()
    upper = table["ci_upper"].to_numpy() - values

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150, layout="constrained")
    ax.plot(positions, values, color=BLUE, linewidth=1.4, alpha=0.5, zorder=1)
    ax.errorbar(
        positions, values, yerr=[lower, upper], fmt="o", color=BLUE,
        markersize=9, capsize=5, capthick=1.5, elinewidth=1.5, zorder=2,
    )

    for x, value, top, count in zip(positions, values, table["ci_upper"], table[size]):
        ax.annotate(
            f"{value:.1%}", xy=(x, top), xytext=(0, 9), textcoords="offset points",
            ha="center", fontsize=9,
        )
        ax.annotate(
            f"n = {count:,}".replace(",", " "), xy=(x, 0), xytext=(0, 8),
            textcoords="offset points", xycoords=("data", "axes fraction"),
            ha="center", fontsize=8, color="#5B6068",
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(table.index)
    ax.set_title(f"{title}\nотрезки — 95% интервалы (бутстреп)")
    ax.set_xlabel("Стаж вождения, лет")
    ax.set_ylabel("Доля пострадавших")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0.25, 0.50)
    ax.margins(x=0.1)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    return fig, ax
