import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE = "#3579B8"
RED = "#B33440"
GREY = "#8D939C"
AMBER = "#C18A16"

SEVERITY_COLORS = {"Легкий": BLUE, "Тяжёлый": AMBER, "С погибшими": RED}
WEEK_DAYS = ["Понедельник", "Вторник", "Среда", "Четверг", "Пятница", "Суббота", "Воскресенье"]

LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=10, r=10, t=50, b=10),
    hoverlabel=dict(font_size=12),
    separators=", ",
)


def _style(fig: go.Figure, title: str, height: int = 480) -> go.Figure:
    fig.update_layout(title=title, height=height, **LAYOUT)
    axis_line = dict(showline=True, linecolor="rgba(90,96,104,0.55)", linewidth=1, ticks="")
    fig.update_xaxes(showgrid=False, **axis_line)
    fig.update_yaxes(gridcolor="rgba(140,146,156,0.25)", zeroline=False, **axis_line)
    return fig


def month_line(counts: pd.DataFrame) -> go.Figure:
    fig = px.line(
        counts, x="year_month", y="count",
        labels={"year_month": "Месяц", "count": "Число ДТП"},
    )
    fig.update_traces(line_color=BLUE, line_width=1.4, name="Месяц", showlegend=True)
    if "rolling" in counts and counts["rolling"].notna().any():
        fig.add_scatter(
            x=counts["year_month"], y=counts["rolling"], mode="lines",
            line=dict(color=RED, width=2.6), name="Скользящее среднее, 12 мес.",
        )
    top = counts["count"].max()
    fig.update_yaxes(range=[0, top * 1.06], rangemode="tozero")
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right", title=None))

    if "partial" in counts:
        partial = counts.loc[counts["partial"], "year_month"]
        for number, month in enumerate(partial):
            label = dict(
                annotation_text="неполный месяц",
                annotation_position="top left",
                annotation_font_size=11,
                annotation_font_color="#5B6068",
            ) if number == len(partial) - 1 else {}
            fig.add_vrect(
                x0=month - pd.Timedelta(days=15), x1=month + pd.Timedelta(days=15),
                fillcolor=GREY, opacity=0.22, line_width=0, layer="below", **label,
            )

    return _style(fig, "Число ДТП по месяцам")


def week_day_hour_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        matrix.to_numpy(),
        x=[str(hour) for hour in matrix.columns],
        y=[WEEK_DAYS[int(day)] for day in matrix.index],
        color_continuous_scale="YlOrRd", zmin=0, aspect="auto",
        labels=dict(x="Час суток", y="", color="Число ДТП"),
    )
    fig.update_traces(hovertemplate="%{y}, %{x}:00<br>ДТП: %{z}<extra></extra>")
    fig.update_xaxes(dtick=2)
    return _style(fig, "ДТП по дням недели и часам", height=460)


def category_bars(counts: pd.DataFrame) -> go.Figure:
    plotted = counts.iloc[::-1]
    fig = px.bar(
        plotted, x="count", y="category", orientation="h",
        labels={"count": "Число ДТП", "category": ""},
        text=plotted["share"].map(lambda value: f"{value:.1%}"),
    )
    fig.update_traces(
        marker_color=[GREY if flag else BLUE for flag in plotted["aggregate"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>ДТП: %{x}<extra></extra>",
    )
    fig.update_xaxes(range=[0, counts["count"].max() * 1.22])
    return _style(fig, "ДТП по типам", height=max(420, 38 * len(plotted) + 110))


def severity_by_light(frame: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        frame, x="share", y="light", color="severity", orientation="h",
        color_discrete_map=SEVERITY_COLORS,
        category_orders={"severity": list(SEVERITY_COLORS)},
        labels={"share": "Доля ДТП", "light": "", "severity": "Последствия"},
    )
    fig.update_traces(
        hovertemplate="%{y}<br>%{fullData.name}: %{x:.1%}<extra></extra>",
        texttemplate="%{x:.0%}", textposition="inside",
        insidetextfont=dict(color="white", size=12),
    )
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right", title=None),
        uniformtext=dict(minsize=10, mode="hide"),
    )
    return _style(fig, "Последствия при разном освещении", height=440)


def experience_ecdf(
    counts: pd.DataFrame,
    group: str | None = None,
    medians: pd.Series | None = None,
) -> go.Figure:
    frame = counts.loc[counts.index.repeat(counts["count"])]
    fig = px.ecdf(
        frame, x="years_of_driving_experience", color=group,
        color_discrete_map={"Мужской": BLUE, "Женский": RED},
        labels={"years_of_driving_experience": "Стаж вождения, лет", "gender": "Пол"},
    )
    if medians is not None:
        fig.add_hline(y=0.5, line_width=1, line_dash="dot", line_color=GREY)
        heights = (0.60, 0.40)
        for number, (name, value) in enumerate(medians.items()):
            fig.add_vline(
                x=value, line_width=1.2, line_dash="dot",
                line_color={"Мужской": BLUE, "Женский": RED}.get(name, GREY),
                annotation_text=f"{name.lower()}: медиана {value:.0f} лет",
                annotation_position="top right",
                annotation_y=heights[number % len(heights)],
                annotation_yanchor="middle",
                annotation_font_size=11,
                annotation_bgcolor="rgba(255,255,255,0.88)",
            )
    fig.update_yaxes(tickformat=".0%", title="Доля не выше значения")
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=1, xanchor="right", title=None))
    return _style(fig, "Стаж вождения водителей", height=480)


def projection_scatter(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: pd.Series | None = None,
    color_label: str = "",
    x_label: str = "",
    y_label: str = "",
) -> go.Figure:
    plotted = frame.copy()
    if color is not None:
        plotted[color_label] = color.reindex(plotted.index)

    fig = px.scatter(
        plotted, x=x, y=y, size="count", size_max=26,
        color=color_label if color is not None else None,
        color_continuous_scale="YlOrRd",
        hover_name=plotted.index,
        labels={x: x_label or x, y: y_label or y, "count": "ДТП в ячейке"},
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="white"), opacity=0.8))
    if color is None:
        fig.update_traces(marker_color=BLUE)

    fig.add_hline(y=0, line_width=1, line_color="rgba(90,96,104,0.45)")
    fig.add_vline(x=0, line_width=1, line_color="rgba(90,96,104,0.45)")

    return _style(fig, title, height=620)


def hex_map(cells: pd.DataFrame, geojson: dict, center: dict, style: str, zoom: float = 6) -> go.Figure:
    cap = float(cells["count"].quantile(0.99))
    fig = px.choropleth_map(
        cells, geojson=geojson, locations="cell", color="count",
        color_continuous_scale="YlOrRd",
        range_color=(0, cap),
        hover_data={"cell": False, "count": True},
        labels={"count": "Число ДТП"},
        opacity=0.65, zoom=zoom, center=center, map_style=style,
    )
    fig.update_layout(
        height=540,
        coloraxis_colorbar=dict(
            title=dict(text=f"Число ДТП<br><sup>шкала обрезана на {cap:.0f}</sup>"),
        ),
        **LAYOUT,
    )
    return fig


def point_map(points: pd.DataFrame, center: dict, style: str, zoom: float = 6) -> go.Figure:
    fig = px.scatter_map(
        points, lat="latitude", lon="longitude", color="severity_display",
        color_discrete_map={**SEVERITY_COLORS, "〈значение отсутствует〉": GREY},
        category_orders={"severity_display": [*SEVERITY_COLORS, "〈значение отсутствует〉"]},
        hover_data=["source_row", "id", "datetime", "category", "region"],
        labels={"severity_display": "Последствия"},
        opacity=0.6, zoom=zoom, center=center, map_style=style,
    )
    fig.update_layout(height=540, **LAYOUT)
    return fig


def category_bars_draft(counts: pd.DataFrame) -> go.Figure:
    plotted = counts.sort_values("category")
    fig = px.bar(
        plotted, x="category", y="count",
        title="Категории ДТП",
        labels={"category": "", "count": "Число ДТП"},
    )
    fig.update_layout(height=700)
    return fig


def cell_difference_strip(
    cells: pd.DataFrame,
    observed: float,
    ci: tuple[float, float],
    title: str = "Разница смертности ночных ДТП внутри ячеек",
) -> go.Figure:
    plotted = cells.sort_values("difference")
    values = plotted["difference"].to_numpy()
    above = int((values > 0).sum())

    fig = go.Figure()
    fig.add_vrect(x0=ci[0], x1=ci[1], fillcolor=GREY, opacity=0.18, line_width=0, layer="below")
    fig.add_bar(
        x=values, y=list(range(len(values))), orientation="h",
        marker_color=[RED if value > 0 else BLUE for value in values],
        customdata=np.column_stack([
            plotted.index,
            plotted["share_unlit"], plotted["accidents_unlit"],
            plotted["share_lit"], plotted["accidents_lit"],
        ]),
        hovertemplate=(
            "%{customdata[0]}<br>"
            "без освещения: %{customdata[1]:.1%} из %{customdata[2]} ДТП<br>"
            "с освещением: %{customdata[3]:.1%} из %{customdata[4]} ДТП<br>"
            "разница: %{x:+.1%}<extra></extra>"
        ),
        showlegend=False,
    )
    fig.add_vline(x=0, line_width=1, line_color="#3B3F45")
    fig.add_vline(
        x=observed, line_width=1.6, line_dash="dash", line_color="#3B3F45",
        annotation_text=(
            f"среднее {observed:+.1%}<br>95% ДИ [{ci[0]:+.1%}; {ci[1]:+.1%}]<br>"
            "серая полоса — интервал"
        ),
        annotation_position="bottom right",
        annotation_font_size=11,
        annotation_bgcolor="rgba(255,255,255,0.88)",
        annotation_align="left",
    )

    fig.update_xaxes(tickformat=".0%", title="Доля смертельных без освещения минус с освещением")
    fig.update_yaxes(showticklabels=False, showgrid=False,
                     title="Ячейки, отсортированы по разнице")
    fig.update_layout(bargap=0.15)

    subtitle = f"без освещения смертельнее в {above} ячейках из {len(values)}"
    styled = _style(fig, f"{title}<br><sup>{subtitle}</sup>", height=520)
    return styled.update_layout(margin=dict(l=10, r=10, t=80, b=10))


def brand_mortality_dots(
    frame: pd.DataFrame,
    highlight: str = "ВАЗ",
    title: str = "Смертность среди ехавших в ТС, по маркам",
) -> go.Figure:
    plotted = frame.sort_values("mortality")

    fig = go.Figure()
    fig.add_scatter(
        x=plotted["mortality"], y=plotted["brand"], mode="markers",
        marker=dict(
            size=11,
            color=[RED if brand == highlight else BLUE for brand in plotted["brand"]],
        ),
        error_x=dict(
            type="data", symmetric=False,
            array=plotted["ci_upper"] - plotted["mortality"],
            arrayminus=plotted["mortality"] - plotted["ci_lower"],
            color=GREY, thickness=2.5, width=0,
        ),
        customdata=np.column_stack([plotted["died"], plotted["occupants"]]),
        hovertemplate=(
            "%{y}<br>смертность: %{x:.2%}<br>"
            "погибло %{customdata[0]} из %{customdata[1]}<extra></extra>"
        ),
        showlegend=False,
    )
    fig.add_scatter(
        x=plotted["ci_upper"], y=plotted["brand"], mode="text",
        text=[f"{died:,.0f} из {total:,.0f}".replace(",", " ")
              for died, total in zip(plotted["died"], plotted["occupants"])],
        textposition="middle right", textfont=dict(size=10, color="#5B6068"),
        hoverinfo="skip", showlegend=False, cliponaxis=False,
    )

    fig.update_xaxes(tickformat=".0%", title="Доля погибших среди ехавших",
                     range=[0, float(plotted["ci_upper"].max()) * 1.35])
    fig.update_yaxes(title="")

    subtitle = "точка — доля погибших, отрезок — 95% интервал (бутстреп)"
    styled = _style(fig, f"{title}<br><sup>{subtitle}</sup>", height=520)
    return styled.update_layout(margin=dict(l=10, r=10, t=80, b=10))


def experience_outcome(
    table: pd.DataFrame,
    title: str = "Доля пострадавших водителей по группам стажа",
) -> go.Figure:
    values = table["share"].to_numpy()
    labels = [f"{group}<br>n = {count:,}".replace(",", " ")
              for group, count in zip(table.index, table["drivers"])]

    fig = go.Figure()
    fig.add_scatter(
        x=labels, y=values, mode="markers",
        marker=dict(size=11, color=BLUE),
        error_y=dict(
            type="data", symmetric=False,
            array=table["ci_upper"].to_numpy() - values,
            arrayminus=values - table["ci_lower"].to_numpy(),
            color=BLUE, thickness=1.5, width=6,
        ),
        hovertemplate="стаж %{x}<br>доля пострадавших: %{y:.1%}<extra></extra>",
        showlegend=False, cliponaxis=False,
    )

    span = float(table["ci_upper"].max() - table["ci_lower"].min())
    fig.add_scatter(
        x=labels, y=table["ci_upper"].to_numpy() + span * 0.08, mode="text",
        text=[f"{value:.1%}" for value in values],
        textposition="top center", textfont=dict(size=12),
        hoverinfo="skip", showlegend=False, cliponaxis=False,
    )
    fig.update_yaxes(
        tickformat=".0%", title="Доля пострадавших",
        range=[float(table["ci_lower"].min()) - span * 0.25,
               float(table["ci_upper"].max()) + span * 0.45],
    )
    fig.update_xaxes(title="Стаж вождения, лет")

    subtitle = "точка — доля пострадавших, отрезок — 95% ДИ доли (бутстреп, 2000 ресэмплов)"
    styled = _style(fig, f"{title}<br><sup>{subtitle}</sup>", height=520)
    return styled.update_layout(margin=dict(l=10, r=10, t=80, b=10))
