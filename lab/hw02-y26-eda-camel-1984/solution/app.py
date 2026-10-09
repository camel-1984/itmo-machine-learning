from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import brands
import charts
import eda
import geo
import hypotheses as hyp
import utils
import pca_tsne
from data_loading import full_frame_to_app, load_records, record_to_full_frame

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
FILTER_PREFIX = "filter_"
MISSING = "〈значение отсутствует〉"
LABELS = {"region": "Территория", "category": "Тип ДТП", "severity": "Последствия"}
ALL_LABELS = {
    "region": "Все территории",
    "category": "Все типы ДТП",
    "severity": "Все последствия",
}
BASEMAP_CREDIT = "Подложка: © OpenStreetMap contributors."
NOTES = {
    "month": (
        "На полной выгрузке в ряду видны две вещи сразу: спад с 2 727 ДТП в 2015 году до 1 848 "
        "в 2021 и полка после него, и устойчивая сезонность с минимумом в феврале "
        "(119 ДТП в среднем за месяц) и максимумом в августе (226)."
    ),
    "heatmap": (
        "На полной выгрузке пик приходится на вечер пятницы, 17 часов, 277 ДТП за все годы. "
        "По дням недели разброс умеренный: суббота выше вторника на 24%."
    ),
    "categories": (
        "На полной выгрузке два типа дают две трети всех происшествий: столкновение 43.9% "
        "и наезд на пешехода 22.8%."
    ),
    "light": (
        "На полной выгрузке доля ДТП с погибшими растёт с 7.8% в светлое время до 22.0% "
        "в темноте без освещения. Проверка на вкладке «Гипотезы» показывает, что больше "
        "половины этого разрыва объясняется не освещением, а местом."
    ),
    "experience": (
        "На полной выгрузке медиана стажа 15 лет у мужчин и 8 лет у женщин, "
        "распределение сильно скошено вправо."
    ),
    "map": (
        "Точки повторяют дорожную сеть: сгустки в городах и цепочки вдоль магистралей. "
        "Цвет по числу ДТП показывает людность и интенсивность движения, а не опасность: "
        "без данных о транспортном потоке количество происшествий нельзя читать как риск."
    ),
    "pca": (
        "PC1 противопоставляет темноту без освещения, съезды с дороги и смертельные исходы "
        "наездам на пешеходов под фонарями, то есть это ось «городская улица против "
        "загородной дороги». Она забирает 34.1% различий между территориями, хотя координат "
        "в признаках нет."
    ),
    "tsne": (
        "Обе картинки показывают одно и то же: ячейки с сотнями ДТП держатся вместе и отдельно "
        "от редких, но разделённых кластеров нет ни при perplexity 30, ни при 5. Территории "
        "образуют континуум, а не устойчивые типы."
    ),
}

FINDINGS = (
    (
        "Аварийность области распадается на два режима: городская улица и загородная дорога",
        "PCA видит это различие в самом составе происшествий: первая компонента забирает 34.1% "
        "различий между территориями и противопоставляет темноту без освещения, съезды с дороги "
        "и смертельные исходы наездам на пешеходов под фонарями, хотя координат в признаках нет. "
        "Гипотеза 1 показывает, во что различие обходится при анализе: разрыв смертности по "
        "освещению падает с 12.15 п.п. до 4.95 п.п., если сравнивать ДТП внутри одной ячейки H3. "
        "Любой вывод о ночной аварийности без учёта места завышен примерно вдвое.",
    ),
    (
        "В ВАЗ погибают чаще, и обстоятельствами аварии это не объясняется",
        "В 3 668 ДТП, где ВАЗ столкнулся с машиной другой марки, доля погибших среди ехавших "
        "в ВАЗ составила 5.4% против 1.5% среди ехавших в других машинах той же аварии. "
        "Эффект +3.94 п.п. с интервалом [+3.20; +4.69], ни одна из 2 000 перестановок такого "
        "результата не дала. Сравнение внутри одного ДТП уравнивает скорость, тип столкновения, "
        "дорогу и погоду. Чем именно вызвана разница, данные не говорят: за маркой стоят возраст "
        "автомобиля, класс и оснащение.",
    ),
    (
        "Стаж связан с исходом, но влияет на него слабо",
        "Доля пострадавших водителей убывает во всех пяти группах подряд, с 44.6% при стаже "
        "до двух лет до 31.4% при стаже больше двадцати, разница крайних групп 13.16 п.п. "
        "с интервалом [+11.25; +15.08]. При этом V Крамера равен 0.092, то есть связь очень "
        "слабая, а p = 4.9e-61 говорит только о размере выборки. Исход для водителя определяется "
        "в основном обстоятельствами самого ДТП, а не тем, сколько лет он за рулём.",
    ),
)

NEXT_QUESTION = (
    "Разделить в эффекте марки два слагаемых: саму марку и возраст автомобиля. Разница между "
    "ВАЗ и другими марками устояла при одинаковых обстоятельствах аварии, но за маркой стоит "
    "и возраст парка. Одного возраста при этом недостаточно: у Mercedes медианный год выпуска "
    "2001, машины старше, чем ВАЗ с медианой 2007, а смертность среди ехавших ниже, 3.77% "
    "против 4.83%. Проверяется на тех же данных: поле year есть у каждого ТС, схема сравнения "
    "внутри одного ДТП остаётся прежней, но в пары берутся машины близкого года выпуска."
)

LIMITATIONS = (
    "По выгрузке видно количество ДТП и тяжесть последствий, но не риск: для оценки риска "
    "нужны данные о транспортном потоке и пробеге.",
    "В выгрузку попадают только ДТП, зарегистрированные ГИБДД, поэтому доли по тяжести "
    "смещены в сторону тяжёлых.",
    "Наблюдения внутри одного ДТП зависимы; там, где это критично, сравнение ведётся внутри "
    "ДТП или внутри ячейки, а где нет, ограничение указано отдельно.",
    "Разбор причин по этим данным невозможен: всё, что ниже уровня «доля такая-то отличается "
    "на столько-то», это предположение, а не результат.",
)

st.set_page_config(page_title="Атлас аварийности", page_icon="🗺️", layout="wide")


@st.cache_data(show_spinner="Читаем выгрузку…")
def read_data(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    records = load_records(Path(path))
    return full_frame_to_app(record_to_full_frame(records))


@st.cache_data(show_spinner="Разворачиваем участников…")
def read_participants(path: str, modified_ns: int, size: int) -> pd.DataFrame:
    return eda.participants_frame(read_data(path, modified_ns, size))


@st.cache_data(show_spinner="Проверяем координаты…")
def read_region_mask(path: str, modified_ns: int, size: int) -> pd.Series:
    return geo.region_mask(read_data(path, modified_ns, size))


@st.cache_data(show_spinner="Считаем гексагоны…")
def read_cells(path: str, modified_ns: int, size: int, resolution: int) -> pd.Series:
    frame = read_data(path, modified_ns, size)
    inside = read_region_mask(path, modified_ns, size)
    return geo.cell_ids(frame[inside], resolution)


@st.cache_data(show_spinner="Считаем проекции…")
def read_projections(path: str, modified_ns: int, size: int) -> dict:
    frame = read_data(path, modified_ns, size)
    profiles = pca_tsne.h3_to_pca(frame)
    coords, variance, weights = pca_tsne.pca(profiles)
    return {
        "profiles": profiles,
        "coords": coords,
        "variance": variance,
        "weights": weights,
        "tsne_main": pca_tsne.tsne(profiles, perplexity=30, random_state=0),
        "tsne_alt": pca_tsne.tsne(profiles, perplexity=5, random_state=0),
    }


@st.cache_data(show_spinner="Проверяем гипотезы…")
def read_hypotheses(path: str, modified_ns: int, size: int) -> dict:
    frame = read_data(path, modified_ns, size)

    night = hyp.night_accidents(frame)
    night_cells = hyp.cell_lighting_table(night, min_each=5)
    raw = night.groupby("lit")["fatal"].mean().reindex([False, True])

    mixed = brands.mixed_brand_occupants(frame)
    drivers = hyp.driver_outcomes(frame)

    return {
        "night_cells": night_cells,
        "night_perm": hyp.lighting_permutation(night, night_cells),
        "night_boot": hyp.lighting_bootstrap(night_cells),
        "night_naive": float(raw[False] - raw[True]) if raw.notna().all() else None,
        "night_total": len(night),
        "brand_mortality": brands.brand_mortality_ci(brands.occupants_frame(frame)),
        "brand_perm": brands.brand_permutation(mixed),
        "brand_boot": brands.brand_bootstrap(brands.accident_brand_table(mixed)),
        "brand_accidents": mixed["source_id"].nunique(),
        "experience": hyp.experience_table_ci(drivers),
        "experience_chi2": hyp.experience_chi2(drivers),
        "experience_diff": hyp.experience_difference(drivers),
    }


def reset_button() -> None:
    st.sidebar.button("Сбросить фильтры", on_click=reset_filters)


def reset_filters() -> None:
    for key in list(st.session_state):
        if str(key).startswith(FILTER_PREFIX):
            del st.session_state[key]


def count_text(value: int) -> str:
    return f"{int(value):,}".replace(",", " ")


def chart_caption(key: str, sample: str = "") -> None:
    note = NOTES.get(key, "").strip()
    prefix = f"Выборка: {sample}  \n" if sample else ""
    st.caption(f"{prefix}Наблюдение: {note}")


def p_value_text(result: dict) -> str:
    p_value = result["p_value"]
    return f"{p_value:.4f}" if p_value else f"< {1 / result['draws']:.4f}"


def apply_category_filters(frame: pd.DataFrame) -> pd.DataFrame:
    for column, title in LABELS.items():
        if st.sidebar.checkbox(ALL_LABELS[column], value=True, key=f"filter_all_{column}"):
            continue

        values = frame[column]
        options = sorted(values.dropna().astype(str).unique().tolist())
        if values.isna().any():
            options.append(MISSING)

        key = f"{FILTER_PREFIX}{column}"
        known = key in st.session_state
        if known:
            st.session_state[key] = [v for v in st.session_state[key] if v in options]

        selected = st.sidebar.multiselect(
            title, options, default=None if known else options, key=key
        )
        mask = frame[column].astype("string").isin([v for v in selected if v != MISSING])
        if MISSING in selected:
            mask |= frame[column].isna()
        frame = frame[mask]

    return frame.copy()


def main() -> None:
    st.title("Атлас аварийности", anchor=False)
    files = sorted(DATA_DIR.glob("*.geojson")) + sorted(DATA_DIR.glob("*.json"))
    files = [p for p in files if not p.name.endswith(".source.json")]
    if not files:
        st.warning("В data/raw пока нет выгрузки. Скачайте файл своего региона.")
        st.code(
            "python data/assign_region.py YOUR_GITHUB_USERNAME\n"
            "python data/import_data.py /path/to/region.geojson.zip",
            language="bash",
        )
        st.link_button("Открыть данные «Карты ДТП»", "https://dtp-stat.ru/opendata/")
        st.markdown("Инструкция находится в `data/README.md` репозитория.")
        return

    selected_path = st.sidebar.selectbox(
        "Выгрузка", files, format_func=lambda p: p.name,
        key="source_file", on_change=reset_filters,
    )
    try:
        stat = selected_path.stat()
        data_key = (str(selected_path), stat.st_mtime_ns, stat.st_size)
        frame = read_data(*data_key)
    except (ValueError, OSError, TypeError) as error:
        st.error(f"Не удалось прочитать выгрузку: {error}")
        return
    if frame.empty:
        st.warning("Файл содержит пустой список записей.")
        return

    invalid_dates = int(frame["datetime"].isna().sum())
    dated = frame[frame["datetime"].notna()]
    if dated.empty:
        st.error("Нет распознанных дат. Проверьте исходный формат и загрузчик.")
        st.dataframe(frame.head(20), hide_index=True)
        return

    first, last = dated["datetime"].min().date(), dated["datetime"].max().date()
    period = st.sidebar.date_input(
        "Период", value=(first, last), min_value=first, max_value=last,
        key="filter_period",
    )
    if len(period) != 2:
        reset_button()
        st.info("Выберите начало и конец периода.")
        return
    start, end = period
    current = dated[
        (dated["datetime"] >= pd.Timestamp(start))
        & (dated["datetime"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    ]
    current = apply_category_filters(current)
    reset_button()

    if invalid_dates:
        st.warning(f"Из фильтра по времени исключено записей без даты: {count_text(invalid_dates)}.")
    if current.empty:
        st.warning("В выбранном срезе нет записей. Измените или сбросьте фильтры.")
        return
    if len(current) < 30:
        st.warning(
            "В выборке меньше 30 записей, отдельные ДТП сильно влияют на любые доли."
        )

    overview, structure, geography, projections, tested, findings, source = st.tabs(
        ["Динамика", "Структура", "Карта", "Проекции", "Гипотезы", "Выводы", "Исходные записи"]
    )

    with overview:
        months = eda.dtp_year_month_frame(current)
        st.plotly_chart(charts.month_line(months), width="stretch")
        partial = int(months["partial"].sum())
        chart_caption(
            "month",
            f"единица ДТП, {count_text(len(current))} за {len(months)} мес. "
            f"({start:%m.%Y} – {end:%m.%Y}). Точка это сумма за календарный месяц, "
            "красная линия это скользящее среднее по 12 месяцам."
            + (f" Затенены неполные месяцы среза: {partial}." if partial else ""),
        )

        matrix = eda.dtp_week_day_hour_pivot(current)
        st.plotly_chart(charts.week_day_hour_heatmap(matrix), width="stretch")
        chart_caption(
            "heatmap",
            f"единица ДТП, {count_text(len(current))} разложены по дню недели и часу начала "
            "(168 ячеек). Цвет это число ДТП в ячейке, шкала общая для всей таблицы.",
        )

    with structure:
        raw_categories = eda.dtp_category_frame(current)
        categories = raw_categories
        if (categories["category"] == "Иной вид ДТП").any():
            categories = utils.collapse_category_below(categories)
        st.plotly_chart(charts.category_bars(categories), width="stretch")
        hidden = len(raw_categories) - len(categories) + 1
        chart_caption(
            "categories",
            f"единица ДТП, {count_text(len(current))}. Проценты считаются от всех ДТП среза."
            + (f" Серым показана сводная группа из {hidden} редких типов." if hidden > 1 else ""),
        )
        with st.expander("Этот же график до оформления"):
            st.plotly_chart(charts.category_bars_draft(raw_categories), width="stretch")
            st.caption(
                "Что изменилось: бары развёрнуты горизонтально, поэтому названия типов читаются "
                "без наклона; сортировка по частоте вместо алфавита; редкие типы собраны в одну "
                "серую группу; рядом с каждым баром доля от всех ДТП среза."
            )

        light = eda.severity_by_light_frame(current)
        if light.empty or light["share"].isna().all():
            st.info("В срезе нет записей с указанным освещением.")
        else:
            st.plotly_chart(charts.severity_by_light(light), width="stretch")
            chart_caption(
                "light",
                f"единица ДТП с указанным освещением, "
                f"{count_text(int(current['light'].notna().sum()))} из "
                f"{count_text(len(current))} среза. Показаны "
                f"{light['light'].nunique()} самых частых условия; доли считаются внутри "
                "каждой строки и в сумме дают 100%.",
            )

        participants = read_participants(*data_key)
        participants = participants[participants["source_id"].isin(current["source_row"])]
        experience = eda.years_of_driving_exp_gender(participants)
        if experience.empty:
            st.info("В срезе нет водителей с указанным стажем.")
        else:
            st.plotly_chart(
                charts.experience_ecdf(
                    experience, group="gender", medians=eda.experience_medians(experience)
                ),
                width="stretch",
            )
            chart_caption(
                "experience",
                f"единица водитель с указанными стажем и полом, "
                f"{count_text(int(experience['count'].sum()))} человек из ДТП среза. "
                "Служебные коды стажа 95, 96 и 99 исключены; по оси Y доля водителей "
                "со стажем не выше значения.",
            )

    with geography:
        in_region = read_region_mask(*data_key).loc[current.index]
        stats = geo.coverage(current, in_region)
        points = current[in_region]
        if points.empty:
            st.info("В срезе нет точек внутри границы региона.")
        else:
            notice = st.empty()
            controls, options = st.columns([1, 2])
            mode = controls.radio("Представление", ["Точки", "Гексагоны H3"], horizontal=True)
            backdrop = controls.checkbox("Показывать подложку карты", value=True)
            style = "open-street-map" if backdrop else "white-bg"
            center = {"lat": float(points["latitude"].median()),
                      "lon": float(points["longitude"].median())}
            zoom = geo.zoom_for(points)

            if mode == "Точки":
                shown = points
                if len(shown) > 5000:
                    shown = shown.sample(5000, random_state=42)
                    notice.info(
                        "Показана случайная подвыборка 5 000 точек. Показатели и выгрузка CSV "
                        "считаются по полному срезу."
                    )
                shown = shown.assign(severity_display=shown["severity"].fillna(MISSING))
                fig = charts.point_map(shown, center, style, zoom)
            else:
                resolution = options.slider(
                    "Размер ячейки H3", min_value=5, max_value=9, value=7,
                    help="Чем больше число, тем мельче гексагон.",
                )
                cells = geo.hex_counts(read_cells(*data_key, resolution).loc[points.index])
                options.caption(
                    f"Ячеек с данными: {count_text(len(cells))} · "
                    f"сторона ≈ {geo.edge_km(resolution)} км · "
                    f"медиана {cells['count'].median():.1f}. "
                )
                fig = charts.hex_map(cells, geo.hex_geojson(cells["cell"]), center, style, zoom)

            st.plotly_chart(fig, width="stretch")
            st.caption(
                f"На карте {count_text(stats['shown'])} из {count_text(stats['total'])} записей среза. "
                f"Без пригодных координат: {count_text(stats['no_coords'])}; "
                f"за границей региона: {count_text(stats['outside'])}. "
                f"Источник данных: dtp-stat.ru. {BASEMAP_CREDIT}"
            )
            chart_caption("map")

    with projections:
        data = read_projections(*data_key)
        variance = data["variance"]
        features = data["profiles"].shape[1] - 1
        st.info(
            "Обе проекции построены на полной выгрузке, фильтры на них не влияют: "
            f"{len(data['profiles'])} гексагонов H3, {features} признаков."
        )
        st.plotly_chart(
            charts.projection_scatter(
                data["coords"], "pc1", "pc2",
                title="PCA: профили территорий",
                x_label=f"PC1 ({variance[0]:.1%} дисперсии)",
                y_label=f"PC2 ({variance[1]:.1%} дисперсии)",
            ),
            width="stretch",
        )
        chart_caption(
            "pca",
            f"единица гексагон H3 (r = 6) с не менее чем 30 ДТП, {len(data['profiles'])} ячеек. "
            f"{features} признаков: доли типов ДТП, тяжести и условий освещения внутри ячейки; "
            "перед разложением признаки стандартизованы. PC1 и PC2 сохраняют "
            f"{variance[0]:.1%} и {variance[1]:.1%} дисперсии.",
        )
        with st.expander("Веса признаков в первых двух компонентах"):
            st.dataframe(data["weights"][["pc1", "pc2"]].round(3), width="stretch")

        left, right = st.columns(2)
        colour = np.log10(data["profiles"]["count"])
        for column, projection, label in (
            (left, data["tsne_main"], "perplexity = 30, seed = 0"),
            (right, data["tsne_alt"], "perplexity = 5, seed = 0"),
        ):
            column.plotly_chart(
                charts.projection_scatter(
                    projection, "tsne1", "tsne2", title=f"t-SNE · {label}",
                    color=colour, color_label="ДТП, log10",
                ),
                width="stretch",
            )

        chart_caption(
            "tsne",
            f"те же {len(data['profiles'])} профилей и {features} признаков. Слева perplexity = 30, "
            "справа = 5; оба запуска: seed 0, инициализация PCA, 1000 итераций. Размер точки это "
            "число ДТП в ячейке, цвет то же в log10. Расстояния между облаками на t-SNE не "
            "имеют масштаба: сравнивать стоит состав групп, а не промежутки между ними.",
        )

    with tested:
        results = read_hypotheses(*data_key)
        st.info("Все три проверки выполнены на полной выгрузке; фильтры на них не влияют.")

        st.subheader("1. Освещение и смертность ночных ДТП", anchor=False)
        boot, perm = results["night_boot"], results["night_perm"]
        a, b, c = st.columns(3)
        a.metric("Эффект", f"{boot['observed']:+.1%}",
                 help="Разница долей смертельных ДТП, сравнение внутри ячеек H3")
        b.metric("95% интервал", f"{boot['ci_lower']:+.1%} … {boot['ci_upper']:+.1%}")
        c.metric("p-value", p_value_text(perm))
        caption = (
            f"{count_text(results['night_total'])} ночных ДТП, "
            f"сравнение внутри {boot['cells']} гексагонов."
        )
        if results["night_naive"] is not None:
            caption += (
                f" Без контроля территории разрыв составлял {results['night_naive']:.1%}, "
                "то есть больше половины объяснялось тем, где происходят аварии."
            )
        st.caption(caption)
        st.plotly_chart(
            charts.cell_difference_strip(
                results["night_cells"], boot["observed"], (boot["ci_lower"], boot["ci_upper"])
            ),
            width="stretch",
        )

        st.subheader("2. Смертность в ВАЗ", anchor=False)
        boot, perm = results["brand_boot"], results["brand_perm"]
        a, b, c = st.columns(3)
        a.metric("Эффект", f"{boot['observed']:+.1%}",
                 help="Разница долей погибших, сравнение внутри одного ДТП")
        b.metric("95% интервал", f"{boot['ci_lower']:+.1%} … {boot['ci_upper']:+.1%}")
        c.metric("p-value", p_value_text(perm))
        st.caption(
            f"{count_text(results['brand_accidents'])} ДТП, где участвовали и ВАЗ, "
            "и машина другой марки. Обстоятельства аварии одинаковы для обеих групп."
        )
        st.plotly_chart(charts.brand_mortality_dots(results["brand_mortality"]), width="stretch")
        with st.expander("Таблица по маркам"):
            st.dataframe(
                results["brand_mortality"][["brand", "occupants", "died", "mortality",
                                            "ci_lower", "ci_upper"]],
                width="stretch", hide_index=True,
                column_config={
                    "brand": "Марка", "occupants": "Ехало", "died": "Погибло",
                    "mortality": st.column_config.NumberColumn("Смертность", format="percent"),
                    "ci_lower": st.column_config.NumberColumn("ДИ снизу", format="percent"),
                    "ci_upper": st.column_config.NumberColumn("ДИ сверху", format="percent"),
                },
            )

        st.subheader("3. Стаж водителя и вероятность пострадать", anchor=False)
        chi2, diff = results["experience_chi2"], results["experience_diff"]
        a, b, c = st.columns(3)
        a.metric("Разница крайних групп", f"{diff['difference']:+.1%}")
        b.metric("95% интервал", f"{diff['ci_lower']:+.1%} … {diff['ci_upper']:+.1%}")
        c.metric("V Крамера", f"{chi2['cramers_v']:.3f}", help="Сила связи, до 0.1 очень слабая")
        st.caption(
            f"Хи-квадрат {chi2['chi2']:.0f}, df {chi2['dof']}, p = {chi2['p_value']:.2g}; "
            f"{count_text(chi2['n'])} водителей. "
            "Доля пострадавших убывает во всех пяти группах подряд. "
            "Связь статистически значима, но по величине слабая."
        )
        st.plotly_chart(charts.experience_outcome(results["experience"]), width="stretch")
        with st.expander("Таблица по группам стажа"):
            st.dataframe(
                results["experience"], width="stretch",
                column_config={
                    "experience_group": "Стаж, лет",
                    "drivers": "Водителей", "hurt": "Пострадало",
                    "share": st.column_config.NumberColumn("Доля", format="percent"),
                    "ci_lower": st.column_config.NumberColumn("ДИ снизу", format="percent"),
                    "ci_upper": st.column_config.NumberColumn("ДИ сверху", format="percent"),
                },
            )

    with findings:
        st.info(
            "Выводы получены на полной выгрузке за 2015–2026 годы; фильтры на них не влияют. "
        )
        for number, (title, body) in enumerate(FINDINGS, 1):
            st.subheader(f"{number}. {title}", anchor=False)
            st.markdown(body)

        st.subheader("Что изучить дальше", anchor=False)
        st.markdown(NEXT_QUESTION)

        st.subheader("Ограничения", anchor=False)
        st.markdown("\n".join(f"- {item}" for item in LIMITATIONS))

    with source:
        columns = ["source_row", "id", "datetime", "region", "category", "severity",
                   "participants_count", "injured_count", "dead_count", "latitude", "longitude"]
        st.dataframe(current[columns], hide_index=True, width="stretch", height=560)
        st.download_button(
            f"Скачать CSV · {count_text(len(current))} записей",
            current[columns].to_csv(index=False).encode("utf-8-sig"),
            file_name="selected_records.csv", mime="text/csv",
        )


if __name__ == "__main__":
    main()
