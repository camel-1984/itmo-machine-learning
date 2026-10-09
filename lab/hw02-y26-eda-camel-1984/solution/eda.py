import h3
import pandas as pd
import utils
import geo
import data_loading

def eda_base(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for feature in df.columns:
        s = df[feature]
        rows.append({
            "feature": feature,
            "dtype": utils.get_dtype(s),
            "unique": utils.get_unique_number(s),
            "dupl": utils.get_duplicated_number(s),
            "filled": utils.get_filled_number(s),
            "missing": utils.get_missing_number(s),
        })
    return pd.DataFrame(rows)

def dtp_year_month_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = utils.get_year_month(df)
    frame = frame["year_month"].value_counts().sort_index().reset_index()
    frame["year_month"] = frame["year_month"].dt.to_timestamp()
    frame["rolling"] = frame["count"].rolling(12, center=True).mean()

    stamps = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    frame["partial"] = False
    if not stamps.empty and not frame.empty:
        first, last = stamps.min(), stamps.max()
        frame.loc[frame.index[0], "partial"] = first.day > 1
        frame.loc[frame.index[-1], "partial"] = last.day < last.days_in_month

    return frame


def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    order = values.sort_values().index
    cumulative = weights.loc[order].cumsum()
    total = cumulative.iloc[-1]
    return float(values.loc[cumulative[cumulative >= total / 2].index[0]])


def experience_medians(counts: pd.DataFrame, group: str = "gender") -> pd.Series:
    medians = {
        name: weighted_median(part["years_of_driving_experience"], part["count"])
        for name, part in counts.groupby(group)
    }
    return pd.Series(medians, name="median")

MONTH_NAMES = ("янв", "фев", "мар", "апр", "май", "июн",
               "июл", "авг", "сен", "окт", "ноя", "дек")


def full_years(df: pd.DataFrame, column: str = "datetime") -> pd.Series:
    stamps = pd.to_datetime(df[column], errors="coerce").dropna()
    complete = [
        year for year, part in stamps.groupby(stamps.dt.year)
        if part.dt.month.nunique() == 12
    ]
    return stamps[stamps.dt.year.isin(complete)]


def dtp_month_profile(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    stamps = full_years(df, column)
    years = stamps.dt.year.nunique()

    frame = (
        stamps.dt.month.value_counts()
        .sort_index()
        .rename("count")
        .rename_axis("month")
        .reset_index()
    )
    frame["name"] = [MONTH_NAMES[month - 1] for month in frame["month"]]
    frame["mean"] = frame["count"] / years
    frame["share"] = frame["count"] / frame["count"].sum()

    return frame


def dtp_month_extremes(df: pd.DataFrame, column: str = "datetime") -> pd.DataFrame:
    stamps = full_years(df, column)
    table = pd.crosstab(stamps.dt.year, stamps.dt.month)

    frame = pd.DataFrame({
        "peak": [MONTH_NAMES[month - 1] for month in table.idxmax(axis="columns")],
        "trough": [MONTH_NAMES[month - 1] for month in table.idxmin(axis="columns")],
    }, index=table.index)
    frame["count"] = table.sum(axis="columns")

    return frame


def season_ratio(
    profile: pd.DataFrame,
    high: tuple = (6, 7, 8),
    low: tuple = (12, 1, 2),
) -> dict:
    high_mean = profile.loc[profile["month"].isin(high), "mean"].mean()
    low_mean = profile.loc[profile["month"].isin(low), "mean"].mean()

    return {
        "high": float(high_mean),
        "low": float(low_mean),
        "ratio": float(high_mean / low_mean),
    }


def dtp_week_day_hour_pivot(df: pd.DataFrame) -> pd.DataFrame:
    frame = utils.get_week_day_hour(df)
    frame = (
        frame.groupby(["week_day", "hour"])
        .size()
        .rename("count")
        .reset_index()
    )
    matrix = frame.pivot(index="week_day", columns="hour", values="count")

    return matrix.reindex(index=range(7), columns=range(24)).fillna(0)

def dtp_category_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = (
        df.groupby("category", dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False, ignore_index=True)
    )
    frame["category"] = frame["category"].fillna("не указано")
    frame["share"] = frame["count"] / frame["count"].sum()
    frame["aggregate"] = frame["category"] == "не указано"
    return frame

def years_of_driving_exp_raw(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.dropna(subset="years_of_driving_experience")
    frame = (
        frame.groupby("years_of_driving_experience")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("years_of_driving_experience", ignore_index=True)
    )
    return frame

def driver_gender(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset="years_of_driving_experience")
    df = (
      df.groupby("gender")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ignore_index=True)
    )
    return df

def years_of_driving_exp(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.dropna(subset="years_of_driving_experience")
    frame = (
        frame.groupby("years_of_driving_experience")
        .size()
        .rename("count")
        .reset_index()
        .sort_values("years_of_driving_experience", ignore_index=True)
    )
    mask = frame["years_of_driving_experience"].astype(str).isin(["95", "96", "99"])

    return frame[~mask]

def years_of_driving_exp_gender(df: pd.DataFrame) -> pd.DataFrame:

    frame = df.dropna(subset=["years_of_driving_experience", "gender"])
    mask = frame["gender"].isin(["Мужской", "Женский"]) & ~(frame["years_of_driving_experience"].
                                                            astype(str).isin(["95", "96", "99"]))
    frame = frame[mask]

    frame = (
        frame.groupby(["years_of_driving_experience", "gender"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("years_of_driving_experience", ignore_index=True)
    )

    return frame

def vehicle_frame(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for sid, vehicles in zip(df["source_id"], df["vehicles"]):
        if not isinstance(vehicles, list):
            continue
        for idx, vehicle in enumerate(vehicles):
            row = {key: value for key, value in vehicle.items() if key != "participants"}
            row["source_id"] = sid
            row["vehicle_idx"] = idx
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    frame["vehicle_idx"] = frame["vehicle_idx"].astype("Int64")

    return frame

def participants_frame(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for sid, vehicles, outside in zip(df["source_id"], df["vehicles"], df["participants"]):
        if isinstance(vehicles, list):
            for vehicle_idx, vehicle in enumerate(vehicles):
                participants = vehicle.get("participants")
                if not isinstance(participants, list):
                    continue
                for idx, participant in enumerate(participants):
                    row = dict(participant)
                    row["source_id"] = sid
                    row["vehicle_idx"] = vehicle_idx
                    row["participant_idx"] = idx
                    row["in_vehicle"] = True
                    rows.append(row)

        if isinstance(outside, list):
            for idx, participant in enumerate(outside):
                row = dict(participant)
                row["source_id"] = sid
                row["vehicle_idx"] = pd.NA
                row["participant_idx"] = idx
                row["in_vehicle"] = False
                rows.append(row)

    frame = pd.DataFrame(rows)
    for column in ("vehicle_idx", "participant_idx", "years_of_driving_experience"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")

    return frame

def severity_by_light_frame(df: pd.DataFrame, top: int = 4) -> pd.DataFrame:
    shares = pd.crosstab(df["light"], df["severity"], normalize="index")
    keep = df["light"].value_counts().head(top).index
    shares = shares.loc[[value for value in keep if value in shares.index]]

    frame = shares.reset_index().melt(id_vars="light", var_name="severity", value_name="share")
    frame["light"] = frame["light"].str.replace("В темное время суток, ", "Темнота, ", regex=False)
    return frame
