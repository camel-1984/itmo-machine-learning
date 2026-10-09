import pandas as pd

def get_duplicated_number(s: pd.Series) -> int:
    na = pd.isna(s)
    return int((s.astype(str).duplicated(keep=False) & ~na).sum())

def get_unique_number(s: pd.Series) -> int:
    return int(s.astype(str).nunique(dropna=False))

def get_dtype(s: pd.Series) -> str:
    return str(s.dtype)

def get_filled_number(s: pd.Series) -> int:
    return int(s.count())

def get_missing_number(s: pd.Series) -> int:
    return int(s.isna().sum())

def get_year_month(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    datetime = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["year_month"] = datetime.dt.to_period("M")
    return frame

def get_week_day_hour(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    datetime = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["week_day"] = datetime.dt.dayofweek
    frame["hour"] = datetime.dt.hour
    return frame

def unpack_point(df: pd.DataFrame) -> pd.DataFrame:
    df["point_longitude"] = df["point"].str.get("long")
    df["point_latitude"] = df["point"].str.get("lat")
    return df

def same_spot_same_time(df: pd.DataFrame) -> pd.DataFrame:
    dupl = df.astype(str).duplicated(subset=["geom_longitude", "geom_latitude", "datetime"], keep=False)
    return df[dupl]

def collapse_category_below(counts: pd.DataFrame) -> pd.DataFrame:
    threshold = counts.loc[counts["category"] == "Иной вид ДТП", "count"].iloc[0]
    rare = counts["count"] < threshold

    rest = pd.DataFrame({
        "category": [f"Прочее ({rare.sum()} кат.)"],
        "count": [counts.loc[rare, "count"].sum()],
        "aggregate": [True],
    })
    frame = pd.concat([counts.loc[~rare], rest], ignore_index=True)
    frame["share"] = frame["count"] / frame["count"].sum()

    return frame

def missing_coords(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reason = {}
    missing_geom = df["geom_longitude"].isna() | df["geom_latitude"].isna()
    missing_point = df["point_longitude"].isna() | df["point_latitude"].isna()
    geom = missing_geom & ~missing_point
    point = missing_point & ~missing_geom
    both = missing_point & missing_geom
    reason["lost"] = int(both.sum())
    reason["geom_only"] = int(point.sum())
    reason["point_only"] = int(geom.sum())
    return df[missing_geom | missing_point], reason

def check_geom_point(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:

    reason = {}

    geom_lat = pd.to_numeric(df["geom_latitude"], errors="coerce")
    geom_lon = pd.to_numeric(df["geom_longitude"], errors="coerce")
    point_lat = pd.to_numeric(df["point_latitude"], errors="coerce")
    point_lon = pd.to_numeric(df["point_longitude"], errors="coerce")

    lat_differs = (geom_lat != point_lat) & ~(geom_lat.isna() & point_lat.isna())
    lon_differs = (geom_lon != point_lon) & ~(geom_lon.isna() & point_lon.isna())
    disagree = lat_differs | lon_differs

    reason["rows"] = len(df)
    reason["lat-differs"] = int(lat_differs.sum())
    reason["lon-differs"] = int(lon_differs.sum())
    reason["disagree"] = int(disagree.sum())

    return df[disagree], reason

def check_coords(
    coord_frame: pd.DataFrame,
    lat_range: tuple[float, float] = (55.0, 57.0),
    lon_range: tuple[float, float] = (38.0, 43.0),
) -> tuple[pd.DataFrame, dict]:

    reason = {}

    lon = pd.to_numeric(coord_frame["geom_longitude"], errors="coerce")
    lat = pd.to_numeric(coord_frame["geom_latitude"], errors="coerce")
    lon = lon.where(lon.notna(), pd.to_numeric(coord_frame["point_longitude"], errors="coerce"))
    lat = lat.where(lat.notna(), pd.to_numeric(coord_frame["point_latitude"], errors="coerce"))

    missing = lon.isna() | lat.isna()
    reason["coords-missing"] = int(missing.sum())

    out_of_region = ~missing & ~(lat.between(*lat_range) & lon.between(*lon_range))
    reason["coords-out-of-region"] = int(out_of_region.sum())

    mask = missing | out_of_region

    return coord_frame[mask], reason


def check_datetime(
    df: pd.DataFrame,
    column: str = "datetime",
    min_year: int = 2000,
    max_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:

    reason = {}

    parsed = pd.to_datetime(df[column], errors="coerce", format="mixed")
    if max_date is None:
        max_date = pd.Timestamp.today().normalize()

    not_parsed = parsed.isna()
    reason["datetime-not-parsed"] = int(not_parsed.sum())

    future = parsed > max_date
    reason["datetime-future"] = int(future.sum())

    too_old = parsed < pd.Timestamp(year=min_year, month=1, day=1)
    reason["datetime-too-old"] = int(too_old.sum())

    mask = not_parsed | future | too_old

    valid = parsed[~mask]
    reason["datetime-min"] = str(valid.min()) if not valid.empty else None
    reason["datetime-max"] = str(valid.max()) if not valid.empty else None

    return df[mask], reason


def monthly_counts(df: pd.DataFrame, column: str = "datetime") -> tuple[pd.Series, dict]:

    reason = {}

    parsed = pd.to_datetime(df[column], errors="coerce", format="mixed").dropna()
    if parsed.empty:
        reason["months-expected"] = 0
        reason["months-empty"] = 0
        reason["records"] = 0
        return pd.Series(dtype="int64", name="records"), reason

    counts = parsed.dt.to_period("M").value_counts().sort_index()
    full_range = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    counts = counts.reindex(full_range, fill_value=0)

    reason["months-expected"] = len(full_range)
    reason["months-empty"] = int((counts == 0).sum())
    reason["records"] = int(counts.sum())

    return counts.rename_axis("month").rename("records"), reason
