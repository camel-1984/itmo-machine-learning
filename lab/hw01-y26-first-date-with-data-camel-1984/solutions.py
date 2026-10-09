import pandas as pd
import numpy as np

def prepare_price_series(prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

    prices_float = np.asarray(prices, dtype=float)

    valid_mask = np.isfinite(prices_float) & (prices_float > 0)
    prices_float_valid = prices_float[valid_mask]

    relative_change = np.diff(prices_float_valid) / prices_float_valid[:-1]

    return prices_float_valid, relative_change



def detect_price_jumps(prices: np.ndarray, threshold: float) -> np.ndarray:

    if threshold < 0:
        raise ValueError

    prices_float = np.asarray(prices, dtype=float)

    valid_mask = np.isfinite(prices_float) & (prices_float > 0)
    prices_float_valid = prices_float[valid_mask]
    relative_change = np.diff(prices_float_valid) / prices_float_valid[:-1]

    valid_change = np.abs(relative_change) >= threshold
    valid_change_index = np.flatnonzero(valid_change)
    valid_change_index_plus = valid_change_index + 1

    prices_float_valid_index = np.flatnonzero(valid_mask)

    return prices_float_valid_index[valid_change_index_plus]


def validate_data(df: pd.DataFrame,) -> dict[str, object]:

    result = {}

    needed_columns = {"date", "market", "category", "commodity", "unit", "pricetype", "currency", "price"}

    missing_columns = sorted(needed_columns - set(df.columns))
    result["missing_columns"] = missing_columns

    dupl = df.duplicated(keep="first").sum()
    result["duplicate_rows"] = int(dupl)

    if "price" in df.columns:
        price = df["price"]
        price_numeric = pd.to_numeric(price, errors="coerce")
        invalid = ~np.isfinite(price_numeric) | (price_numeric <= 0)
        result["invalid_prices"] = int(invalid.sum())
    else:
        result["invalid_prices"] = None

    if "market" in df.columns:
        market = df["market"]
        blank = market.astype("string").str.strip() == ""
        invalid = pd.isna(market) | blank
        result["missing_markets"] = int(invalid.sum())
    else:
        result["missing_markets"] = None

    if "date" in df.columns:
        date = df["date"]
        parsed = pd.to_datetime(date, errors="coerce")
        invalid = pd.isna(parsed)
        result["invalid_dates"] = int(invalid.sum())
    else:
        result["invalid_dates"] = None

    result["is_valid"] = not result["missing_columns"] and all(
        result[k] == 0 for k in ("duplicate_rows", "invalid_prices", "missing_markets", "invalid_dates")
    )

    return result

def clean_data(df: pd.DataFrame, ) -> pd.DataFrame:

    pos = ["market", "commodity", "category", "unit", "pricetype", "currency"]
    df_copy = df.copy()

    df_copy[pos] = df_copy[pos].apply(lambda s: s.astype("string").str.split().str.join(" "))

    df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
    df_copy = df_copy.dropna(subset=["date"])

    df_copy["price"] = pd.to_numeric(df_copy["price"], errors="coerce")
    valid = np.isfinite(df_copy["price"]) & (df_copy["price"] > 0)
    df_copy = df_copy[valid]

    pos = ["market", "commodity", "unit"]
    df_copy[pos] = df_copy[pos].apply(lambda s: s.astype("string").str.strip()).replace("", pd.NA)
    df_copy = df_copy.dropna(subset=pos)

    pos = ["date", "market_id", "commodity_id", "unit", "pricetype"]
    df_copy = df_copy.drop_duplicates(subset=pos, keep="last")

    df_copy = df_copy.sort_values(by=["date", "market"]).reset_index(drop=True)

    return df_copy

def filter_data(
    df: pd.DataFrame,
    *,
    commodities: str | list[str] | None = None,
    markets: str | list[str] | None = None,
    units: str | list[str] | None = None,
    pricetype: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,) -> pd.DataFrame:

    if date_from is not None:
        date_from = pd.to_datetime(date_from, errors="raise")
    if date_to is not None:
        date_to = pd.to_datetime(date_to, errors="raise")

    if date_from is not None and date_to is not None and date_from > date_to:
        raise ValueError

    df_copy = df.copy()

    for values, col in ((commodities, "commodity"), (markets, "market"),
                        (units, "unit"), (pricetype, "pricetype")):
        if values is None:
            continue
        if isinstance(values, str):
            values = [values]
        valid = df_copy[col].isin(values)
        df_copy = df_copy[valid]

    if date_from is not None:
        df_copy = df_copy[df_copy["date"] >= date_from]
    if date_to is not None:
        df_copy = df_copy[df_copy["date"] <= date_to]

    df_copy = df_copy.reset_index(drop=True)

    return df_copy

def build_price_report(
    df: pd.DataFrame,
    group_by: list[str],) -> pd.DataFrame:

    allowed = {"admin1", "admin2", "market", "category", "commodity", "unit", "pricetype"}
    if not group_by or len(group_by) != len(set(group_by)) or (set(group_by) - allowed):
        raise ValueError

    df_copy = df.copy()

    df_copy = df_copy.groupby(group_by).agg(
        observations=("market", "size"),
        markets=("market", "nunique"),
        min_price=("price", "min"),
        max_price=("price", "max"),
        mean_price=("price", "mean"),
        median_price=("price", "median")
    )
    df_copy["price_range"] = df_copy["max_price"] - df_copy["min_price"]

    df_copy = df_copy.reset_index()

    df_copy = df_copy.sort_values(by=["observations"] + group_by, ascending=[False] + [True] * len(group_by))
    df_copy = df_copy.reset_index(drop=True)

    return df_copy


def compare_markets(
    df: pd.DataFrame,
    commodity: str,
    unit: str,
    pricetype: str,
    markets: list[str],) -> pd.DataFrame:

    if not markets or len(markets) != len(set(markets)):
        raise ValueError

    df_copy = df.copy()

    valid_commodity = df_copy["commodity"] == commodity
    valid_unit = df_copy["unit"] == unit
    valid_pricetype = df_copy["pricetype"] == pricetype
    valid_markets = df_copy["market"].isin(markets)

    valid_rampage = valid_commodity & valid_unit & valid_pricetype & valid_markets
    df_copy = df_copy[valid_rampage]

    df_copy["year"] = df_copy["date"].dt.year
    df_copy["month"] = df_copy["date"].dt.month

    df_copy = df_copy.groupby(["year", "month", "market"])["price"].median().unstack("market")
    df_copy = df_copy.rename_axis(columns=None)

    df_copy = df_copy.dropna()
    df_copy = df_copy[markets]
    df_copy = df_copy.reset_index()
    df_copy = df_copy.sort_values(by=["year", "month"]).reset_index(drop=True)

    return df_copy


def market_coverage(
    df: pd.DataFrame,
    commodity: str,
    unit: str,
    year: int,
) -> pd.DataFrame:

    df_copy = df.copy()

    valid_commodity = df_copy["commodity"] == commodity
    valid_unit = df_copy["unit"] == unit
    valid_year = df_copy["date"].dt.year == year

    valid_rampage = valid_commodity & valid_unit & valid_year
    df_copy = df_copy[valid_rampage]

    df_copy["month"] = df_copy["date"].dt.month

    df_copy = df_copy.groupby("market").agg(
        months_observed=("month", "nunique"),
        first_month=("month", "min"),
        last_month=("month", "max")
    )

    coverage = df_copy["months_observed"] / 12
    df_copy["coverage"] = coverage

    df_copy = df_copy.reset_index()

    df_copy = df_copy.sort_values(by=["coverage", "market"], ascending=[False] + [True])
    df_copy = df_copy.reset_index(drop=True)

    return df_copy


def find_missing_months(
    df: pd.DataFrame,
    commodity: str,
    unit: str,
    market: str,
    year: int,
) -> np.ndarray:

    df_copy = df.copy()

    valid_commodity = df_copy["commodity"] == commodity
    valid_unit = df_copy["unit"] == unit
    valid_market = df_copy["market"] == market
    valid_year = df_copy["date"].dt.year == year

    valid_rampage = valid_commodity & valid_unit & valid_market & valid_year

    df_copy = df_copy[valid_rampage]

    df_copy["months"] = df_copy["date"].dt.month

    i_need_you = set(range(1, 13))
    mia = i_need_you - set(df_copy["months"])

    mia_list_sorted = sorted(mia)
    return np.array(mia_list_sorted, dtype=int)


def build_market_snapshot(
    df: pd.DataFrame,
    date: str,
    max_age_days: int = 30,
) -> pd.DataFrame:

    date = pd.to_datetime(date, errors="raise")
    if max_age_days < 0:
        raise ValueError

    df_copy = df.copy()

    valid = df_copy["date"] <= date
    df_copy = df_copy[valid]

    df_copy = df_copy.sort_values(by="date")
    df_copy = df_copy.drop_duplicates(subset=["market", "commodity", "unit", "pricetype"], keep="last")

    a = (date - df_copy["date"]).dt.days
    df_copy["days_since_observation"] = a

    valid = a <= max_age_days
    df_copy = df_copy[valid]

    df_copy = df_copy[["market", "commodity", "unit", "pricetype", "date", "price", "days_since_observation"]]
    df_copy = df_copy.sort_values(["market", "commodity", "unit", "pricetype"])
    df_copy = df_copy.reset_index(drop=True)

    return df_copy
