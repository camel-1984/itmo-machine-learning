import numpy as np
import pandas as pd

import eda

DIED_PREFIX = "Скончался"
UNKNOWN_STATUS = "Не определен"
AGGREGATE_BRANDS = ("Прочие марки ТС",)


def occupants_frame(df: pd.DataFrame) -> pd.DataFrame:

    participants = eda.participants_frame(df)
    vehicles = eda.vehicle_frame(df)

    inside = participants[participants["in_vehicle"]]
    frame = inside.merge(
        vehicles[["source_id", "vehicle_idx", "brand", "year", "category"]],
        on=["source_id", "vehicle_idx"],
        how="inner",
        validate="many_to_one",
    )

    status = frame["health_status"]
    frame["died"] = status.str.startswith(DIED_PREFIX, na=False)
    frame["status_known"] = status.notna() & status.ne(UNKNOWN_STATUS)

    return frame


def brand_mortality(
        occupants: pd.DataFrame,
        min_count: int = 200,
        top: int = 12,
) -> pd.DataFrame:

    known = occupants[occupants["status_known"] & occupants["brand"].notna()]
    known = known[~known["brand"].isin(AGGREGATE_BRANDS)]

    frame = (
        known.groupby("brand")
        .agg(occupants=("died", "size"), died=("died", "sum"), median_year=("year", "median"))
        .reset_index()
    )
    frame = frame[frame["occupants"] >= min_count]
    frame["mortality"] = frame["died"] / frame["occupants"]

    return frame.nlargest(top, "occupants").sort_values("mortality", ignore_index=True)


TARGET_BRAND = "ВАЗ"


def mixed_brand_occupants(df: pd.DataFrame, brand: str = TARGET_BRAND) -> pd.DataFrame:

    frame = occupants_frame(df)
    frame = frame[frame["status_known"] & frame["brand"].notna()]
    frame = frame[~frame["brand"].isin(AGGREGATE_BRANDS)]

    frame = frame.copy()
    frame["target"] = frame["brand"] == brand

    both = frame.groupby("source_id")["target"].nunique()
    mixed = both[both == 2].index

    return frame[frame["source_id"].isin(mixed)]


def accident_brand_table(occupants: pd.DataFrame) -> pd.DataFrame:

    table = (
        occupants.groupby(["source_id", "target"])
        .agg(people=("died", "size"), died=("died", "sum"))
        .unstack("target")
    )
    table.columns = [f"{name}_{'target' if flag else 'other'}" for name, flag in table.columns]
    table = table.fillna(0).astype(int)

    table["share_target"] = table["died_target"] / table["people_target"]
    table["share_other"] = table["died_other"] / table["people_other"]
    table["difference"] = table["share_target"] - table["share_other"]

    return table.sort_values("difference")


def _accident_vehicles(occupants: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, int]]:
    rows = []
    for _, part in occupants.groupby("source_id", sort=True):
        vehicles = (
            part.groupby("vehicle_idx")
            .agg(people=("died", "size"), died=("died", "sum"), target=("target", "first"))
            .sort_values("target", ascending=False)
        )
        rows.append((
            vehicles["people"].to_numpy(dtype=float),
            vehicles["died"].to_numpy(dtype=float),
            int(vehicles["target"].sum()),
        ))
    return rows


def _mean_pair_difference(rows: list[tuple[np.ndarray, np.ndarray, int]]) -> float:
    differences = [
        died[:k].sum() / people[:k].sum() - died[k:].sum() / people[k:].sum()
        for people, died, k in rows
    ]
    return float(np.mean(differences))


def brand_permutation(
        occupants: pd.DataFrame,
        draws: int = 2000,
        seed: int = 0,
) -> dict:

    rows = _accident_vehicles(occupants)
    observed = _mean_pair_difference(rows)

    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        shuffled = []
        for people, died, k in rows:
            order = rng.permutation(people.size)
            shuffled.append((people[order], died[order], k))
        null[i] = _mean_pair_difference(shuffled)

    return {
        "observed": observed,
        "p_value": float((np.abs(null) >= abs(observed)).mean()),
        "accidents": len(rows),
        "draws": draws,
        "seed": seed,
        "null": null,
    }


def brand_bootstrap(
        pairs: pd.DataFrame,
        draws: int = 2000,
        seed: int = 0,
        level: float = 0.95,
) -> dict:

    differences = pairs["difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(draws, differences.size), replace=True).mean(axis=1)

    tail = (1 - level) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])

    return {
        "observed": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "level": level,
        "accidents": int(differences.size),
        "draws": draws,
        "seed": seed,
        "samples": samples,
    }


def brand_mortality_ci(
        occupants: pd.DataFrame,
        min_count: int = 500,
        top: int = 12,
        draws: int = 2000,
        seed: int = 0,
        level: float = 0.95,
) -> pd.DataFrame:

    frame = brand_mortality(occupants, min_count=min_count, top=top)
    known = occupants[occupants["status_known"] & occupants["brand"].notna()]

    rng = np.random.default_rng(seed)
    tail = (1 - level) / 2
    bounds = []
    for brand in frame["brand"]:
        died = known.loc[known["brand"] == brand, "died"].to_numpy(dtype=float)
        samples = rng.choice(died, size=(draws, died.size), replace=True).mean(axis=1)
        bounds.append(np.quantile(samples, [tail, 1 - tail]))

    frame[["ci_lower", "ci_upper"]] = np.array(bounds)

    return frame
