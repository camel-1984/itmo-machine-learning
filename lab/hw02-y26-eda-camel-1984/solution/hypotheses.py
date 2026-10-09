import h3
import numpy as np
import pandas as pd
from scipy import stats

import data_loading
import eda
import geo

LIT = "В темное время суток, освещение включено"
UNLIT = "В темное время суток, освещение отсутствует"


def night_accidents(df: pd.DataFrame, resolution: int = 6) -> pd.DataFrame:

    frame = df if "latitude" in df.columns else data_loading.full_frame_to_app(df)
    frame = frame[geo.region_mask(frame)]
    frame = frame[frame["light"].isin([LIT, UNLIT])]

    frame = frame.copy()
    frame["lit"] = frame["light"] == LIT
    frame["fatal"] = pd.to_numeric(frame["dead_count"], errors="coerce").fillna(0) > 0
    frame["cell"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(frame["latitude"], frame["longitude"])
    ]

    return frame


def cell_lighting_table(night: pd.DataFrame, min_each: int = 5) -> pd.DataFrame:

    table = (
        night.groupby(["cell", "lit"])
        .agg(accidents=("fatal", "size"), fatal=("fatal", "sum"))
        .unstack("lit")
    )
    table.columns = [f"{name}_{'lit' if flag else 'unlit'}" for name, flag in table.columns]
    table = table.fillna(0).astype(int)

    table = table[
        (table["accidents_lit"] >= min_each) & (table["accidents_unlit"] >= min_each)
    ]

    table["share_lit"] = table["fatal_lit"] / table["accidents_lit"]
    table["share_unlit"] = table["fatal_unlit"] / table["accidents_unlit"]
    table["difference"] = table["share_unlit"] - table["share_lit"]

    return table.sort_values("difference")


def _cell_arrays(night: pd.DataFrame, cells: pd.Index) -> list[tuple[np.ndarray, int]]:
    inside = night[night["cell"].isin(cells)]
    arrays = []
    for _, part in inside.groupby("cell", sort=True):
        part = part.sort_values("lit", ascending=False)
        fatal = part["fatal"].to_numpy(dtype=float)
        arrays.append((fatal, int(part["lit"].sum())))
    return arrays


def _mean_difference(arrays: list[tuple[np.ndarray, int]]) -> float:
    differences = [
        fatal[lit_n:].mean() - fatal[:lit_n].mean()
        for fatal, lit_n in arrays
    ]
    return float(np.mean(differences))


def lighting_permutation(
        night: pd.DataFrame,
        cells: pd.DataFrame,
        draws: int = 2000,
        seed: int = 0,
) -> dict:

    arrays = _cell_arrays(night, cells.index)
    observed = _mean_difference(arrays)

    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        shuffled = [(rng.permutation(fatal), lit_n) for fatal, lit_n in arrays]
        null[i] = _mean_difference(shuffled)

    p_value = float((np.abs(null) >= abs(observed)).mean())

    return {
        "observed": observed,
        "p_value": p_value,
        "cells": len(arrays),
        "draws": draws,
        "seed": seed,
        "null": null,
    }


def lighting_bootstrap(
        cells: pd.DataFrame,
        draws: int = 2000,
        seed: int = 0,
        level: float = 0.95,
) -> dict:

    differences = cells["difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, size=(draws, differences.size), replace=True).mean(axis=1)

    tail = (1 - level) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])

    return {
        "observed": float(differences.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "level": level,
        "cells": int(differences.size),
        "draws": draws,
        "seed": seed,
        "samples": samples,
    }


EXPERIENCE_BINS = (-1, 2, 5, 10, 20, 100)
EXPERIENCE_LABELS = ("0-2", "3-5", "6-10", "11-20", "21+")
SERVICE_VALUES = ("95", "96", "99")
UNKNOWN_STATUS = "Не определен"
UNHURT_STATUS = "Не пострадал"


def driver_outcomes(df: pd.DataFrame) -> pd.DataFrame:

    participants = eda.participants_frame(df)

    frame = participants[
        participants["role"].eq("Водитель")
        & participants["years_of_driving_experience"].notna()
        & participants["health_status"].notna()
    ]
    frame = frame[~frame["years_of_driving_experience"].astype(str).isin(SERVICE_VALUES)]
    frame = frame[frame["health_status"] != UNKNOWN_STATUS]

    frame = frame.copy()
    frame["hurt"] = ~frame["health_status"].str.startswith(UNHURT_STATUS)
    frame["experience_group"] = pd.cut(
        frame["years_of_driving_experience"],
        bins=list(EXPERIENCE_BINS), labels=list(EXPERIENCE_LABELS),
    )

    return frame


def experience_table(drivers: pd.DataFrame) -> pd.DataFrame:

    table = (
        drivers.groupby("experience_group", observed=True)
        .agg(drivers=("hurt", "size"), hurt=("hurt", "sum"))
    )
    table["share"] = table["hurt"] / table["drivers"]

    return table


def experience_chi2(drivers: pd.DataFrame) -> dict:

    table = pd.crosstab(drivers["experience_group"], drivers["hurt"])
    chi2, p_value, dof, expected = stats.chi2_contingency(table)
    n = int(table.to_numpy().sum())

    shares = experience_table(drivers)["share"].to_numpy()
    monotone = bool(np.all(np.diff(shares) < 0))

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "n": n,
        "cramers_v": float(np.sqrt(chi2 / (n * (min(table.shape) - 1)))),
        "min_expected": float(expected.min()),
        "monotone": monotone,
    }


def experience_difference(
        drivers: pd.DataFrame,
        low: str = "0-2",
        high: str = "21+",
        draws: int = 2000,
        seed: int = 0,
        level: float = 0.95,
) -> dict:

    groups = {
        name: drivers.loc[drivers["experience_group"] == name, "hurt"].to_numpy(dtype=float)
        for name in (low, high)
    }
    observed = groups[low].mean() - groups[high].mean()

    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for i in range(draws):
        a = rng.choice(groups[low], size=groups[low].size, replace=True)
        b = rng.choice(groups[high], size=groups[high].size, replace=True)
        samples[i] = a.mean() - b.mean()

    tail = (1 - level) / 2
    lower, upper = np.quantile(samples, [tail, 1 - tail])

    return {
        "low": low, "high": high,
        "low_n": int(groups[low].size), "high_n": int(groups[high].size),
        "low_share": float(groups[low].mean()), "high_share": float(groups[high].mean()),
        "difference": float(observed),
        "ci_lower": float(lower), "ci_upper": float(upper),
        "level": level, "draws": draws, "seed": seed,
    }


def experience_table_ci(
        drivers: pd.DataFrame,
        draws: int = 2000,
        seed: int = 0,
        level: float = 0.95,
) -> pd.DataFrame:

    table = experience_table(drivers)
    rng = np.random.default_rng(seed)
    tail = (1 - level) / 2

    bounds = []
    for group in table.index:
        hurt = drivers.loc[drivers["experience_group"] == group, "hurt"].to_numpy(dtype=float)
        samples = rng.choice(hurt, size=(draws, hurt.size), replace=True).mean(axis=1)
        bounds.append(np.quantile(samples, [tail, 1 - tail]))

    table[["ci_lower", "ci_upper"]] = np.array(bounds)

    return table
