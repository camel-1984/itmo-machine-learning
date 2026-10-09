import pandas as pd
import pytest

from solutions import build_market_snapshot


def snapshot_df():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-10",
            "2024-01-20",
            "2024-02-05",
            "2023-12-01",
            "2024-01-24",
        ]),
        "market": ["Nairobi", "Nairobi", "Nairobi", "Mombasa", "Mombasa"],
        "commodity": ["Rice", "Rice", "Rice", "Rice", "Beans"],
        "unit": ["KG"] * 5,
        "pricetype": ["Retail"] * 5,
        "price": [100, 110, 150, 90, 50],
    })


def test_build_market_snapshot():
    result = build_market_snapshot(
        snapshot_df(),
        date="2024-01-25",
        max_age_days=30,
    )

    assert result.columns.tolist() == [
        "market", "commodity", "unit", "pricetype",
        "date", "price", "days_since_observation"
    ]

    # Mombasa/Rice is older than 30 days and is excluded.
    assert len(result) == 2

    nairobi = result[
        (result["market"] == "Nairobi") &
        (result["commodity"] == "Rice")
    ].iloc[0]
    assert nairobi["date"] == pd.Timestamp("2024-01-20")
    assert nairobi["price"] == 110
    assert nairobi["days_since_observation"] == 5

    mombasa_beans = result[
        (result["market"] == "Mombasa") &
        (result["commodity"] == "Beans")
    ].iloc[0]
    assert mombasa_beans["days_since_observation"] == 1


def test_snapshot_rejects_invalid_arguments():
    df = snapshot_df()

    with pytest.raises(ValueError):
        build_market_snapshot(df, date="not-a-date")

    with pytest.raises(ValueError):
        build_market_snapshot(df, date="2024-01-01", max_age_days=-1)
