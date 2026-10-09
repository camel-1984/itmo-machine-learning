import numpy as np
import pandas as pd

from solutions import validate_data, clean_data


def make_valid_df():
    return pd.DataFrame({
        "date": ["2024-01-01", "2024-02-01"],
        "market": ["Nairobi", "Mombasa"],
        "market_id": [1, 2],
        "category": ["cereals", "cereals"],
        "commodity": ["Maize", "Maize"],
        "commodity_id": [10, 10],
        "unit": ["KG", "KG"],
        "pricetype": ["Retail", "Retail"],
        "currency": ["KES", "KES"],
        "price": [100.0, 120.0],
    })


def test_validate_data_valid_table():
    df = make_valid_df()
    original = df.copy(deep=True)

    result = validate_data(df)

    assert result == {
        "missing_columns": [],
        "duplicate_rows": 0,
        "invalid_prices": 0,
        "missing_markets": 0,
        "invalid_dates": 0,
        "is_valid": True,
    }
    pd.testing.assert_frame_equal(df, original)


def test_validate_data_counts_bad_values():
    df = make_valid_df()
    bad = pd.DataFrame({
        "date": ["bad-date", None, "2024-03-01"],
        "market": [" ", None, "Kisumu"],
        "market_id": [3, 4, 5],
        "category": ["cereals"] * 3,
        "commodity": ["Rice"] * 3,
        "commodity_id": [20] * 3,
        "unit": ["KG"] * 3,
        "pricetype": ["Retail"] * 3,
        "currency": ["KES"] * 3,
        "price": ["unknown", np.inf, -5],
    })
    df = pd.concat([df, bad], ignore_index=True)

    result = validate_data(df)

    assert result["invalid_prices"] == 3
    assert result["missing_markets"] == 2
    assert result["invalid_dates"] == 2
    assert result["is_valid"] is False


def test_clean_data_normalizes_and_keeps_last_observation():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "bad"],
        "market": ["  Nairobi  ", "Nairobi", "Mombasa"],
        "market_id": [1, 1, 2],
        "category": ["  cereals   and tubers ", "cereals and tubers", "cereals"],
        "commodity": ["  Maize   flour ", "Maize flour", "Rice"],
        "commodity_id": [10, 10, 20],
        "unit": [" KG ", "KG", "KG"],
        "pricetype": [" Retail ", "Retail", "Retail"],
        "currency": [" KES ", "KES", "KES"],
        "price": ["100", "120", "50"],
    })

    result = clean_data(df)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["market"] == "Nairobi"
    assert row["commodity"] == "Maize flour"
    assert row["category"] == "cereals and tubers"
    assert row["unit"] == "KG"
    assert row["pricetype"] == "Retail"
    assert row["currency"] == "KES"
    assert row["price"] == 120
    assert pd.api.types.is_datetime64_any_dtype(result["date"])
    assert result.index.tolist() == [0]
