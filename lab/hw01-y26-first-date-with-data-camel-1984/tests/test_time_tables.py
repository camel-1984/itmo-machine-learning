import numpy as np
import pandas as pd

from solutions import compare_markets, market_coverage, find_missing_months


def time_df():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-05", "2024-01-20", "2024-02-10", "2024-03-10",
            "2024-01-08", "2024-03-12",
            "2024-01-09", "2024-02-11", "2024-03-13",
        ]),
        "market": ["A", "A", "A", "A", "B", "B", "C", "C", "C"],
        "commodity": ["Rice"] * 9,
        "unit": ["KG"] * 9,
        "pricetype": ["Retail"] * 9,
        "price": [10, 14, 20, 30, 12, 36, 11, 22, 33],
    })


def test_compare_markets_keeps_only_common_months():
    result = compare_markets(
        time_df(), "Rice", "KG", "Retail", ["A", "B"]
    )

    assert result.columns.tolist() == ["year", "month", "A", "B"]
    assert result[["year", "month"]].values.tolist() == [
        [2024, 1],
        [2024, 3],
    ]
    # A January median = median(10, 14) = 12
    assert result.loc[0, "A"] == 12
    assert result.loc[0, "B"] == 12


def test_market_coverage_counts_unique_months():
    result = market_coverage(time_df(), "Rice", "KG", 2024)

    a = result[result["market"] == "A"].iloc[0]
    b = result[result["market"] == "B"].iloc[0]

    assert a["months_observed"] == 3
    assert a["first_month"] == 1
    assert a["last_month"] == 3
    assert a["coverage"] == 3 / 12

    assert b["months_observed"] == 2
    assert b["coverage"] == 2 / 12


def test_find_missing_months():
    result = find_missing_months(time_df(), "Rice", "KG", "B", 2024)

    np.testing.assert_array_equal(
        result,
        np.array([2, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    )
