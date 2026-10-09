import pandas as pd
import pytest

from solutions import filter_data, build_price_report


def sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"
        ]),
        "admin1": ["A", "A", "B", "B"],
        "admin2": ["A1", "A1", "B1", "B1"],
        "market": ["M1", "M1", "M2", "M3"],
        "category": ["cereals", "cereals", "pulses", "cereals"],
        "commodity": ["Rice", "Rice", "Beans", "Rice"],
        "unit": ["KG", "KG", "KG", "50 KG"],
        "pricetype": ["Retail", "Wholesale", "Retail", "Retail"],
        "price": [10.0, 20.0, 30.0, 100.0],
    })


def test_filter_data_combines_filters():
    df = sample_df()

    result = filter_data(
        df,
        commodities="Rice",
        markets=["M1"],
        units="KG",
        pricetype="Retail",
        date_from="2024-01-01",
        date_to="2024-01-31",
    )

    assert len(result) == 1
    assert result.iloc[0]["market"] == "M1"
    assert result.iloc[0]["price"] == 10.0
    assert result.index.tolist() == [0]


def test_filter_data_empty_list_means_empty_result():
    df = sample_df()

    result = filter_data(df, commodities=[])

    assert result.empty
    assert result.columns.tolist() == df.columns.tolist()


def test_filter_data_invalid_date_range():
    df = sample_df()

    with pytest.raises(ValueError):
        filter_data(df, date_from="2024-03-01", date_to="2024-01-01")


def test_build_price_report_dynamic_grouping():
    df = sample_df()

    result = build_price_report(df, ["commodity", "unit"])

    assert result.columns.tolist() == [
        "commodity", "unit", "observations", "markets",
        "min_price", "max_price", "mean_price",
        "median_price", "price_range"
    ]

    rice_kg = result[
        (result["commodity"] == "Rice") & (result["unit"] == "KG")
    ].iloc[0]

    assert rice_kg["observations"] == 2
    assert rice_kg["markets"] == 1
    assert rice_kg["min_price"] == 10.0
    assert rice_kg["max_price"] == 20.0
    assert rice_kg["mean_price"] == 15.0
    assert rice_kg["median_price"] == 15.0
    assert rice_kg["price_range"] == 10.0


@pytest.mark.parametrize(
    "group_by",
    [[], ["commodity", "commodity"], ["not_a_column"]],
)
def test_build_price_report_rejects_bad_grouping(group_by):
    with pytest.raises(ValueError):
        build_price_report(sample_df(), group_by)
