import pandas as pd
import matplotlib.pyplot as plt

def commodity_date_price_change(df: pd.DataFrame,
                           commodity: str,
                           market: str,
                           unit: str,
                           pricetype: str) -> pd.DataFrame:

    valid_market = df["market"] == market
    valid_unit = df["unit"] == unit
    valid_pricetype = df["pricetype"] == pricetype
    valid_commodity = df["commodity"] == commodity

    mask = valid_market & valid_unit & valid_pricetype & valid_commodity
    df_valid = df[mask]

    df_valid = df_valid.groupby("date")["price"].median()
    df_valid = df_valid.reset_index()

    return df_valid

def commodity_market_price_change(df: pd.DataFrame,
                           commodity: str,
                           unit: str,
                           pricetype: str) -> pd.DataFrame:

    valid_unit = df["unit"] == unit
    valid_pricetype = df["pricetype"] == pricetype
    valid_commodity = df["commodity"] == commodity

    mask = valid_unit & valid_pricetype & valid_commodity
    df_valid = df[mask]

    df_valid["market"] = df_valid["market"].astype("string").str.split().str[:2].str.join(" ")

    df_valid = df_valid.groupby("market")["price"].median()
    df_valid = df_valid.reset_index()
    df_valid = df_valid.sort_values(by="price")

    return df_valid

def visualize_date_commodity_price_change(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.set_title("Изменение цены товара по годам")
    ax.set_xlabel("Год")
    ax.set_ylabel("Цена")
    ax.plot(df["date"], df["price"])
    fig.autofmt_xdate()
    plt.show()
    fig.savefig("date_price.png", dpi=200)

def visualize_market_commodity_price_change(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    ax.set_title("Медиана цены товара по рынкам")
    ax.set_xlabel("Рынок")
    ax.set_ylabel("Цена")
    ax.scatter(df["market"], df["price"])
    ax.tick_params(axis="x", rotation=15)
    plt.show()
    fig.savefig("market_price.png", dpi=200)

if __name__ == "__main__":
    df = pd.read_csv("data/wfp_food_prices_ken.csv")
    commodity = "Beans"
    market = "Mombasa"
    unit = "KG"
    pricetype = "Wholesale"

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price"])

    valid_df_one = commodity_date_price_change(df,
                                      commodity,
                                      market,
                                      unit,
                                      pricetype)

    valid_df_two = commodity_market_price_change(df,
                                               commodity,
                                               unit,
                                               pricetype)

    visualize_date_commodity_price_change(valid_df_one)
    visualize_market_commodity_price_change(valid_df_two)