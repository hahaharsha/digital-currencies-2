import pandas as pd
import numpy as np
from pathlib import Path


project_root = Path(__file__).resolve().parent

prices_path = project_root / "dataset" / "prices.csv"
output_path = project_root / "dataset" / "literature_features.csv"

prices = pd.read_csv(
    prices_path,
    parse_dates=["date"]
)

prices = prices.sort_values(
    ["coin", "date"]
).copy()


# extract btc and eth

btc = (
    prices[prices["coin"] == "BTC"]
    .sort_values("date")
    .copy()
)

eth = (
    prices[prices["coin"] == "ETH"]
    .sort_values("date")
    .copy()
)


# btc features

# 24-hour BTC price change
btc["BTC_percent_change_24h"] = (
    btc["close"].pct_change(1)
)

# 24-hour BTC volume change
btc["BTC_volume_percent_change_24h"] = (
    btc["volume_usd"].pct_change(1)
)

# 30-day BTC volume change
btc["BTC_volume_percent_change_30d"] = (
    btc["volume_usd"].pct_change(30)
)

# Daily BTC return
btc["BTC_daily_return"] = (
    btc["close"].pct_change(1)
)

# prelim backward-looking btc volatility proxy
# 30-day rolling standard deviation of daily returns
btc["BTC_realized_volatility_30d_proxy"] = (
    btc["BTC_daily_return"]
    .rolling(
        window=30,
        min_periods=30
    )
    .std()
)


#eth features

# 24-hour ETH price change
eth["ETH_percent_change_24h"] = (
    eth["close"].pct_change(1)
)

# 30-day ETH volume change
eth["ETH_volume_percent_change_30d"] = (
    eth["volume_usd"].pct_change(30)
)

# Daily ETH return
eth["ETH_daily_return"] = (
    eth["close"].pct_change(1)
)

# preliminary backward-looking eth volatility proxy
eth["ETH_realized_volatility_30d_proxy"] = (
    eth["ETH_daily_return"]
    .rolling(
        window=30,
        min_periods=30
    )
    .std()
)


# using required features only

btc_features = btc[
    [
        "date",
        "BTC_percent_change_24h",
        "BTC_volume_percent_change_24h",
        "BTC_volume_percent_change_30d",
        "BTC_realized_volatility_30d_proxy"
    ]
].copy()

eth_features = eth[
    [
        "date",
        "ETH_percent_change_24h",
        "ETH_volume_percent_change_30d",
        "ETH_realized_volatility_30d_proxy"
    ]
].copy()


# merge btc+eth features by date
features = btc_features.merge(
    eth_features,
    on="date",
    how="inner"

)

features = (

    features

    .sort_values("date")

    .reset_index(drop=True)

)

# diagnostics

print("\n=== LITERATURE FEATURE DATASET ===")

print(
    "Observations:",
    len(features)
)

print(
    "Date range:",
    features["date"].min(),
    "to",
    features["date"].max()
)

print("\n=== FEATURE MISSINGNESS ===")

print(
    features
    .isna()
    .sum()
)

print("\n=== FIRST COMPLETE FEATURE DATE ===")

complete = features.dropna().copy()

if len(complete) > 0:
    print(
        complete["date"].min()
    )

else:
    print(
        "No complete rows found."
    )

print("\n=== FEATURE SUMMARY ===")

print(
    features
    .drop(columns=["date"])
    .describe()
    .T
)

# save outputs

features.to_csv(
    output_path,
    index=False
)

print("\nSaved:")
print(output_path)