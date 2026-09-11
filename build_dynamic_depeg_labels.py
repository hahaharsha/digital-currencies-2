import pandas as pd
from pathlib import Path


# paths and params

project_root = Path(__file__).resolve().parent

prices_path = (
    project_root
    / "dataset"
    / "prices.csv"
)

output_path = (
    project_root
    / "processed_data"
    / "depeg_labels_dynamic.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# literature-based dynamic threshold parameter
alpha = 1 / 3


stablecoins = [
    "USDT",
    "USDC",
    "DAI",
    "BUSD",
]


# load price data

prices = pd.read_csv(
    prices_path,
    parse_dates=["date"],
)


prices = (
    prices[
        prices["coin"].isin(stablecoins)
    ]
    .copy()
    .sort_values(
        [
            "coin",
            "date",
        ]
    )
    .reset_index(drop=True)
)


print("\n=== RAW STABLECOIN PRICE DATA ===")

print(
    f"Observations: {len(prices):,}"
)

print(
    "Date range:",
    prices["date"].min().date(),
    "to",
    prices["date"].max().date(),
)

print("\nObservations by coin:")

print(
    prices["coin"]
    .value_counts()
    .sort_index()
)


# 30-day rolling volume window is used to construct the literature-based dynamic depeg threshold.
prices["rolling_volume_30d"] = (
    prices
    .groupby("coin")["volume_usd"]
    .transform(
        lambda x: x.rolling(
            window=30,
            min_periods=30,
        ).sum()
    )
)


# dynamic depeg threshold
# adapted from Lee et al.:

# lower threshold: 1 - 10 / V_monthly^(1/3)
# upper threshold: 1 + 10 / V_monthly^(1/3)
# V_monthly is operationalised here using trailing 30-day trading volume available at time t.

prices["threshold_distance"] = (
    10
    /
    (
        prices["rolling_volume_30d"]
        ** alpha
    )
)


prices["threshold_down"] = (
    1
    -
    prices["threshold_distance"]
)


prices["threshold_up"] = (
    1
    +
    prices["threshold_distance"]
)



# next-day price extremes
# information observed at day t is used to define whether a depeg occurs on day t+1.

prices["next_day_low"] = (
    prices
    .groupby("coin")["low"]
    .shift(-1)
)


prices["next_day_high"] = (
    prices
    .groupby("coin")["high"]
    .shift(-1)
)


prices["next_day_close"] = (
    prices
    .groupby("coin")["close"]
    .shift(-1)
)


# dynamic depeg label
# Y(i,t+1) = 1 if the next-day price breaches either the lower or upper dynamic threshold.

prices["depeg_dynamic"] = (
    (
        prices["next_day_low"]
        <= prices["threshold_down"]
    )
    |
    (
        prices["next_day_high"]
        >= prices["threshold_up"]
    )
).astype(int)



# remove obs without required info 
# first 29 observations per coin do not have a complete trailing 30-day volume window.
# last observation per coin has no next-day price.

valid = (
    prices["rolling_volume_30d"].notna()
    &
    prices["next_day_low"].notna()
    &
    prices["next_day_high"].notna()
)


labels = (
    prices
    .loc[valid]
    .copy()
    .reset_index(drop=True)
)


# data diagnostics

print("\n========================================")
print("DYNAMIC DEPEG LABEL DATASET")
print("========================================")


print(
    f"\nObservations: {len(labels):,}"
)

print(
    "Date range:",
    labels["date"].min().date(),
    "to",
    labels["date"].max().date(),
)


print("\n=== OBSERVATIONS BY COIN ===")

print(
    labels["coin"]
    .value_counts()
    .sort_index()
)


# dynamic threshold summary

print("\n=== DYNAMIC THRESHOLD SUMMARY ===")

print(
    labels[
        [
            "threshold_distance",
            "threshold_down",
            "threshold_up",
        ]
    ]
    .describe()
    .T
    .round(6)
)


# summary by coin

print(
    "\n=== DYNAMIC THRESHOLD SUMMARY BY COIN ==="
)

threshold_by_coin = (
    labels
    .groupby("coin")
    .agg(
        observations=(
            "date",
            "size",
        ),

        median_30d_volume=(
            "rolling_volume_30d",
            "median",
        ),

        mean_threshold_distance=(
            "threshold_distance",
            "mean",
        ),

        median_threshold_distance=(
            "threshold_distance",
            "median",
        ),

        min_threshold_distance=(
            "threshold_distance",
            "min",
        ),

        max_threshold_distance=(
            "threshold_distance",
            "max",
        ),
    )
)

print(
    threshold_by_coin.round(6)
)


# class balance

print("\n=== DYNAMIC DEPEG CLASS BALANCE ===")

class_balance = (
    labels["depeg_dynamic"]
    .value_counts()
    .sort_index()
)

print(
    class_balance
)


positive_count = int(
    labels["depeg_dynamic"].sum()
)

positive_rate = (
    labels["depeg_dynamic"].mean()
)


print(
    f"\nDynamic depeg observations: "
    f"{positive_count:,}"
)

print(
    f"Dynamic depeg rate: "
    f"{positive_rate:.4f}"
)

print(
    f"Dynamic depeg rate (%): "
    f"{positive_rate * 100:.2f}%"
)


#  dynamic depeg rate by coin

depeg_by_coin = (
    labels
    .groupby("coin")["depeg_dynamic"]
    .agg(
        events="sum",
        observations="size",
        rate="mean",
    )
)

depeg_by_coin[
    "rate_pct"
] = (
    depeg_by_coin["rate"]
    * 100
)


print("\n=== DYNAMIC DEPEG RATE BY COIN ===")

print(
    depeg_by_coin.round(4)
)


# depeg rate by coin and year

labels["year"] = (
    labels["date"]
    .dt.year
)


depeg_by_coin_year = (
    labels
    .groupby(
        [
            "coin",
            "year",
        ]
    )["depeg_dynamic"]
    .agg(
        events="sum",
        observations="size",
        rate="mean",
    )
)

depeg_by_coin_year[
    "rate_pct"
] = (
    depeg_by_coin_year["rate"]
    * 100
)


print(
    "\n=== DYNAMIC DEPEG RATE BY COIN AND YEAR ==="
)

print(
    depeg_by_coin_year.round(4)
)


# sample positive observations

print(
    "\n=== FIRST 20 DYNAMIC DEPEG OBSERVATIONS ==="
)

print(
    labels.loc[
        labels[
            "depeg_dynamic"
        ] == 1,
        [
            "date",
            "coin",
            "rolling_volume_30d",
            "threshold_down",
            "threshold_up",
            "next_day_low",
            "next_day_high",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)


# save output

columns_to_save = [
    "date",
    "coin",
    "rolling_volume_30d",
    "threshold_distance",
    "threshold_down",
    "threshold_up",
    "next_day_low",
    "next_day_high",
    "next_day_close",
    "depeg_dynamic",
]


labels[
    columns_to_save
].to_csv(
    output_path,
    index=False,
)

print("SAVED OUTPUT")

print(
    output_path
)