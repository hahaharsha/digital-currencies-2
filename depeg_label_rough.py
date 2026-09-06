import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------

project_root = Path(__file__).resolve().parent

prices_path = project_root / "dataset" / "prices.csv"
output_path = project_root / "dataset" / "depeg_labels.csv"


# load data

prices = pd.read_csv(
    prices_path,
    parse_dates=["date"]
)

prices = prices.sort_values(
    ["coin", "date"]
).copy()


#keep stablecoins only

stablecoins = [
    "USDC",
    "USDT",
    "DAI",
    "BUSD"
]

stable = prices[
    prices["coin"].isin(stablecoins)
].copy()


# define depeg band
# primary threshold:
# downside depeg = price falls more than 1% below $1 peg

lower_band = 0.99
upper_band = 1.01


# create next day variables

# shift(-1) gives the price information for day t+1
# while keeping the row indexed at day t

stable["next_day_low"] = (
    stable.groupby("coin")["low"]
    .shift(-1)
)

stable["next_day_high"] = (
    stable.groupby("coin")["high"]
    .shift(-1)
)

stable["next_day_close"] = (
    stable.groupby("coin")["close"]
    .shift(-1)
)

# the last observation for each coin has no t+1 price, so its future depeg label is undefined.

stable = stable.dropna(
    subset=[
        "next_day_low",
        "next_day_high",
        "next_day_close"
    ]
).copy()


# depeg deviation measures

stable["next_day_max_downside_deviation"] = (
    1 - stable["next_day_low"]
)

stable["next_day_max_upside_deviation"] = (
    stable["next_day_high"] - 1
)

stable["next_day_max_abs_deviation"] = stable[
    [
        "next_day_max_downside_deviation",
        "next_day_max_upside_deviation"
    ]
].max(axis=1)


# primary depeg label

# Primary outcome: Y(i,t) = 1 if the stablecoin trades below $0.99 at any point during day t+1.
# this captures downside depegging / run risk rather than treating an upside premium as the same event.

stable["depeg_downside_1pct"] = (
    stable["next_day_low"] < lower_band
).astype(int)

# robustness label 1: symmetric +/-1% DEPEG

# counts both downside depegs and upside premiums.

stable["depeg_symmetric_1pct"] = (
    (stable["next_day_low"] < lower_band)
    | (stable["next_day_high"] > upper_band)
).astype(int)

# labels for more severe downside depegs

stable["depeg_downside_2pct"] = (
    stable["next_day_low"] < 0.98
).astype(int)

stable["depeg_downside_3pct"] = (
    stable["next_day_low"] < 0.97
).astype(int)


# present depeg status at time = t
# to identify whether the coin is ALREADY below the primary peg threshold on day t.

stable["depeg_today"] = (
    stable["low"] < lower_band
).astype(int)


# depeg onset label

# this is a early-warning outcome which equals 1 only when:
# 1. the coin is NOT depegged on day t, AND
# 2. the coin depegs on day t+1.
#
# this prevents an ongoing multi-day depeg from being counted as a new warning event every day.

stable["depeg_onset_next_day"] = (
    (stable["depeg_today"] == 0)
    & (stable["next_day_low"] < lower_band)
).astype(int)


# BACKWARDS-COMPATIBLE PRIMARY LABEL NAME

# IMPORTANT: it means DOWNSIDE 1% depeg, not
# symmetric +/-1% deviation.

stable["depeg_next_day"] = (
    stable["depeg_downside_1pct"]
)


# save outputs

output_cols = [
    "date",
    "coin",
    "open",
    "high",
    "low",
    "close",

    "next_day_low",
    "next_day_high",
    "next_day_close",

    "next_day_max_downside_deviation",
    "next_day_max_upside_deviation",
    "next_day_max_abs_deviation",

    "depeg_today",

    "depeg_downside_1pct",
    "depeg_symmetric_1pct",
    "depeg_downside_2pct",
    "depeg_downside_3pct",

    "depeg_onset_next_day",

    "depeg_next_day"
]

stable[output_cols].to_csv(
    output_path,
    index=False
)

print("Saved:")
print(output_path)


##robustness checks
# 1. label sensitivity

label_cols = [
    "depeg_downside_1pct",
    "depeg_symmetric_1pct",
    "depeg_downside_2pct",
    "depeg_downside_3pct",
    "depeg_onset_next_day"
]

print("\n=== LABEL SENSITIVITY ===")

for col in label_cols:

    positives = int(
        stable[col].sum()
    )

    percentage = (
        stable[col].mean() * 100
    )

    print(
        f"{col}: "
        f"{positives} positives "
        f"({percentage:.2f}%)"
    )


# 2. positive label count by coin

print(
    "\n=== POSITIVE LABEL COUNTS BY COIN ==="
)

print(
    stable.groupby("coin")[label_cols]
    .sum()
)


# 3. all deopeg obs
print(
    "\n=== NEXT-DAY DOWNSIDE DEPEG ROWS ==="
)

print(
    stable[
        stable["depeg_downside_1pct"] == 1
    ][
        [
            "date",
            "coin",
            "low",
            "next_day_low",
            "next_day_high",
            "depeg_today",
            "depeg_downside_1pct",
            "depeg_onset_next_day"
        ]
    ].to_string(index=False)
)


# CHECK 4: NEW DEPEG ONSET EVENTS ONLY
print(
    "\n=== NEW DEPEG ONSET ROWS ==="
)

print(
    stable[
        stable["depeg_onset_next_day"] == 1
    ][
        [
            "date",
            "coin",
            "low",
            "next_day_low",
            "next_day_high",
            "depeg_onset_next_day"
        ]
    ].to_string(index=False)
)