import pandas as pd
import numpy as np
from pathlib import Path


# paths

project_root = Path(__file__).resolve().parent.parent

prices_path = project_root / "dataset" / "prices.csv"
sentiment_path = project_root / "dataset" / "sentiment.csv"
output_path = (
    project_root
    / "processed_data"
    / "market_regime.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)


# helper functions
def calculate_rsi(series, window=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_adx(df, window=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # True range
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ],
        axis=1
    ).max(axis=1)

    atr = tr.rolling(window).mean()

    plus_di = 100 * (
        plus_dm.rolling(window).mean() / atr
    )

    minus_di = 100 * (
        minus_dm.rolling(window).mean() / atr
    )

    dx = (
        100
        * (plus_di - minus_di).abs()
        / (plus_di + minus_di)
    )

    adx = dx.rolling(window).mean()

    return adx


def prepare_coin(prices, coin_name):
    coin = (
        prices[prices["coin"] == coin_name]
        .sort_values("date")
        .copy()
    )

    # Daily return
    coin["return"] = coin["close"].pct_change()

    # Moving averages
    coin["sma20"] = coin["close"].rolling(20).mean()
    coin["sma50"] = coin["close"].rolling(50).mean()

    # RSI
    coin["rsi14"] = calculate_rsi(
        coin["close"],
        window=14
    )

    # ADX
    coin["adx14"] = calculate_adx(
        coin,
        window=14
    )

    # Optional rolling volatility for EDA
    coin["volatility_20"] = (
        coin["return"]
        .rolling(20)
        .std()
    )

    # Bearish SMA signal
    coin["sma_bearish"] = (
        (coin["close"] < coin["sma50"])
        & (coin["sma20"] < coin["sma50"])
    ).astype(int)

    # Bearish RSI signal
    coin["rsi_bearish"] = (
        coin["rsi14"] < 50
    ).astype(int)

    # Keep only the columns we need
    coin = coin[
        [
            "date",
            "close",
            "return",
            "sma20",
            "sma50",
            "rsi14",
            "adx14",
            "volatility_20",
            "sma_bearish",
            "rsi_bearish"
        ]
    ].copy()

    # rename for clarity
    coin = coin.rename(
        columns={
            "close": f"{coin_name.lower()}_close",
            "return": f"{coin_name.lower()}_return",
            "sma20": f"{coin_name.lower()}_sma20",
            "sma50": f"{coin_name.lower()}_sma50",
            "rsi14": f"{coin_name.lower()}_rsi14",
            "adx14": f"{coin_name.lower()}_adx14",
            "volatility_20": f"{coin_name.lower()}_volatility20",
            "sma_bearish": f"{coin_name.lower()}_sma_bearish",
            "rsi_bearish": f"{coin_name.lower()}_rsi_bearish"
        }
    )

    return coin


# load data

prices = pd.read_csv(
    prices_path,
    parse_dates=["date"]
)

sentiment = pd.read_csv(
    sentiment_path,
    parse_dates=["date"]
)

print("Prices loaded:", prices.shape)
print("Sentiment loaded:", sentiment.shape)


# prepare btc and eth then merge data

btc = prepare_coin(prices, "BTC")
eth = prepare_coin(prices, "ETH")


market = btc.merge(
    eth,
    on="date",
    how="inner"
)


market = market.merge(
    sentiment[
        [
            "date",
            "fear_greed",
            "fear_greed_label"
        ]
    ],
    on="date",
    how="left"
)


# f&g signal

# initial rule: f&g < 40 = stressed sentiment

market["fear_signal"] = (
    market["fear_greed"] < 40
).astype(int)


# market stress components

# BTC contributes one vote if both its SMA and RSI are bearish
market["btc_bearish"] = (
    (
        market["btc_sma_bearish"]
        + market["btc_rsi_bearish"]
    ) >= 2
).astype(int)


# ETH contributes one vote if both its SMA and RSI are bearish
market["eth_bearish"] = (
    (
        market["eth_sma_bearish"]
        + market["eth_rsi_bearish"]
    ) >= 2
).astype(int)


# Stress score ranges from 0 to 3:
# BTC bearish + ETH bearish + Fear & Greed stress

market["stress_score"] = (
    market["btc_bearish"]
    + market["eth_bearish"]
    + market["fear_signal"]
)


# binary regime

# Majority rule: at least 2 out of 3 stress components agree

market["stress"] = (
    market["stress_score"] >= 2
).astype(int)

market["regime"] = np.where(
    market["stress"] == 1,
    "stressed",
    "normal"
)


# drop initial warm up -  SMA50 needs 50 observations.


required_cols = [
    "btc_sma50",
    "btc_rsi14",
    "btc_adx14",
    "eth_sma50",
    "eth_rsi14",
    "eth_adx14",
    "fear_greed"
]

market = market.dropna(
    subset=required_cols
).copy()


# save outputs

market.to_csv(
    output_path,
    index=False
)

print("\nSaved regime data to:")
print(output_path)


# checks

print("\nRegime counts:")
print(market["regime"].value_counts())

print("\nRegime percentages:")
print(
    market["regime"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nMay 2022:")
print(
    market[
        (market["date"] >= "2022-05-01")
        & (market["date"] <= "2022-05-20")
    ][
        [
            "date",
            "btc_close",
            "eth_close",
            "fear_greed",
            "stress_score",
            "regime"
        ]
    ]
)

print("\nMarch 2023:")
print(
    market[
        (market["date"] >= "2023-03-01")
        & (market["date"] <= "2023-03-20")
    ][
        [
            "date",
            "btc_close",
            "eth_close",
            "fear_greed",
            "stress_score",
            "regime"
        ]
    ]
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

cols = [
    "date",
    "btc_close",
    "btc_sma_bearish",
    "btc_rsi_bearish",
    "btc_adx14",
    "eth_close",
    "eth_sma_bearish",
    "eth_rsi_bearish",
    "eth_adx14",
    "fear_greed",
    "fear_signal",
    "btc_bearish",
    "eth_bearish",
    "stress_score",
    "regime"
]

print("\n=== MAY 2022 DIAGNOSTIC ===")
print(
    market[
        (market["date"] >= "2022-05-01")
        & (market["date"] <= "2022-05-15")
    ][cols].to_string(index=False)
)

print("\n=== MARCH 2023 DIAGNOSTIC ===")
print(
    market[
        (market["date"] >= "2023-03-01")
        & (market["date"] <= "2023-03-20")
    ][cols].to_string(index=False)
)