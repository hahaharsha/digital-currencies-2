import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)
#set paths and load data

project_root = Path(__file__).resolve().parent

labels_path = (
    project_root / "dataset" / "depeg_labels.csv"
)

features_path = (
    project_root / "dataset" / "literature_features.csv"
)

regime_path = (
    project_root / "dataset" / "market_regime.csv"
)

labels = pd.read_csv(
    labels_path,
    parse_dates=["date"]
)

literature = pd.read_csv(
    features_path,
    parse_dates=["date"]
)

regime = pd.read_csv(
    regime_path,
    parse_dates=["date"]
)


# merge data -- starting from stablecoin-specific depeg labels
df = labels[
    [
        "date",
        "coin",
        "depeg_downside_1pct"
    ]
].copy()


# lit features are broad-market vars so merge them onto every stablecoin by date L
df = df.merge(
    literature,
    on="date",
    how="inner"
)


# add independently constructed market regime.
df = df.merge(
    regime[
        [
            "date",
            "stress"
        ]
    ],
    on="date",
    how="inner"
)


df = (
    df.sort_values(["date", "coin"])
    .reset_index(drop=True)
)


# select features

features = [
    "BTC_realized_volatility_30d_proxy",
    "BTC_volume_percent_change_30d",
    "BTC_percent_change_24h",
    "ETH_percent_change_24h",
    "ETH_volume_percent_change_30d"
]


# remove missing observations
df = df.dropna(
    subset=features + ["stress", "depeg_downside_1pct"]
).copy()


# sample diagnostics

print("\n=== MODEL SAMPLE ===")

print("Observations:", len(df))

print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max()
)


print("\n=== CLASS BALANCE ===")

print(
    df["depeg_downside_1pct"]
    .value_counts()
)


print("\n=== POSITIVE OBSERVATIONS ===")

print(
    df.loc[
        df["depeg_downside_1pct"] == 1,
        [
            "date",
            "coin",
            "stress"
        ]
    ].to_string(index=False)
)


# define x and y

X_base = df[features].copy()

X_regime = df[
    features + ["stress"]
].copy()

y = df[
    "depeg_downside_1pct"
].copy()


# standardise continuous features

scaler = StandardScaler()

X_base_scaled = X_base.copy()
X_regime_scaled = X_regime.copy()


X_base_scaled[features] = (
    scaler.fit_transform(
        X_base[features]
    )
)


X_regime_scaled[features] = (
    scaler.transform(
        X_regime[features]
    )
)


#regularised logit

benchmark_logit = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=1000
)

regime_logit = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=1000
)


benchmark_logit.fit(
    X_base_scaled,
    y
)

regime_logit.fit(
    X_regime_scaled,
    y
)


# fitted probabilities

p_base = benchmark_logit.predict_proba(
    X_base_scaled
)[:, 1]

p_regime = regime_logit.predict_proba(
    X_regime_scaled
)[:, 1]


# eval framework

def evaluate_model(y_true, probabilities):

    return {
        "PR-AUC": average_precision_score(
            y_true,
            probabilities
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            probabilities
        ),

        "Brier": brier_score_loss(
            y_true,
            probabilities
        )
    }


results = pd.DataFrame(
    [
        evaluate_model(y, p_base),
        evaluate_model(y, p_regime)
    ],
    index=[
        "Literature Benchmark",
        "Literature + Regime"
    ]
)


print(
    "\n=== PRELIMINARY IN-SAMPLE MODEL COMPARISON ==="
)

print(
    results.round(4)
)


# return coefficients 

benchmark_coef = pd.Series(
    benchmark_logit.coef_[0],
    index=X_base_scaled.columns
)

regime_coef = pd.Series(
    regime_logit.coef_[0],
    index=X_regime_scaled.columns
)


print(
    "\n=== BENCHMARK COEFFICIENTS ==="
)

print(
    benchmark_coef
    .sort_values(ascending=False)
    .round(4)
)


print(
    "\n=== BENCHMARK + REGIME COEFFICIENTS ==="
)

print(
    regime_coef
    .sort_values(ascending=False)
    .round(4)
)


# positive event predictions

df["p_benchmark"] = p_base
df["p_regime"] = p_regime


print(
    "\n=== PREDICTED RISK FOR DEPEG OBSERVATIONS ==="
)

print(
    df.loc[
        df["depeg_downside_1pct"] == 1,
        [
            "date",
            "coin",
            "stress",
            "p_benchmark",
            "p_regime"
        ]
    ].to_string(index=False)
)


# regime/depeg diagnostics

print(
    "\n=== DEPEG x REGIME CONTINGENCY TABLE ==="
)

print(
    pd.crosstab(
        df["stress"],
        df["depeg_downside_1pct"],
        margins=True
    )
)


print(
    "\n=== DEPEG RATE BY REGIME ==="
)

regime_rates = (
    df.groupby("stress")[
        "depeg_downside_1pct"
    ]
    .agg(["mean", "sum", "count"])
)

regime_rates["mean_pct"] = (
    regime_rates["mean"] * 100
)

print(regime_rates)