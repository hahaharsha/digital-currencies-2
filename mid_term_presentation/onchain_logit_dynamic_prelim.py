import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


# set paths and load data

PROJECT_ROOT = Path(__file__).resolve().parent

ONCHAIN_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "onchain_2022_2023_clean.csv"
)

LABELS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "depeg_labels_dynamic.csv"
)

REGIME_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "market_regime.csv"
)

onchain = pd.read_csv(
    ONCHAIN_PATH,
    parse_dates=["date"],
)

labels = pd.read_csv(
    LABELS_PATH,
    parse_dates=["date"],
)

regime = pd.read_csv(
    REGIME_PATH,
    parse_dates=["date"],
)


# settings

TARGET = "depeg_dynamic"

MODEL_FEATURES = [
    "log1p_n_transfers",
    "log1p_onchain_volume",
    "log1p_n_unique_senders",
    "log1p_n_unique_receivers",
    "log1p_minted",
    "log1p_burned",
    "log1p_volume_large_transfers",
    "signed_log_net_issuance",
]

# check and select required columns

def check_required_columns(df, required_columns, dataset_name):

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


check_required_columns(
    onchain,
    ["date", "coin"] + MODEL_FEATURES,
    "On-chain dataset",
)

check_required_columns(
    labels,
    ["date", "coin", TARGET],
    "Dynamic depeg labels",
)

check_required_columns(
    regime,
    ["date", "stress"],
    "Market regime dataset",
)


onchain = (
    onchain[
        ["date", "coin"] + MODEL_FEATURES
    ]
    .copy()
)

labels = (
    labels[
        [
            "date",
            "coin",
            TARGET,
        ]
    ]
    .copy()
)

regime = (
    regime[
        [
            "date",
            "stress",
        ]
    ]
    .copy()
)


# merge modelling data
# X(i,t): stablecoin-specific on-chain predictors
# R(t): broad crypto-market regime (takes either 0 or 1)
# Y(i,t+1): dynamic future depeg indicator

df = (
    onchain
    .merge(
        labels,
        on=[
            "date",
            "coin",
        ],
        how="inner",
    )
    .merge(
        regime,
        on="date",
        how="inner",
    )
    .sort_values(
        [
            "date",
            "coin",
        ]
    )
    .reset_index(drop=True)
)


# sample diagnostics - before and after dropping missing vals

print("\n=== MERGED MODEL SAMPLE ===")

print(
    f"Observations before dropping missing values: "
    f"{len(df):,}"
)

required_model_columns = (
    MODEL_FEATURES
    +
    [
        TARGET,
        "stress",
    ]
)

before_drop = len(df)

df = (
    df
    .dropna(
        subset=required_model_columns
    )
    .copy()
)

after_drop = len(df)


print(
    f"Observations after dropping missing values: "
    f"{after_drop:,}"
)

print(
    f"Observations dropped: "
    f"{before_drop - after_drop:,}"
)

print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date(),
)


print("\n=== OBSERVATIONS BY COIN ===")

print(
    df[
        "coin"
    ]
    .value_counts()
    .sort_index()
)


# class balance

print("\n=== DYNAMIC DEPEG CLASS BALANCE ===")

class_balance = (
    df[TARGET]
    .value_counts()
    .sort_index()
)

print(
    class_balance
)


positive_count = int(
    df[TARGET].sum()
)

positive_rate = (
    df[TARGET].mean()
)


print(
    f"\nPositive observations: "
    f"{positive_count:,}"
)

print(
    f"Positive rate: "
    f"{positive_rate:.4f}"
)

print(
    f"Positive rate (%): "
    f"{positive_rate * 100:.2f}%"
)


# dynamic depeg rate by coin

coin_rates = (
    df
    .groupby("coin")[TARGET]
    .agg(
        events="sum",
        observations="size",
        rate="mean",
    )
)

coin_rates["rate_pct"] = (
    coin_rates["rate"]
    * 100
)


print("\n=== DYNAMIC DEPEG RATE BY COIN ===")

print(
    coin_rates.round(4)
)


# dynamic depeg with regime awareness

print("\n=== DYNAMIC DEPEG x MARKET REGIME ===")

contingency = pd.crosstab(
    df["stress"],
    df[TARGET],
    margins=True,
)

print(
    contingency
)


# dynamic depeg rate with regime awareness

regime_rates = (
    df
    .groupby("stress")[TARGET]
    .agg(
        events="sum",
        observations="size",
        rate="mean",
    )
)

regime_rates["rate_pct"] = (
    regime_rates["rate"]
    * 100
)


print("\n=== DYNAMIC DEPEG RATE BY MARKET REGIME ===")

print(
    regime_rates.round(4)
)


# dynamic depeg rate by coin (with regime awareness)

coin_regime_rates = (
    df
    .groupby(
        [
            "coin",
            "stress",
        ]
    )[TARGET]
    .agg(
        events="sum",
        observations="size",
        rate="mean",
    )
)

coin_regime_rates["rate_pct"] = (
    coin_regime_rates["rate"]
    * 100
)


print(
    "\n=== DYNAMIC DEPEG RATE BY COIN + MARKET REGIME ==="
)

print(
    coin_regime_rates.round(4)
)


# define models
#
# M1: Dynamic depeg ~ on-chain predictors
#
# M2: Dynamic depeg ~ on-chain predictors + market regime


X1 = (
    df[
        MODEL_FEATURES
    ]
    .copy()
)

X2 = (
    df[
        MODEL_FEATURES
        + ["stress"]
    ]
    .copy()
)

y = (
    df[TARGET]
    .astype(int)
    .copy()
)


# standardise continuous onchain features
# note: stress remains binary and is not standardised.

scaler = StandardScaler()

X1_scaled = X1.copy()
X2_scaled = X2.copy()


X1_scaled[
    MODEL_FEATURES
] = scaler.fit_transform(
    X1[
        MODEL_FEATURES
    ]
)

X2_scaled[
    MODEL_FEATURES
] = scaler.transform(
    X2[
        MODEL_FEATURES
    ]
)


# fit logit -- in sample data only

model_1 = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=2000,
)

model_2 = LogisticRegression(
    C=1.0,
    solver="liblinear",
    max_iter=2000,
)


model_1.fit(
    X1_scaled,
    y,
)

model_2.fit(
    X2_scaled,
    y,
)


# predicted probabilities

p1 = (
    model_1
    .predict_proba(
        X1_scaled
    )[:, 1]
)

p2 = (
    model_2
    .predict_proba(
        X2_scaled
    )[:, 1]
)


# evaluation functions

def evaluate(y_true, probabilities):

    return {
        "PR-AUC":
            average_precision_score(
                y_true,
                probabilities,
            ),

        "ROC-AUC":
            roc_auc_score(
                y_true,
                probabilities,
            ),

        "Brier":
            brier_score_loss(
                y_true,
                probabilities,
            ),
    }


# model comparison

results = pd.DataFrame(
    [
        evaluate(
            y,
            p1,
        ),

        evaluate(
            y,
            p2,
        ),
    ],
    index=[
        "M1: On-chain Benchmark",
        "M2: On-chain + Regime",
    ],
)


print(
    "\n=== PRELIMINARY IN-SAMPLE MODEL COMPARISON ==="
)

print(
    results.round(4)
)


# changes after regime awareness

delta_pr = (
    results.loc[
        "M2: On-chain + Regime",
        "PR-AUC"
    ]
    -
    results.loc[
        "M1: On-chain Benchmark",
        "PR-AUC"
    ]
)

delta_roc = (
    results.loc[
        "M2: On-chain + Regime",
        "ROC-AUC"
    ]
    -
    results.loc[
        "M1: On-chain Benchmark",
        "ROC-AUC"
    ]
)

delta_brier = (
    results.loc[
        "M2: On-chain + Regime",
        "Brier"
    ]
    -
    results.loc[
        "M1: On-chain Benchmark",
        "Brier"
    ]
)


print("\n=== CHANGE AFTER ADDING REGIME ===")

print(
    f"Delta PR-AUC: "
    f"{delta_pr:.4f}"
)

print(
    f"Delta ROC-AUC: "
    f"{delta_roc:.4f}"
)

print(
    f"Delta Brier: "
    f"{delta_brier:.4f}"
)


# model coefficients

coef_m1 = pd.Series(
    model_1.coef_[0],
    index=X1_scaled.columns,
)

coef_m2 = pd.Series(
    model_2.coef_[0],
    index=X2_scaled.columns,
)


print("\n=== M1 COEFFICIENTS ===")

print(
    coef_m1
    .sort_values(
        ascending=False
    )
    .round(4)
)


print("\n=== M2 COEFFICIENTS ===")

print(
    coef_m2
    .sort_values(
        ascending=False
    )
    .round(4)
)


# add predicted probabilities to df

df["p_m1"] = p1
df["p_m2"] = p2


# observed and predicted risk by regime

risk_by_regime = (
    df
    .groupby("stress")
    .agg(
        observed_rate=(
            TARGET,
            "mean",
        ),

        m1_predicted_rate=(
            "p_m1",
            "mean",
        ),

        m2_predicted_rate=(
            "p_m2",
            "mean",
        ),

        events=(
            TARGET,
            "sum",
        ),

        observations=(
            TARGET,
            "size",
        ),
    )
)


print(
    "\n=== OBSERVED + PREDICTED RISK BY MARKET REGIME ==="
)

print(
    risk_by_regime.round(6)
)


# observed and predicted risk by regime by coin

risk_by_coin_regime = (
    df
    .groupby(
        [
            "coin",
            "stress",
        ]
    )
    .agg(
        observed_rate=(
            TARGET,
            "mean",
        ),

        m1_predicted_rate=(
            "p_m1",
            "mean",
        ),

        m2_predicted_rate=(
            "p_m2",
            "mean",
        ),

        events=(
            TARGET,
            "sum",
        ),

        observations=(
            TARGET,
            "size",
        ),
    )
)


print(
    "\n=== OBSERVED + PREDICTED RISK BY COIN + REGIME ==="
)

print(
    risk_by_coin_regime.round(6)
)


# interpretation

# M1: Dynamic depeg ~ on-chain signals
# M2: Dynamic depeg ~ on-chain signals + market regime

# M1 provides the baseline relationship between stablecoin-specific on-chain activity at time t and dynamic depeg outcomes at t+1.
# M2 adds the independently constructed broad crypto-market regime indicator R_t.
# The M1 -> M2 comparison therefore evaluates whether market regime provides incremental predictive information beyond stablecoin-specific on-chain activity.

# Do note that these models are currently evaluated in-sample
# As such, they do not yet establish:
#       1. whether on-chain predictor relationships differ between normal and stressed regimes; or
#       2.whether the models generalise out-of-sample.
# Regime dependence will require interaction terms: Y(i,t+1) ~ X(i,t) + R(t) + X(i,t) * R(t)
# Out-of-sample warning performance will require a time-ordered / walk-forward evaluation.