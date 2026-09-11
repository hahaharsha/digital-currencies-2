import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


# paths and load data

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


# keep required columns

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


# merge model data
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


# drop missing data and do sample diagnostics

df = (
    df
    .dropna(
        subset=(
            MODEL_FEATURES
            + [
                TARGET,
                "stress",
            ]
        )
    )
    .copy()
)


print("\n=== MODEL SAMPLE ===")

print(
    f"Observations: {len(df):,}"
)

print(
    "Date range:",
    df["date"].min().date(),
    "to",
    df["date"].max().date(),
)


print("\n=== DEPEG RATE BY REGIME ===")

print(
    df
    .groupby("stress")[TARGET]
    .agg(
        events="sum",
        observations="size",
        rate="mean",
    )
)


# standardise base onchain vars
# imp: standardise X first, THEN construct X * stress. 
# this makes interaction coefficients easier to compare.

scaler = StandardScaler()

X_standardised = pd.DataFrame(
    scaler.fit_transform(
        df[
            MODEL_FEATURES
        ]
    ),
    columns=MODEL_FEATURES,
    index=df.index,
)


# add regime

X_standardised[
    "stress"
] = (
    df["stress"]
    .astype(int)
)

# create interaction term
# for each X: interaction = standardised X * stress
# if stress = 0: interaction = 0
# if stress = 1: interaction = X

INTERACTION_FEATURES = []

for feature in MODEL_FEATURES:

    interaction_name = (
        f"{feature}_x_stress"
    )

    X_standardised[
        interaction_name
    ] = (
        X_standardised[feature]
        *
        X_standardised["stress"]
    )

    INTERACTION_FEATURES.append(
        interaction_name
    )


# define m1, m2, m3

X1 = (
    X_standardised[
        MODEL_FEATURES
    ]
    .copy()
)


X2 = (
    X_standardised[
        MODEL_FEATURES
        + ["stress"]
    ]
    .copy()
)


X3 = (
    X_standardised[
        MODEL_FEATURES
        + ["stress"]
        + INTERACTION_FEATURES
    ]
    .copy()
)


y = (
    df[TARGET]
    .astype(int)
    .copy()
)


# fit models

def fit_logit(X, y):

    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2000,
    )

    model.fit(
        X,
        y,
    )

    return model


model_1 = fit_logit(
    X1,
    y,
)

model_2 = fit_logit(
    X2,
    y,
)

model_3 = fit_logit(
    X3,
    y,
)


# predicted probabilities

p1 = (
    model_1
    .predict_proba(
        X1
    )[:, 1]
)

p2 = (
    model_2
    .predict_proba(
        X2
    )[:, 1]
)

p3 = (
    model_3
    .predict_proba(
        X3
    )[:, 1]
)


# evaluation framework
def evaluate(
    y_true,
    probabilities,
):

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

        evaluate(
            y,
            p3,
        ),
    ],
    index=[
        "M1: On-chain",
        "M2: On-chain + Regime",
        "M3: On-chain + Regime + Interactions",
    ],
)


print(
    "\n=== IN-SAMPLE MODEL COMPARISON ==="
)

print(
    results.round(4)
)


# m3 coefficients 

coef_m3 = pd.Series(
    model_3.coef_[0],
    index=X3.columns,
)


print(
    "\n=== M3 COEFFICIENTS ==="
)

print(
    coef_m3
    .sort_values(
        ascending=False
    )
    .round(4)
)


# normal vs stressed slopes
# normal regime:beta_k
# stressed regime: beta_k + delta_k

regime_effects = []


for feature in MODEL_FEATURES:

    interaction = (
        f"{feature}_x_stress"
    )

    beta_normal = (
        coef_m3[
            feature
        ]
    )

    delta = (
        coef_m3[
            interaction
        ]
    )

    beta_stressed = (
        beta_normal
        +
        delta
    )

    regime_effects.append(
        {
            "feature":
                feature,

            "normal_coefficient":
                beta_normal,

            "interaction_change":
                delta,

            "stressed_coefficient":
                beta_stressed,
        }
    )


regime_effects = pd.DataFrame(
    regime_effects
)


print(
    "\n=== NORMAL VS STRESSED COEFFICIENTS ==="
)

print(
    regime_effects
    .set_index("feature")
    .round(4)
)


# change in performance between m2 --> m3

delta_pr = (
    results.loc[
        "M3: On-chain + Regime + Interactions",
        "PR-AUC"
    ]
    -
    results.loc[
        "M2: On-chain + Regime",
        "PR-AUC"
    ]
)


delta_roc = (
    results.loc[
        "M3: On-chain + Regime + Interactions",
        "ROC-AUC"
    ]
    -
    results.loc[
        "M2: On-chain + Regime",
        "ROC-AUC"
    ]
)


delta_brier = (
    results.loc[
        "M3: On-chain + Regime + Interactions",
        "Brier"
    ]
    -
    results.loc[
        "M2: On-chain + Regime",
        "Brier"
    ]
)


print(
    "\n=== CHANGE FROM M2 TO M3 ==="
)

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


# interpretation!!

# M1: Y(i,t+1) ~ X(i,t)
# M2: Y(i,t+1) ~ X(i,t) + R(t)
# M3: Y(i,t+1) ~ X(i,t) + R(t) + X(i,t) * R(t)

# For each on-chain predictor:
#   beta_k = coefficient under NORMAL market conditions
#   delta_k = change in that coefficient under STRESSED market conditions
#   beta_k + delta_k = coefficient under STRESSED conditions
# Therefore the interaction terms are key quantities for the regime-dependence hypothesis.

# A non-zero interaction coefficient suggests that the relationship between that on-chain predictor and future depegging differs between normal and stressed regimes.
# However, sklearn coefficients alone do NOT provide formal statistical significance tests.

# For formal inference, the next step is to estimate the same M3 specification using statsmodels and jointly test:
#   H0: delta_1 = delta_2 = ... = delta_K = 0
# This is the joint test of whether the on-chain predictive relationships differ across market regimes.
# Current model comparison is IN-SAMPLE only. Temporal OOS evaluation must be carried out separately.