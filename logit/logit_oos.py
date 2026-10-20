import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)


# settings

RAW = [
    "n_transfers",
    "onchain_volume",
    "n_unique_senders",
    "n_unique_receivers",
    "minted",
    "burned",
    "volume_large_transfers",
]

FEATURES = ["log1p_" + c for c in RAW]
TARGET = "depeg_dynamic"


# input validation

def required(d, cols, name):
    missing = set(cols) - set(d.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def unique(d, keys, name):
    if d.duplicated(keys).any():
        raise ValueError(
            f"{name}: duplicate {keys}; resolve explicitly."
        )


def dates(d, name):
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"], errors="raise")

    if d["date"].isna().any() or d["date"].dt.tz is not None:
        raise ValueError(
            f"{name}: missing or timezone-aware dates; "
            "align daily boundaries explicitly."
        )

    if not d["date"].eq(d["date"].dt.normalize()).all():
        raise ValueError(
            f"{name}: expected daily dates at midnight."
        )

    return d


# load raw data

def load_year(path, year):
    d = pd.read_csv(path)
    required(d, ["coin", "date"] + RAW, path.name)
    d = dates(d, path.name)

    if not d["date"].dt.year.eq(year).all():
        raise ValueError(
            f"{path.name}: contains dates outside {year}."
        )

    if d["coin"].isna().any():
        raise ValueError(
            f"{path.name}: missing coin identifier."
        )

    d["coin"] = (
        d["coin"].astype(str).str.strip().str.upper()
    )

    unique(d, ["coin", "date"], path.name)

    invalid = []

    for raw, feature in zip(RAW, FEATURES):
        values = pd.to_numeric(d[raw], errors="raise")
        valid = np.isfinite(values) & values.ge(0)

        invalid.append({
            "file": path.name,
            "feature": raw,
            "invalid_rows": int((~valid).sum()),
        })

        # deterministic transformation.
        # no zero-filling, backward filling, interpolation, or full-sample imputation.
        d[feature] = np.log1p(values.where(valid))

    return d[["coin", "date"] + FEATURES].copy(), invalid


def load_inputs(root, lag):
    path22 = root / "dataset/onchain_2022_base.csv"

    if not path22.exists():
        path22 = root / "dataset/onchain_2022_base.csc"

    path23 = root / "dataset/onchain_2023_base.csv"

    x22, bad22 = load_year(path22, 2022)
    x23, bad23 = load_year(path23, 2023)

    label_path = root / "depeg_labeller/results/depeg_labels_dynamic.csv"
    regime_path = root / "regime_classifier/outputs/market_regime.csv"

    y = pd.read_csv(label_path)

    required(
        y,
        ["coin", "date", "event_date", TARGET],
        "labels",
    )

    y = dates(y, "labels")

    if y["coin"].isna().any():
        raise ValueError("Labels contain missing coin identifiers.")

    y["coin"] = (
        y["coin"].astype(str).str.strip().str.upper()
    )

    y["event_date"] = pd.to_datetime(
        y["event_date"], errors="raise"
    )

    if not (
        y["event_date"] - y["date"]
    ).eq(pd.Timedelta(days=1)).all():
        raise ValueError(
            "Labels must map predictor date t "
            "to event date t+1."
        )

    if not y[TARGET].dropna().isin([0, 1]).all():
        raise ValueError("Labels must be binary or missing.")

    unique(y, ["coin", "date"], "labels")

    r = pd.read_csv(regime_path)
    required(r, ["date", "stress"], "regime")
    r = dates(r, "regime")

    unique(r, ["date"], "regime")

    if not r["stress"].dropna().isin([0, 1]).all():
        raise ValueError("Stress must be binary or missing.")

    # concatenation is only for source-date lookup.
    # training/test membership is determined by actual dates.
    x = pd.concat([x22, x23], ignore_index=True)
    unique(x, ["coin", "date"], "on-chain source lookup")

    # cxact calendar-day lag: source data from day s become predictors on day s + lag.
    x["feature_date"] = x["date"]
    x["date"] = x["date"] + pd.Timedelta(days=lag)

    r = r[["date", "stress"]].copy()
    r["regime_date"] = r["date"]
    r["date"] = r["date"] + pd.Timedelta(days=lag)

    merged = (
        y[["coin", "date", "event_date", TARGET]]
        .merge(
            x,
            on=["coin", "date"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            r,
            on="date",
            how="left",
            validate="many_to_one",
        )
    )

    finite = np.isfinite(
        merged[FEATURES].to_numpy(dtype=float)
    ).all(axis=1)

    merged["eligible"] = (
        finite
        & merged[TARGET].notna()
        & merged["stress"].notna()
    )

    coverage = merged.groupby(
        [merged["date"].dt.year.rename("year"), "coin"]
    ).agg(
        label_rows=("date", "size"),
        model_rows=("eligible", "sum"),
    )

    panel = (
        merged.loc[merged["eligible"]]
        .copy()
        .sort_values(["date", "coin"])
    )

    panel[TARGET] = panel[TARGET].astype(int)
    panel["stress"] = panel["stress"].astype(int)

    if not panel["feature_date"].le(panel["date"]).all():
        raise AssertionError("Future-dated on-chain predictor.")

    if not panel["regime_date"].le(panel["date"]).all():
        raise AssertionError("Future-dated regime predictor.")

    paths = [
        path22,
        path23,
        label_path,
        regime_path,
    ]

    return (
        panel,
        coverage,
        pd.DataFrame(bad22 + bad23),
        paths,
    )


# model matrices

def design(frame, scaler, fit=False):
    if fit:
        values = scaler.fit_transform(frame[FEATURES])
    else:
        values = scaler.transform(frame[FEATURES])

    # m1: on-chain predictors.
    a = pd.DataFrame(
        values,
        columns=FEATURES,
        index=frame.index,
    )

    # m2: same predictors + regime intercept shift.
    b = a.assign(stress=frame["stress"])

    return [a, b]


# fit models and decompose m2

def fit_predict(train, test, fold, phase):
    if train.empty or test.empty:
        raise ValueError("Empty training or prediction panel.")

    # training OUTCOMES must finish before the first validation/test predictor day.
    if not train["event_date"].max() < test["date"].min():
        raise AssertionError(
            "Training outcomes overlap the prediction period."
        )

    if not train["date"].max() < test["date"].min():
        raise AssertionError(
            "Training predictor dates overlap the prediction period."
        )

    if train[TARGET].nunique() != 2:
        raise ValueError("Training sample has only one class.")

    # fit scaling parameters on training data only.
    scaler = StandardScaler()
    xt = design(train, scaler, fit=True)
    xv = design(test, scaler, fit=False)

    pred = test[
        [
            "date",
            "event_date",
            "coin",
            TARGET,
            "stress",
            "feature_date",
            "regime_date",
        ]
    ].copy()

    pred["fold"] = fold
    pred["test_month"] = pred["date"].dt.strftime("%Y-%m")
    pred["phase"] = phase

    # train_end is the latest training OUTCOME date.
    pred["train_end"] = train["event_date"].max()
    pred["train_predictor_end"] = train["date"].max()

    coefficients = []

    for i, (a, b) in enumerate(zip(xt, xv), start=1):
        model = LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=2000,
            random_state=0,
        )

        model.fit(a, train[TARGET])

        if model.n_iter_.max() >= model.max_iter:
            raise RuntimeError(
                f"M{i} did not converge; inspect training data."
            )

        pred[f"p_m{i}_oos"] = model.predict_proba(b)[:, 1]

        # m2 prob decomp
        # use exactly the same fitted model and on-chain inputs but change only the stress flag; do not refit.

        if i == 2:
            stress_zero = b.copy()
            stress_one = b.copy()

            stress_zero["stress"] = 0
            stress_one["stress"] = 1

            pred["p_m2_stress0"] = model.predict_proba(
                stress_zero
            )[:, 1]

            pred["p_m2_stress1"] = model.predict_proba(
                stress_one
            )[:, 1]

            # contribution of the observed flag relative to the same M2 model with stress forced to zero.
            pred["m2_regime_contribution"] = (
                pred["p_m2_oos"] - pred["p_m2_stress0"]
            )

            # flag=1 versus flag=0 contrast for every observation.
            pred["m2_stress_contrast"] = (
                pred["p_m2_stress1"] - pred["p_m2_stress0"]
            )

            reconstructed = np.where(
                test["stress"].eq(1),
                pred["p_m2_stress1"],
                pred["p_m2_stress0"],
            )

            if not np.allclose(
                pred["p_m2_oos"],
                reconstructed,
                atol=1e-12,
                rtol=0,
            ):
                raise AssertionError(
                    "M2 decomposition does not reconstruct predictions."
                )

            if not np.allclose(
                pred.loc[
                    pred["stress"].eq(0),
                    "m2_regime_contribution",
                ],
                0,
                atol=1e-12,
                rtol=0,
            ):
                raise AssertionError(
                    "Regime contribution must be zero when stress=0."
                )

        # save intercept as well as slopes.
        coefficients.append({
            "phase": phase,
            "fold": fold,
            "model": f"M{i}",
            "feature": "intercept",
            "coefficient": float(model.intercept_[0]),
        })

        coefficients.extend([
            {
                "phase": phase,
                "fold": fold,
                "model": f"M{i}",
                "feature": feature,
                "coefficient": float(value),
            }
            for feature, value in zip(
                a.columns,
                model.coef_[0],
            )
        ])

    log = {
        "phase": phase,
        "fold": fold,
        "status": "fitted",
        "train_n": len(train),
        "train_first": str(train["date"].min().date()),
        "train_predictor_end": str(train["date"].max().date()),
        "train_outcome_end": str(train["event_date"].max().date()),
        "test_first": str(test["date"].min().date()),
        "test_last": str(test["date"].max().date()),
        "test_n": len(test),
        "train_breaches": int(train[TARGET].sum()),
        "test_breaches": int(test[TARGET].sum()),
        "train_stressed_days": train.loc[
            train["stress"].eq(1), "date"
        ].nunique(),
        "test_stressed_days": test.loc[
            test["stress"].eq(1), "date"
        ].nunique(),
        "train_regime_classes": train["stress"].nunique(),
    }

    scaling = {
        "phase": phase,
        "fold": fold,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": FEATURES,
    }

    return pred, log, coefficients, scaling


# validation metrics

def metrics(pred):
    if pred.empty:
        return pd.DataFrame()

    groups = [("pooled", "all", pred)]

    groups += [
        (str(fold), "all", group)
        for fold, group in pred.groupby("fold")
    ]

    groups += [
        ("pooled", str(stress), group)
        for stress, group in pred.groupby("stress")
    ]

    rows = []

    for fold, stress, group in groups:
        y = group[TARGET]
        two_classes = y.nunique() == 2

        for i in [1, 2]:
            probabilities = group[f"p_m{i}_oos"]

            rows.append({
                "fold": fold,
                "stress": stress,
                "model": f"M{i}",
                "n": len(group),
                "positives": int(y.sum()),
                "prevalence": y.mean(),
                "average_precision": (
                    average_precision_score(y, probabilities)
                    if two_classes else np.nan
                ),
                "roc_auc": (
                    roc_auc_score(y, probabilities)
                    if two_classes else np.nan
                ),
                "brier": brier_score_loss(y, probabilities),
            })

    return pd.DataFrame(rows)


# svb decomp plots

def plot_decomposition(oos, out):
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    event = oos.loc[
        oos["coin"].isin(["USDC", "DAI"])
        & oos["event_date"].between(
            "2023-03-01",
            "2023-03-20",
        )
    ].copy()

    event.to_csv(
        out / "SVB_decomposition_daily.csv",
        index=False,
    )

    for coin in ["USDC", "DAI"]:
        group = event.loc[
            event["coin"].eq(coin)
        ].sort_values("event_date")

        if group.empty:
            print(f"No SVB predictions available for {coin}.")
            continue

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1, 1]},
        )
        ax = axes[0]

        ax.plot(
            group["event_date"],
            group["p_m1_oos"],
            color="black",
            linestyle="--",
            label="M1 baseline",
        )

        ax.plot(
            group["event_date"],
            group["p_m2_oos"],
            color="blue",
            marker=".",
            label="M2 actual",
        )

        ax.plot(
            group["event_date"],
            group["p_m2_stress0"],
            color="orange",
            linestyle="--",
            label="Same M2, stress forced to zero",
        )

        ax.fill_between(
            group["event_date"],
            group["p_m2_stress0"],
            group["p_m2_oos"],
            color="blue",
            alpha=0.15,
            label="Contribution from observed stress flag",
        )

        breach = group[TARGET].eq(1)

        # outcome markers use axes coordinates so they do not force the probability scale to extend to one.
        ax.scatter(
            group.loc[breach, "event_date"],
            np.full(int(breach.sum()), 0.03),
            transform=ax.get_xaxis_transform(),
            color="red",
            marker="v",
            label="Observed breach",
            zorder=5,
        )

        ax.axvline(
            pd.Timestamp("2023-03-10"),
            color="gray",
            linestyle=":",
            label="10 March reference",
        )

        ax.set(
            title=f"{coin}: frozen M2 probability decomposition",
            ylabel="Predicted probability",
        )

        ax.legend(fontsize=8)

        # probability-point contribution.
        axes[1].bar(
            group["event_date"],
            100 * group["m2_regime_contribution"],
            color="steelblue",
        )

        axes[1].axhline(
            0,
            color="black",
            linewidth=0.7,
        )

        axes[1].set_ylabel(
            "Flag contribution\n(percentage points)"
        )

        # regime actually supplied to each forecast.
        axes[2].step(
            group["event_date"],
            group["stress"],
            where="mid",
            color="purple",
        )

        axes[2].set(
            yticks=[0, 1],
            yticklabels=["Normal", "Stressed"],
            ylim=(-0.1, 1.1),
            ylabel="Forecast flag",
            xlabel="Target event date (t+1)",
        )

        fig.tight_layout()

        fig.savefig(
            out / f"{coin}_M2_decomposition.png",
            dpi=160,
        )

        plt.close(fig)


# run evaluation

def run(
    root,
    out,
    lag=0,
    validation_start="2022-07-01",
    min_train_days=90,
):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Output directory is not empty. "
            "Choose a new --out to preserve prior results."
        )

    panel, coverage, invalid, paths = load_inputs(root, lag)

    predictions22 = []
    logs = []
    coefs = []
    scalers = []

    start = pd.Timestamp(validation_start)

    if start.year != 2022 or start.day != 1:
        raise ValueError(
            "--validation-start must be a 2022 month start."
        )

    eligible22 = panel.loc[
        panel["event_date"].lt("2023-01-01")
        & panel["date"].dt.year.eq(2022)
    ].copy()

    # 2022 sequential validation
    for begin in pd.date_range(
        start,
        "2022-12-01",
        freq="MS",
    ):
        end = begin + pd.offsets.MonthBegin(1)
        train = eligible22.loc[
            eligible22["event_date"].lt(begin)
        ].copy()

        test = eligible22.loc[
            eligible22["date"].ge(begin)
            & eligible22["date"].lt(end)
        ].copy()

        fold = begin.strftime("%Y-%m")

        reasons = []

        if test.empty:
            reasons.append("empty validation sample")

        if train["date"].nunique() < min_train_days:
            reasons.append("insufficient historical training dates")

        if train[TARGET].nunique() < 2:
            reasons.append("single-class training")

        if reasons:
            logs.append({
                "phase": "2022_sequential_validation",
                "fold": fold,
                "status": "skipped",
                "train_n": len(train),
                "test_n": len(test),
                "reason": "; ".join(reasons),
            })
            continue

        pred, log, coef, scale = fit_predict(
            train,
            test,
            fold,
            "2022_sequential_validation",
        )

        predictions22.append(pred)
        logs.append(log)
        coefs.extend(coef)
        scalers.append(scale)

    validation = (
        pd.concat(predictions22, ignore_index=True)
        if predictions22
        else pd.DataFrame()
    )

    # 2023 event study
    # freeze parameters/scaler from 2022.
    # no 2023 labels enter model fitting.

    train = eligible22.copy()

    test = panel.loc[
        panel["date"].dt.year.eq(2023)
    ].copy()

    if train.empty or test.empty:
        raise ValueError(
            "Need eligible 2022 training and 2023 prediction rows."
        )

    oos, log, coef, scale = fit_predict(
        train,
        test,
        "frozen_2022",
        "2023_event_study",
    )

    if not pd.to_datetime(
        oos["train_end"]
    ).dt.year.eq(2022).all():
        raise AssertionError(
            "Frozen model includes outcomes outside 2022."
        )

    logs.append(log)
    coefs.extend(coef)
    scalers.append(scale)

    # save outputs

    out.mkdir(parents=True, exist_ok=True)

    panel.to_csv(
        out / "model_panel.csv",
        index=False,
    )

    coverage.to_csv(
        out / "input_coverage.csv"
    )

    invalid.to_csv(
        out / "invalid_raw_features.csv",
        index=False,
    )

    if not validation.empty:
        validation.to_csv(
            out / "logit_validation_2022_predictions.csv",
            index=False,
        )

        metrics(validation).to_csv(
            out / "logit_validation_2022_metrics.csv",
            index=False,
        )

    oos.to_csv(
        out / "logit_oos_predictions.csv",
        index=False,
    )

    # descriptive 2023 coverage
    oos.groupby(["test_month", "stress"]).agg(
        coin_days=("date", "size"),
        market_days=("date", "nunique"),
        breaches=(TARGET, "sum"),
    ).to_csv(
        out / "2023_event_study_coverage.csv"
    )

    pd.DataFrame(logs).to_csv(
        out / "fold_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(coefs).to_csv(
        out / "coefficients.csv",
        index=False,
    )

    (out / "training_scalers.json").write_text(
        json.dumps(scalers, indent=2)
    )

    manifest = {
        "design": (
            "2022 monthly sequential validation; "
            "frozen-2022 2023 forecast"
        ),
        "availability_lag_calendar_days": lag,
        "forecast_cutoff_assumption": (
            "After completion of predictor day t; "
            "all used aggregates available."
        ),
        "historical_publication_and_revision_timing_verified": False,
        "models": [
            "M1: on-chain",
            "M2: on-chain + stress",
        ],
        "C": 1.0,
        "tuning": "None; fixed specification",
        "validation_start": validation_start,
        "minimum_training_dates": min_train_days,
        "files": {
            str(path): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in paths
        },
        "validation_rows": len(validation),
        "test_rows": len(oos),
        "decomposition": {
            "method": (
                "Same fitted M2 and on-chain inputs; "
                "set stress to zero or one without refitting."
            ),
            "contribution": (
                "p_m2_oos minus p_m2_stress0"
            ),
            "interpretation": (
                "Model-based probability decomposition, "
                "not a causal effect."
            ),
        },
    }

    (out / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    plot_decomposition(oos, out)

    print(coverage.to_string())

    print(
        f"2022 validation: {len(validation)} rows. "
        f"Frozen-2022 test: {len(oos)} rows."
    )

    print(f"Outputs: {out.resolve()}")

    print(
        "Saved USDC_M2_decomposition.png "
        "and DAI_M2_decomposition.png where data are available."
    )

    print(
        "Decomposition holds fitted M2 parameters and "
        "on-chain inputs fixed; only the stress flag changes."
    )

    print(
        "Publication-time assumptions and retrospective "
        "revisions remain unverified."
    )

    return validation, oos


# command

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    p.add_argument(
        "--out",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--availability-lag-days",
        type=int,
        default=0,
    )

    p.add_argument(
        "--validation-start",
        default="2022-07-01",
    )

    p.add_argument(
        "--min-train-days",
        type=int,
        default=90,
    )

    args = p.parse_args()

    if args.availability_lag_days < 0:
        p.error("Availability lag must be nonnegative.")

    if args.min_train_days < 1:
        p.error("Minimum training dates must be positive.")

    # each default run gets a fresh folder.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    dest = args.out or (
        args.root
        / "logit"
        / "outputs"
        / f"logit_decomposition_{timestamp}"
    )

    run(
        args.root,
        dest,
        args.availability_lag_days,
        args.validation_start,
        args.min_train_days,
    )