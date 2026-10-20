"""Random-forest counterpart of logit/logit_oos.py.

Same panel, folds, timing guards and M2 decomposition as the logit script;
only the classifier changes. Data loading and metrics are imported from the
logit script so both models see identical observations.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logit"))

from logit_oos import (  # noqa: E402
    FEATURES,
    TARGET,
    load_inputs,
    metrics,
    plot_decomposition,
)


# fixed specification; no tuning (mirrors the logit's fixed C=1).
RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "bootstrap": True,
    "class_weight": None,
    "random_state": 0,
    "n_jobs": -1,
}


def make_forest(**overrides):
    return RandomForestClassifier(**{**RF_PARAMS, **overrides})


def design(frame):
    # trees are invariant to monotone rescaling, so no scaler is fitted.
    a = frame[FEATURES].copy()          # M1: on-chain predictors
    b = a.assign(stress=frame["stress"])  # M2: + regime flag
    return [a, b]


def fit_predict(train, test, fold, phase, **rf_overrides):
    if train.empty or test.empty:
        raise ValueError("Empty training or prediction panel.")

    if not train["event_date"].max() < test["date"].min():
        raise AssertionError("Training outcomes overlap the prediction period.")

    if not train["date"].max() < test["date"].min():
        raise AssertionError("Training predictor dates overlap the prediction period.")

    if train[TARGET].nunique() != 2:
        raise ValueError("Training sample has only one class.")

    xt = design(train)
    xv = design(test)

    pred = test[
        ["date", "event_date", "coin", TARGET, "stress", "feature_date", "regime_date"]
    ].copy()
    pred["fold"] = fold
    pred["test_month"] = pred["date"].dt.strftime("%Y-%m")
    pred["phase"] = phase
    pred["train_end"] = train["event_date"].max()
    pred["train_predictor_end"] = train["date"].max()

    importances = []

    for i, (a, b) in enumerate(zip(xt, xv), start=1):
        model = make_forest(**rf_overrides)
        model.fit(a, train[TARGET])

        pred[f"p_m{i}_oos"] = model.predict_proba(b)[:, 1]

        if i == 2:
            # same fitted forest and on-chain inputs; only the flag changes.
            stress_zero = b.assign(stress=0)
            stress_one = b.assign(stress=1)
            pred["p_m2_stress0"] = model.predict_proba(stress_zero)[:, 1]
            pred["p_m2_stress1"] = model.predict_proba(stress_one)[:, 1]
            pred["m2_regime_contribution"] = pred["p_m2_oos"] - pred["p_m2_stress0"]
            pred["m2_stress_contrast"] = pred["p_m2_stress1"] - pred["p_m2_stress0"]

            reconstructed = np.where(
                test["stress"].eq(1), pred["p_m2_stress1"], pred["p_m2_stress0"]
            )
            if not np.allclose(pred["p_m2_oos"], reconstructed, atol=1e-12, rtol=0):
                raise AssertionError("M2 decomposition does not reconstruct predictions.")
            if not np.allclose(
                pred.loc[pred["stress"].eq(0), "m2_regime_contribution"], 0, atol=1e-12, rtol=0
            ):
                raise AssertionError("Regime contribution must be zero when stress=0.")

        # impurity (MDI) importances stand in for logit coefficients.
        importances.extend(
            {
                "phase": phase,
                "fold": fold,
                "model": f"M{i}",
                "feature": feature,
                "mdi_importance": float(value),
            }
            for feature, value in zip(a.columns, model.feature_importances_)
        )

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
        "train_stressed_days": train.loc[train["stress"].eq(1), "date"].nunique(),
        "test_stressed_days": test.loc[test["stress"].eq(1), "date"].nunique(),
        "train_regime_classes": train["stress"].nunique(),
    }

    return pred, log, importances


def sequential_run(panel, validation_start="2022-07-01", min_train_days=90, fitter=None):
    """2022 monthly sequential validation + frozen-2022 2023 forecast.

    Returns (validation_predictions, oos_predictions, logs, importances).
    `fitter` defaults to the RF fit_predict; it must return (pred, log, extra).
    """
    fitter = fitter or fit_predict
    start = pd.Timestamp(validation_start)
    if start.year != 2022 or start.day != 1:
        raise ValueError("--validation-start must be a 2022 month start.")

    eligible22 = panel.loc[
        panel["event_date"].lt("2023-01-01") & panel["date"].dt.year.eq(2022)
    ].copy()

    predictions22, logs, extras = [], [], []

    for begin in pd.date_range(start, "2022-12-01", freq="MS"):
        end = begin + pd.offsets.MonthBegin(1)
        train = eligible22.loc[eligible22["event_date"].lt(begin)]
        test = eligible22.loc[eligible22["date"].ge(begin) & eligible22["date"].lt(end)]
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

        pred, log, extra = fitter(train, test, fold, "2022_sequential_validation")
        predictions22.append(pred)
        logs.append(log)
        extras.extend(extra)

    validation = pd.concat(predictions22, ignore_index=True) if predictions22 else pd.DataFrame()

    test = panel.loc[panel["date"].dt.year.eq(2023)]
    if eligible22.empty or test.empty:
        raise ValueError("Need eligible 2022 training and 2023 prediction rows.")

    oos, log, extra = fitter(eligible22, test, "frozen_2022", "2023_event_study")
    if not pd.to_datetime(oos["train_end"]).dt.year.eq(2022).all():
        raise AssertionError("Frozen model includes outcomes outside 2022.")
    logs.append(log)
    extras.extend(extra)

    return validation, oos, logs, extras


def run(root, out, lag=0, validation_start="2022-07-01", min_train_days=90):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Output directory is not empty. Choose a new --out to preserve prior results."
        )

    panel, coverage, invalid, paths = load_inputs(root, lag)
    validation, oos, logs, importances = sequential_run(panel, validation_start, min_train_days)

    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "model_panel.csv", index=False)
    coverage.to_csv(out / "input_coverage.csv")
    invalid.to_csv(out / "invalid_raw_features.csv", index=False)

    if not validation.empty:
        validation.to_csv(out / "rf_validation_2022_predictions.csv", index=False)
        metrics(validation).to_csv(out / "rf_validation_2022_metrics.csv", index=False)

    oos.to_csv(out / "rf_oos_predictions.csv", index=False)

    oos.groupby(["test_month", "stress"]).agg(
        coin_days=("date", "size"),
        market_days=("date", "nunique"),
        breaches=(TARGET, "sum"),
    ).to_csv(out / "2023_event_study_coverage.csv")

    pd.DataFrame(logs).to_csv(out / "fold_diagnostics.csv", index=False)
    pd.DataFrame(importances).to_csv(out / "feature_importances.csv", index=False)

    manifest = {
        "design": "2022 monthly sequential validation; frozen-2022 2023 forecast",
        "availability_lag_calendar_days": lag,
        "models": ["M1: on-chain", "M2: on-chain + stress"],
        "classifier": "sklearn RandomForestClassifier",
        "rf_params": RF_PARAMS,
        "scaling": "None (trees are invariant to monotone transforms)",
        "tuning": "None; fixed specification",
        "validation_start": validation_start,
        "minimum_training_dates": min_train_days,
        "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "validation_rows": len(validation),
        "test_rows": len(oos),
        "decomposition": {
            "method": "Same fitted M2 forest and on-chain inputs; set stress to zero or one without refitting.",
            "contribution": "p_m2_oos minus p_m2_stress0",
            "interpretation": "Model-based probability decomposition, not a causal effect.",
        },
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

    plot_decomposition(oos, out)

    print(coverage.to_string())
    print(f"2022 validation: {len(validation)} rows. Frozen-2022 test: {len(oos)} rows.")
    if not validation.empty:
        m = metrics(validation)
        print(m[(m.fold == "pooled") & (m.stress == "all")].to_string(index=False))
    print(f"Outputs: {out.resolve()}")
    return validation, oos


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--availability-lag-days", type=int, default=0)
    p.add_argument("--validation-start", default="2022-07-01")
    p.add_argument("--min-train-days", type=int, default=90)
    args = p.parse_args()

    if args.availability_lag_days < 0:
        p.error("Availability lag must be nonnegative.")
    if args.min_train_days < 1:
        p.error("Minimum training dates must be positive.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = args.out or (args.root / "random_forest" / "outputs" / f"rf_decomposition_{timestamp}")
    run(args.root, dest, args.availability_lag_days, args.validation_start, args.min_train_days)
