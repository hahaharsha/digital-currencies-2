"""Random-forest counterpart of logit/regime_lag_sensitivity.py.

Regime flag lagged 0, 1 or 3 calendar days on a common sample; on-chain
predictors stay at lag 0, so M1 must be identical across lags.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logit_oos import load_inputs, metrics, dates, unique, TARGET  # noqa: E402
from rf_oos import fit_predict, RF_PARAMS  # noqa: E402

LAGS = [0, 1, 3]


def run(root, out):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output folder is not empty; choose a fresh --out.")

    panel, _, _, paths = load_inputs(root, lag=0)
    raw = dates(pd.read_csv(root / "regime_classifier/outputs/market_regime.csv"), "regime")
    unique(raw, ["date"], "regime")
    if not raw.stress.dropna().isin([0, 1]).all():
        raise ValueError("Invalid raw regime flags.")

    # exact calendar-date merges, never row shifts: gaps stay missing.
    base = panel.drop(columns=["stress", "regime_date"]).copy()
    for lag in LAGS:
        r = raw[["date", "stress"]].copy()
        r[f"regime_source_date_lag{lag}"] = r.date
        r["date"] = r.date + pd.Timedelta(days=lag)
        r = r.rename(columns={"stress": f"stress_lag{lag}"})
        base = base.merge(r, on="date", how="left", validate="many_to_one")

    stress_cols = [f"stress_lag{lag}" for lag in LAGS]
    common = base.loc[base[stress_cols].notna().all(axis=1)].copy()
    if common.empty:
        raise ValueError("No common sample across regime lags.")
    coverage = base.groupby(base.date.dt.year).agg(rows_before_common_filter=("date", "size"))
    coverage["common_sample_rows"] = common.groupby(common.date.dt.year).size()

    predictions, diagnostics, validation_scores, importances = [], [], [], []

    for lag in LAGS:
        d = common.copy()
        d["stress"] = d[f"stress_lag{lag}"].astype(int)
        d["regime_date"] = d[f"regime_source_date_lag{lag}"]
        assert (d.date - d.regime_date).dt.days.eq(lag).all()
        assert d.feature_date.eq(d.date).all()

        train22 = d[d.date.dt.year.eq(2022) & d.event_date.lt("2023-01-01")]
        fold_predictions = []

        for begin in pd.date_range("2022-07-01", "2022-12-01", freq="MS"):
            train = train22[train22.event_date.lt(begin)]
            test = train22[train22.date.ge(begin) & train22.date.lt(begin + pd.offsets.MonthBegin(1))]
            if test.empty or train.date.nunique() < 90 or train[TARGET].nunique() < 2:
                diagnostics.append({"lag_days": lag, "phase": "2022_validation",
                                    "fold": begin.strftime("%Y-%m"), "status": "skipped",
                                    "train_n": len(train), "test_n": len(test)})
                continue
            pred, log, imp = fit_predict(train, test, begin.strftime("%Y-%m"), "2022_validation")
            pred["regime_lag_days"] = lag
            fold_predictions.append(pred)
            predictions.append(pred)
            diagnostics.append({"lag_days": lag, **log})
            importances.extend({"lag_days": lag, **c} for c in imp)

        if fold_predictions:
            score = metrics(pd.concat(fold_predictions, ignore_index=True))
            score["regime_lag_days"] = lag
            validation_scores.append(score)

        test23 = d[d.date.dt.year.eq(2023)]
        pred, log, imp = fit_predict(train22, test23, "frozen_2022", "2023_event_study")
        pred["regime_lag_days"] = lag
        predictions.append(pred)
        diagnostics.append({"lag_days": lag, **log})
        importances.extend({"lag_days": lag, **c} for c in imp)

    all_pred = pd.concat(predictions, ignore_index=True)

    # M1 never receives stress and the forest seed is fixed, so it must not move.
    check = all_pred.pivot(index=["phase", "coin", "date"], columns="regime_lag_days", values="p_m1_oos")
    if check.isna().any().any() or not np.allclose(
        check.to_numpy(), check.iloc[:, [0]].to_numpy(), atol=1e-12, rtol=0
    ):
        raise AssertionError("Baseline forecasts/sample changed across regime lags.")

    out.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out / "common_sample_coverage.csv")
    common.to_csv(out / "common_model_panel.csv", index=False)
    all_pred.to_csv(out / "all_lag_predictions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "fold_diagnostics.csv", index=False)
    pd.DataFrame(importances).to_csv(out / "feature_importances.csv", index=False)

    if validation_scores:
        scores = pd.concat(validation_scores, ignore_index=True)
        scores.to_csv(out / "2022_validation_metrics.csv", index=False)
        print("2022 validation: pooled metrics on identical observations")
        print(scores[(scores.fold == "pooled") & (scores.stress == "all")].to_string(index=False))

    for lag in LAGS:
        dest = out / f"lag_{lag}"
        dest.mkdir()
        p = all_pred[all_pred.regime_lag_days.eq(lag)]
        p[p.phase.eq("2022_validation")].to_csv(dest / "rf_validation_2022_predictions.csv", index=False)
        p[p.phase.eq("2023_event_study")].to_csv(dest / "rf_oos_predictions.csv", index=False)

    event = all_pred[all_pred.phase.eq("2023_event_study")
                     & all_pred.event_date.between("2023-03-01", "2023-03-20")
                     & all_pred.coin.isin(["USDC", "DAI"])].copy()
    event.to_csv(out / "SVB_probability_paths.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for coin, g in event.groupby("coin"):
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        baseline = g[g.regime_lag_days.eq(0)].sort_values("event_date")
        axes[0].plot(baseline.event_date, baseline.p_m1_oos, color="black", ls="--", label="M1 baseline")
        for lag, line in g.groupby("regime_lag_days"):
            line = line.sort_values("event_date")
            axes[0].plot(line.event_date, line.p_m2_oos, marker=".", label=f"M2: R(t-{lag})")
            axes[1].step(line.event_date, line.stress + int(lag), where="mid", label=f"Lag {lag}")
        hit = baseline[TARGET].eq(1)
        axes[0].scatter(baseline.loc[hit, "event_date"], np.full(hit.sum(), 1.02),
                        color="red", marker="v", label="Observed breach")
        axes[0].axvline(pd.Timestamp("2023-03-10"), color="gray", ls=":", label="10 March reference")
        axes[0].set(title=f"{coin}: RF regime-only lag sensitivity", ylabel="Predicted probability", ylim=(0, 1.08))
        axes[0].legend(fontsize=8)
        axes[1].set(ylabel="Flag + offset", xlabel="Target event date (t+1)")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"{coin}_SVB_lag_sensitivity.png", dpi=160)
        plt.close(fig)

    manifest = {"regime_lags_calendar_days": LAGS, "onchain_lag_days": 0,
                "common_sample_across_lags": True, "2023_models_frozen_from_2022": True,
                "validation_start": "2022-07-01", "minimum_training_dates": 90,
                "rf_params": RF_PARAMS, "lag_selection_using_2023": False,
                "point_in_time_publication_verified": False,
                "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(coverage.to_string())
    print("PASS: M1 forecasts identical across all regime lags.")
    print("Outputs:", out.resolve())
    return all_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    run(args.root, args.out or args.root / "random_forest/outputs/regime_lag_sensitivity")
