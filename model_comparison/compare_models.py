"""Does the regime flag add out-of-sample value? Logit vs random forest.

For each model family (logit from logit/logit_oos.py, RF from
random_forest/rf_oos.py) this script runs the identical design:
2022 monthly sequential validation + frozen-2022 forecast of 2023.

Outputs
  real_metrics.csv       M1/M2 metrics with the real regime flag
  bootstrap_deltas.csv   moving-block (by date) bootstrap CIs for M2 - M1
  placebo_deltas.csv     M2 - M1 when the flag is replaced by a placebo
                         (regime series circularly shifted along its calendar)
  placebo_summary.csv    real gain vs placebo distribution, with p-values
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "logit"))
sys.path.insert(0, str(ROOT / "random_forest"))

from logit_oos import load_inputs, TARGET, fit_predict as logit_fit_predict  # noqa: E402
from rf_oos import sequential_run, fit_predict as rf_fit_predict, RF_PARAMS  # noqa: E402
from rf_interaction_test import regime_calendar, placebo_shifts, shifted_stress  # noqa: E402


def logit_fitter(train, test, fold, phase):
    pred, log, coefs, _ = logit_fit_predict(train, test, fold, phase)
    return pred, log, coefs


FITTERS = {"Logit": logit_fitter, "Random forest": rf_fit_predict}
PHASES = {"2022 validation": "validation", "2023 frozen test": "oos"}
METRICS = ["average_precision", "roc_auc", "brier"]


def score(y, p):
    two = len(np.unique(y)) == 2
    return {
        "average_precision": average_precision_score(y, p) if two else np.nan,
        "roc_auc": roc_auc_score(y, p) if two else np.nan,
        "brier": brier_score_loss(y, p),
    }


def deltas(pred):
    y = pred[TARGET].to_numpy()
    m1, m2 = score(y, pred.p_m1_oos), score(y, pred.p_m2_oos)
    return {f"delta_{k}": m2[k] - m1[k] for k in METRICS}


def run_all(panel, fitter):
    validation, oos, _, _ = sequential_run(panel, fitter=fitter)
    return {"2022 validation": validation, "2023 frozen test": oos}


def block_bootstrap(pred, reps=2000, block=7, seed=0):
    """Moving-block bootstrap over dates; all coins on a date move together."""
    rng = np.random.default_rng(seed)
    days = np.sort(pred.date.unique())
    by_day = {d: g for d, g in pred.groupby("date")}
    n_blocks = int(np.ceil(len(days) / block))
    out = []
    for _ in range(reps):
        starts = rng.integers(0, max(1, len(days) - block + 1), n_blocks)
        picked = np.concatenate([days[s:s + block] for s in starts])[: len(days)]
        sample = pd.concat([by_day[d] for d in picked], ignore_index=True)
        if sample[TARGET].nunique() < 2:
            continue
        out.append(deltas(sample))
    return pd.DataFrame(out)


def main(out, reps):
    out.mkdir(parents=True, exist_ok=True)
    panel, _, _, _ = load_inputs(ROOT, lag=0)
    raw = regime_calendar(ROOT)
    shifts = placebo_shifts(len(raw))

    real_rows, boot_rows, placebo_rows = [], [], []

    for model, fitter in FITTERS.items():
        real = run_all(panel, fitter)
        for phase, pred in real.items():
            y = pred[TARGET].to_numpy()
            for spec in ["M1", "M2"]:
                for regime, g in [("all", pred)] + [(str(s), gg) for s, gg in pred.groupby("stress")]:
                    real_rows.append({
                        "model": model, "phase": phase, "spec": spec, "stress": regime,
                        "n": len(g), "positives": int(g[TARGET].sum()),
                        **score(g[TARGET].to_numpy(), g[f"p_{spec.lower()}_oos"].to_numpy()),
                    })
            b = block_bootstrap(pred, reps=reps)
            point = deltas(pred)
            for k in METRICS:
                col = f"delta_{k}"
                boot_rows.append({
                    "model": model, "phase": phase, "metric": k,
                    "delta_m2_minus_m1": point[col],
                    "ci_low": b[col].quantile(0.025), "ci_high": b[col].quantile(0.975),
                    "share_boot_m2_better": float((b[col] < 0).mean() if k == "brier" else (b[col] > 0).mean()),
                    "valid_reps": int(b[col].notna().sum()),
                })
            pred.to_csv(out / f"{model.lower().replace(' ', '_')}_{PHASES[phase]}_predictions.csv", index=False)

        for k in shifts:
            p = panel.copy()
            p["stress"] = shifted_stress(raw, k, p.regime_date)
            res = run_all(p, fitter)
            for phase, pred in res.items():
                placebo_rows.append({"model": model, "phase": phase, "shift_days": k,
                                     "stressed_share": float(pred.stress.mean()), **deltas(pred)})
            print(f"{model}: shift {k}d done")

    real_df = pd.DataFrame(real_rows)
    boot_df = pd.DataFrame(boot_rows)
    placebo_df = pd.DataFrame(placebo_rows)

    summary = []
    for (model, phase), g in placebo_df.groupby(["model", "phase"]):
        r = real_df[(real_df.model == model) & (real_df.phase == phase) & (real_df.stress == "all")]
        r = r.set_index("spec")
        for k in METRICS:
            real_delta = r.loc["M2", k] - r.loc["M1", k]
            col = g[f"delta_{k}"].dropna()
            # "better" = higher AP/AUC or lower Brier
            as_good = (col <= real_delta) if k == "brier" else (col >= real_delta)
            summary.append({
                "model": model, "phase": phase, "metric": k,
                "real_delta": real_delta, "placebo_mean": col.mean(),
                "placebo_q05": col.quantile(0.05), "placebo_q95": col.quantile(0.95),
                "placebo_p_value": float((1 + as_good.sum()) / (1 + len(col))),
                "n_placebos": len(col),
            })

    real_df.to_csv(out / "real_metrics.csv", index=False)
    boot_df.to_csv(out / "bootstrap_deltas.csv", index=False)
    placebo_df.to_csv(out / "placebo_deltas.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "placebo_summary.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "design": "identical panel/folds for both models; lag 0",
        "bootstrap": f"moving-block over dates, block=7 days, reps={reps}",
        "placebo": "regime series circularly shifted along its daily calendar; M1 unchanged",
        "placebo_shifts_days": shifts,
        "rf_params": RF_PARAMS,
        "logit": "LogisticRegression C=1, liblinear, standardised on training data",
    }, indent=2))
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "model_comparison/outputs")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args()
    main(args.out, args.reps)
