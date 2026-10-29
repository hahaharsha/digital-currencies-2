"""M2-score variant: replace the 0/1 stress flag with stress_score (0-3 votes).

Same panel, folds, frozen-2022 test, bootstrap and placebo design as
compare_models.py; only M2's extra column changes. Logit standardises the
on-chain features on training data and enters stress_score unscaled (as the
flag is); the RF uses raw inputs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "logit"))
sys.path.insert(0, str(ROOT / "random_forest"))
sys.path.insert(0, str(ROOT / "model_comparison"))

from logit_oos import load_inputs, dates, FEATURES, TARGET  # noqa: E402
from rf_oos import sequential_run, make_forest  # noqa: E402
from rf_interaction_test import placebo_shifts  # noqa: E402
from compare_models import block_bootstrap, deltas, score, METRICS  # noqa: E402

OUT = ROOT / "model_comparison/outputs_score"


def make_fitter(kind):
    def fit(train, test, fold, phase):
        if not train["event_date"].max() < test["date"].min():
            raise AssertionError("Training outcomes overlap the prediction period.")
        pred = test[["date", "event_date", "coin", TARGET, "stress", "stress_score"]].copy()
        pred["fold"], pred["phase"] = fold, phase
        pred["train_end"] = train["event_date"].max()
        for i, extra in [(1, []), (2, ["stress_score"])]:
            if kind == "logit":
                sc = StandardScaler().fit(train[FEATURES])
                xt = pd.DataFrame(sc.transform(train[FEATURES]), columns=FEATURES, index=train.index)
                xv = pd.DataFrame(sc.transform(test[FEATURES]), columns=FEATURES, index=test.index)
                for c in extra:
                    xt[c], xv[c] = train[c], test[c]
                model = LogisticRegression(C=1.0, solver="liblinear", max_iter=2000, random_state=0)
            else:
                xt, xv = train[FEATURES + extra], test[FEATURES + extra]
                model = make_forest()
            model.fit(xt, train[TARGET])
            pred[f"p_m{i}_oos"] = model.predict_proba(xv)[:, 1]
        return pred, {"phase": phase, "fold": fold}, []
    return fit


FITTERS = {"Logit": make_fitter("logit"), "Random forest": make_fitter("rf")}


def run_all(panel, fitter):
    v, o, _, _ = sequential_run(panel, fitter=fitter)
    return {"2022 validation": v, "2023 frozen test": o}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel, _, _, _ = load_inputs(ROOT, lag=0)
    raw = dates(pd.read_csv(ROOT / "regime_classifier/outputs/market_regime.csv"), "regime")
    raw = raw.sort_values("date").reset_index(drop=True)
    score_by_date = raw.set_index("date")["stress_score"]
    panel["stress_score"] = score_by_date.loc[panel.regime_date].to_numpy().astype(float)
    pos = pd.Series(np.arange(len(raw)), index=raw.date)

    rows, placebo = [], []
    for model, fitter in FITTERS.items():
        for phase, pred in run_all(panel, fitter).items():
            y = pred[TARGET].to_numpy()
            m1, m2 = score(y, pred.p_m1_oos), score(y, pred.p_m2_oos)
            b = block_bootstrap(pred)
            for k in METRICS:
                rows.append({"model": model, "phase": phase, "metric": k, "M1": m1[k], "M2_score": m2[k],
                             "delta": m2[k] - m1[k],
                             "ci_low": b[f"delta_{k}"].quantile(0.025),
                             "ci_high": b[f"delta_{k}"].quantile(0.975)})
            pred.to_csv(OUT / f"{model.lower().replace(' ', '_')}_{phase[:4]}_predictions.csv", index=False)
        for k in placebo_shifts(len(raw)):
            p = panel.copy()
            idx = (pos.loc[p.regime_date].to_numpy() + k) % len(raw)
            p["stress_score"] = raw.stress_score.to_numpy()[idx].astype(float)
            for phase, pred in run_all(p, fitter).items():
                placebo.append({"model": model, "phase": phase, "shift_days": k, **deltas(pred)})
        print(f"{model} done", flush=True)

    res, placebo = pd.DataFrame(rows), pd.DataFrame(placebo)
    pvals = []
    for r in res.itertuples():
        col = placebo[(placebo.model == r.model) & (placebo.phase == r.phase)][f"delta_{r.metric}"].dropna()
        better = (col <= r.delta) if r.metric == "brier" else (col >= r.delta)
        pvals.append((1 + better.sum()) / (1 + len(col)))
    res["placebo_p"] = pvals

    flag = pd.read_csv(ROOT / "model_comparison/outputs/real_metrics.csv")
    flag = flag[(flag.stress == "all") & (flag.spec == "M2")]
    res["M2_flag"] = [flag[(flag.model == r.model) & (flag.phase == r.phase)][r.metric].iloc[0]
                      for r in res.itertuples()]
    fp = pd.read_csv(ROOT / "model_comparison/outputs/placebo_summary.csv")
    res["flag_placebo_p"] = [fp[(fp.model == r.model) & (fp.phase == r.phase) & (fp.metric == r.metric)]
                             .placebo_p_value.iloc[0] for r in res.itertuples()]
    res = res[["model", "phase", "metric", "M1", "M2_flag", "M2_score", "delta", "ci_low", "ci_high",
               "placebo_p", "flag_placebo_p"]]
    res.to_csv(OUT / "score_vs_flag.csv", index=False)
    placebo.to_csv(OUT / "placebo_deltas.csv", index=False)
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
