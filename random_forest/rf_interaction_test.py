"""Random-forest counterpart of logit/interaction_test.py.

The logit M3 adds explicit X-by-stress terms and Wald-tests them. A forest
has no coefficients but can learn interactions on its own, so here we ask the
same question of the fitted M2 forest (on-chain + stress, 2022 sample):

  Does the effect of each on-chain feature depend on the regime?

For feature j we compute partial dependence on a quantile grid with stress
forced to 0 and to 1. Delta_j(g) = PD_j(g, stress=1) - PD_j(g, stress=0).
Without an interaction Delta_j is flat (a pure level shift), so
I_j = SD over the grid of Delta_j(g) measures regime-dependence of feature j;
the joint statistic is sum_j I_j.

Inference uses placebo regimes: the real stress series is circularly shifted
along the regime calendar (keeping its spell lengths and persistence), the
forest is refitted, and the statistics are recomputed. p = (1 + #placebo >=
real) / (1 + #placebo). This is a descriptive in-sample test, not evidence of
better out-of-sample prediction.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logit"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from logit_oos import load_inputs, dates, unique, FEATURES, TARGET  # noqa: E402
from rf_oos import make_forest, RF_PARAMS  # noqa: E402

GRID_QUANTILES = np.linspace(0.05, 0.95, 19)
MIN_SHIFT_DAYS = 30
SHIFT_STEP_DAYS = 7


def regime_calendar(root):
    raw = dates(pd.read_csv(root / "regime_classifier/outputs/market_regime.csv"), "regime")
    unique(raw, ["date"], "regime")
    raw = raw.sort_values("date")[["date", "stress"]].dropna().reset_index(drop=True)
    expected = pd.date_range(raw.date.min(), raw.date.max(), freq="D")
    if len(expected) != len(raw) or not raw.date.eq(expected).all():
        raise ValueError("Regime calendar has gaps; circular shifts need a continuous daily series.")
    return raw


def placebo_shifts(n_days):
    return list(range(MIN_SHIFT_DAYS, n_days - MIN_SHIFT_DAYS + 1, SHIFT_STEP_DAYS))


def shifted_stress(raw, k, regime_dates):
    """Stress on regime date d replaced by stress k days later (circularly)."""
    pos = pd.Series(np.arange(len(raw)), index=raw.date)
    idx = (pos.loc[regime_dates].to_numpy() + k) % len(raw)
    return raw.stress.to_numpy()[idx].astype(int)


def partial_dependence(model, X):
    """Per-feature PD curves with stress forced to 0 and 1."""
    rows = []
    n = len(X)
    base = X.to_numpy(dtype=float)
    s_col = X.columns.get_loc("stress")
    for f in FEATURES:
        j = X.columns.get_loc(f)
        grid = np.unique(np.quantile(X[f], GRID_QUANTILES))
        blocks = []
        for s in (0, 1):
            for g in grid:
                b = base.copy()
                b[:, j] = g
                b[:, s_col] = s
                blocks.append(b)
        p = model.predict_proba(pd.DataFrame(np.vstack(blocks), columns=X.columns))[:, 1]
        p = p.reshape(2, len(grid), n).mean(axis=2)
        for gi, g in enumerate(grid):
            rows.append({"feature": f, "grid_value": g, "pd_normal": p[0, gi],
                         "pd_stressed": p[1, gi], "delta": p[1, gi] - p[0, gi]})
    return pd.DataFrame(rows)


def statistics(model, X, y):
    pdp = partial_dependence(model, X)
    by_feature = pdp.groupby("feature")["delta"].std(ddof=0).reindex(FEATURES)
    p0 = model.predict_proba(X.assign(stress=0))[:, 1]
    p1 = model.predict_proba(X.assign(stress=1))[:, 1]
    oob = model.oob_decision_function_[:, 1]
    stats = {
        "joint_interaction": float(by_feature.sum()),
        "mean_stress_contrast": float(np.mean(p1 - p0)),
        "oob_average_precision_m2": float(average_precision_score(y, oob)),
    }
    stats.update({f"I_{f}": float(v) for f, v in by_feature.items()})
    return stats, pdp


def run(root, out, sample="2022", lag=0):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose an empty output directory to preserve prior results.")
    if lag != 0:
        raise ValueError("Placebo shifts are defined for the lag-0 regime alignment only.")

    panel, _, _, paths = load_inputs(root, lag)
    d = panel.copy()
    if sample == "2022":
        d = d[d.date.dt.year.eq(2022) & d.event_date.lt("2023-01-01")].copy()
    if d.empty or d.stress.nunique() < 2 or d[TARGET].nunique() < 2:
        raise ValueError("Need observations, both regimes, and both outcome classes.")

    diagnostic = d.groupby("stress").agg(coin_days=("date", "size"),
                                         date_clusters=("date", "nunique"),
                                         breaches=(TARGET, "sum"))
    print("Inference sample:", sample)
    print(diagnostic.to_string())

    y = d[TARGET].to_numpy()
    X = d[FEATURES].assign(stress=d.stress.to_numpy())

    m1 = make_forest(oob_score=True).fit(d[FEATURES], y)
    oob_ap_m1 = float(average_precision_score(y, m1.oob_decision_function_[:, 1]))

    model = make_forest(oob_score=True).fit(X, y)
    real, pdp = statistics(model, X, y)
    real["oob_ap_gain"] = real["oob_average_precision_m2"] - oob_ap_m1

    raw = regime_calendar(root)
    shifts = placebo_shifts(len(raw))
    placebo = []
    for k in shifts:
        Xk = X.assign(stress=shifted_stress(raw, k, d.regime_date))
        if Xk.stress.nunique() < 2:
            continue
        mk = make_forest(oob_score=True).fit(Xk, y)
        s, _ = statistics(mk, Xk, y)
        s["oob_ap_gain"] = s["oob_average_precision_m2"] - oob_ap_m1
        s["shift_days"] = k
        s["placebo_stressed_share"] = float(Xk.stress.mean())
        placebo.append(s)
        print(f"shift {k:>3}d  joint={s['joint_interaction']:.4f}  oob_gain={s['oob_ap_gain']:+.3f}")
    placebo = pd.DataFrame(placebo)

    def p_value(col):
        return float((1 + (placebo[col] >= real[col]).sum()) / (1 + len(placebo)))

    tested = ["joint_interaction", "oob_ap_gain"] + [f"I_{f}" for f in FEATURES]
    summary = pd.DataFrame({
        "statistic": tested,
        "real": [real[c] for c in tested],
        "placebo_mean": [placebo[c].mean() for c in tested],
        "placebo_95pct": [placebo[c].quantile(0.95) for c in tested],
        "placebo_p_value": [p_value(c) for c in tested],
    })

    importances = pd.DataFrame({"feature": X.columns, "mdi_importance": model.feature_importances_})

    report = {
        "sample": sample,
        "hypothesis": "regime-dependence of on-chain feature effects is no larger than under placebo regimes",
        "statistic": "sum_j SD_grid[PD_j(g, stress=1) - PD_j(g, stress=0)]",
        "joint_interaction": real["joint_interaction"],
        "joint_placebo_p_value": p_value("joint_interaction"),
        "oob_ap_m1": oob_ap_m1,
        "oob_ap_m2": real["oob_average_precision_m2"],
        "oob_ap_gain_placebo_p_value": p_value("oob_ap_gain"),
        "mean_stress_contrast": real["mean_stress_contrast"],
        "n_placebos": len(placebo),
        "placebo_design": f"circular shifts of the daily regime series by {MIN_SHIFT_DAYS}.."
                          f"{len(raw) - MIN_SHIFT_DAYS} days in steps of {SHIFT_STEP_DAYS}",
        "observations": len(d),
        "date_clusters": int(d.date.nunique()),
        "rf_params": RF_PARAMS,
        "limitation": "In-sample/OOB statistics on 2022 only. Few stress spells and few breaches "
                      "make the placebo distribution coarse; placebo flags are not independent of each other.",
        "inputs_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }

    out.mkdir(parents=True, exist_ok=True)
    pdp.to_csv(out / "regime_specific_partial_dependence.csv", index=False)
    summary.to_csv(out / "interaction_summary.csv", index=False)
    placebo.to_csv(out / "placebo_statistics.csv", index=False)
    importances.to_csv(out / "m2_feature_importances.csv", index=False)
    diagnostic.to_csv(out / "inference_sample_by_regime.csv")
    (out / "interaction_test.json").write_text(json.dumps(report, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, f in zip(axes.flat, FEATURES):
        g = pdp[pdp.feature.eq(f)]
        ax.plot(g.grid_value, g.pd_normal, color="#1f77b4", label="stress = 0")
        ax.plot(g.grid_value, g.pd_stressed, color="#d62728", label="stress = 1")
        ax.set_title(f.replace("log1p_", ""), fontsize=9)
        ax.tick_params(labelsize=7)
    axes.flat[0].legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("RF M2 partial dependence by regime (2022 sample)")
    fig.tight_layout()
    fig.savefig(out / "regime_specific_partial_dependence.png", dpi=160)
    plt.close(fig)

    print(summary.round(4).to_string(index=False))
    print("Descriptive inference only; this is not evidence of improved OOS prediction.")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample", choices=["2022", "all"], default="2022")
    args = parser.parse_args()
    run(args.root, args.out or args.root / "random_forest/outputs/interaction_test", args.sample)
