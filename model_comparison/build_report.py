"""Build the logit-vs-random-forest regime report as a PDF.

Reads only saved outputs:
  logit/outputs/{logit_decomposition, interaction_weekly_initial, regime_lag_sensitivity}
  random_forest/outputs/{rf_decomposition, regime_lag_sensitivity}
  model_comparison/outputs/   (from compare_models.py)
"""

import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / "model_comparison/outputs"
LOGIT = ROOT / "logit/outputs"
RF = ROOT / "random_forest/outputs"
TARGET = "depeg_dynamic"
# Reports are written outside the repo
REPORT_DIR = Path.home() / "Downloads" / "stablecoin_regime_report"

# categorical slots (reference palette): model identity follows the entity
LOGIT_C, RF_C = "#2a78d6", "#eb6834"
MODEL_C = {"Logit": LOGIT_C, "Random forest": RF_C}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
A4 = (8.27, 11.69)

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.titleweight": "bold", "axes.titlesize": 10,
    "legend.frameon": False, "lines.linewidth": 2,
})


# ---------------------------------------------------------------- helpers

def page(pdf, title, blocks, figsize=A4):
    """Text page. blocks: list of (kind, content) with kind in h, p, b (bullet)."""
    fig = plt.figure(figsize=figsize)
    y = 0.95
    fig.text(0.08, y, title, fontsize=15, fontweight="bold", color=INK, va="top")
    y -= 0.045
    for kind, content in blocks:
        need = 0.06 if kind == "h" else 0.0185 * len(textwrap.wrap(content, 92)) + 0.01
        if y - need < 0.05:  # continue on a new page
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=figsize)
            y = 0.95
            fig.text(0.08, y, title + " (cont.)", fontsize=15, fontweight="bold", color=INK, va="top")
            y -= 0.045
        if kind == "h":
            y -= 0.012
            fig.text(0.08, y, content, fontsize=11, fontweight="bold", color=INK, va="top")
            y -= 0.03
            continue
        width = 92 if kind == "p" else 88
        lines = textwrap.wrap(content, width)
        x = 0.08 if kind == "p" else 0.11
        if kind == "b":
            fig.text(0.085, y, "•", fontsize=9.5, color=INK, va="top")
        for line in lines:
            fig.text(x, y, line, fontsize=9.5, color=INK, va="top")
            y -= 0.0185
        y -= 0.008
    pdf.savefig(fig)
    plt.close(fig)


def table(ax, df, title=None, col_widths=None, fontsize=8, highlight_rows=()):
    ax.axis("off")
    t = ax.table(cellText=df.values, colLabels=df.columns, loc="upper center",
                 cellLoc="center", colLoc="center", colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.35)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_facecolor("#f1f0ec")
            cell.set_text_props(fontweight="bold", color=INK)
        elif r - 1 in highlight_rows:
            cell.set_facecolor("#fbf6ee")
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=4)
    return t


def fmt(x, d=3):
    return "" if pd.isna(x) else f"{x:.{d}f}"


def score(y, p):
    two = len(np.unique(y)) == 2
    return (average_precision_score(y, p) if two else np.nan,
            roc_auc_score(y, p) if two else np.nan,
            brier_score_loss(y, p))


# ---------------------------------------------------------------- load

real = pd.read_csv(CMP / "real_metrics.csv")
boot = pd.read_csv(CMP / "bootstrap_deltas.csv")
plac = pd.read_csv(CMP / "placebo_deltas.csv")
plac_sum = pd.read_csv(CMP / "placebo_summary.csv")
preds = {
    ("Logit", "2022 validation"): pd.read_csv(CMP / "logit_validation_predictions.csv", parse_dates=["date", "event_date"]),
    ("Logit", "2023 frozen test"): pd.read_csv(CMP / "logit_oos_predictions.csv", parse_dates=["date", "event_date"]),
    ("Random forest", "2022 validation"): pd.read_csv(CMP / "random_forest_validation_predictions.csv", parse_dates=["date", "event_date"]),
    ("Random forest", "2023 frozen test"): pd.read_csv(CMP / "random_forest_oos_predictions.csv", parse_dates=["date", "event_date"]),
}
# Logit M3 from logit/interaction_test.py: within-coin scaling, coin effects, week-clustered SEs
INTERACTION = LOGIT / "interaction_weekly_initial"
_w = json.loads((INTERACTION / "wald_test.json").read_text())
wald = {"wald_chi_square": _w["wald_chi_square"]["statistic"], "df": _w["wald_chi_square"]["df"],
        "p_value": _w["wald_chi_square"]["p_value"], "observations": _w["observations"],
        "weekly_clusters": _w["weekly_clusters"]}
slopes = pd.read_csv(INTERACTION / "regime_specific_slopes.csv").rename(columns={
    "normal_week_clustered_se": "normal_se", "stressed_week_clustered_se": "stressed_se"})
rf_imp = pd.read_csv(RF / "rf_decomposition/feature_importances.csv")
logit_coef = pd.read_csv(LOGIT / "logit_decomposition/coefficients.csv")
lag_preds = {
    "Logit": pd.read_csv(LOGIT / "regime_lag_sensitivity/all_lag_predictions.csv"),
    "Random forest": pd.read_csv(RF / "regime_lag_sensitivity/all_lag_predictions.csv"),
}


def rm(model, phase, spec, stress="all", col="average_precision"):
    r = real[(real.model == model) & (real.phase == phase) & (real.spec == spec) & (real.stress == stress)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def bt(model, phase, metric):
    return boot[(boot.model == model) & (boot.phase == phase) & (boot.metric == metric)].iloc[0]


def ps(model, phase, metric):
    return plac_sum[(plac_sum.model == model) & (plac_sum.phase == phase) & (plac_sum.metric == metric)].iloc[0]


MODELS = ["Logit", "Random forest"]
PHASES = ["2022 validation", "2023 frozen test"]


def build(out_pdf, findings):
    with PdfPages(out_pdf) as pdf:
        # ---------------------------------------------------------- 1 summary
        page(pdf, "Does the regime classifier add value?", findings)

        # ---------------------------------------------------------- 2 design
        v = preds[("Logit", "2022 validation")]
        o = preds[("Logit", "2023 frozen test")]
        page(pdf, "Setup", [
            ("h", "Question"),
            ("p", "Two nested specifications are compared within each model family. M1 uses seven "
                  "log1p on-chain features (transfers, volume, unique senders/receivers, minted, burned, "
                  "large-transfer volume) for coin c on day t. M2 adds the binary market-stress flag from "
                  "regime_classifier for day t. The target is depeg_dynamic on day t+1. The regime "
                  "classifier 'adds value' if M2 forecasts breaches better than M1 out of sample."),
            ("h", "Models"),
            ("b", "Logit: sklearn LogisticRegression, C = 1 (L2), liblinear, features standardised on "
                  "training data only. Identical to logit/logit_oos.py (re-run; results match the saved run)."),
            ("b", "Random forest: 500 trees, max_features = sqrt, min_samples_leaf = 5, bootstrap, "
                  "no class weights, random_state = 0, no scaling. Fixed in advance, no tuning, mirroring the "
                  "logit's fixed C. Code: random_forest/rf_oos.py."),
            ("h", "Evaluation design (identical for both models)"),
            ("b", f"Panel: {v.coin.nunique()} coins (DAI, USDC, USDT); 2022 predictor days Apr to Nov, "
                  f"2023 predictor days Feb to Apr."),
            ("b", f"2022 sequential validation: monthly folds Jul to Nov 2022, each trained only on outcomes "
                  f"before the fold starts (expanding window, at least 90 training dates). {len(v)} forecasts, "
                  f"{int(v[TARGET].sum())} breaches, all in July."),
            ("b", f"2023 frozen test: models trained on all of 2022 and frozen; no 2023 labels are used. "
                  f"{len(o)} forecasts, {int(o[TARGET].sum())} breaches, but only "
                  f"{o.loc[o.stress.eq(1), 'date'].nunique()} stressed days (all in the SVB week)."),
            ("b", "Metrics: average precision (AP, main metric for rare events), ROC AUC, Brier score "
                  "(lower is better). M2 minus M1 is the value added by the flag."),
            ("h", "How 'value added' is tested"),
            ("b", "Block bootstrap: dates are resampled in 7-day blocks (coins on a date stay together) "
                  "to give a 95% interval for M2 minus M1."),
            ("b", "Placebo regimes: the real stress series is circularly shifted along its own calendar by "
                  "30 to 646 days (89 shifts, step 7). A placebo keeps the flag's share of stressed days and spell "
                  "lengths but breaks its timing. Each placebo goes through the full pipeline. If the real flag "
                  "carries information, its M2 gain should beat most placebo gains. p = share of placebos "
                  "at least as good."),
            ("b", "Interaction test: does the effect of on-chain activity differ by regime? Logit M3 with "
                  "seven feature-by-stress terms, coin fixed effects and a week-clustered Wald test "
                  "(logit/interaction_test.py)."),
        ])

        # ---------------------------------------------------------- 3 headline metrics table
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.96, "Out-of-sample metrics: M1 (on-chain) vs M2 (+ regime flag)",
                 fontsize=13, fontweight="bold", va="top")
        rows = []
        for phase in PHASES:
            for model in MODELS:
                for spec in ["M1", "M2"]:
                    r = real[(real.model == model) & (real.phase == phase) & (real.spec == spec) & (real.stress == "all")].iloc[0]
                    rows.append([phase, model, spec, int(r.n), int(r.positives),
                                 fmt(r.average_precision), fmt(r.roc_auc), fmt(r.brier, 4)])
        ax = fig.add_axes([0.04, 0.60, 0.92, 0.32])
        table(ax, pd.DataFrame(rows, columns=["Phase", "Model", "Spec", "n", "Breaches", "AP", "ROC AUC", "Brier"]),
              highlight_rows=[1, 3, 5, 7], col_widths=[0.2, 0.17, 0.07, 0.07, 0.1, 0.1, 0.1, 0.1])

        rows = []
        for phase in PHASES:
            for model in MODELS:
                for metric, label in [("average_precision", "AP"), ("roc_auc", "AUC"), ("brier", "Brier")]:
                    b = bt(model, phase, metric)
                    p = ps(model, phase, metric)
                    rows.append([phase, model, label, f"{b.delta_m2_minus_m1:+.3f}" if metric != "brier" else f"{b.delta_m2_minus_m1:+.4f}",
                                 f"[{b.ci_low:+.3f}, {b.ci_high:+.3f}]" if metric != "brier" else f"[{b.ci_low:+.4f}, {b.ci_high:+.4f}]",
                                 f"{b.share_boot_m2_better:.0%}", f"{p.placebo_p_value:.2f}"])
        ax = fig.add_axes([0.04, 0.10, 0.92, 0.44])
        table(ax, pd.DataFrame(rows, columns=["Phase", "Model", "Metric", "M2 - M1", "95% bootstrap CI",
                                              "M2 better", "Placebo p"]),
              title="Value added by the regime flag (M2 minus M1). Brier: negative = M2 better.",
              col_widths=[0.18, 0.16, 0.08, 0.1, 0.22, 0.11, 0.1])
        fig.text(0.06, 0.07, textwrap.fill(
            "M2 better: share of bootstrap draws in which M2 beats M1. "
            "Placebo p: share of 89 placebo flags whose M2 gain is at least as good as the real flag's. "
            "Small p means the real regime timing matters; p near 0.5 or above means a random flag with the "
            "same persistence does as well. The 2022 validation has only 8 breaches (all July), so its "
            "intervals are wide.", 120), fontsize=8, color=INK2, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 4 AP bar chart
        fig, axes = plt.subplots(1, 2, figsize=(A4[0], 4.6))
        for ax, phase in zip(axes, PHASES):
            x = np.arange(2)
            for k, model in enumerate(MODELS):
                vals = [rm(model, phase, s) for s in ["M1", "M2"]]
                bars = ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=MODEL_C[model],
                              edgecolor="white", linewidth=2, label=model)
                for bx, vv in zip(bars, vals):
                    ax.text(bx.get_x() + bx.get_width() / 2, vv + 0.01, f"{vv:.2f}", ha="center",
                            fontsize=8, color=INK)
            ax.set_xticks(x, ["M1: on-chain", "M2: + regime"])
            ax.set_ylim(0, 1)
            ax.set_title(phase)
            ax.set_ylabel("Average precision")
            ax.grid(axis="x", visible=False)
        prev = preds[("Logit", "2023 frozen test")][TARGET].mean()
        axes[1].axhline(prev, color=MUTED, ls=":", lw=1)
        axes[1].text(1.45, prev + 0.01, "base rate", fontsize=7, color=INK2, ha="right")
        axes[0].legend(loc="upper left", fontsize=8)
        fig.suptitle("Average precision by model and specification", fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 5 placebo histograms
        fig, axes = plt.subplots(2, 2, figsize=(A4[0], 7.2), sharex="col")
        for i, phase in enumerate(PHASES):
            for j, model in enumerate(MODELS):
                ax = axes[j, i]
                g = plac[(plac.model == model) & (plac.phase == phase)]["delta_average_precision"].dropna()
                real_d = rm(model, phase, "M2") - rm(model, phase, "M1")
                ax.hist(g, bins=20, color="#c9c8c3", edgecolor="white", linewidth=1)
                ax.axvline(real_d, color=MODEL_C[model], lw=2.5)
                ax.axvline(0, color=MUTED, lw=0.8, ls=":")
                p = ps(model, phase, "average_precision").placebo_p_value
                ax.text(0.98, 0.95, f"real flag: {real_d:+.3f}\nplacebo p = {p:.2f}",
                        transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)
                ax.set_title(f"{model}: {phase}", fontsize=9)
                ax.set_xlabel("AP gain, M2 - M1")
                ax.set_ylabel("Placebo flags")
        fig.suptitle("Real regime flag vs 89 placebo flags (grey)", fontweight="bold")
        fig.text(0.5, 0.005, "Placebo flags: the real stress series shifted 30 to 646 days along its calendar.",
                 ha="center", fontsize=8, color=INK2)
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 6 per-fold + by regime
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.96, "Where the gains come from", fontsize=13, fontweight="bold", va="top")
        rows = []
        for model in MODELS:
            pv = preds[(model, "2022 validation")]
            for fold, g in pv.groupby("fold"):
                y = g[TARGET].to_numpy()
                a1, _, b1 = score(y, g.p_m1_oos)
                a2, _, b2 = score(y, g.p_m2_oos)
                rows.append([model, fold, len(g), int(y.sum()), int(g.stress.sum()), fmt(a1), fmt(a2), fmt(b1, 4), fmt(b2, 4)])
        ax = fig.add_axes([0.04, 0.62, 0.92, 0.30])
        table(ax, pd.DataFrame(rows, columns=["Model", "Fold", "n", "Breaches", "Stressed",
                                              "AP M1", "AP M2", "Brier M1", "Brier M2"]),
              title="2022 validation by fold (AP undefined without breaches)",
              col_widths=[0.15, 0.1, 0.07, 0.1, 0.1, 0.1, 0.1, 0.11, 0.11], fontsize=7.5)
        rows = []
        for phase in PHASES:
            for model in MODELS:
                for s in ["0", "1"]:
                    r1 = real[(real.model == model) & (real.phase == phase) & (real.spec == "M1") & (real.stress == s)]
                    r2 = real[(real.model == model) & (real.phase == phase) & (real.spec == "M2") & (real.stress == s)]
                    if r1.empty:
                        continue
                    r1, r2 = r1.iloc[0], r2.iloc[0]
                    rows.append([phase, model, "stressed" if s == "1" else "normal", int(r1.n), int(r1.positives),
                                 fmt(r1.average_precision), fmt(r2.average_precision), fmt(r1.brier, 4), fmt(r2.brier, 4)])
        ax = fig.add_axes([0.04, 0.16, 0.92, 0.40])
        table(ax, pd.DataFrame(rows, columns=["Phase", "Model", "Regime", "n", "Breaches",
                                              "AP M1", "AP M2", "Brier M1", "Brier M2"]),
              title="Split by the regime on the forecast day",
              col_widths=[0.15, 0.14, 0.09, 0.06, 0.09, 0.09, 0.09, 0.1, 0.1], fontsize=7.5)
        fig.text(0.06, 0.13, textwrap.fill(
            "Within a regime the M2 flag is constant, so M2 can only re-rank observations through the "
            "intercept shift (logit) or through flag-dependent splits (RF). Gains therefore show up mostly "
            "in Brier (calibration) and in the pooled ranking across regimes.", 120), fontsize=8, color=INK2, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 7 SVB paths
        fig, axes = plt.subplots(2, 1, figsize=(A4[0], 8.4), sharex=True)
        for ax, coin in zip(axes, ["USDC", "DAI"]):
            for model in MODELS:
                p = preds[(model, "2023 frozen test")]
                g = p[p.coin.eq(coin) & p.event_date.between("2023-03-01", "2023-03-20")].sort_values("event_date")
                ax.plot(g.event_date, g.p_m1_oos, color=MODEL_C[model], ls="--", lw=1.5, label=f"{model} M1")
                ax.plot(g.event_date, g.p_m2_oos, color=MODEL_C[model], lw=2, marker="o", ms=4, label=f"{model} M2")
            hit = g[g[TARGET].eq(1)]
            ax.scatter(hit.event_date, np.full(len(hit), 1.03), marker="v", color=INK, s=28,
                       label="Observed breach", zorder=5, clip_on=False)
            st = g[g.stress.eq(1)]
            for d in st.event_date:
                ax.axvspan(d - pd.Timedelta(hours=12), d + pd.Timedelta(hours=12), color="#ece9f7", zorder=0)
            ax.set_ylim(0, 1.06)
            ax.set_ylabel("Predicted breach probability")
            ax.set_title(f"{coin}, SVB window (shaded = stress flag on forecast day)")
        axes[0].legend(fontsize=7.5, ncol=3, loc="upper left")
        axes[1].set_xlabel("Target event date (t+1)")
        fig.autofmt_xdate()
        fig.suptitle("Frozen-2022 forecasts through the SVB episode", fontweight="bold")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 8 decomposition
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.96, "How much of M2's forecast comes from the flag?", fontsize=13, fontweight="bold", va="top")
        rows = []
        for model in MODELS:
            for phase in PHASES:
                p = preds[(model, phase)]
                contrast = p.p_m2_stress1 - p.p_m2_stress0
                contrib = p.p_m2_oos - p.p_m2_stress0
                svb = p[p.coin.isin(["USDC", "DAI"]) & p.event_date.between("2023-03-01", "2023-03-20") & p.stress.eq(1)]
                rows.append([model, phase, f"{100 * contrast.mean():+.1f}", f"{100 * contrast.median():+.1f}",
                             f"{100 * contrib[p.stress.eq(1)].mean():+.1f}" if p.stress.eq(1).any() else "",
                             f"{100 * (svb.p_m2_oos - svb.p_m2_stress0).mean():+.1f}" if len(svb) else ""])
        ax = fig.add_axes([0.04, 0.72, 0.92, 0.18])
        table(ax, pd.DataFrame(rows, columns=["Model", "Phase", "Mean contrast", "Median contrast",
                                              "Contrib. (stressed)", "Contrib. (SVB, USDC/DAI)"]),
              col_widths=[0.14, 0.16, 0.15, 0.15, 0.17, 0.21], fontsize=7.5)
        ax.text(0, -0.02, "All values in percentage points.", transform=ax.transAxes, fontsize=7, color=INK2)
        fig.text(0.06, 0.70, textwrap.fill(
            "Contrast = P(breach | stress = 1) - P(breach | stress = 0) from the same fitted M2 with on-chain inputs "
            "held fixed (pp = percentage points). Contribution = actual M2 - M2 with stress forced to 0; it is zero "
            "on normal days by construction. Model-based decomposition, not a causal effect.", 120),
            fontsize=8, color=INK2, va="top")
        ax = fig.add_axes([0.1, 0.10, 0.84, 0.5])
        for model in MODELS:
            p = preds[(model, "2023 frozen test")]
            g = p[p.coin.isin(["USDC", "DAI"]) & p.event_date.between("2023-03-01", "2023-03-20")]
            g = g.groupby("event_date")["m2_stress_contrast"].mean()
            ax.plot(g.index, 100 * g.values, color=MODEL_C[model], marker="o", ms=4, label=model)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("Flag contrast, pp (USDC/DAI mean)")
        ax.set_title("What the frozen M2 would add if stress were on, SVB window")
        ax.legend(fontsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_ha("right")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 9 lag sensitivity
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.96, "Regime lag sensitivity (flag from t, t-1 or t-3)", fontsize=13, fontweight="bold", va="top")
        rows = []
        for model in MODELS:
            lp = lag_preds[model]
            for phase_key, phase in [("2022_validation", "2022 validation"), ("2023_event_study", "2023 frozen test")]:
                for lag, g in lp[lp.phase.eq(phase_key)].groupby("regime_lag_days"):
                    y = g[TARGET].to_numpy()
                    a1, u1, b1 = score(y, g.p_m1_oos)
                    a2, u2, b2 = score(y, g.p_m2_oos)
                    rows.append([model, phase, int(lag), fmt(a1), fmt(a2), f"{a2 - a1:+.3f}", fmt(b1, 4), fmt(b2, 4)])
        ax = fig.add_axes([0.04, 0.45, 0.92, 0.46])
        table(ax, pd.DataFrame(rows, columns=["Model", "Phase", "Lag (days)", "AP M1", "AP M2", "AP gain",
                                              "Brier M1", "Brier M2"]))
        fig.text(0.06, 0.42, textwrap.fill(
            "On-chain inputs stay at lag 0, so M1 is identical across rows (checked in code). Only the regime flag's "
            "timing changes. Lag choice was not tuned on 2023. If the gain depends heavily on whether the flag is "
            "from t or t-1, the evidence for the regime signal is fragile.", 120), fontsize=8, color=INK2, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 10 interaction test (logit)
        fig = plt.figure(figsize=A4)
        fig.text(0.06, 0.96, "Do on-chain effects differ by regime? (logit interaction test, 2022 sample)",
                 fontsize=12.5, fontweight="bold", va="top")
        fig.text(0.06, 0.925, textwrap.fill(
            f"Logit M3: joint Wald chi-square({wald['df']}) = {wald['wald_chi_square']:.1f}, "
            f"p = {wald['p_value']:.1e}, week-clustered, {wald['observations']} obs / {wald['weekly_clusters']} weeks. "
            f"Warnings from the script: one regime has fewer breaches than parameters, and there are fewer than 50 "
            f"week clusters, so the asymptotic test may be unreliable.",
            115), fontsize=8.5, color=INK, va="top")
        sl = slopes.copy()
        sl["feature"] = sl.feature.str.replace("log1p_", "")
        ax = fig.add_axes([0.04, 0.66, 0.92, 0.22])
        table(ax, pd.DataFrame({
            "Feature": sl.feature,
            "Normal slope (SE)": [f"{a:.2f} ({b:.2f})" for a, b in zip(sl.normal_log_odds_slope, sl.normal_se)],
            "Stressed slope (SE)": [f"{a:.2f} ({b:.2f})" for a, b in zip(sl.stressed_log_odds_slope, sl.stressed_se)],
            "Interaction": sl.interaction_delta.map(lambda v: f"{v:+.2f}"),
        }), title="Logit M3: regime-specific log-odds slopes (within-coin standardised features)", fontsize=7.5)
        fig.text(0.06, 0.635, "This is an in-sample description of 2022, not out-of-sample evidence.",
                 fontsize=8, color=INK2, va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 11 coefficients vs importances (frozen)
        fig, axes = plt.subplots(1, 2, figsize=(A4[0], 4.8))
        c = logit_coef[(logit_coef.fold == "frozen_2022") & (logit_coef.feature != "intercept")]
        for k, spec in enumerate(["M1", "M2"]):
            cc = c[c.model == spec].set_index("feature").coefficient
            y = np.arange(len(cc))
            axes[0].barh(y + (k - 0.5) * 0.38, cc.values, height=0.36, color=LOGIT_C, alpha=0.5 + 0.5 * k,
                         edgecolor="white", label=spec)
        axes[0].set_yticks(np.arange(len(c[c.model == "M2"])),
                           c[c.model == "M2"].feature.str.replace("log1p_", ""), fontsize=7.5)
        axes[0].axvline(0, color=MUTED, lw=0.8)
        axes[0].set_title("Logit coefficients (frozen 2022)")
        axes[0].legend(fontsize=7)
        axes[0].grid(axis="y", visible=False)
        r = rf_imp[rf_imp.fold == "frozen_2022"]
        for k, spec in enumerate(["M1", "M2"]):
            rr = r[r.model == spec].set_index("feature").mdi_importance.reindex(r[r.model == "M2"].feature)
            y = np.arange(len(rr))
            axes[1].barh(y + (k - 0.5) * 0.38, rr.values, height=0.36, color=RF_C, alpha=0.5 + 0.5 * k,
                         edgecolor="white", label=spec)
        axes[1].set_yticks(np.arange(len(r[r.model == "M2"])),
                           r[r.model == "M2"].feature.str.replace("log1p_", ""), fontsize=7.5)
        axes[1].set_title("RF impurity importance (frozen 2022)")
        axes[1].legend(fontsize=7)
        axes[1].grid(axis="y", visible=False)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ---------------------------------------------------------- 12 caveats / files
        page(pdf, "Caveats and reproduction", [
            ("h", "Caveats"),
            ("b", "Very few events. The 2022 validation has 8 breaches, all in July 2022; Aug to Nov folds have none, "
                  "so they contribute only to Brier. Any 2022 out-of-sample conclusion rests on one month."),
            ("b", "The 2023 test has 49 breaches but the regime flag is on for only 3 days (SVB week). M2 and M1 "
                  "differ only on those days, so the 2023 test is close to a test of 'what happens on 3 days'."),
            ("b", "Placebo flags are overlapping shifts of one series, so they are not independent draws; treat "
                  "placebo p-values as descriptive. With 89 placebos the smallest attainable p is about 0.011."),
            ("b", "Both models use fixed, untuned specifications. A tuned forest could look different, but "
                  "tuning on this few events would mostly fit noise."),
            ("b", "Publication timing of the on-chain aggregates and regime inputs is assumed (lag 0), not verified; "
                  "the lag-sensitivity page shows how much this matters for the flag."),
            ("h", "Code and outputs"),
            ("b", "logit/: logit_oos.py, regime_lag_sensitivity.py, interaction_test.py."),
            ("b", "random_forest/: rf_oos.py, rf_regime_lag_sensitivity.py, with outputs "
                  "in random_forest/outputs/."),
            ("b", "model_comparison/: compare_models.py (bootstrap + placebo for both models) and build_report.py "
                  "(this PDF)."),
            ("p", "Run order from the repository root: python random_forest/rf_oos.py --out "
                  "random_forest/outputs/rf_decomposition; python random_forest/rf_regime_lag_sensitivity.py; "
                  "python model_comparison/compare_models.py; "
                  "python model_comparison/build_report.py."),
        ])


def findings():
    """Summary page. Numbers are pulled from the outputs; the reading is ours."""
    def d(model, phase, metric="average_precision"):
        return rm(model, phase, "M2", col=metric) - rm(model, phase, "M1", col=metric)

    o_l = preds[("Logit", "2023 frozen test")]
    o_r = preds[("Random forest", "2023 frozen test")]
    norm_l, norm_r = o_l[o_l.stress.eq(0)], o_r[o_r.stress.eq(0)]
    return [
        ("h", "Short answer"),
        ("p", "The regime flag adds a small amount of ranking power in both model families, but the gains "
              "are within what a placebo flag with the same persistence produces. The evidence that the "
              "regime classifier adds out-of-sample value is weak. The flag also worsens calibration in 2023. "
              "The choice of model matters more than the flag, and which model wins depends on the period."),
        ("h", "1. Value of the regime flag (M2 vs M1)"),
        ("b", f"2022 validation AP: logit {rm('Logit', PHASES[0], 'M1'):.2f} -> {rm('Logit', PHASES[0], 'M2'):.2f} "
              f"({d('Logit', PHASES[0]):+.3f}); RF {rm('Random forest', PHASES[0], 'M1'):.2f} -> "
              f"{rm('Random forest', PHASES[0], 'M2'):.2f} ({d('Random forest', PHASES[0]):+.3f}). Both gains have "
              f"bootstrap intervals spanning zero by a wide margin and placebo p = "
              f"{ps('Logit', PHASES[0], 'average_precision').placebo_p_value:.2f} (logit) and "
              f"{ps('Random forest', PHASES[0], 'average_precision').placebo_p_value:.2f} (RF). Only 8 breaches, all in July 2022."),
        ("b", f"2022 validation Brier: the flag improves calibration for both models; for RF this beats almost "
              f"every placebo (p = {ps('Random forest', PHASES[0], 'brier').placebo_p_value:.3f}), for logit p = "
              f"{ps('Logit', PHASES[0], 'brier').placebo_p_value:.2f}. This is the strongest pro-regime result, "
              f"It comes from July and August, where M2 lowers false alarms; in September (mostly "
              f"stressed, no breaches) M2 is worse for both models."),
        ("b", f"2023 frozen test AP: logit {d('Logit', PHASES[1]):+.3f} (bootstrap CI "
              f"[{bt('Logit', PHASES[1], 'average_precision').ci_low:+.3f}, {bt('Logit', PHASES[1], 'average_precision').ci_high:+.3f}], "
              f"placebo p = {ps('Logit', PHASES[1], 'average_precision').placebo_p_value:.2f}); RF {d('Random forest', PHASES[1]):+.3f} "
              f"(placebo p = {ps('Random forest', PHASES[1], 'average_precision').placebo_p_value:.2f}). Logit AUC gain "
              f"{d('Logit', PHASES[1], 'roc_auc'):+.3f} is the only interval excluding zero (placebo p = "
              f"{ps('Logit', PHASES[1], 'roc_auc').placebo_p_value:.2f})."),
        ("b", f"2023 Brier gets worse with the flag (logit {d('Logit', PHASES[1], 'brier'):+.4f}, CI excludes zero; "
              f"RF {d('Random forest', PHASES[1], 'brier'):+.4f}). Reason: the flag is on for only 3 days in 2023. "
              f"M2 learned from 2022 that breaches cluster in stress, so it cuts normal-day probabilities "
              f"(logit mean {norm_l.p_m1_oos.mean():.1%} -> {norm_l.p_m2_oos.mean():.1%}, RF "
              f"{norm_r.p_m1_oos.mean():.1%} -> {norm_r.p_m2_oos.mean():.1%}), but "
              f"{int(norm_l[TARGET].sum())} of the {int(o_l[TARGET].sum())} 2023 breaches happened on normal-regime days."),
        ("b", "Lag sensitivity: the 2022 AP gain moves a lot with the flag's timing (largest at lag 1 for both "
              "models) while 2023 barely changes. A real signal should not hinge on a one-day shift; this points "
              "to noise from the few July 2022 breaches."),
        ("h", "2. Logit vs random forest"),
        ("b", f"2022 validation: RF ranks breaches much better (M1 AP {rm('Random forest', PHASES[0], 'M1'):.2f} vs "
              f"{rm('Logit', PHASES[0], 'M1'):.2f}) and is better calibrated (Brier "
              f"{rm('Random forest', PHASES[0], 'M1', col='brier'):.3f} vs {rm('Logit', PHASES[0], 'M1', col='brier'):.3f})."),
        ("b", f"2023 frozen test: logit generalises better (M2 AP {rm('Logit', PHASES[1], 'M2'):.2f} vs "
              f"{rm('Random forest', PHASES[1], 'M2'):.2f}; AUC {rm('Logit', PHASES[1], 'M2', col='roc_auc'):.2f} vs "
              f"{rm('Random forest', PHASES[1], 'M2', col='roc_auc'):.2f}). The forest fits 2022 patterns closely and "
              f"extrapolates poorly to the 2023 regime of on-chain activity; the logit's smooth slopes carry over better."),
        ("b", "On the SVB stressed days the flag adds about +3 percentage points for USDC/DAI in both M2s. "
              "For logit this lifts M2 above M1; for RF, M2's stress-off baseline is lower, so RF M2 ends up "
              "roughly equal to RF M1. Neither model gives DAI or USDC a high probability during SVB (RF peaks "
              "near 25% for USDC, logit near 6%); both mainly flag USDT."),
        ("h", "3. Does the effect of on-chain activity depend on the regime?"),
        ("b", f"Yes, in-sample. Logit M3 Wald p = {wald['p_value']:.1e} (but with a sparse-data warning). "
              f"This heterogeneity does not turn into better out-of-sample forecasts."),
        ("h", "Bottom line for the paper"),
        ("p", "Report the regime flag as a descriptive moderator (on-chain effects differ under stress), not as "
              "a forecasting improvement. Out of sample, the flag's gains are small, timing-sensitive and "
              "indistinguishable from placebo flags, and the classifier marked almost none of the 2023 depeg "
              "period as stressed. RF wins in 2022 and logit wins in 2023, so neither model choice changes this "
              "conclusion."),
    ]


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "regime_value_logit_vs_rf.pdf"
    build(out, findings())
    print("Wrote", out)
