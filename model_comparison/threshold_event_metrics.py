"""Threshold-based and event-level metrics for logit and RF (M1 vs M2).

Thresholds: chosen once per model/spec on the pooled 2022 sequential-validation
forecasts (the probability cut-off that maximises F1), then applied unchanged
to the frozen 2023 forecasts.

Events: a new depeg episode starts on an event date where a coin breaches
after a non-breach day (or after a gap in the data). Warning coverage asks
whether any forecast targeting the onset day or the 3 days before it was
above the threshold; lead time is how many days before the onset the first
such warning targeted (0 = the onset day itself).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
CMP = ROOT / "model_comparison/outputs"
TARGET = "depeg_dynamic"
WINDOW = 3
FILES = {"Logit": "logit", "Random forest": "random_forest"}


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1)
    return float(thr[np.nanargmax(f1[:-1])])


def confusion(y, p, t):
    hit = p >= t
    tp, fp = int((hit & (y == 1)).sum()), int((hit & (y == 0)).sum())
    fn = int((~hit & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "true_alarms": tp,
            "false_alarms": fp, "missed_breaches": fn, "warnings": int(hit.sum())}


def onsets(pred):
    d = pred.sort_values(["coin", "event_date"]).copy()
    prev = d.groupby("coin")[TARGET].shift(1)
    gap = d.groupby("coin")["event_date"].diff().dt.days
    d["onset"] = d[TARGET].eq(1) & (prev.ne(1) | gap.ne(1))
    return d


def main():
    thr_rows, conf_rows, event_rows = [], [], []
    for model, stem in FILES.items():
        val = pd.read_csv(CMP / f"{stem}_validation_predictions.csv", parse_dates=["date", "event_date"])
        oos = pd.read_csv(CMP / f"{stem}_oos_predictions.csv", parse_dates=["date", "event_date"])
        oos = onsets(oos)
        for spec in ["M1", "M2"]:
            col = f"p_{spec.lower()}_oos"
            t = best_f1_threshold(val[TARGET].to_numpy(), val[col].to_numpy())
            thr_rows.append({"model": model, "spec": spec, "threshold": t})
            for phase, d in [("2022 validation", val), ("2023 frozen test", oos)]:
                conf_rows.append({"model": model, "spec": spec, "phase": phase, "threshold": t,
                                  **confusion(d[TARGET].to_numpy(), d[col].to_numpy(), t)})
            for r in oos[oos.onset].itertuples():
                win = oos[oos.coin.eq(r.coin)
                          & oos.event_date.between(r.event_date - pd.Timedelta(days=WINDOW), r.event_date)]
                warned = win[win[col] >= t]
                lead = (r.event_date - warned.event_date.min()).days if len(warned) else np.nan
                event_rows.append({"model": model, "spec": spec, "coin": r.coin,
                                   "onset_event_date": r.event_date.date(), "stress_on_forecast_day": r.stress,
                                   "p_on_onset_forecast": getattr(r, col),
                                   "warned_on_onset_day": bool(getattr(r, col) >= t),
                                   "warned_within_window": bool(len(warned)), "lead_days": lead})
    ev = pd.DataFrame(event_rows)
    summary = ev.groupby(["model", "spec"]).agg(
        episodes=("coin", "size"), warned_on_onset_day=("warned_on_onset_day", "sum"),
        warned_within_3d=("warned_within_window", "sum"),
        mean_lead_days_if_warned=("lead_days", "mean")).reset_index()
    pd.DataFrame(thr_rows).to_csv(CMP / "thresholds.csv", index=False)
    pd.DataFrame(conf_rows).to_csv(CMP / "threshold_metrics.csv", index=False)
    ev.to_csv(CMP / "event_level_detail.csv", index=False)
    summary.to_csv(CMP / "event_level_summary.csv", index=False)
    print(pd.DataFrame(conf_rows).round(3).to_string(index=False))
    print(summary.round(2).to_string(index=False))
    print(ev[ev.spec == "M2"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
