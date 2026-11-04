"""Section 2 (data) summary statistics and figures for the final report.

Run from the repository root after steps 1-2 of the README run order:

    python model_comparison/data_summary.py --out model_comparison/outputs_data_summary

Reads only saved files:
  dataset/onchain_2022_base.csv, dataset/onchain_2023_base.csv
  depeg_labeller/results/depeg_labels_dynamic.csv
  regime_classifier/outputs/market_regime.csv
  logit/outputs/logit_decomposition/model_panel.csv   (the modelling panel)

Writes:
  onchain_medians.csv           Table 2: median daily on-chain activity per coin and period
  breaches_by_coin_regime.csv   Table 3: coin-days, breach days and breach rate by coin, period, regime
  band_widths.csv               median dynamic band width (bp) per coin and period
  regime_signals.csv            share of days with each stress signal, per period
  fig_prices_depegs.png         Figure 1: next-day close and dynamic depeg days, 2022-2023
  fig_btc_regime.png            Figure 2: BTC price with stressed regime, March-July 2022
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

COINS = ["DAI", "USDC", "USDT"]
RAW = ["n_transfers", "onchain_volume", "n_unique_senders", "n_unique_receivers",
       "minted", "burned", "volume_large_transfers"]
WINDOWS = {"2022": ("2022-04-01", "2022-11-01"), "2023": ("2023-02-01", "2023-04-30")}
COLOURS = {"DAI": "#C25B1A", "USDC": "#2E6DA4", "USDT": "#1A9E77"}


def period(dates):
    return np.where(dates.dt.year == 2022, "2022", "2023")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model_comparison/outputs_data_summary")
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv("logit/outputs/logit_decomposition/model_panel.csv", parse_dates=["date"])
    labels = pd.read_csv("depeg_labeller/results/depeg_labels_dynamic.csv", parse_dates=["date", "event_date"])
    regime = pd.read_csv("regime_classifier/outputs/market_regime.csv", parse_dates=["date"])
    onchain = pd.concat([pd.read_csv(f"dataset/onchain_{y}_base.csv", parse_dates=["date"])
                         for y in (2022, 2023)], ignore_index=True)

    # Table 2: raw on-chain levels for the coin-days in the modelling panel
    m = panel[["coin", "date"]].merge(onchain[["coin", "date"] + RAW], on=["coin", "date"], how="left")
    m["period"] = period(m.date)
    m.groupby(["period", "coin"])[RAW].median().round(0).to_csv(out / "onchain_medians.csv")

    # Table 3: breaches by coin, period and regime on the predictor date
    panel["period"] = period(panel.date)
    t3 = panel.groupby(["period", "coin", "stress"]).depeg_dynamic.agg(
        coin_days="size", breach_days="sum", breach_rate="mean")
    t3.round(4).to_csv(out / "breaches_by_coin_regime.csv")

    # Band widths in the modelling panel
    b = panel[["coin", "event_date"]].assign(event_date=lambda d: pd.to_datetime(d.event_date)).merge(
        labels[["coin", "event_date", "threshold_bp"]], on=["coin", "event_date"], how="left")
    b["period"] = np.where(b.event_date.dt.year == 2022, "2022", "2023")
    b.groupby(["period", "coin"]).threshold_bp.median().round(1).to_csv(out / "band_widths.csv")

    # Regime signal shares in each on-chain window
    rows = []
    for name, (a, z) in WINDOWS.items():
        g = regime[(regime.date >= a) & (regime.date <= z)]
        rows.append({"period": name, "days": len(g), "btc_bearish": g.btc_bearish.mean(),
                     "eth_bearish": g.eth_bearish.mean(), "fear_greed_below_40": g.fear_signal.mean(),
                     "stressed": g.stress.mean(), "mean_fear_greed": g.fear_greed.mean()})
    pd.DataFrame(rows).round(3).to_csv(out / "regime_signals.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # Figure 1: prices and depeg days, full label sample
    lab = labels[labels.coin.isin(COINS)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.2), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for c in ["USDT", "USDC", "DAI"]:
        g = lab[lab.coin == c].sort_values("event_date")
        a1.plot(g.event_date, g.next_day_close, lw=1, color=COLOURS[c], label=c)
    for a in (a1, a2):
        for s, e in [("2022-04-01", "2022-11-02"), ("2023-02-01", "2023-05-01")]:
            a.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#E8ECF2", zorder=0)
    a1.axhline(1, color="#888", lw=0.6, ls="--")
    a1.set_ylim(0.966, 1.012)
    a1.set_ylabel("Next-day close (US$)")
    a1.legend(loc="lower left", frameon=False, ncol=3)
    a1.annotate("USDC and DAI fall to ~0.97\nafter SVB (11 March 2023)", xy=(pd.Timestamp("2023-03-10"), 0.9725),
                xytext=(pd.Timestamp("2022-08-20"), 0.976), fontsize=9, arrowprops=dict(arrowstyle="->", color="#555"))
    a1.set_title("Next-day closing prices stay near US$1, but breaches of the dynamic band are frequent",
                 fontsize=11, loc="left")
    for i, c in enumerate(COINS):
        g = lab[(lab.coin == c) & (lab.depeg_dynamic == 1)]
        a2.scatter(g.event_date, [i] * len(g), marker="|", s=60, color=COLOURS[c])
        a2.text(pd.Timestamp("2024-01-05"), i, f"{len(g)} of {int((lab.coin == c).sum())}", va="center", fontsize=9)
    a2.set_yticks([0, 1, 2])
    a2.set_yticklabels(COINS)
    a2.set_ylim(-0.6, 2.6)
    a2.set_title("Dynamic depeg days (shaded = on-chain modelling windows)", fontsize=10, loc="left")
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a2.set_xlim(pd.Timestamp("2022-01-25"), pd.Timestamp("2024-02-20"))
    fig.tight_layout()
    fig.savefig(out / "fig_prices_depegs.png", dpi=200)
    plt.close(fig)

    # Figure 2: BTC with stressed regime around the Terra/UST collapse
    w = regime[(regime.date >= "2022-03-15") & (regime.date <= "2022-07-15")].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(w.date, w.btc_close / 1000, color="#1F3A5F", lw=1.6)
    start = None
    for i in range(len(w)):
        if w.stress[i] == 1 and start is None:
            start = w.date[i]
        if start is not None and (w.stress[i] == 0 or i == len(w) - 1):
            ax.axvspan(start, w.date[i], color="#5B4BC4", alpha=0.15, lw=0)
            start = None
    ax.axvline(pd.Timestamp("2022-05-07"), color="#C25B1A", ls="--", lw=1.2)
    ax.text(pd.Timestamp("2022-05-08"), 46.5, "UST depeg begins\n(7 May 2022)", color="#C25B1A", fontsize=9, va="top")
    ax.set_ylabel("BTC close (US$ thousands)")
    ax.set_ylim(17, 48)
    ax.set_title("Bitcoin price with stressed regime shaded (purple), March–July 2022", fontsize=11, loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.tight_layout()
    fig.savefig(out / "fig_btc_regime.png", dpi=200)
    plt.close(fig)
    print(f"Wrote data summary to {out}")


if __name__ == "__main__":
    main()
