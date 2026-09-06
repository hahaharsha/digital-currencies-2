"""
Replication of Tables 3 and 4 from:

    Lee, Y.-H., Chiu, Y.-F., & Hsieh, M.-H. (2025).
    Stablecoin depegging risk prediction.
    Pacific-Basin Finance Journal, 90, 102640.

Scope is deliberately narrow: Table 3 and Table 4 only. No figures, no feature
audit, no models.

    python replication.py
    python replication.py --folder dataset
    python replication.py --diagnostics        # extra columns explaining any gap

Input
    A price table with columns: coin, date, open, high, low, close, volume_usd.
    Found automatically as prices.{csv,xlsx,xls,parquet} in --folder, that
    folder's dataset/ subfolder, or the repo's dataset/ directory.

Output (--outdir, default replication_output/)
    table3_descriptive_statistics.csv
    table4_depegging_statistics.csv
    Both are written as CSV so they diff cleanly in git.

--------------------------------------------------------------------------
TABLE 3 -- Descriptive statistics (paper p. 9)
    Computed from the daily CLOSING price only. Min and Max are the extremes
    of the closing series across the sample, NOT the daily low and high.
    Stablecoins rounded to 4 dp, BTC/ETH to 1 dp, matching the paper.

TABLE 4 -- Depegging statistics (paper p. 11)
    Computed from high, low and volume_usd only. Never uses close.

    Depegging definition, paper Section 3.1, extending Carey (2023):

        Y = 1  if  P_L <= Thresh_D  or  P_H >= Thresh_U
        Y = 0  otherwise

        Thresh_D = 1 - 10 / V^alpha
        Thresh_U = 1 + 10 / V^alpha
        V        = rolling 30-day sum of daily trading volume
        alpha    = 1/3          (Carey's optimal setting)
        P_L, P_H = daily low and daily high

    The bilateral form is the paper's own extension; Carey treated downward
    depegging only. "Number of depeg event" counts DAYS with Y = 1, not
    episodes.

Note on sources: the paper uses CoinMarketCap (Section 3.1). If this dataset
comes from another vendor, closing prices generally agree closely (so Table 3
reproduces well) while the intraday high/low range does not (so Table 4 may
not). Run with --diagnostics to measure that rather than assume it.
--------------------------------------------------------------------------
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Parameters (paper Section 3.1)
# --------------------------------------------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2023-12-31"

STABLECOINS = ["USDT", "USDC", "BUSD", "DAI"]
MAJORS = ["BTC", "ETH"]
T3_ORDER = ["USDT", "USDC", "DAI", "BUSD", "BTC", "ETH"]   # paper's row order
T4_ORDER = ["USDT", "USDC", "DAI", "BUSD"]

ALPHA = 1 / 3
KCONST = 10.0
WINDOW = 30

REQUIRED = ["coin", "date", "open", "high", "low", "close", "volume_usd"]

# --------------------------------------------------------------------------
# Published values, transcribed from the paper
# --------------------------------------------------------------------------
PUBLISHED_T3 = pd.DataFrame(
    [
        ["USDT", 730, 1.0001, 0.0007, 0.9959, 1.0000, 1.0001, 1.0003, 1.0077],
        ["USDC", 730, 1.0000, 0.0011, 0.9715, 0.9999, 1.0000, 1.0001, 1.0008],
        ["DAI", 730, 0.9998, 0.0011, 0.9739, 0.9996, 0.9999, 1.0001, 1.0023],
        ["BUSD", 730, 1.0002, 0.0006, 0.9980, 0.9999, 1.0002, 1.0005, 1.0037],
        ["BTC", 730, 28528.7, 8332.1, 15787.3, 21529.6, 27272.5, 35074.0, 47686.8],
        ["ETH", 730, 1891.3, 577.4, 993.6, 1561.8, 1791.0, 2044.7, 3829.6],
    ],
    columns=["Token Symbol", "Number of Record", "Mean", "STD", "Min",
             "25 %", "50 %", "75 %", "Max"],
).set_index("Token Symbol")

PUBLISHED_T4 = pd.DataFrame(
    [["USDT", 730, 168, 0.2301],
     ["USDC", 730, 16, 0.0219],
     ["DAI", 730, 14, 0.0192],
     ["BUSD", 730, 193, 0.2644]],
    columns=["Token symbol", "Number of record",
             "Number of depeg event", "Depeg ratio"],
).set_index("Token symbol")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def find_prices(folder, explicit=False):
    """
    Look for a prices file. Any of csv, xlsx, xls, parquet.

    When --folder was given explicitly we search only that folder and its
    dataset/ subfolder, so a typo fails loudly instead of silently falling back
    to the repo's own data.
    """
    here = Path(__file__).resolve().parent
    bases = ([folder, folder / "dataset"] if explicit
             else [folder, folder / "dataset", here / "dataset", here])
    seen = []
    for base in bases:
        if base in seen or not base.is_dir():
            continue
        seen.append(base)
        for p in sorted(base.iterdir()):
            if p.is_file() and p.stem.lower() == "prices" \
                    and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".parquet"):
                return p
    sys.exit("Could not find a prices file (csv/xlsx/parquet).\n"
             "Looked in:\n  " + "\n  ".join(str(b) for b in seen) +
             "\nPass --folder to point at the right directory.")


def read_any(path):
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    xl = pd.ExcelFile(path)
    sheet = next((s for s in xl.sheet_names if s.lower() == "prices"),
                 xl.sheet_names[0])
    return xl.parse(sheet)


def load_prices(folder, start, end, explicit=False):
    path = find_prices(folder, explicit=explicit)
    d = read_any(path)
    d.columns = [str(c).strip().lower() for c in d.columns]

    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        sys.exit(f"{path.name} is missing required columns: {missing}\n"
                 f"found: {list(d.columns)}")

    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    if getattr(d["date"].dtype, "tz", None) is not None:
        d["date"] = d["date"].dt.tz_localize(None)
    d["coin"] = d["coin"].astype(str).str.strip().str.upper()
    for c in ("open", "high", "low", "close", "volume_usd"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["date", "coin"])
    d = d[d.date.between(start, end)]
    d = d.drop_duplicates(["coin", "date"]).sort_values(["coin", "date"])

    print(f"input   : {path}")
    print(f"          {len(d)} rows, {d.date.min().date()} to {d.date.max().date()}")
    print(f"          coins: {', '.join(sorted(d.coin.unique()))}")
    if "price_source" in d.columns:
        srcs = sorted(d.price_source.dropna().astype(str).unique())
        print(f"          vendor: {', '.join(srcs)}  (paper used CoinMarketCap)")

    for a in T3_ORDER:
        n = int((d.coin == a).sum())
        if n == 0:
            print(f"  WARNING: {a} is absent; it will be blank in the output")
        elif abs(n - 730) > 5:
            print(f"  WARNING: {a} has {n} rows against the paper's 730")
    return d.reset_index(drop=True)


# --------------------------------------------------------------------------
# Table 3
# --------------------------------------------------------------------------
def table3(prices):
    rows = []
    for a in T3_ORDER:
        s = prices.loc[prices.coin == a, "close"].dropna()
        if s.empty:
            continue
        rows.append({"Token Symbol": a,
                     "Number of Record": len(s),
                     "Mean": s.mean(), "STD": s.std(), "Min": s.min(),
                     "25 %": s.quantile(0.25), "50 %": s.quantile(0.50),
                     "75 %": s.quantile(0.75), "Max": s.max()})
    if not rows:
        sys.exit("No usable closing prices for any asset.")

    t3 = pd.DataFrame(rows).set_index("Token Symbol")
    num = [c for c in t3.columns if c != "Number of Record"]
    for group, dp in [(STABLECOINS, 4), (MAJORS, 1)]:
        idx = t3.index.intersection(group)
        t3.loc[idx, num] = t3.loc[idx, num].round(dp)

    out = t3.join(PUBLISHED_T3, lsuffix="_ours", rsuffix="_paper", how="outer")
    out = out[sorted(out.columns, key=lambda c: c.rsplit("_", 1)[0])]
    return t3, out.loc[[a for a in T3_ORDER if a in out.index]]


# --------------------------------------------------------------------------
# Table 4
# --------------------------------------------------------------------------
def apply_threshold(d):
    """Paper Section 3.1. One coin, sorted by date."""
    d = d.sort_values("date").copy()
    V = d["volume_usd"].rolling(WINDOW, min_periods=1).sum()
    band = KCONST / V ** ALPHA
    d["thresh_d"] = 1 - band
    d["thresh_u"] = 1 + band
    d["band_bp"] = band * 1e4
    d["breach_lower"] = (d["low"] <= d["thresh_d"]).astype(int)
    d["breach_upper"] = (d["high"] >= d["thresh_u"]).astype(int)
    d["Y"] = ((d["breach_lower"] == 1) | (d["breach_upper"] == 1)).astype(int)
    # close-only variant, used by --diagnostics; NOT the paper's rule
    d["Y_close_only"] = ((d["close"] <= d["thresh_d"])
                         | (d["close"] >= d["thresh_u"])).astype(int)
    d["intraday_range_bp"] = (d["high"] - d["low"]) * 1e4
    return d


def table4(prices, diagnostics=False):
    flagged = pd.concat([apply_threshold(d) for _, d in prices.groupby("coin")],
                        ignore_index=True)

    rows = []
    for a in T4_ORDER:
        d = flagged[flagged.coin == a]
        if d.empty:
            continue
        r = {"Token symbol": a,
             "Number of record": len(d),
             "Number of depeg event": int(d["Y"].sum()),
             "Depeg ratio": round(float(d["Y"].mean()), 4)}
        if diagnostics:
            r.update({
                "breach_lower": int(d["breach_lower"].sum()),
                "breach_upper": int(d["breach_upper"].sum()),
                "both_sides_same_day": int(((d.breach_lower == 1)
                                            & (d.breach_upper == 1)).sum()),
                "depeg_events_close_only": int(d["Y_close_only"].sum()),
                "median_band_bp": round(float(d["band_bp"].median()), 2),
                "median_intraday_range_bp": round(
                    float(d["intraday_range_bp"].median()), 2),
                "range_over_band": round(
                    float(d["intraday_range_bp"].median()
                          / d["band_bp"].median()), 2),
            })
        rows.append(r)

    t4 = pd.DataFrame(rows).set_index("Token symbol")
    out = t4.join(PUBLISHED_T4, lsuffix="_ours", rsuffix="_paper", how="outer")
    out["event_gap"] = (out["Number of depeg event_ours"]
                        - out["Number of depeg event_paper"])
    out["ratio_gap_pct_points"] = (
        (out["Depeg ratio_ours"] - out["Depeg ratio_paper"]) * 100).round(2)
    return t4, out.loc[[a for a in T4_ORDER if a in out.index]], flagged


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=None,
                    help="folder holding the prices file "
                         "(default: the repo's dataset/)")
    ap.add_argument("--outdir", default="replication_output",
                    help="where to write the CSVs (default: replication_output)")
    ap.add_argument("--start", default=START_DATE)
    ap.add_argument("--end", default=END_DATE)
    ap.add_argument("--diagnostics", action="store_true",
                    help="add columns that localise any Table 4 gap "
                         "(breach side, close-only count, range vs band)")
    args, _ = ap.parse_known_args()

    explicit = args.folder is not None
    folder = Path(args.folder if explicit else "dataset").expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    print("=" * 76)
    print("Lee, Chiu & Hsieh (2025), Pacific-Basin Finance Journal 90, 102640")
    print("Replication of Table 3 and Table 4")
    print("=" * 76)
    prices = load_prices(folder, start, end, explicit=explicit)

    print("\n" + "-" * 76)
    print("TABLE 3  Descriptive statistics of daily closing prices")
    print("-" * 76)
    t3, t3_vs = table3(prices)
    print(t3.to_string())

    print("\n" + "-" * 76)
    print("TABLE 4  Depegging statistics")
    print("  Y = 1 if low <= 1 - 10/V^(1/3) or high >= 1 + 10/V^(1/3)")
    print("  V = rolling 30-day sum of trading volume")
    print("-" * 76)
    t4, t4_vs, _ = table4(prices, diagnostics=args.diagnostics)
    print(t4_vs.to_string())

    if args.diagnostics:
        print("\n  range_over_band near or above 1 means the daily high-low range")
        print("  is as wide as the threshold itself, so the label is being driven")
        print("  by intraday noise rather than by depegging. Compare")
        print("  depeg_events_close_only to see how much of the count survives")
        print("  when the same threshold is applied to the closing price.")

    p3 = outdir / "table3_descriptive_statistics.csv"
    p4 = outdir / "table4_depegging_statistics.csv"
    t3_vs.reset_index().to_csv(p3, index=False)
    t4_vs.reset_index().to_csv(p4, index=False)
    print(f"\nwrote {p3}")
    print(f"wrote {p4}")


if __name__ == "__main__":
    main()