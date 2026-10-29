"""Placebo regime flags: the real stress series circularly shifted along its calendar.

Shared by compare_models.py and score_variant.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "logit"))
from logit_oos import dates, unique  # noqa: E402

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
