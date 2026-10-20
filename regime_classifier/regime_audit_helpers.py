#### read-only regime audit helpers. episodes are consecutive calendar-day runs
import numpy as np
import pandas as pd

FEATURES = ['log1p_n_transfers', 'log1p_onchain_volume', 'log1p_n_unique_senders',
            'log1p_n_unique_receivers', 'log1p_minted', 'log1p_burned',
            'log1p_volume_large_transfers']


def unique(d, keys, name):
    if d.duplicated(keys).any():
        raise ValueError(f'{name}: duplicate {keys}')


def episode_ids(d, date_col, flag, group=None):
    """Assign full-series positive-run IDs before dividing into windows.

    IDs are per coin when grouped; gaps break episodes. Missing labels cannot
    bridge episodes. A clipped sample boundary is not proof of event onset.
    """
    x = d.sort_values(([group] if group else []) + [date_col]).copy()
    def assign(g):
        positive = g[flag].eq(1).fillna(False)
        start = positive & (~positive.shift(fill_value=False)
                            | g[date_col].diff().ne(pd.Timedelta(days=1)))
        return start.cumsum().where(positive).astype('Int64')
    x['episode_id'] = pd.Series(pd.NA, index=x.index, dtype='Int64')
    if group:
        for _, g in x.groupby(group):
            x.loc[g.index, 'episode_id'] = assign(g)
    else:
        x['episode_id'] = assign(x)
    return x


def summary(panel):
    rows = []
    for (window, stress), g in panel.groupby(['window', 'stress']):
        positive = g[g.depeg_dynamic.eq(1)]
        # coin episodes may overlap in calendar time; they are not independent shocks.
        rows.append(dict(window=window, stress=int(stress),
                         market_days=g.date.nunique(), coin_days=len(g),
                         breach_coin_days=int(g.depeg_dynamic.sum()),
                         breach_market_days=positive.event_date.nunique(),
                         coin_episodes_overlapping=len(positive[['coin', 'episode_id']].drop_duplicates()),
                         regime_spells_overlapping=g.regime_spell_id.nunique(),
                         breach_rate=g.depeg_dynamic.mean()))
    return pd.DataFrame(rows)


def load_panel(root):
    regime = pd.read_csv(root / 'regime_classifier/outputs/market_regime.csv', parse_dates=['date'])
    labels = pd.read_csv(root / 'depeg_labeller/results/depeg_labels_dynamic.csv', parse_dates=['date', 'event_date'])
    import sys
    logit_dir = str(root / 'logit')
    if logit_dir not in sys.path:
        sys.path.insert(0, logit_dir)
    from logit_oos import load_inputs
    raw_panel, _, _, _ = load_inputs(root, 0)
    onchain = raw_panel[['coin', 'date'] + FEATURES].copy()
    unique(regime, ['date'], 'regime'); unique(labels, ['coin', 'date'], 'labels')
    unique(onchain, ['coin', 'date'], 'onchain')
    if not regime.stress.isin([0, 1]).all():
        raise ValueError('Invalid regime flags; do not treat missing values as normal.')
    regime = regime.sort_values('date')
    regime['regime_spell_id'] = (regime.stress.ne(regime.stress.shift())
                                  | regime.date.diff().ne(pd.Timedelta(days=1))).cumsum()
    labels = episode_ids(labels, 'event_date', 'depeg_dynamic', 'coin')
    panel = labels.merge(onchain[['coin', 'date'] + FEATURES], on=['coin', 'date'],
                          how='inner', validate='one_to_one').merge(
        regime[['date', 'stress', 'regime_spell_id']], on='date', how='inner', validate='many_to_one')
    finite = np.isfinite(panel[FEATURES].to_numpy(dtype=float)).all(axis=1)
    panel = panel.loc[finite].dropna(subset=['depeg_dynamic', 'stress']).copy()
    if not (panel.event_date - panel.date).eq(pd.Timedelta(days=1)).all():
        raise ValueError('Labels must use the next calendar day.')
    panel['window'] = np.where(panel.date.dt.year.eq(2022), '2022 training',
                               np.where(panel.date.dt.year.eq(2023), '2023 calendar', 'other'))
    return regime, labels, panel


def check_predictions(pred, panel):
    pred = pred.copy()
    pred['date'] = pd.to_datetime(pred.date)
    unique(pred, ['coin', 'date'], 'OOS predictions')
    joined = pred.merge(panel[['coin', 'date', 'event_date', 'depeg_dynamic', 'stress',
                               'episode_id', 'regime_spell_id']], on=['coin', 'date'],
                        suffixes=('_saved', ''), how='left', validate='one_to_one', indicator=True)
    joined['label_matches'] = joined.depeg_dynamic_saved.eq(joined.depeg_dynamic)
    joined['regime_matches'] = joined.stress_saved.eq(joined.stress)
    return joined
