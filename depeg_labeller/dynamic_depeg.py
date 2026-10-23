from pathlib import Path
import argparse
import numpy as np
import pandas as pd

COINS = ['USDT', 'USDC', 'DAI', 'BUSD']


def load_prices(path):
    d = pd.read_csv(path)
    required = {'coin', 'date', 'low', 'high', 'volume_usd'}
    if not required.issubset(d.columns):
        raise ValueError(f'Missing columns: {required - set(d.columns)}')
    d['coin'] = d.coin.astype(str).str.strip().str.upper()
    d = d[d.coin.isin(COINS)].copy()
    if d.empty:
        raise ValueError('No selected stablecoins found.')
    dates = pd.to_datetime(d.date, errors='raise')
    if dates.dt.tz is not None:
        raise ValueError('Convert timestamps to a documented common daily boundary first.')
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError('Expected daily dates at midnight; verify source boundaries first.')
    d['date'] = dates
    if d.duplicated(['coin', 'date']).any():
        raise ValueError('Duplicate coin/date rows; resolve them explicitly.')
    for c in ['low', 'high', 'volume_usd', 'close']:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors='raise')
    return d.sort_values(['coin', 'date']).reset_index(drop=True)


def monthly_volumes(d):
    x = d.assign(month=d.date.dt.to_period('M'))
    x['valid_volume'] = np.isfinite(x.volume_usd) & x.volume_usd.ge(0)
    m = x.groupby(['coin', 'month'], as_index=False).agg(
        monthly_volume_usd=('volume_usd', 'sum'),
        observed_days=('date', 'nunique'),
        valid_volume_days=('valid_volume', 'sum'))
    m['expected_days'] = m.month.dt.days_in_month
    m['complete'] = (m.observed_days.eq(m.expected_days)
                     & m.valid_volume_days.eq(m.expected_days)
                     & np.isfinite(m.monthly_volume_usd)
                     & m.monthly_volume_usd.gt(0))
    return m


def breach(low, high, distance, valid):
    y = pd.Series(pd.NA, index=low.index, dtype='Int64')
    y.loc[valid] = (low.loc[valid].le(1 - distance.loc[valid])
                   | high.loc[valid].ge(1 + distance.loc[valid])).astype(int)
    return y


def build_labels(d):
    m = monthly_volumes(d)
    ref = m.loc[m.complete, ['coin', 'month', 'monthly_volume_usd']].copy()
    ref['volume_month'] = ref.month
    ref['month'] = ref.month + 1
    ref = ref.rename(columns={'monthly_volume_usd': 'previous_month_volume_usd'})
    e = d.assign(month=d.date.dt.to_period('M')).merge(
        ref, on=['coin', 'month'], how='left', validate='many_to_one')
    e = e.rename(columns={'date': 'event_date', 'low': 'next_day_low',
                          'high': 'next_day_high', 'close': 'next_day_close'})
    e['date'] = e.event_date - pd.Timedelta(days=1)
    e['threshold_distance'] = 10 / e.previous_month_volume_usd.pow(1 / 3)
    e['threshold_bp'] = e.threshold_distance * 10000
    e['threshold_down'] = 1 - e.threshold_distance
    e['threshold_up'] = 1 + e.threshold_distance
    e['valid_extremes'] = (np.isfinite(e.next_day_low) & np.isfinite(e.next_day_high)
                           & e.next_day_low.gt(0) & e.next_day_high.ge(e.next_day_low))
    e['eligible_dynamic'] = (e.valid_extremes & np.isfinite(e.threshold_distance)
                              & e.threshold_distance.gt(0))
    # close is never required for the high/low outcome.
    e['depeg_dynamic'] = breach(e.next_day_low, e.next_day_high,
                                e.threshold_distance, e.eligible_dynamic)
    # Downside-only robustness label
    e['depeg_downside'] = pd.Series(pd.NA, index=e.index, dtype='Int64')
    ok = e.eligible_dynamic
    e.loc[ok, 'depeg_downside'] = (e.loc[ok, 'next_day_low'] <= e.loc[ok, 'threshold_down']).astype(int)
    e['depeg_static_25bp'] = breach(e.next_day_low, e.next_day_high,
                                   pd.Series(.0025, index=e.index), e.valid_extremes)
    # reproduce the original 30-observation window through predictor day t.
    x = d[['coin', 'date', 'volume_usd']].copy()
    x['usable_volume'] = x.volume_usd.where(np.isfinite(x.volume_usd) & x.volume_usd.ge(0))
    x['rolling_volume_30d'] = x.groupby('coin').usable_volume.transform(
        lambda s: s.rolling(30, min_periods=30).sum())
    x['predictor_observed'] = True
    e = e.merge(x[['coin', 'date', 'rolling_volume_30d', 'predictor_observed']],
                on=['coin', 'date'], how='left', validate='one_to_one')
    e['original_threshold_distance'] = 10 / e.rolling_volume_30d.pow(1 / 3)
    ok = (e.valid_extremes & np.isfinite(e.original_threshold_distance)
          & e.original_threshold_distance.gt(0))
    e['depeg_original_30obs'] = breach(e.next_day_low, e.next_day_high,
                                      e.original_threshold_distance, ok)
    e['original_threshold_bp'] = e.original_threshold_distance * 10000
    e['common_sample'] = (e.predictor_observed.eq(True)
                          & e[['depeg_dynamic', 'depeg_static_25bp',
                                'depeg_original_30obs']].notna().all(axis=1))
    labels = e.loc[e.eligible_dynamic & e.predictor_observed.eq(True)].copy()
    return labels, e, m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prices', type=Path, default=Path(__file__).resolve().parents[1] / 'dataset/prices.csv')
    p.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = p.parse_args()
    labels, events, monthly = build_labels(load_prices(args.prices))
    args.out.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out / 'depeg_labels_dynamic.csv', index=False)
    events.to_csv(args.out / 'depeg_label_all_events.csv', index=False)
    monthly.to_csv(args.out / 'depeg_monthly_volume.csv', index=False)
    print(f'Saved {len(labels):,} eligible next-day labels to {args.out.resolve()}')
    print('Interpretation: volume-scaled abnormal deviations; counts are days, not episodes.')


if __name__ == '__main__':
    main()
