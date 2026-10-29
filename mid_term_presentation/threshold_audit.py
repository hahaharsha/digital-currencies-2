from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / 'dataset/prices.csv', parse_dates=['date'])
df = df[df.coin.isin(['USDT', 'USDC', 'DAI'])].sort_values(['coin', 'date']).copy()

# Original: trailing 30 observed days, including the current day.
df['original_bp'] = 100000 / df.groupby('coin').volume_usd.transform(lambda s: s.rolling(30, min_periods=30).sum()).pow(1/3)

# Revised: volume from the previous completed calendar month.
df['month'] = df.date.dt.to_period('M')
monthly = df.groupby(['coin', 'month'], as_index=False).volume_usd.sum()
monthly['previous_volume'] = monthly.groupby('coin').volume_usd.shift()
df = df.merge(monthly[['coin', 'month', 'previous_volume']], on=['coin', 'month'])
df['revised_bp'] = 100000 / df.previous_volume.pow(1/3)
df['static_bp'] = 25.0

out = Path(__file__).resolve().parent / 'audit_label'
out.mkdir(exist_ok=True)
df[['coin', 'date', 'original_bp', 'revised_bp', 'static_bp']].to_csv(out / 'threshold_daily.csv', index=False)
summary = df.groupby('coin')[['original_bp', 'revised_bp', 'static_bp']].median().round(2)
summary.to_csv(out / 'threshold_medians.csv')
print(summary)

fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
for ax, (coin, group) in zip(axes, df.groupby('coin', sort=False)):
    for col, label in [('original_bp', 'Original (30-day)'), ('revised_bp', 'Revised (previous month)')]:
        ax.plot(group.date, group[col], label=label, linewidth=1)
    ax.axhline(25, linestyle='--', color='gray', label='Static (25 bp)')
    ax.set(title=coin, ylabel='Band half-width (bp)')
    ax.legend(loc='upper right', fontsize=8)
fig.tight_layout()
fig.savefig(out / 'threshold_comparison.png', dpi=160)
