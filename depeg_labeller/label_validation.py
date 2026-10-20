from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]  
df = pd.read_csv(root / 'dataset/prices.csv', parse_dates=['date'])
df = df[df.coin.isin(['USDT', 'USDC', 'DAI'])].sort_values(['coin', 'date']).copy()

# thresholds at forecast date t; label uses prices on t+1.
df['original_bp'] = 1e5 / df.groupby('coin').volume_usd.transform(lambda s: s.rolling(30, min_periods=30).sum()).pow(1/3)
df['month'] = df.date.dt.to_period('M')
monthly = df.groupby(['coin', 'month'], as_index=False).volume_usd.sum()
monthly['previous_volume'] = monthly.groupby('coin').volume_usd.shift()
df = df.merge(monthly[['coin', 'month', 'previous_volume']], on=['coin', 'month'])
df['revised_bp'] = 1e5 / df.previous_volume.pow(1/3)
df['static_bp'] = 25.0
for field in ['high', 'low']:
    df['next_' + field] = df.groupby('coin')[field].shift(-1)
df = df.dropna(subset=['original_bp', 'revised_bp', 'next_high', 'next_low']).copy()

for method in ['original', 'revised', 'static']:
    band = df[method + '_bp'] / 1e4
    df[method] = (df.next_low <= 1 - band) | (df.next_high >= 1 + band)

df['downside'] = df.next_low <= 1 - df.revised_bp / 1e4
df['upward_only'] = df.revised & ~df.downside

out = Path(__file__).resolve().parent / 'results'
out.mkdir(exist_ok=True)
df.to_csv(out / 'label_comparison_daily.csv', index=False)
print('\nPositive coin-days (same forecast dates):')
print(df.groupby('coin')[['original', 'revised', 'static', 'downside']].sum().astype(int).to_string())
print('\nPairwise overlap (flagged by both / flagged by either):')
for a, b in [('original', 'revised'), ('revised', 'static')]:
    both = (df[a] & df[b]).sum()
    either = (df[a] | df[b]).sum()
    print(f'{a} vs {b}: {both}/{either} = {both/either:.1%}' if either else 'No positives')

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, coin in zip(axes, ['USDT', 'USDC', 'DAI']):
    g = df[df.coin == coin]
    dates = g.date + pd.Timedelta(days=1)  # actual outcome date
    ax.plot(dates, g.next_low, lw=.7, label='Daily low')
    ax.plot(dates, g.next_high, lw=.7, label='Daily high')
    ax.scatter(dates[g.downside], g.loc[g.downside, 'next_low'], s=12, label='Revised downside', color='tab:red')
    ax.scatter(dates[g.upward_only], g.loc[g.upward_only, 'next_high'], s=12, label='Revised upward only', color='tab:orange')
    ax.axhline(1, ls='--', lw=.8, color='gray')
    ax.set(title=coin, ylabel='USD')
    ax.legend(fontsize=8, loc='best')
axes[-1].set_xlabel('Outcome date (t+1)')
fig.tight_layout()
fig.savefig(out / 'price_labels.png', dpi=160)

# summary table 
# deviation magnitude on outcome date (t+1)
df["deviation_bp"] = 10000 * np.maximum(
    (1 - df["next_low"]).clip(lower=0),
    (df["next_high"] - 1).clip(lower=0)
)

flagged = df[df["revised"]].copy()

bins = [0, 10, 20, 25, 100, float("inf")]
labels = ["0-10", "10-20", "20-25", "25-100", ">100"]

flagged["range"] = pd.cut(
    flagged["deviation_bp"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

summary = pd.crosstab(flagged["range"], flagged["coin"])

# plot deviation magnitude distribution
ax = summary[['USDT', 'USDC', 'DAI']].plot(
    kind='bar', figsize=(9, 5), width=0.8
)

ax.set(
    title='Magnitude of Revised Dynamic-Band Breaches',
    xlabel='Maximum intraday deviation from $1 (bp)',
    ylabel='Number of positive coin-days'
)

ax.tick_params(axis='x', rotation=0)
ax.legend(title='Stablecoin')

plt.tight_layout()
plt.savefig(out / 'deviation_magnitude.png', dpi=160)
plt.close()

# downside-only robustness
summary3 = df.groupby('coin')[['downside', 'upward_only']].sum()

summary3['bilateral'] = summary3.sum(axis=1)
summary3['upward_share_%'] = (
    100 * summary3['upward_only'] / summary3['bilateral']
)

print('\nDownside-only robustness:')
print(summary3.round(1))

ax = summary3[['downside', 'upward_only']].plot(
    kind='bar', stacked=False, figsize=(8, 5)
)

ax.set(
    title='Revised Dynamic-Band Breaches by Direction',
    xlabel='Stablecoin',
    ylabel='Positive coin-days'
)

ax.tick_params(axis='x', rotation=0)
ax.legend(['Downside', 'Upward only'])
plt.tight_layout()
plt.savefig(out / 'downside_robustness.png', dpi=160)
plt.close()

# identify new and continuing breaches
df = df.sort_values(['coin', 'date']).copy()

for label in ['revised', 'downside']:
    previous = df.groupby('coin')[label].shift(fill_value=False)

    df[label + '_new'] = df[label] & ~previous
    df[label + '_continuing'] = df[label] & previous

summary4 = df.groupby('coin')[
    ['revised', 'revised_new', 'revised_continuing',
     'downside', 'downside_new', 'downside_continuing']
].sum().astype(int)

print('\nBreach episode summary:')
print(summary4)

summary4.to_csv(out / 'breach_episode_summary.csv')
df.to_csv(out / 'label_comparison_daily.csv', index=False)

ax = summary4[['revised_new', 'revised_continuing']].plot(
    kind='bar', figsize=(8, 5)
)

ax.set(
    title='New vs Continuing Dynamic-Band Breaches',
    xlabel='Stablecoin',
    ylabel='Positive coin-days'
)

ax.legend(['New episodes', 'Continuing days'])
ax.tick_params(axis='x', rotation=0)

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()
plt.savefig(out / 'breach_episodes.png', dpi=160)
plt.close()