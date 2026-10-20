
import argparse
import hashlib
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from logit_oos import load_inputs, FEATURES, TARGET


def run(root, out, sample='2022', lag=0):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory to preserve prior results.')
    panel, coverage, invalid, paths = load_inputs(root, lag)
    d = panel.copy()
    if sample == '2022':
        d = d[d.date.dt.year.eq(2022) & d.event_date.lt('2023-01-01')].copy()
    if d.empty or d.stress.nunique() < 2 or d[TARGET].nunique() < 2:
        raise ValueError('Need observations, both regimes, and both outcome classes.')
    diagnostic = d.groupby('stress').agg(
        coin_days=('date', 'size'), date_clusters=('date', 'nunique'),
        breaches=(TARGET, 'sum'))
    print('Inference sample:', sample)
    print(diagnostic.to_string())
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(d[FEATURES]), columns=FEATURES, index=d.index)
    x['stress'] = d.stress
    interactions = []
    for f in FEATURES:
        name = f + '_x_stress'
        interactions.append(name)
        x[name] = x[f] * x.stress
    x = add_constant(x, has_constant='add')
    if np.linalg.matrix_rank(x.to_numpy()) < x.shape[1]:
        raise ValueError('Rank-deficient M3 design; interactions cannot be tested as specified.')
    groups = pd.factorize(d.date)[0]
    if len(np.unique(groups)) <= x.shape[1]:
        raise ValueError('Too few date clusters relative to model parameters.')
    # ordinary unpenalized likelihood: do not attach a standard Wald test to
    # sklearn's penalized coefficients or fall back silently to a penalty.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = GLM(d[TARGET], x, family=Binomial()).fit(
            maxiter=1000, tol=1e-9, cov_type='cluster',
            cov_kwds={'groups': groups, 'use_correction': True}, use_t=False)
    notes = [str(w.message) for w in caught]
    if not result.converged or any('separation' in t.lower() for t in notes):
        raise RuntimeError('M3 failed convergence/separation checks. No valid ordinary Wald test. '
                           + '; '.join(notes))
    cov = result.cov_params().to_numpy()
    if not np.isfinite(result.params).all() or not np.isfinite(cov).all() or (np.diag(cov) <= 0).any():
        raise RuntimeError('Nonfinite or nonpositive parameter variances; no valid Wald test.')
    R = np.zeros((len(interactions), x.shape[1]))
    for j, name in enumerate(interactions):
        R[j, x.columns.get_loc(name)] = 1
    restriction_cov = R @ cov @ R.T
    if np.linalg.matrix_rank(restriction_cov) < len(interactions):
        raise RuntimeError('Singular interaction covariance; cannot test all restrictions.')
    if np.linalg.eigvalsh(restriction_cov).min() <= 0:
        raise RuntimeError('Interaction covariance is not positive definite.')
    wald = result.wald_test(R, use_f=False, scalar=True)
    ci = result.conf_int()
    coefficients = pd.DataFrame({'coefficient': result.params, 'date_clustered_se': result.bse,
        'z': result.tvalues, 'p_value': result.pvalues, 'ci_lower': ci[0], 'ci_upper': ci[1]})
    slopes = []
    for f, interaction in zip(FEATURES, interactions):
        contrast = np.zeros(x.shape[1]); contrast[x.columns.get_loc(f)] = 1
        contrast[x.columns.get_loc(interaction)] = 1
        test = result.t_test(contrast)
        slopes.append({'feature': f, 'normal_log_odds_slope': result.params[f],
            'normal_clustered_se': result.bse[f], 'interaction_delta': result.params[interaction],
            'stressed_log_odds_slope': float(np.asarray(test.effect).item()),
            'stressed_clustered_se': float(np.asarray(test.sd).item()),
            'stressed_p_value': float(np.asarray(test.pvalue).item())})
    report = {'sample': sample, 'hypothesis': 'all seven X-by-stress interaction coefficients equal zero',
        'wald_chi_square': float(wald.statistic), 'df': len(interactions),
        'p_value': float(wald.pvalue), 'observations': len(d),
        'date_clusters': d.date.nunique(), 'first_predictor_date': str(d.date.min().date()),
        'last_predictor_date': str(d.date.max().date()), 'availability_lag_days': lag,
        'design_condition_number': float(np.linalg.cond(x.to_numpy())),
        'warnings': notes,
        'limitation': 'Date clustering allows within-day cross-coin dependence, not dependence across dates. '
                      'Rare events, few stress spells, or near separation can undermine asymptotic Wald inference.',
        'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    if diagnostic.breaches.min() < len(FEATURES) + 1:
        report['warnings'].append('One regime has fewer breaches than regime-specific slope/intercept parameters; '
                                  'inference is sparse and potentially unstable.')
    out.mkdir(parents=True, exist_ok=True)
    coefficients.to_csv(out / 'm3_date_clustered_coefficients.csv')
    coefficients.loc[interactions].to_csv(out / 'interaction_coefficients.csv')
    pd.DataFrame(slopes).to_csv(out / 'regime_specific_slopes.csv', index=False)
    diagnostic.to_csv(out / 'inference_sample_by_regime.csv')
    d.to_csv(out / 'inference_panel.csv', index=False)
    (out / 'wald_test.json').write_text(json.dumps(report, indent=2))
    (out / 'm3_summary.txt').write_text(result.summary().as_text())
    print(coefficients.loc[interactions].round(4).to_string())
    print(f"Joint interaction Wald chi-square({len(interactions)}) = {float(wald.statistic):.4f}; p={float(wald.pvalue):.6g}")
    for note in report['warnings']:
        print('Inference note:', note)
    print('Descriptive inference only; this is not evidence of improved OOS prediction.')
    return result, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--out', type=Path, default=None)
    parser.add_argument('--sample', choices=['2022', 'all'], default='2022')
    parser.add_argument('--availability-lag-days', type=int, default=0)
    args = parser.parse_args()
    if args.availability_lag_days < 0:
        parser.error('Lag must be nonnegative.')
    run(args.root, args.out or args.root / 'logit/outputs/interaction_test', args.sample, args.availability_lag_days)
