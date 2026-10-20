"""Descriptive M3 logit: within-coin scaling, coin fixed effects, weekly clusters.

Default inference: 2022 predictor AND outcome dates. All-sample inference is
an explicit retrospective option. This script does not fit predictive M1/M2.
Weeks run Monday–Sunday, grouped by predictor date t, across all coins.
"""
import argparse
import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Binomial
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import StandardScaler
from logit_oos import load_inputs, FEATURES, TARGET


def make_design(d,reference_coin=None):
    coins=sorted(d.coin.unique())
    reference_coin=reference_coin or coins[0]
    if reference_coin not in coins:raise ValueError('Reference coin not in inference sample.')
    x=pd.DataFrame(index=d.index,columns=FEATURES,dtype=float)
    records=[]
    for coin,g in d.groupby('coin'):
        scaler=StandardScaler()
        x.loc[g.index,FEATURES]=scaler.fit_transform(g[FEATURES])
        for j,f in enumerate(FEATURES):
            records.append(dict(coin=coin,feature=f,observations=len(g),
                mean_log1p=float(scaler.mean_[j]),scale_log1p=float(scaler.scale_[j]),
                variance_log1p=float(scaler.var_[j])))
    # R is binary and is not standardised. Interactions use within-coin Z.
    x['stress']=d.stress.astype(float)
    interactions=[]
    for f in FEATURES:
        name=f+'_x_stress';interactions.append(name);x[name]=x[f]*x.stress
    for coin in coins:
        if coin!=reference_coin:x['coin_'+coin]=d.coin.eq(coin).astype(float)
    return add_constant(x,has_constant='add'),interactions,pd.DataFrame(records),reference_coin


def run(root,out,sample='2022',lag=0,reference_coin=None):
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose an empty output directory.')
    panel,_,_,paths=load_inputs(root,lag)
    d=panel.copy()
    if sample=='2022':d=d[d.date.dt.year.eq(2022)&d.event_date.lt('2023-01-01')].copy()
    d=d.sort_values(['date','coin']).reset_index(drop=True)
    if d.empty or d.stress.nunique()<2 or d[TARGET].nunique()<2:
        raise ValueError('Need nonempty data, both regimes and both outcome classes.')
    # A W-SUN period ends Sunday and starts Monday. Not week numbers alone:
    # use actual start dates to avoid mixing the same week number across years.
    d['cluster_week_start']=d.date.dt.to_period('W-SUN').dt.start_time
    d['cluster_week_end']=d.cluster_week_start+pd.Timedelta(days=6)
    groups=pd.factorize(d.cluster_week_start)[0]
    n_clusters=d.cluster_week_start.nunique()
    x,interactions,scaling,reference_coin=make_design(d,reference_coin)
    out.mkdir(parents=True,exist_ok=True)
    diagnostic=d.groupby('stress').agg(coin_days=('date','size'),market_days=('date','nunique'),
        weekly_clusters=('cluster_week_start','nunique'),breaches=(TARGET,'sum'))
    positive_weeks=d[d[TARGET].eq(1)].groupby('stress').cluster_week_start.nunique()
    diagnostic['weeks_with_breaches']=positive_weeks.reindex(diagnostic.index,fill_value=0)
    diagnostic.to_csv(out/'inference_sample_by_regime.csv')
    d.groupby(['coin','stress']).agg(coin_days=('date','size'),breaches=(TARGET,'sum')).to_csv(out/'inference_sample_by_coin_regime.csv')
    scaling.to_csv(out/'within_coin_scaling.csv',index=False)
    d.to_csv(out/'inference_panel.csv',index=False)
    x.to_csv(out/'design_matrix.csv',index=False)
    print('Inference sample:',sample,'| reference coin:',reference_coin)
    print(f'{len(d)} observations; {x.shape[1]} parameters; {n_clusters} weekly clusters; {int(d[TARGET].sum())} positives.')
    print(diagnostic.to_string())
    if np.linalg.matrix_rank(x.to_numpy())<x.shape[1]:
        raise ValueError('Rank-deficient M3 design. Diagnostics saved; no Wald test reported.')
    if n_clusters<=len(interactions):
        raise ValueError('Too few weekly clusters for seven joint restrictions.')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result=GLM(d[TARGET],x,family=Binomial()).fit(maxiter=1000,tol=1e-9,
            cov_type='cluster',cov_kwds={'groups':groups,'use_correction':True},use_t=False)
    notes=[str(w.message) for w in caught]
    constants=scaling.loc[scaling.variance_log1p.eq(0),['coin','feature']]
    if not constants.empty:
        notes.append('Some predictors are constant within a coin and become zero after centering: '+str(constants.to_dict('records')))
    if not result.converged or any('separation' in s.lower() for s in notes):
        (out/'inference_failure.txt').write_text('Convergence/separation failure. '+ '; '.join(notes))
        raise RuntimeError('Convergence/separation checks failed; no valid ordinary Wald inference.')
    cov=result.cov_params().to_numpy()
    if not np.isfinite(result.params).all() or not np.isfinite(cov).all() or (np.diag(cov)<=0).any():
        raise RuntimeError('Nonfinite estimates or nonpositive variances; no Wald test reported.')
    ci=result.conf_int(alpha=.05)
    coefficients=pd.DataFrame(dict(coefficient=result.params,week_clustered_se=result.bse,
        z=result.tvalues,p_value=result.pvalues,ci_lower=ci[0],ci_upper=ci[1]))
    coefficients.index.name='parameter'
    coefficients.to_csv(out/'m3_week_clustered_coefficients.csv')
    coefficients.loc[interactions].to_csv(out/'interaction_coefficients.csv')
    slopes=[]
    for feature,interaction in zip(FEATURES,interactions):
        c=np.zeros(x.shape[1]);c[x.columns.get_loc(feature)]=1;c[x.columns.get_loc(interaction)]=1
        test=result.t_test(c)
        bounds=np.asarray(test.conf_int(alpha=.05)).reshape(-1)
        slopes.append(dict(feature=feature,normal_log_odds_slope=result.params[feature],
            normal_week_clustered_se=result.bse[feature],normal_ci_lower=ci.loc[feature,0],normal_ci_upper=ci.loc[feature,1],
            interaction_delta=result.params[interaction],interaction_ci_lower=ci.loc[interaction,0],
            interaction_ci_upper=ci.loc[interaction,1],
            stressed_log_odds_slope=float(np.asarray(test.effect).item()),
            stressed_week_clustered_se=float(np.asarray(test.sd).item()),
            stressed_ci_lower=float(bounds[0]),stressed_ci_upper=float(bounds[1])))
    pd.DataFrame(slopes).to_csv(out/'regime_specific_slopes.csv',index=False)
    R=np.zeros((len(interactions),x.shape[1]))
    for j,name in enumerate(interactions):R[j,x.columns.get_loc(name)]=1
    rcov=R@cov@R.T
    valid_wald=np.linalg.matrix_rank(rcov)==len(interactions) and np.linalg.eigvalsh(rcov).min()>0
    if n_clusters<50:notes.append('Fewer than 50 weekly clusters: asymptotic reference distributions may be inaccurate; 50 is a caution flag, not a validity cutoff.')
    if diagnostic.breaches.min()<len(FEATURES)+1:
        notes.append('One regime has fewer positive outcomes than its eight slope/intercept terms; estimates may be unstable even when convergence checks pass.')
    if (abs(result.params)>20).any():notes.append('Very large log-odds coefficients: inspect sparse outcomes and possible near separation.')
    notes.append('Week clustering allows arbitrary dependence within a week across coins/days; dependence across weeks remains unaddressed.')
    wald_chi=None;wald_f=None
    if valid_wald:
        wald=result.wald_test(R,use_f=False,scalar=True)
        wald_chi=dict(statistic=float(wald.statistic),df=len(interactions),p_value=float(wald.pvalue))
        # Supplementary F reference, not a wild-cluster bootstrap or exact test.
        test_f=result.wald_test(R,use_f=True,scalar=True)
        wald_f=dict(statistic=float(test_f.statistic),df_num=float(test_f.df_num),
                    df_denom=float(test_f.df_denom),p_value=float(test_f.pvalue))
    else:notes.append('Joint interaction covariance is singular or nonpositive definite; joint Wald test withheld.')
    report=dict(sample=sample,scaling='Within-coin z scores of log1p predictors, fitted on this inference sample; population SD',
        coin_fixed_effects=True,reference_coin=reference_coin,constant_within_coin_predictors=constants.to_dict('records'),coin_dummy_parameters=[c for c in x if c.startswith('coin_')],
        cluster_definition='Monday–Sunday calendar week of predictor date t; all coins share a week cluster',
        covariance='Cluster sandwich with finite-sample multiplier; normal/z 95% coefficient intervals',
        hypothesis='All seven on-chain-by-stress interactions equal zero; coin dummies and stress intercept unrestricted',
        wald_chi_square=wald_chi,secondary_f_reference=wald_f,
        observations=len(d),parameters=x.shape[1],positive_coin_days=int(d[TARGET].sum()),
        weekly_clusters=n_clusters,weeks_with_breaches=d.loc[d[TARGET].eq(1),'cluster_week_start'].nunique(),
        first_predictor_date=str(d.date.min().date()),last_predictor_date=str(d.date.max().date()),
        availability_lag_days=lag,design_condition_number=float(np.linalg.cond(x.to_numpy())),
        warnings=notes,descriptive_only=True,
        inputs_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (out/'wald_test.json').write_text(json.dumps(report,indent=2))
    (out/'m3_summary.txt').write_text(result.summary().as_text())
    print('\nInteraction coefficients and normal-reference 95% CIs:')
    print(coefficients.loc[interactions].round(4).to_string())
    if wald_chi:print(f"Joint weekly-clustered Wald chi-square(7)={wald_chi['statistic']:.4f}; p={wald_chi['p_value']:.6g}")
    else:print('Joint Wald test withheld: covariance diagnostics failed.')
    if wald_f:print(f"Supplementary F(7,{int(wald_f['df_denom'])})={wald_f['statistic']:.4f}; p={wald_f['p_value']:.6g}")
    for note in notes:print('Descriptive-inference note:',note)
    print('Output:',out.resolve())
    return result,report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--out',type=Path)
    p.add_argument('--sample',choices=['2022','all'],default='2022')
    p.add_argument('--availability-lag-days',type=int,default=0)
    p.add_argument('--reference-coin',default=None)
    a=p.parse_args()
    if a.availability_lag_days<0:p.error('Lag must be nonnegative.')
    dest=a.out or a.root/'logit/outputs'/('interaction_weekly_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    run(a.root,dest,a.sample,a.availability_lag_days,a.reference_coin)
