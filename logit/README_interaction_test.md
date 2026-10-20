# Descriptive interaction logit: revised specification

Run from the repository root:

```sh
python logit/interaction_test.py
```

The default sample retains predictor dates and outcome dates in 2022. For an
explicitly retrospective pooled 2022–2023 analysis, use `--sample all`.
Results go to a fresh `logit/outputs/interaction_weekly_<timestamp>/` folder.
Existing date-clustered results are retained but belong to an older model.

## Within-coin scaling and coin effects

The seven predictors are log1p-transformed by the existing loader. For each
coin and predictor, the INFERENCE SAMPLE mean is subtracted and the result
is divided by that coin's population standard deviation (StandardScaler).
`within_coin_scaling.csv` records the means, variances, scales and sample sizes.
A predictor constant within a coin is centered to zero (scale set to one), not divided by zero. In the current sample USDT minted is constant at zero; it contributes no within-USDT variation. This is retrospective descriptive scaling, not an OOS forecast pipeline.

The model contains an intercept, seven within-coin-standardised predictors,
a binary stress intercept shift, seven predictor-by-stress interactions and
two coin dummies: 18 parameters for three coins. DAI is the default alphabetical
reference; override using `--reference-coin USDT` if desired. Coin effects and
within-coin scaling address level/scale differences, but do not establish a
causal effect or eliminate other omitted variables.

Each normal slope describes the conditional log-odds association of a
one-within-coin-SD increase in that log1p predictor. The stressed slope is
beta_j + delta_j. The interaction delta_j is the difference in these slopes,
not a probability marginal effect. Dummies are constant coin intercept shifts;
the slope coefficients are otherwise pooled across coins.

## Weekly dependence

Clusters use predictor-date calendar weeks, Monday through Sunday. The full
week start date is used, so week numbers are never mixed across years. All
coins and dates within that week belong to one cluster. `inference_panel.csv`
records week start/end for every observation. No labels or stablecoin prices
are used to select week boundaries. Changing to outcome-date weeks would be
a different prespecified boundary sensitivity, not what this default does.

The covariance is a weekly cluster sandwich with the standard finite-sample
multiplier. Coefficient CIs are normal-reference 95% intervals, and the primary
joint Wald test uses a chi-square reference. A supplementary F reference with
cluster-based denominator degrees of freedom is also reported. Neither is a
wild-cluster bootstrap or exact small-sample test. Week clustering addresses
cross-coin and cross-day dependence within each week, but not between weeks.

## Outputs

- `m3_week_clustered_coefficients.csv`: all 18 estimates, clustered SEs,
  individual z statistics/p-values and 95% CIs.
- `interaction_coefficients.csv`: the seven interactions and CIs.
- `regime_specific_slopes.csv`: normal/stressed slopes and CIs; stressed SEs
  include covariance between the normal slope and its interaction.
- `within_coin_scaling.csv`, `design_matrix.csv`, `inference_panel.csv`.
- `inference_sample_by_regime.csv` and `inference_sample_by_coin_regime.csv`.
- `wald_test.json`: joint null, weekly cluster count, positive counts, source
  hashes and warnings. The stress intercept and coin dummies are NOT jointly
  restricted in the interaction-only test.

The current default data give 645 observations, 62 positive coin-days,
18 parameters and 32 weekly clusters. There are only three normal-regime
positives, in three weeks. Weekly cluster counts within normal/stressed groups
can overlap because one week may contain both regimes; do not sum them.

Design-rank, convergence, separation-warning and covariance checks are applied.
If joint covariance is invalid, coefficient outputs remain available but the
joint test is withheld. Passing numerical checks does not guarantee stable
inference under rare events or near separation. Individual interaction tests
are unadjusted for multiple testing.

Use descriptive/exploratory language. A joint rejection does not establish
predictive improvement, causality, or large economic effects. Differences from
the older results reflect THREE changes (scaling, coin effects and clustering),
not clustering alone. M1/M2 prediction scripts have not been changed.

Implementation references:
https://www.statsmodels.org/stable/glm.html
https://github.com/statsmodels/statsmodels/blob/main/statsmodels/base/covtype.py
