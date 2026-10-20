# Stablecoin depeg and regime analysis

## Layout

- `dataset/`: shared raw data. CMC and VIX CSVs remain local/ignored for now.
- `replication/replication.py`: reference-paper replication. Historical tables remain in the ignored local archive.
- `depeg_labeller/dynamic_depeg.py`: current bilateral labeller.
- `depeg_labeller/audit.ipynb`: outcome audit; outputs and labels in `depeg_labeller/results/`.
- `depeg_labeller/label_validation.py`: retained legacy comparison diagnostic. It uses its original timing/window conventions and is not the authoritative label generator.
- `regime_classifier/regime_classifier.py`: current market-regime builder.
- `regime_classifier/regime_audit.ipynb`: regime and event-study audit.
- `regime_classifier/outputs/`: regime data and audit outputs.
- `logit/logit_oos.py`: M1/M2 prediction and probability decomposition.
- `logit/interaction_test.py`: separate descriptive M3/date-clustered Wald inference.
- `logit/regime_lag_sensitivity.py`: regime-only lag checks.
- `logit/outputs/logit_decomposition/`: retained decomposition run.
- `logit/outputs/interaction_test/` and `regime_lag_sensitivity/`: retained inference/sensitivity results.
- `mid_term_presentation/`: preserved earlier scripts, EDA and processed-data snapshots.
- `local_rf/`: RF, LASSO-RF and audits/results; ignored by Git.
- `local_archive/`: retired scripts and older outputs; ignored by Git.
- `run.py`: raw on-chain coverage utility.

## Run from the repository root

```sh
python depeg_labeller/dynamic_depeg.py
python regime_classifier/regime_classifier.py
python logit/logit_oos.py
python logit/interaction_test.py --out logit/outputs/interaction_test_new
python logit/regime_lag_sensitivity.py --out logit/outputs/regime_lag_sensitivity_new
python replication/replication.py
```

The model script creates new timestamped runs. The regime notebook initially
reads `logit/outputs/logit_decomposition/`; update its RESULTS setting when
using a newer run. Both active notebooks can open from the root or their own
folders. Re-run them after moving: stored cell outputs were cleared.

Raw datasets and preserved numerical outputs are not rewritten during the
cleanup. Historical run manifests retain original paths and hashes as provenance.
The archived files are kept locally, not deleted. Changes have not been committed
or pushed automatically.
