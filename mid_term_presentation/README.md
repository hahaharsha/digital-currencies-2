# Preserved earlier project files

The processed_data directory contains the files as they existed immediately
before repository reorganisation. These are preservation copies, not verified
original mid-term vintages. Active models use depeg_labeller/results and
regime_classifier/outputs instead. Scripts/notebooks here retain historical
paths and are not active entry points.

Early utilities moved here from the repository root (they read `dataset/` at
the root and are not part of the report pipeline):

- `build_onchain_data.py`: merges the 2022 and 2023 on-chain files into
  `processed_data/onchain_2022_2023_clean.csv`.
- `threshold_audit.py`: early depeg-band comparison, superseded by
  `depeg_labeller/label_validation.py`.
- `read_prices_csv.py`: prints coins and date range in `dataset/prices.csv`.
