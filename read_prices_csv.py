import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent
prices_path = project_root / "dataset" / "prices.csv"

prices = pd.read_csv(prices_path)

print("\nCoins:")
print(prices["coin"].unique())

print("\nCounts by coin:")
print(prices["coin"].value_counts())

print("\nDate range:")
print("Start:", prices["date"].min())
print("End:", prices["date"].max())