import pandas as pd
from pathlib import Path

# Location of dataset folder
data_dir = Path("dataset")

# Convert every .xlsx file in dataset/ to .csv
for xlsx_file in data_dir.glob("*.xlsx"):

    df = pd.read_excel(xlsx_file)

    csv_file = xlsx_file.with_suffix(".csv")

    df.to_csv(csv_file, index=False)

    print(f"Converted: {xlsx_file.name} -> {csv_file.name}")