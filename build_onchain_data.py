import pandas as pd
import numpy as np
from pathlib import Path


# set paths

PROJECT_ROOT = Path(__file__).resolve().parent

ONCHAIN_2022_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "onchain_2022_base.csv"
)

ONCHAIN_2023_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "onchain_2023_base.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "onchain_2022_2023_clean.csv"
)


# find all the common columns

COMMON_COLUMNS = [
    "date",
    "coin",
    "n_transfers",
    "onchain_volume",
    "n_unique_senders",
    "n_unique_receivers",
    "minted",
    "burned",
    "volume_large_transfers",
]


RAW_FEATURES = [
    "n_transfers",
    "onchain_volume",
    "n_unique_senders",
    "n_unique_receivers",
    "minted",
    "burned",
    "volume_large_transfers",
]


# load data

def load_onchain_data():

    onchain_2022 = pd.read_csv(
        ONCHAIN_2022_PATH,
        parse_dates=["date"],
    )

    onchain_2023 = pd.read_csv(
        ONCHAIN_2023_PATH,
        parse_dates=["date"],
    )

    print("\n=== RAW ON-CHAIN DATA ===")

    print(
        f"2022 observations: {len(onchain_2022):,}"
    )

    print(
        f"2023 observations: {len(onchain_2023):,}"
    )

    return onchain_2022, onchain_2023


# check the columsn

def check_required_columns(df, required_columns, dataset_name):

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


# harmonise columns and retain required columns 

def harmonise_data(onchain_2022, onchain_2023):

    check_required_columns(
        onchain_2022,
        COMMON_COLUMNS,
        "2022 dataset",
    )

    check_required_columns(
        onchain_2023,
        COMMON_COLUMNS,
        "2023 dataset",
    )

    onchain_2022 = (
        onchain_2022[
            COMMON_COLUMNS
        ]
        .copy()
    )

    onchain_2023 = (
        onchain_2023[
            COMMON_COLUMNS
        ]
        .copy()
    )

    # keep source year for diagnostics
    onchain_2022["data_year"] = 2022
    onchain_2023["data_year"] = 2023

    return onchain_2022, onchain_2023


# combine data by years

def combine_data(onchain_2022, onchain_2023):

    onchain = pd.concat(
        [
            onchain_2022,
            onchain_2023,
        ],
        ignore_index=True,
    )

    onchain = (
        onchain
        .sort_values(
            [
                "coin",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    print("\n=== COMBINED RAW PANEL ===")

    print(
        f"Observations before duplicate removal: "
        f"{len(onchain):,}"
    )

    duplicate_count = (
        onchain
        .duplicated(
            subset=[
                "coin",
                "date",
            ]
        )
        .sum()
    )

    print(
        f"Duplicate coin-date observations: "
        f"{duplicate_count:,}"
    )

    # keep first observation if duplicate coin-date rows exist
    onchain = (
        onchain
        .drop_duplicates(
            subset=[
                "coin",
                "date",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print(
        f"Observations after duplicate removal: "
        f"{len(onchain):,}"
    )

    return onchain


# clean variables

def clean_numeric_features(onchain):

    for col in RAW_FEATURES:

        onchain[col] = pd.to_numeric(
            onchain[col],
            errors="coerce",
        )

        # counts and volumes should not be negative.
        # invalid negative observations are converted to NaN.
        negative_count = (
            onchain[col] < 0
        ).sum()

        if negative_count > 0:

            print(
                f"Warning: {negative_count} negative "
                f"values found in {col}. "
                f"Converted to NaN."
            )

            onchain.loc[
                onchain[col] < 0,
                col,
            ] = np.nan

    return onchain


# feature engineering

def engineer_features(onchain):

    # net issuance

    onchain["net_issuance"] = (
        onchain["minted"]
        -
        onchain["burned"]
    )


    # log trainsformation 

    for col in RAW_FEATURES:

        onchain[
            f"log1p_{col}"
        ] = np.log1p(
            onchain[col]
        )


    # log transformation with sign for net issuance
    # net issuance may be positive, negative, or zero.
    # sign(x) * log(1 + |x|)
    # this preserves direction while compressing magnitude.
   
    onchain[
        "signed_log_net_issuance"
    ] = (
        np.sign(
            onchain["net_issuance"]
        )
        *
        np.log1p(
            np.abs(
                onchain["net_issuance"]
            )
        )
    )

    return onchain


# summary of dataset

def print_dataset_summary(onchain):

    print("\n=== CLEANED ON-CHAIN PANEL ===")

    print(
        f"Observations: {len(onchain):,}"
    )

    print(
        "Date range:",
        onchain["date"].min().date(),
        "to",
        onchain["date"].max().date(),
    )


    print("\n=== OBSERVATIONS BY COIN ===")

    print(
        onchain[
            "coin"
        ]
        .value_counts()
        .sort_index()
    )


    print("\n=== OBSERVATIONS BY SOURCE YEAR ===")

    print(
        onchain[
            "data_year"
        ]
        .value_counts()
        .sort_index()
    )


# coverage by coin

def print_coin_coverage(onchain):

    coverage = (
        onchain
        .groupby("coin")
        .agg(
            first_date=(
                "date",
                "min",
            ),
            last_date=(
                "date",
                "max",
            ),
            observations=(
                "date",
                "size",
            ),
            unique_dates=(
                "date",
                "nunique",
            ),
        )
        .sort_index()
    )

    print("\n=== DATE COVERAGE BY COIN ===")

    print(
        coverage
    )


# missing values -- IMP

def print_missing_values(onchain):

    columns_to_check = (
        RAW_FEATURES
        +
        ["net_issuance"]
    )

    missing = (
        onchain[
            columns_to_check
        ]
        .isna()
        .sum()
    )

    missing_pct = (
        onchain[
            columns_to_check
        ]
        .isna()
        .mean()
        * 100
    )

    missing_table = pd.DataFrame(
        {
            "missing_n":
                missing,

            "missing_pct":
                missing_pct,
        }
    )

    print("\n=== MISSING VALUES ===")

    print(
        missing_table.round(3)
    )


# descriptive stats

def print_descriptive_statistics(onchain):

    columns = (
        RAW_FEATURES
        +
        ["net_issuance"]
    )

    summary = (
        onchain[
            columns
        ]
        .describe(
            percentiles=[
                0.01,
                0.25,
                0.50,
                0.75,
                0.99,
            ]
        )
        .T
    )

    print("\n=== RAW FEATURE SUMMARY ===")

    print(
        summary.round(3)
    )


# stats by coin

def print_coin_statistics(onchain):

    print("\n===MEDIAN ON-CHAIN ACTIVITY BY COIN===")

    coin_summary = (
        onchain
        .groupby("coin")[
            RAW_FEATURES
            +
            ["net_issuance"]
        ]
        .median()
    )

    print(
        coin_summary.round(3)
    )


# zero counts -- for minted/burned vars


def print_zero_counts(onchain):

    zero_count = (
        onchain[
            RAW_FEATURES
        ]
        .eq(0)
        .sum()
    )

    zero_pct = (
        onchain[
            RAW_FEATURES
        ]
        .eq(0)
        .mean()
        * 100
    )

    zero_table = pd.DataFrame(
        {
            "zero_n":
                zero_count,

            "zero_pct":
                zero_pct,
        }
    )

    print("\n=== ZERO VALUES ===")

    print(
        zero_table.round(3)
    )


# correlation diagnostic 

def print_feature_correlations(onchain):

    model_features = [
        "log1p_n_transfers",
        "log1p_onchain_volume",
        "log1p_n_unique_senders",
        "log1p_n_unique_receivers",
        "log1p_minted",
        "log1p_burned",
        "log1p_volume_large_transfers",
        "signed_log_net_issuance",
    ]

    correlation_matrix = (
        onchain[
            model_features
        ]
        .corr()
    )

    print("\n=== CORRELATION MATRIX: MODEL FEATURES ===")

    print(
        correlation_matrix.round(3)
    )


# data integrity check

def final_integrity_check(onchain):

    print("\n=== FINAL INTEGRITY CHECK ===")

    duplicates = (
        onchain
        .duplicated(
            subset=[
                "coin",
                "date",
            ]
        )
        .sum()
    )

    print(
        f"Duplicate coin-date rows: {duplicates:,}"
    )


    invalid_dates = (
        onchain["date"]
        .isna()
        .sum()
    )

    print(
        f"Missing dates: {invalid_dates:,}"
    )


    missing_coins = (
        onchain["coin"]
        .isna()
        .sum()
    )

    print(
        f"Missing coin identifiers: {missing_coins:,}"
    )


    negative_raw_values = (
        onchain[
            RAW_FEATURES
        ]
        .lt(0)
        .sum()
        .sum()
    )

    print(
        f"Remaining negative raw values: "
        f"{negative_raw_values:,}"
    )


# save to dataset

def save_dataset(onchain):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    onchain.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n=== DATA SAVED ===")

    print(
        OUTPUT_PATH
    )


# main

def main():

    # load
    onchain_2022, onchain_2023 = (
        load_onchain_data()
    )


    # harmonise schemas
    onchain_2022, onchain_2023 = (
        harmonise_data(
            onchain_2022,
            onchain_2023,
        )
    )


    # combine years
    onchain = combine_data(
        onchain_2022,
        onchain_2023,
    )


    # clean numeric features
    onchain = clean_numeric_features(
        onchain
    )


    # engineer features
    onchain = engineer_features(
        onchain
    )


    # eda and diagnostics
    print_dataset_summary(
        onchain
    )

    print_coin_coverage(
        onchain
    )

    print_missing_values(
        onchain
    )

    print_descriptive_statistics(
        onchain
    )

    print_coin_statistics(
        onchain
    )

    print_zero_counts(
        onchain
    )

    print_feature_correlations(
        onchain
    )

    final_integrity_check(
        onchain
    )


    # Save final panel
    save_dataset(
        onchain
    )


if __name__ == "__main__":
    main()