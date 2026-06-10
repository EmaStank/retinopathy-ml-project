#!/usr/bin/env python
"""Explore the Messidor features dataset."""

import argparse
import os

import pandas as pd
from scipy.io import arff


def load_dataset(path: str) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")

    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)

    return df


def explore_dataset(df: pd.DataFrame, save_results: bool = True) -> None:
    print("\n===================")
    print("DATASET EXPLORATION")
    print("===================")

    print("\nFirst rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values by column:")
    print(df.isnull().sum())

    print("\nNumber of duplicated rows:")
    print(df.duplicated().sum())

    if "Class" in df.columns:
        print("\nTarget distribution:")
        target_counts = df["Class"].value_counts().sort_index()
        target_percentages = df["Class"].value_counts(normalize=True).sort_index() * 100

        print(target_counts)
        print("\nTarget distribution in percentages:")
        print(target_percentages.round(2))

    print("\nBasic statistics:")
    print(df.describe())

    if "Class" in df.columns:
        print("\nCorrelation with target:")
        correlations = df.corr(numeric_only=True)["Class"].sort_values(ascending=False)
        print(correlations)

    if save_results:
        os.makedirs("results", exist_ok=True)

        df.describe().to_csv("results/basic_statistics.csv")

        if "Class" in df.columns:
            target_summary = pd.DataFrame({
                "count": target_counts,
                "percentage": target_percentages.round(2)
            })
            target_summary.to_csv("results/target_distribution.csv")

            correlations.to_csv("results/correlation_with_target.csv")

        print("\nSaved exploration files:")
        print("results/basic_statistics.csv")
        print("results/target_distribution.csv")
        print("results/correlation_with_target.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore the Messidor features dataset."
    )
    parser.add_argument(
        "--input",
        default="messidor_features.arff",
        help="Path to the ARFF dataset file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.input)
    explore_dataset(df)


if __name__ == "__main__":
    main()
