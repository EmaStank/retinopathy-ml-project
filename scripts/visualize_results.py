#!/usr/bin/env python
"""Create visualizations for the Messidor retinopathy ML project."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(path: str) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")

    df["Class"] = df["Class"].astype(int)

    return df


def create_plots(input_path: str) -> None:
    os.makedirs("results/plots", exist_ok=True)

    df = load_dataset(input_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    )

    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # 1. ROC CURVE
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(
        log_reg,
        X_test,
        y_test,
        name="Logistic Regression",
        ax=ax,
    )
    RocCurveDisplay.from_estimator(
        rf,
        X_test,
        y_test,
        name="Random Forest",
        ax=ax,
    )
    ax.set_title("ROC Curve")
    fig.savefig("results/plots/roc_curve.png", bbox_inches="tight")
    plt.close(fig)

    # 2. CONFUSION MATRIX - LOGISTIC REGRESSION
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        log_reg,
        X_test,
        y_test,
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Logistic Regression")
    fig.savefig(
        "results/plots/confusion_matrix_logistic_regression.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # 3. CONFUSION MATRIX - RANDOM FOREST
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        rf,
        X_test,
        y_test,
        ax=ax,
    )
    ax.set_title("Confusion Matrix - Random Forest")
    fig.savefig(
        "results/plots/confusion_matrix_random_forest.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    # 4. FEATURE IMPORTANCE
    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_,
    }).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(feature_importance["Feature"], feature_importance["Importance"])
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest Feature Importance")
    fig.savefig("results/plots/feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    print("\nPlots saved successfully:")
    print("results/plots/roc_curve.png")
    print("results/plots/confusion_matrix_logistic_regression.png")
    print("results/plots/confusion_matrix_random_forest.png")
    print("results/plots/feature_importance.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create ML project visualizations.")
    parser.add_argument(
        "--input",
        default="messidor_features.arff",
        help="Path to the ARFF dataset file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_plots(args.input)


if __name__ == "__main__":
    main()
