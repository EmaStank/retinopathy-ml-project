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


def save_model_comparison_plot():
    path = "results/model_comparison.csv"

    if not os.path.exists(path):
        print("Skipping model comparison plot: results/model_comparison.csv not found.")
        return

    df = pd.read_csv(path)
    df = df.sort_values("CV Mean ROC-AUC", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(df["Model"], df["CV Mean ROC-AUC"])
    ax.set_xlabel("CV Mean ROC-AUC")
    ax.set_ylabel("Model")
    ax.set_title("Model Comparison by CV ROC-AUC")
    ax.set_xlim(0, 1)

    fig.savefig("results/plots/model_comparison_roc_auc.png", bbox_inches="tight")
    plt.close(fig)


def save_threshold_plots():
    path = "results/logistic_regression_threshold_analysis.csv"

    if not os.path.exists(path):
        print(
            "Skipping threshold plots: "
            "results/logistic_regression_threshold_analysis.csv not found."
        )
        return

    df = pd.read_csv(path)

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["Threshold"], df[metric], marker="o")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric)
        ax.set_title(f"Logistic Regression: Threshold vs {metric}")
        ax.set_ylim(0, 1)
        ax.grid(True)

        filename = metric.lower().replace("-", "_")
        fig.savefig(
            f"results/plots/threshold_vs_{filename}.png",
            bbox_inches="tight",
        )
        plt.close(fig)


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
        ("classifier", LogisticRegression(max_iter=2000)),
    ])

    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    )

    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    # ROC curve
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

    # Confusion matrix - Logistic Regression
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

    # Confusion matrix - Random Forest
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

    # Feature importance
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

    # New plots from saved CSV files
    save_model_comparison_plot()
    save_threshold_plots()

    print("\nPlots saved successfully:")
    print("results/plots/roc_curve.png")
    print("results/plots/confusion_matrix_logistic_regression.png")
    print("results/plots/confusion_matrix_random_forest.png")
    print("results/plots/feature_importance.png")
    print("results/plots/model_comparison_roc_auc.png")
    print("results/plots/threshold_vs_accuracy.png")
    print("results/plots/threshold_vs_precision.png")
    print("results/plots/threshold_vs_recall.png")
    print("results/plots/threshold_vs_f1_score.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ML project visualizations."
    )
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