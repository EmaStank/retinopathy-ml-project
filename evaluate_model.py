#!/usr/bin/env python
"""Evaluate multiple ML models on the Messidor retinopathy dataset."""

import argparse
import os

import pandas as pd
from scipy.io import arff

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_dataset(path: str) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.select_dtypes([object]).columns:
        df[col] = df[col].str.decode("utf-8")

    df["Class"] = df["Class"].astype(int)
    return df


def evaluate_model(name, model, X_test, y_test, threshold=None):
    probabilities = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        predictions = model.predict(X_test)
        threshold_label = "default"
    else:
        predictions = (probabilities >= threshold).astype(int)
        threshold_label = threshold

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n===================")
    print(f"{name} | threshold: {threshold_label}")
    print("===================")

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)
    print("ROC-AUC:", roc_auc)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    return {
        "Model": name,
        "Threshold": threshold_label,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
    }


def threshold_analysis(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    results = []

    print("\n===================")
    print("LOGISTIC REGRESSION THRESHOLD ANALYSIS")
    print("===================")

    for threshold in [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)

        print(
            f"Threshold={threshold:.2f} | "
            f"Accuracy={accuracy:.3f} | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f}"
        )

        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ML models.")
    parser.add_argument(
        "--input",
        default="messidor_features.arff",
        help="Path to the ARFF dataset file.",
    )
    args = parser.parse_args()

    df = load_dataset(args.input)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
        ),

        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=SVC(kernel="rbf", random_state=42),
                cv=3,
            )),
        ]),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),

        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test))

        if name == "Logistic Regression":
            results.append(
                evaluate_model(
                    name,
                    model,
                    X_test,
                    y_test,
                    threshold=0.35,
                )
            )

            threshold_df = threshold_analysis(model, X_test, y_test)

    os.makedirs("results", exist_ok=True)

    evaluation_results = pd.DataFrame(results)
    evaluation_results.to_csv("results/evaluation_results.csv", index=False)

    threshold_df.to_csv(
        "results/logistic_regression_threshold_analysis.csv",
        index=False,
    )

    print("\nEvaluation results saved to:")
    print("results/evaluation_results.csv")
    print("results/logistic_regression_threshold_analysis.csv")


if __name__ == "__main__":
    main()