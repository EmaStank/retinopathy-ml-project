#!/usr/bin/env python
"""Train and compare multiple ML models on the Messidor retinopathy dataset."""

import argparse
import os

import pandas as pd
from scipy.io import arff

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
from sklearn.model_selection import cross_val_score, train_test_split
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


def evaluate_model(name, model, X, y, X_test, y_test):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test, probabilities)

    cv_scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc",
    )

    print("\n===================")
    print(name.upper())
    print("===================")

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1-score:", f1_score(y_test, predictions))
    print("Test ROC-AUC:", test_roc_auc)

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("CV ROC-AUC scores:")
    print(cv_scores)
    print("CV Mean ROC-AUC:", cv_scores.mean())
    print("CV Std ROC-AUC:", cv_scores.std())

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1-score": f1_score(y_test, predictions),
        "Test ROC-AUC": test_roc_auc,
        "CV Mean ROC-AUC": cv_scores.mean(),
        "CV Std ROC-AUC": cv_scores.std(),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare ML models."
    )
    parser.add_argument(
        "--input",
        default="messidor_features.arff",
        help="Path to the ARFF dataset file.",
    )
    args = parser.parse_args()

    df = load_dataset(args.input)

    print("\n===================")
    print("DATASET INFO")
    print("===================")

    print("Shape:", df.shape)

    print("\nTarget distribution:")
    print(df["Class"].value_counts())
    print(df["Class"].value_counts(normalize=True))

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
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42)),
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

        result = evaluate_model(
            name=name,
            model=model,
            X=X,
            y=y,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

    os.makedirs("results", exist_ok=True)

    model_comparison = pd.DataFrame(results)
    model_comparison = model_comparison.sort_values(
        by="CV Mean ROC-AUC",
        ascending=False,
    )

    model_comparison.to_csv(
        "results/model_comparison.csv",
        index=False,
    )

    print("\n===================")
    print("MODEL COMPARISON")
    print("===================")

    print(model_comparison)

    print("\nCSV file saved successfully:")
    print("results/model_comparison.csv")


if __name__ == "__main__":
    main()