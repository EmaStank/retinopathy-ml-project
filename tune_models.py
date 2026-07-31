#!/usr/bin/env python
"""Tune the four strongest classifiers without using the final test set."""

import argparse
import json
import os
from pathlib import Path

# Avoid a harmless joblib warning on Windows systems where physical-core
# detection is unavailable. This does not limit --jobs chosen by the user.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import numpy as np
import pandas as pd
from scipy.io import arff
from scipy.stats import loguniform, randint
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42


def load_dataset(path: str) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for column in df.select_dtypes([object]).columns:
        df[column] = df[column].str.decode("utf-8")

    df["Class"] = df["Class"].astype(int)
    return df


def build_search_spaces():
    """Return estimators and intentionally broad, computationally modest spaces."""
    return {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        solver="saga",
                        l1_ratio=0.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            {
                "classifier__C": loguniform(1e-4, 1e3),
                # 0.0 is pure L2, 1.0 is pure L1; intermediate values
                # test elastic-net regularization in scikit-learn >= 1.8.
                "classifier__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "SVM": (
            Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            kernel="rbf",
                            random_state=RANDOM_STATE,
                        ),
                        cv=3,
                        ensemble=False,
                    ),
                ),
            ]),
            {
                "classifier__estimator__C": loguniform(1e-3, 1e3),
                "classifier__estimator__gamma": loguniform(1e-5, 1e1),
                "classifier__estimator__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {
                "n_estimators": randint(200, 1001),
                "max_depth": [None, 3, 5, 8, 12, 16, 24],
                "max_features": ["sqrt", "log2", None, 0.5, 0.75],
                "min_samples_split": randint(2, 21),
                "min_samples_leaf": randint(1, 11),
                "class_weight": [None, "balanced", "balanced_subsample"],
                "criterion": ["gini", "entropy", "log_loss"],
            },
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "learning_rate": loguniform(0.01, 0.3),
                "max_iter": randint(100, 501),
                "max_leaf_nodes": randint(7, 64),
                "max_depth": [None, 3, 5, 8, 12],
                "min_samples_leaf": randint(5, 51),
                "l2_regularization": loguniform(1e-5, 10.0),
                "class_weight": [None, "balanced"],
            },
        ),
    }


def select_threshold(y_true, probabilities, minimum_recall: float):
    """Maximize F1 among thresholds that satisfy the requested recall."""
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "Threshold": threshold,
            "Precision": precision_score(y_true, predictions, zero_division=0),
            "Recall": recall_score(y_true, predictions, zero_division=0),
            "F1-score": f1_score(y_true, predictions, zero_division=0),
            "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        })

    analysis = pd.DataFrame(rows)
    eligible = analysis[analysis["Recall"] >= minimum_recall]
    candidates = eligible if not eligible.empty else analysis
    best_row = candidates.sort_values(
        ["F1-score", "Balanced Accuracy", "Threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["Threshold"]), analysis


def calculate_metrics(name, variant, y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

    return {
        "Model": name,
        "Prediction variant": variant,
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall (Sensitivity)": recall_score(
            y_true, predictions, zero_division=0
        ),
        "Specificity": tn / (tn + fp),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def json_safe(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Tune Logistic Regression, SVM, Random Forest and HistGB."
    )
    parser.add_argument("--input", default="messidor_features.arff")
    parser.add_argument("--output-dir", default="results/tuning")
    parser.add_argument("--model-dir", default="models/tuned")
    parser.add_argument(
        "--iterations",
        type=int,
        default=40,
        help="Random parameter combinations tested for each model.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--minimum-recall",
        type=float,
        default=0.85,
        help="Recall constraint used when selecting a decision threshold.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=-1,
        help="Parallel RandomizedSearchCV jobs; use 1 for low-memory systems.",
    )
    args = parser.parse_args()

    if not 0 < args.minimum_recall <= 1:
        parser.error("--minimum-recall must be in (0, 1].")
    if args.iterations < 1 or args.folds < 2:
        parser.error("--iterations must be >= 1 and --folds must be >= 2.")

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.input)
    X = df.drop(columns="Class")
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        "f1": "f1",
        "balanced_accuracy": "balanced_accuracy",
    }
    comparison_rows = []
    best_parameters = {}

    for name, (estimator, distributions) in build_search_spaces().items():
        print(f"\nTuning {name} ({args.iterations} candidates)...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=distributions,
            n_iter=args.iterations,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=args.jobs,
            return_train_score=False,
            verbose=1,
        )
        search.fit(X_train, y_train)

        safe_name = name.lower().replace(" ", "_")
        cv_results = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_roc_auc"
        )
        cv_results.to_csv(
            output_dir / f"{safe_name}_cv_results.csv", index=False
        )

        best_model = search.best_estimator_
        oof_probabilities = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=args.jobs,
        )[:, 1]
        selected_threshold, threshold_analysis = select_threshold(
            y_train,
            oof_probabilities,
            args.minimum_recall,
        )
        threshold_analysis.to_csv(
            output_dir / f"{safe_name}_threshold_analysis.csv",
            index=False,
        )

        test_probabilities = best_model.predict_proba(X_test)[:, 1]
        comparison_rows.append(
            calculate_metrics(
                name, "Default threshold", y_test, test_probabilities, 0.5
            )
        )
        comparison_rows.append(
            calculate_metrics(
                name,
                "CV-selected threshold",
                y_test,
                test_probabilities,
                selected_threshold,
            )
        )

        best_parameters[name] = {
            "best_cv_roc_auc": search.best_score_,
            "selected_threshold": selected_threshold,
            "parameters": {
                key: json_safe(value)
                for key, value in search.best_params_.items()
            },
        }
        joblib.dump(best_model, model_dir / f"{safe_name}.joblib")
        print(
            f"Best CV ROC-AUC: {search.best_score_:.4f}; "
            f"OOF-selected threshold: {selected_threshold:.3f}"
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["ROC-AUC", "F1-score"], ascending=False
    )
    comparison.to_csv(output_dir / "tuned_model_comparison.csv", index=False)
    with open(
        output_dir / "best_parameters.json", "w", encoding="utf-8"
    ) as file:
        json.dump(best_parameters, file, indent=2, ensure_ascii=False)

    print("\nFinal untouched-test comparison:")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to {os.fspath(output_dir)}")
    print(f"Models saved to {os.fspath(model_dir)}")


if __name__ == "__main__":
    main()
