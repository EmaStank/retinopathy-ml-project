#!/usr/bin/env python
"""Tune additional tabular classifiers and a soft-voting ensemble."""

import argparse
import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tune_models import (
    calculate_metrics,
    json_safe,
    load_dataset,
    select_threshold,
)


RANDOM_STATE = 42


def elastic_net_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                solver="saga",
                l1_ratio=0.5,
                max_iter=5000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def xgboost_classifier():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def build_search_spaces():
    """Return each additional estimator and its valid search space."""
    extra_trees = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=1)
    xgboost = xgboost_classifier()
    elastic_net = elastic_net_pipeline()

    soft_voting = VotingClassifier(
        estimators=[
            (
                "extra",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=250,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
            ("elastic", elastic_net_pipeline()),
        ],
        voting="soft",
        n_jobs=1,
    )

    return {
        "Extra Trees": (
            extra_trees,
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
        "XGBoost": (
            xgboost,
            {
                "n_estimators": randint(100, 601),
                "max_depth": randint(2, 9),
                "learning_rate": loguniform(0.01, 0.3),
                "min_child_weight": loguniform(0.1, 10.0),
                "subsample": uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "gamma": loguniform(1e-5, 5.0),
                "reg_alpha": loguniform(1e-5, 10.0),
                "reg_lambda": loguniform(1e-3, 20.0),
            },
        ),
        "Linear Discriminant Analysis": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", LinearDiscriminantAnalysis()),
            ]),
            [
                {
                    "classifier__solver": ["svd"],
                    "classifier__tol": loguniform(1e-6, 1e-2),
                },
                {
                    "classifier__solver": ["lsqr", "eigen"],
                    "classifier__shrinkage": [None, "auto", 0.01, 0.1, 0.3, 0.5, 0.8],
                    "classifier__tol": loguniform(1e-6, 1e-2),
                },
            ],
        ),
        "Quadratic Discriminant Analysis": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", QuadraticDiscriminantAnalysis()),
            ]),
            {
                "classifier__reg_param": uniform(0.0, 1.0),
                "classifier__tol": loguniform(1e-6, 1e-2),
            },
        ),
        "Gaussian Naive Bayes": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]),
            {
                "classifier__var_smoothing": loguniform(1e-12, 1e-4),
            },
        ),
        "Elastic-net Logistic Regression": (
            elastic_net,
            {
                "classifier__C": loguniform(1e-4, 1e3),
                "classifier__l1_ratio": uniform(0.0, 1.0),
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Soft Voting Ensemble": (
            soft_voting,
            {
                "weights": [
                    (1, 1, 1),
                    (2, 1, 1),
                    (1, 2, 1),
                    (1, 1, 2),
                    (2, 1, 2),
                    (2, 2, 1),
                ],
                "extra__max_depth": [None, 5, 8, 12],
                "extra__min_samples_leaf": [1, 2, 4, 8],
                "xgb__max_depth": [2, 3, 4, 5],
                "xgb__learning_rate": [0.02, 0.05, 0.1],
                "elastic__classifier__C": loguniform(1e-3, 1e2),
                "elastic__classifier__l1_ratio": uniform(0.0, 1.0),
            },
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune additional tabular models for retinopathy detection."
    )
    parser.add_argument("--input", default="messidor_features.arff")
    parser.add_argument("--output-dir", default="results/tuning_models_2")
    parser.add_argument("--model-dir", default="models/tuned_models_2")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--minimum-recall", type=float, default=0.85)
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args()

    if args.iterations < 1 or args.folds < 2:
        parser.error("--iterations must be >= 1 and --folds must be >= 2.")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be in (0, 1).")
    if not 0 < args.minimum_recall <= 1:
        parser.error("--minimum-recall must be in (0, 1].")

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

    warnings.filterwarnings("ignore", category=FitFailedWarning)

    for name, (estimator, distributions) in build_search_spaces().items():
        print(f"\nTuning {name} ({args.iterations} candidates)...", flush=True)
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
            error_score="raise",
            verbose=1,
        )
        search.fit(X_train, y_train)

        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_roc_auc"
        ).to_csv(output_dir / f"{safe_name}_cv_results.csv", index=False)

        best_model = search.best_estimator_
        oof_probabilities = cross_val_predict(
            clone(best_model),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=args.jobs,
        )[:, 1]
        threshold, threshold_analysis = select_threshold(
            y_train, oof_probabilities, args.minimum_recall
        )
        threshold_analysis.to_csv(
            output_dir / f"{safe_name}_threshold_analysis.csv", index=False
        )

        test_probabilities = best_model.predict_proba(X_test)[:, 1]
        comparison_rows.extend([
            calculate_metrics(
                name, "Default threshold", y_test, test_probabilities, 0.5
            ),
            calculate_metrics(
                name,
                "CV-selected threshold",
                y_test,
                test_probabilities,
                threshold,
            ),
        ])

        best_parameters[name] = {
            "best_cv_roc_auc": search.best_score_,
            "selected_threshold": threshold,
            "parameters": {
                key: json_safe(value)
                for key, value in search.best_params_.items()
            },
        }
        joblib.dump(best_model, model_dir / f"{safe_name}.joblib")
        print(
            f"Best CV ROC-AUC: {search.best_score_:.4f}; "
            f"OOF-selected threshold: {threshold:.3f}",
            flush=True,
        )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["ROC-AUC", "F1-score"], ascending=False
    )
    comparison.to_csv(output_dir / "tuned_model_comparison.csv", index=False)
    with open(output_dir / "best_parameters.json", "w", encoding="utf-8") as file:
        json.dump(best_parameters, file, indent=2, ensure_ascii=False)

    print("\nFinal untouched-test comparison:")
    print(comparison.to_string(index=False))
    print(f"\nResults saved to {output_dir}")
    print(f"Models saved to {model_dir}")


if __name__ == "__main__":
    main()
