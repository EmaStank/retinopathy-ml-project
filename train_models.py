from scipy.io import arff
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ===================
# 1. LOAD DATA
# ===================

data, meta = arff.loadarff("messidor_features.arff")
df = pd.DataFrame(data)

for col in df.select_dtypes([object]).columns:
    df[col] = df[col].str.decode("utf-8")

print("\n===================")
print("DATASET INFO")
print("===================")

print("Shape:", df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nTarget distribution:")
print(df["Class"].value_counts())
print(df["Class"].value_counts(normalize=True))

# ===================
# 2. FEATURES / TARGET
# ===================

X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

# ===================
# 3. TRAIN / TEST SPLIT
# ===================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===================
# 4. LOGISTIC REGRESSION
# ===================

log_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

log_reg.fit(X_train, y_train)

log_pred = log_reg.predict(X_test)
log_prob = log_reg.predict_proba(X_test)[:, 1]

print("\n===================")
print("LOGISTIC REGRESSION")
print("===================")

print("Accuracy:", accuracy_score(y_test, log_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, log_pred))

print("\nClassification report:")
print(classification_report(y_test, log_pred))

print("ROC-AUC:", roc_auc_score(y_test, log_prob))

# ===================
# 5. THRESHOLD = 0.35
# ===================

custom_threshold = 0.35
custom_pred = (log_prob >= custom_threshold).astype(int)

print("\n===================")
print("LOGISTIC REGRESSION - THRESHOLD 0.35")
print("===================")

print("Accuracy:", accuracy_score(y_test, custom_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, custom_pred))

print("\nClassification report:")
print(classification_report(y_test, custom_pred))

# ===================
# 6. THRESHOLD ANALYSIS
# ===================

print("\n===================")
print("THRESHOLD ANALYSIS")
print("===================")

for threshold in [0.10, 0.20, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]:
    pred = (log_prob >= threshold).astype(int)

    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print(
        f"Threshold={threshold:.2f} | "
        f"Precision={precision:.3f} | "
        f"Recall={recall:.3f} | "
        f"F1={f1:.3f}"
    )

# ===================
# 7. LOGISTIC REGRESSION CROSS-VALIDATION
# ===================

log_cv_scores = cross_val_score(
    log_reg,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

print("\n===================")
print("LOGISTIC REGRESSION CROSS-VALIDATION")
print("===================")

print("CV ROC-AUC scores:")
print(log_cv_scores)

print("Mean ROC-AUC:", log_cv_scores.mean())
print("Std:", log_cv_scores.std())

# ===================
# 8. RANDOM FOREST
# ===================

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

print("\n===================")
print("RANDOM FOREST")
print("===================")

print("Accuracy:", accuracy_score(y_test, rf_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification report:")
print(classification_report(y_test, rf_pred))

print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

# ===================
# 9. RANDOM FOREST CROSS-VALIDATION
# ===================

rf_cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring="roc_auc"
)

print("\n===================")
print("RANDOM FOREST CROSS-VALIDATION")
print("===================")

print("CV ROC-AUC scores:")
print(rf_cv_scores)

print("Mean ROC-AUC:", rf_cv_scores.mean())
print("Std:", rf_cv_scores.std())

# ===================
# 10. FEATURE IMPORTANCE
# ===================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n===================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("===================")

print(feature_importance)

# ===================
# 11. MODEL COMPARISON
# ===================

print("\n===================")
print("MODEL COMPARISON")
print("===================")

print("Logistic Regression Test ROC-AUC:", roc_auc_score(y_test, log_prob))
print("Logistic Regression CV Mean ROC-AUC:", log_cv_scores.mean())

print("Random Forest Test ROC-AUC:", roc_auc_score(y_test, rf_prob))
print("Random Forest CV Mean ROC-AUC:", rf_cv_scores.mean())