import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import statsmodels.api as sm

import os

# Create results folder path
results_dir = "results/logistic_regression"
os.makedirs(results_dir, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Load Preprocessed Train/Test Sets from CSV
# ---------------------------------------------------------------------
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()

X_final = pd.concat([X_train, X_test])
y_final = np.concatenate([y_train, y_test])
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# ---------------------------------------------------------------------
# 2. Train Logistic Regression
# ---------------------------------------------------------------------
log_reg = LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------
# 3. Evaluate Model
# ---------------------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()

print("Logistic Regression Performance:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {roc_auc:.4f}")
print("Confusion Matrix:\n", conf_matrix)

# ---------------------------------------------------------------------
# 4. Export Metrics to CSV
# ---------------------------------------------------------------------
results = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "AUC-ROC", "TP", "FP", "FN", "TN"],
    "Value": [accuracy, precision, recall, roc_auc, TP, FP, FN, TN]
})
results.to_csv(f"{results_dir}/logistic_regression_results.csv", index=False)

# ---------------------------------------------------------------------
# 5. Feature Importance
# ---------------------------------------------------------------------
importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Coefficient": log_reg.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x="Coefficient", y="Feature", data=importance, palette="coolwarm")
plt.axvline(x=0, color="black", linestyle="--")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# 6. ROC Curve & Precision-Recall Curve
# ---------------------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, color="red", lw=2)
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.grid()
plt.show()

# ---------------------------------------------------------------------
# 7. Hyperparameter Tuning
# ---------------------------------------------------------------------
param_grid = {
    "penalty": ["l1", "l2"],
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}
grid_search = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42), param_grid, cv=5, scoring="roc_auc")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
y_best_proba = best_model.predict_proba(X_test)[:, 1]

print("Best Parameters:", grid_search.best_params_)
print("Best AUC from GridSearchCV:", roc_auc_score(y_test, y_best_proba))

# ---------------------------------------------------------------------
# 8. Regularization Comparison
# ---------------------------------------------------------------------
models = {
    "L1 (Lasso)": LogisticRegression(penalty="l1", solver="liblinear", random_state=42),
    "L2 (Ridge)": LogisticRegression(penalty="l2", solver="liblinear", random_state=42),
    "ElasticNet": LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, random_state=42)
}
performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    performance[name] = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        roc_auc_score(y_test, y_proba)
    ]

perf_df = pd.DataFrame(performance, index=["Accuracy", "Precision", "Recall", "AUC-ROC"])
perf_df.to_csv(f"{results_dir}/regularization_comparison.csv", index=False)
print("Regularization Comparison:\n", perf_df)

# ---------------------------------------------------------------------
# 9. Cross-Validation for Each Regularization
# ---------------------------------------------------------------------
for name, model in models.items():
    cv_scores = cross_val_score(model, X_final, y_final, cv=kfold, scoring="roc_auc")
    print(f"{name} CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------
# 10. Statsmodels Summary
# ---------------------------------------------------------------------
X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
print(result.summary())
