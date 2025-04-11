import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import joblib  # For saving and loading models

# Load preprocessed CSV data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")['Default']
y_test = pd.read_csv("data/y_test.csv")['Default']


# Initialize a list to store metrics
metrics_list = []

# Decision Tree Classifier (Default/Imbalanced)
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)
dt_default_preds = dt_default.predict(X_test)
dt_default_probs = dt_default.predict_proba(X_test)[:, 1]

dt_default_accuracy = accuracy_score(y_test, dt_default_preds)
dt_default_precision = precision_score(y_test, dt_default_preds)
dt_default_recall = recall_score(y_test, dt_default_preds)
dt_default_auc = roc_auc_score(y_test, dt_default_probs)

metrics_list.append({
    "Model": "Decision Tree (Default)",
    "Accuracy": dt_default_accuracy,
    "Precision": dt_default_precision,
    "Recall": dt_default_recall,
    "AUC": dt_default_auc
})

plt.figure(figsize=(20, 10))
plot_tree(dt_default,
          feature_names=X_train.columns,
          class_names=[str(cls) for cls in np.unique(y_train)],
          filled=True, rounded=True)
plt.title("Decision Tree (Default)")
plt.savefig("tree_images/decision_tree_default.png")
# plt.show()

# Decision Tree Classifier (Balanced)
dt_balanced = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_balanced.fit(X_train, y_train)
dt_balanced_preds = dt_balanced.predict(X_test)
dt_balanced_probs = dt_balanced.predict_proba(X_test)[:, 1]

dt_balanced_accuracy = accuracy_score(y_test, dt_balanced_preds)
dt_balanced_precision = precision_score(y_test, dt_balanced_preds)
dt_balanced_recall = recall_score(y_test, dt_balanced_preds)
dt_balanced_auc = roc_auc_score(y_test, dt_balanced_probs)

metrics_list.append({
    "Model": "Decision Tree (Balanced)",
    "Accuracy": dt_balanced_accuracy,
    "Precision": dt_balanced_precision,
    "Recall": dt_balanced_recall,
    "AUC": dt_balanced_auc
})

plt.figure(figsize=(20, 10))
plot_tree(dt_balanced,
          feature_names=X_train.columns,
          class_names=[str(cls) for cls in np.unique(y_train)],
          filled=True, rounded=True)
plt.title("Decision Tree (Balanced)")
plt.savefig("tree_images/decision_tree_balanced.png")
# plt.show()

# Random Forest Classifier (Default/Imbalanced)
rf_default = RandomForestClassifier(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)
rf_default_preds = rf_default.predict(X_test)
rf_default_probs = rf_default.predict_proba(X_test)[:, 1]

rf_default_accuracy = accuracy_score(y_test, rf_default_preds)
rf_default_precision = precision_score(y_test, rf_default_preds)
rf_default_recall = recall_score(y_test, rf_default_preds)
rf_default_auc = roc_auc_score(y_test, rf_default_probs)

metrics_list.append({
    "Model": "Random Forest (Default)",
    "Accuracy": rf_default_accuracy,
    "Precision": rf_default_precision,
    "Recall": rf_default_recall,
    "AUC": rf_default_auc
})

importances_default = rf_default.feature_importances_
indices_default = np.argsort(importances_default)[::-1]
features = X_train.columns

plt.figure(figsize=(10, 6))
plt.title("Random Forest (Default) Feature Importances")
plt.bar(range(len(importances_default)), importances_default[indices_default], align="center")
plt.xticks(range(len(importances_default)), features[indices_default], rotation=90)
plt.tight_layout()
plt.savefig("tree_images/rf_default_feature_importance.png")
# plt.show()

# Random Forest Classifier (Balanced)
rf_balanced = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_balanced.fit(X_train, y_train)
rf_balanced_preds = rf_balanced.predict(X_test)
rf_balanced_probs = rf_balanced.predict_proba(X_test)[:, 1]

rf_balanced_accuracy = accuracy_score(y_test, rf_balanced_preds)
rf_balanced_precision = precision_score(y_test, rf_balanced_preds)
rf_balanced_recall = recall_score(y_test, rf_balanced_preds)
rf_balanced_auc = roc_auc_score(y_test, rf_balanced_probs)

metrics_list.append({
    "Model": "Random Forest (Balanced)",
    "Accuracy": rf_balanced_accuracy,
    "Precision": rf_balanced_precision,
    "Recall": rf_balanced_recall,
    "AUC": rf_balanced_auc
})

importances_balanced = rf_balanced.feature_importances_
indices_balanced = np.argsort(importances_balanced)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Random Forest (Balanced) Feature Importances")
plt.bar(range(len(importances_balanced)), importances_balanced[indices_balanced], align="center")
plt.xticks(range(len(importances_balanced)), features[indices_balanced], rotation=90)
plt.tight_layout()
plt.savefig("tree_images/rf_balanced_feature_importance.png")
# plt.show()

# Create a comparison metrics table
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv("model_comparison_metrics.csv", index=False)
print("Model Comparison Metrics:")
print(metrics_df)

# Save the trained models for future use
joblib.dump(dt_default, 'saved_trees/decision_tree_default.pkl')
joblib.dump(dt_balanced, 'saved_trees/decision_tree_balanced.pkl')
joblib.dump(rf_default, 'saved_trees/random_forest_default.pkl')
joblib.dump(rf_balanced, 'saved_trees/random_forest_balanced.pkl')


# Later, to load them using:
# dt_model = joblib.load('decision_tree_default.pkl')
# rf_model = joblib.load('decision_tree_balanced.pkl')
# etc...


# Note to self: Interpretability & Ensemble Concepts 

# Decision Trees are highly interpretable because their decision process can be visualized
# as a tree structure, showing how each feature is used to split the data. This makes it easier
# to understand the rationale behind predictions.
#
# Random Forests, on the other hand, are an ensemble of decision trees built using bagging (bootstrap aggregation).
# Bagging involves training each tree on a random subset of the data (with replacement) which reduces variance
# and helps prevent overfitting. While individual trees are interpretable, the ensemble average is more complex,
# but overall it typically achieves better performance due to its robustness and improved generalization.

