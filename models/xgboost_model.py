import pandas as pd
import os
import shutil
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, RocCurveDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Set fixed results directory
results_dir = "results/xgboost"
os.makedirs(results_dir, exist_ok=True)

# Clear existing files in directory
for file in os.listdir(results_dir):
    file_path = os.path.join(results_dir, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Error deleting {file_path}: {e}")

# Load preprocessed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")['Default']
y_test = pd.read_csv("data/y_test.csv")['Default']

# Convert to DMatrix (optimized for XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Base parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',  # Faster for large datasets
    'scale_pos_weight': len(y_train[y_train==0])/len(y_train[y_train==1]),  # Handle imbalance
    'seed': 42
}

# Simplified hyperparameter grid
param_grid = {
    'max_depth': [3, 6],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(**params)

# Reduced grid search for faster training
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Training XGBoost model...")
grid_search.fit(X_train, y_train)

# Get best estimator
best_xgb = grid_search.best_estimator_

# Make predictions
print("Making predictions...")
y_pred = best_xgb.predict(X_test)
y_probs = best_xgb.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Calculating metrics...")
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_probs),
    'Best max_depth': grid_search.best_params_['max_depth'],
    'Best learning_rate': grid_search.best_params_['learning_rate'],
    'Best subsample': grid_search.best_params_['subsample'],
    'Best colsample_bytree': grid_search.best_params_['colsample_bytree']
}
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

# Save results
print("Saving results...")
# 1. Save metrics and best parameters
metrics_df.to_csv(f"{results_dir}/metrics.csv")

# 2. Save full classification report
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{results_dir}/classification_report.csv")

# 3. Save predictions and probabilities
pd.DataFrame({
    'True': y_test,
    'Predicted': y_pred,
    'Probability': y_probs
}).to_csv(f"{results_dir}/predictions.csv", index=False)

# 4. Save feature importance
importance = best_xgb.get_booster().get_score(importance_type='weight')
pd.DataFrame({
    'Feature': list(importance.keys()),
    'Importance': list(importance.values())
}).sort_values('Importance', ascending=False).to_csv(f"{results_dir}/feature_importance.csv", index=False)

# 5. Save grid search results
pd.DataFrame(grid_search.cv_results_).to_csv(f"{results_dir}/grid_search_results.csv", index=False)

# Visualizations
print("Generating visualizations...")
# 1. Confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(f"{results_dir}/confusion_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

# 2. ROC Curve
RocCurveDisplay.from_estimator(best_xgb, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.savefig(f"{results_dir}/roc_curve.png", bbox_inches='tight', dpi=300)
plt.close()

# 3. Feature importance plot
fig, ax = plt.subplots(figsize=(10, 8))
xgb.plot_importance(best_xgb, ax=ax, max_num_features=20)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig(f"{results_dir}/feature_importance_plot.png", bbox_inches='tight', dpi=300)
plt.close()

# Save complete configuration
with open(f"{results_dir}/config.txt", 'w') as f:
    f.write("XGBoost Configuration:\n")
    f.write(f"Best parameters: {grid_search.best_params_}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write("\nFull parameter grid:\n")
    for k, v in param_grid.items():
        f.write(f"{k}: {v}\n")

# Save model
best_xgb.save_model(f"{results_dir}/xgboost_model.json")

print(f"\nAll XGBoost results saved to: {results_dir}/")
print("Files created:")
for file in os.listdir(results_dir):
    print(f"- {file}")