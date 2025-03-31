import pandas as pd
import os
import shutil
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, RocCurveDisplay)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import datetime

# Set fixed results directory
results_dir = "results/svm_rbf_fast"
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

# Subsample for faster prototyping (use 10% of data)
sample_size = int(0.1 * len(X_train))
X_sample = X_train[:sample_size]
y_sample = y_train[:sample_size]

# Parameter distribution for randomized search
param_dist = {
    'C': stats.loguniform(1e-2, 1e2),  # Wider but smarter range
    'gamma': stats.loguniform(1e-4, 1e-1),  # Continuous values
    'kernel': ['rbf']
}

# Initialize optimized RBF SVM
print("Training optimized RBF SVM model...")
rbf_svm = SVC(
    class_weight='balanced',
    random_state=42,
    probability=True,
    cache_size=1000,  # Larger cache size
    shrinking=True,  # Use shrinking heuristic
    tol=0.01,  # Higher tolerance for faster convergence
    verbose=True  # Show training progress
)

# Randomized search with reduced iterations
search = RandomizedSearchCV(
    rbf_svm,
    param_dist,
    n_iter=10,  # Only 10 combinations
    cv=2,  # 2-fold CV
    n_jobs=-1,
    random_state=42,
    scoring='roc_auc'
)

search.fit(X_sample, y_sample)

# Get best estimator
best_svm = search.best_estimator_

# Make predictions on full test set
print("Making predictions...")
y_pred = best_svm.predict(X_test)
y_probs = best_svm.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Calculating metrics...")
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_probs),
    'Best C': search.best_params_['C'],
    'Best gamma': search.best_params_['gamma'],
    'Training Sample Size': sample_size
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

# 4. Save search results
pd.DataFrame(search.cv_results_).to_csv(f"{results_dir}/search_results.csv", index=False)

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
RocCurveDisplay.from_estimator(best_svm, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.savefig(f"{results_dir}/roc_curve.png", bbox_inches='tight', dpi=300)
plt.close()

# Save complete configuration
with open(f"{results_dir}/config.txt", 'w') as f:
    f.write("Optimized RBF SVM Configuration:\n")
    f.write(f"Training timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Best parameters: {search.best_params_}\n")
    f.write(f"Training samples used: {sample_size}/{len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write("\nParameter distributions:\n")
    for k, v in param_dist.items():
        f.write(f"{k}: {v.ppf([0.1, 0.9])}\n")  # Show 10th and 90th percentiles

print(f"\nAll optimized RBF SVM results saved to: {results_dir}/")
print("Files created:")
for file in os.listdir(results_dir):
    print(f"- {file}")