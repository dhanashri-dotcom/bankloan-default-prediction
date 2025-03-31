import pandas as pd
import os
import shutil
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, RocCurveDisplay)
import matplotlib.pyplot as plt
import seaborn as sns

# Set fixed results directory
results_dir = "results/svm"

# Clear existing directory if it exists
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

# Load preprocessed data
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")['Default']
y_test = pd.read_csv("data/y_test.csv")['Default']

# Initialize and train Linear SVM model
print("Training Linear SVM model...")
linear_svm = LinearSVC(
    C=1.0, 
    penalty='l2', 
    loss='squared_hinge', 
    dual=False,
    class_weight='balanced',
    random_state=42,
    max_iter=1000,
    verbose=1
)
linear_svm.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = linear_svm.predict(X_test)
y_scores = linear_svm.decision_function(X_test)

# Evaluation metrics
print("Calculating metrics...")
metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'ROC AUC': roc_auc_score(y_test, y_scores)
}
metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])

# Save results
print("Saving results...")
# 1. Save metrics
metrics_df.to_csv(f"{results_dir}/metrics.csv")

# 2. Save full classification report
report = classification_report(y_test, y_pred, output_dict=True)
pd.DataFrame(report).transpose().to_csv(f"{results_dir}/classification_report.csv")

# 3. Save predictions and scores
pd.DataFrame({
    'True': y_test,
    'Predicted': y_pred,
    'Decision_Scores': y_scores
}).to_csv(f"{results_dir}/predictions.csv", index=False)

# 4. Save model coefficients
pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': linear_svm.coef_[0]
}).to_csv(f"{results_dir}/coefficients.csv", index=False)

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
RocCurveDisplay.from_estimator(linear_svm, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'k--')
plt.savefig(f"{results_dir}/roc_curve.png", bbox_inches='tight', dpi=300)
plt.close()

# 3. Top 20 important features
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': linear_svm.coef_[0]
}).sort_values('Importance', key=abs, ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=coef_df.head(20), palette='viridis')
plt.title('Top 20 Most Important Features')
plt.tight_layout()
plt.savefig(f"{results_dir}/feature_importance.png", bbox_inches='tight', dpi=300)
plt.close()

# Save complete configuration
with open(f"{results_dir}/config.txt", 'w') as f:
    f.write("Linear SVM Configuration:\n")
    f.write(f"Model parameters:\n{linear_svm.get_params()}\n")
    f.write(f"Training samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")

print(f"\nAll SVM results saved to: {results_dir}/")
print("Files created:")
for file in os.listdir(results_dir):
    print(f"- {file}")