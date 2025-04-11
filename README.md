# Bank Loan Default Prediction

This project builds a robust machine learning pipeline to predict loan defaults using a variety of models and resampling strategies. The dataset is highly imbalanced, and techniques like SMOTE, SMOTE-ENN, random oversampling, and undersampling are used to improve performance across classifiers. The final solution compares 13 models and incorporates hyperparameter tuning and ensembling.

## Project Highlights

- Full data pipeline: loading, cleaning, preprocessing

- Resampling strategies: SMOTE, RandomOverSampling, RandomUnderSampling, SMOTE-ENN

- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC

- Hyperparameter tuning

- Models: Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, MLP, CatBoost, XGBoost, Stacking

**Loan Default Dataset Preprocessing Pipeline** 


### Summary:
-----------
This script performs preprocessing on the loan default dataset downloaded via `kagglehub`. It includes:
- Exploratory Data Analysis (EDA)
- Dropping irrelevant features (like LoanID)
- One-hot encoding of categorical features
- Train-test split (stratified)
- Handling class imbalance using SMOTE on training data
- Feature scaling using StandardScaler
- Saving processed datasets for model training

### Output:
----------
The following files are saved to the `data/` directory:
- data/X_train.csv : Scaled features for training
- data/X_test.csv  : Scaled features for testing
- data/y_train.csv : Target values for training (SMOTE balanced)
- data/y_test.csv  : Target values for testing (original distribution)

### How to Use in Other Models:
------------------------------
You can load these files in any model training script like this:

```python
import pandas as pd

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").values.ravel()
y_test = pd.read_csv("data/y_test.csv").values.ravel()
