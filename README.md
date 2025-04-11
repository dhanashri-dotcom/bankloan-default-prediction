# Bank Loan Default Prediction

This project develops a machine learning pipeline to predict loan defaults using various classification models. The dataset is highly imbalanced, and the pipeline addresses this using hybrid resampling techniques. The project compares multiple models, visualizes their performance, and concludes with an advanced stacking ensemble for optimal results.

## Data Pre-processing


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
