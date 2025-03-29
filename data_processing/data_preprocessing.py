import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Create 'data' directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Load Dataset
dataset_path = kagglehub.dataset_download("nikhil1e9/Loan-default")
df = pd.read_csv(f"{dataset_path}/loan_default.csv")

# EDA
print(df.info())
print(df.describe())
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

sns.countplot(x='Default', data=df, palette='coolwarm')
plt.title("Class Distribution")
plt.show()

# Numerical Feature Analysis
df_num = df.select_dtypes(exclude=['object'])
for col in df_num:
    sns.boxplot(x='Default', y=col, data=df, palette='coolwarm')
    plt.title(f"Boxplot of {col} by Default")
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df_num.corr(), annot=True, cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income', y='LoanAmount', data=df, hue='Default')
plt.title("Scatterplot: Income vs Loan Amount (Colored by Default)")
plt.show()

# Drop ID column
df.drop('LoanID', axis=1, inplace=True)

# One-hot encode categorical columns
onehot_cols = ["EmploymentType", "MaritalStatus", "LoanPurpose", "Education", "HasMortgage", "HasDependents", "HasCoSigner"]
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[onehot_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(onehot_cols), index=df.index)

df = df.drop(columns=onehot_cols)
df = pd.concat([df, encoded_df], axis=1)

# Define features and target
X = df.drop('Default', axis=1)
y = df['Default']

# Train-test split (before SMOTE!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to training set only
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Print Class Distribution
print("Before Resampling:\n", y.value_counts())
print("\nAfter Resampling (train only):\n", y_train_res.value_counts())

# Class Distribution Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x=y, ax=axes[0], palette="coolwarm")
axes[0].set_title("Class Distribution Before Resampling")
axes[0].set_xlabel("Default")
axes[0].set_ylabel("Count")

sns.countplot(x=y_train_res, ax=axes[1], palette="coolwarm")
axes[1].set_title("Training Set After SMOTE")
axes[1].set_xlabel("Default")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

# Save datasets
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv("data/X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("data/X_test.csv", index=False)
pd.Series(y_train_res, name="Default").to_csv("data/y_train.csv", index=False)
pd.Series(y_test, name="Default").to_csv("data/y_test.csv", index=False)

print("Preprocessing complete. Files saved in 'data/' folder.")
