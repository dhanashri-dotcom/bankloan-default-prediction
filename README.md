# Bank Loan Default Prediction

This project builds a robust machine learning pipeline to predict loan defaults using a variety of models and resampling strategies. The dataset is highly imbalanced, and techniques like SMOTE, SMOTE-ENN, random oversampling, and undersampling are used to improve performance across classifiers. The final solution compares 13 models and incorporates hyperparameter tuning and ensembling.

## Project Highlights

- Full data pipeline: loading, cleaning, preprocessing

- Resampling strategies: SMOTE, RandomOverSampling, RandomUnderSampling, SMOTE-ENN

- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC AUC

- Hyperparameter tuning

- Models: Logistic Regression, Naive Bayes, KNN, SVM, Decision Tree, Random Forest, MLP, CatBoost, XGBoost, Stacking

## Files Used

| Notebook / Script                            | Purpose                                                                 |
|---------------------------------------------|-------------------------------------------------------------------------|
| `notebooks/Loan_Default_Prediction.ipynb`              | Data loading, SMOTE-ENN, and model training for Logistic Regression, SVM (Linear), and advanced models like XGBoost, CatBoost, MLP, and Stacking |
| `notebooks/KNN_SMOTE_UnderSampling_OverSampling.ipynb` | KNN with various resampling strategies                                  |
| `notebooks/naive_bayes.ipynb`                   | Naive Bayes classifier with priors and SMOTE                            |
| `models/decision_trees.py`                          | Decision Tree (Default and Balanced) model comparisons                  |



## Model Comparison

### Baseline Models

| Model                    | Accuracy | AUC    | Precision | Recall | F1 Score | Notes                                       |
|-------------------------|----------|--------|-----------|--------|----------|---------------------------------------------|
| **Decision Tree**        | 0.8199   | 0.8120 | 0.8354    | 0.8601 | 0.8476   | Weak recall and AUC despite decent accuracy |
| **Logistic Regression**  | 0.7500   | 0.8165 | 0.77      | 0.80   | 0.79     | Balanced performance, strong baseline       |
| **SVM (Linear)**         | 0.7100   | 0.7806 | 0.69      | 0.89   | 0.78     | High recall, but imbalanced precision       |
| **Naive Bayes (Priors)** | 0.6719   | 0.7499 | 0.2183    | 0.7070 | 0.3335   | Very high recall, low precision             |
| **Naive Bayes (SMOTE)**  | 0.7743   | 0.7071 | 0.2418    | 0.4416 | 0.3125   | Improved balance but lower discriminative   |
| **KNN (SMOTE)**          | 0.6848   | 0.7033 | 0.208     | 0.6105 | 0.3103   | Missed defaults; weaker overall             |
| **KNN (OverSampling)**   | 0.6551   | 0.7494 | 0.2119    | 0.7243 | 0.3278   | Best KNN performer in terms of F1           |
| **KNN (UnderSampling)**  | 0.6507   | 0.7504 | 0.2108    | 0.7316 | 0.3272   | Highest recall among KNNs, least accurate   |

### Advanced Models

| Model                    | Accuracy | AUC    | Precision | Recall | F1 Score | Notes                                          |
|-------------------------|----------|--------|-----------|--------|----------|------------------------------------------------|
| **Random Forest (Tuned)**| 0.8300   | 0.9076 | 0.83      | 0.83   | 0.83     | Balanced and reliable performance              |
| **XGBoost**              | 0.8900   | 0.9576 | 0.93      | 0.88   | 0.91     | Strong precision and generalization            |
| **CatBoost**             | 0.9200   | 0.9681 | 0.97      | 0.89   | 0.93     | Top performer as a single model                |
| **MLP (Neural Net)**     | 0.8400   | 0.9153 | 0.88      | 0.84   | 0.86     | Performs well, but slower training             |
| **Stacking Classifier**  | 0.9200   | 0.9689 | 0.96      | 0.90   | 0.93     | Combines best models (RF + XGB + CatBoost)     |


## Pipeline Overview

1. **Preprocessing**
   - Dropped identifier columns
   - Encoded categorical features
   - Normalized where required (e.g., NB, MLP)

2. **Imbalance Handling**
   - SMOTE-ENN, SMOTE, RandomOver/UnderSampling
   - Class priors for Naive Bayes

3. **Training**
   - Baseline models: Logistic Regression, Naive Bayes, Decision Tree, KNN
   - Advanced models: Random Forest, XGBoost, CatBoost, MLP, Stacking

4. **Evaluation**
   - Confusion Matrix
   - Precision / Recall / F1-score
   - ROC-AUC and thresholds (esp. for NB)

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost catboost kagglehub
```
Then, execute the project in the following order:

### Step-by-Step Execution

#### 1. Run Notebooks (for model training and evaluation)

```bash
# Loan Default Prediction - Main Models
jupyter nbconvert --to notebook --execute notebooks/Loan_Default_Prediction.ipynb --output notebooks/Loan_Default_Prediction_output.ipynb

# Naive Bayes Model
jupyter nbconvert --to notebook --execute notebooks/naive_bayes.ipynb --output notebooks/naive_bayes_output.ipynb

# KNN with Resampling Methods
jupyter nbconvert --to notebook --execute notebooks/KNN_SMOTE_UnderSampling_OverSampling.ipynb --output notebooks/KNN_SMOTE_output.ipynb
```

#### 2. Run Decision Tree (Python-based)

First, process the data:

```bash
python data_processing/data_preprocessing.py
```

Then run the Decision Tree model:

```bash
python models/decision_trees.py
```

**> You can also open and run the notebooks manually using Jupyter Notebook or JupyterLab if preferred.**


## Key Observations

- **CatBoost & Stacking** achieved the best overall performance (AUC ≈ 0.97), showing strong precision, recall, and F1.

- **Random Forest (Tuned)** offered a strong balance between accuracy (83%), AUC (0.9076), and F1-score (0.83), making it a solid general-purpose baseline.

- **XGBoost** delivered excellent performance (AUC: 0.9576, F1: 0.91). Its gradient boosting mechanism with regularization made it more robust than standalone trees, especially in terms of overfitting.

- **CatBoost** slightly outperformed XGBoost (AUC: 0.9681 vs. 0.9576) because:
  - It **natively handles categorical features** without manual encoding (your dataset has a large number of categorical variables).
  - It avoids **overfitting** better due to built-in support for ordered boosting and handling of missing values.

- **SVM (Linear Kernel)** achieved high **recall (0.89)** but suffered from **low precision (0.69)** and overall lower F1-score (0.78), suggesting it focused too much on catching positives at the expense of false alarms. SVM is also sensitive to class imbalance without scaling and cost adjustment.

- **MLP (Neural Net)** achieved solid results (AUC: 0.9153, F1: 0.86) and learned non-linear patterns well. However, it required more preprocessing (scaling) and longer training time.

- **Stacking Classifier** combined **Random Forest**, **XGBoost**, and **CatBoost** into a single ensemble model, resulting in **the best performance overall**:
  - Accuracy: 92%
  - AUC: 0.9689
  - F1-Score: 0.93
  - This strategy captured both deep tree-based patterns and ensemble diversity for maximum generalization.

- **Naive Bayes with priors** had better recall and F1 than when used with SMOTE. It benefits more from class prior adjustments than oversampling.

- **KNN with OverSampling** gave the best recall (0.7243) among all KNN approaches. However, precision and overall performance still lagged behind tree-based models.

- **Decision Tree** underperformed with an AUC of just 0.558, highlighting its inability to handle imbalanced data and generalize well.



## Future Directions

- Perform **time-based validation** for real-world simulation
- Include **cost-sensitive learning** (business impact, fraud cost)
- Apply **hyperparameter tuning** using `Optuna` or `GridSearchCV`
