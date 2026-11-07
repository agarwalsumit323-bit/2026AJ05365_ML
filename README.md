# 2026AJ05365: Sumit Agarwal: ML Assignment 2

## a. Problem statement

A telecom company loses revenue every time a subscriber cancels service ("churns"). The goal is to predict, from a customer's account, service and billing details, **whether that customer will churn (Yes/No)**. This is a **binary classification** problem.

Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data and compared using Accuracy, AUC, Precision, Recall, F1 and Matthews Correlation Coefficient (MCC). The trained models are served through an interactive Streamlit app that evaluates them on an uploaded test CSV.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Telco Customer Churn (IBM sample data) |
| Source | Kaggle: <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> (original IBM copy: `IBM/telco-customer-churn-on-icp4d`) |
| Instances | 7,043 customers |
| Input features | 19 (3 numeric, 16 categorical); `customerID` is dropped as an identifier |
| Target | `Churn` (Yes = 1,869 / 26.5%, No = 5,174 / 73.5%) — imbalanced |
| Missing values | 11 blank `TotalCharges` values (customers with tenure 0), imputed with the median |

**Features**

- **Numeric:** `tenure` (months), `MonthlyCharges`, `TotalCharges`
- **Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Account/billing:** `Contract`, `PaperlessBilling`, `PaymentMethod`

**Preprocessing:** numeric columns get median imputation and standard scaling; categorical columns get one-hot encoding. Preprocessing is stored inside each saved scikit-learn `Pipeline`, so the app applies exactly the same transformations. The data is split 80/20 with stratification on `Churn` (`random_state=42`): 5,634 training rows and 1,409 test rows. The test split is saved as `test_data.csv`.

## c. GitHub repository link

https://github.com/agarwalsumit323-bit/2026AJ05365_ML

Live Streamlit app: https://2026aj05356-mlassignment2.streamlit.app/

**Repository structure**

```
project-folder/
│-- app.py                 # Streamlit app (loads saved models, no retraining)
│-- train_models.py        # Preprocessing, training, evaluation, model export
│-- requirements.txt
│-- README.md
│-- test_data.csv          # Held-out 20% test split (19 features + Churn)
│-- data/
│   └-- Telco-Customer-Churn.csv
└-- model/
    │-- logistic_regression.joblib
    │-- decision_tree.joblib
    │-- knn.joblib
    │-- naive_bayes.joblib
    │-- svm.joblib
    └-- metrics.json
```

## d. Models used

All metrics are computed on the held-out test set (1,409 rows). Precision, Recall, F1 and MCC treat **Churn = Yes** as the positive class. AUC uses predicted probabilities (or the SVM decision function).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.8419 | 0.6572 | 0.5588 | 0.6040 | 0.4790 |
| Decision Tree | 0.7970 | 0.8383 | 0.6679 | 0.4679 | 0.5503 | 0.4357 |
| KNN | 0.7814 | 0.8223 | 0.5948 | 0.5535 | 0.5734 | 0.4272 |
| Naive Bayes | 0.6955 | 0.8075 | 0.4597 | 0.8396 | 0.5941 | 0.4268 |
| Support Vector Machine | 0.7906 | 0.7932 | 0.6386 | 0.4866 | 0.5524 | 0.4255 |

**Model settings:** Logistic Regression (`max_iter=1000`); Decision Tree (`max_depth=6`, `min_samples_leaf=20`); KNN (`k=15`); Gaussian Naive Bayes (defaults); SVM (RBF kernel, `C=1`).

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest Accuracy (0.806), AUC (0.842), F1 (0.604) and MCC (0.479). Once categorical features are one-hot encoded, churn is close to linearly separable: contract type, tenure and fibre-optic internet shift the odds in a mostly additive way. The model is also stable and interpretable. |
| Decision Tree | Second-best Accuracy (0.797) and AUC (0.838), and the highest Precision (0.668). It is conservative: Recall is only 0.468, so it misses more than half of the churners. Limiting depth to 6 avoided overfitting; it captures the clear splits (month-to-month contract, short tenure) but not finer patterns. |
| KNN | Middle of the pack (Accuracy 0.781, MCC 0.427). After scaling and one-hot encoding, distance is dominated by the many binary columns, which blurs neighbourhoods. A large k (15) smooths the predictions; precision and recall are balanced but neither is strong. |
| Naive Bayes | Lowest Accuracy (0.696) but by far the highest Recall (0.840): it flags most churners at the cost of many false alarms (Precision 0.460). The independence assumption is violated here (for example, tenure, MonthlyCharges and TotalCharges are strongly correlated) and Gaussian likelihoods suit 0/1 dummies poorly. Still, its F1 (0.594) is second-best, so it is useful when missing a churner costs more than a false alarm. |
| Support Vector Machine | Reasonable Accuracy (0.791) but the lowest AUC (0.793) and MCC (0.426), with low Recall (0.487). With default C and gamma and no class weighting, the RBF kernel leans toward the majority "No churn" class. Tuning C and gamma or using `class_weight='balanced'` would probably help, but it is also the slowest model to train. |
| **Overall winner** | **Logistic Regression** leads on Accuracy, AUC, F1 and MCC. MCC matters most because the classes are imbalanced (26.5% churn). It is also the simplest and most explainable model. If the business wants to catch as many churners as possible, Naive Bayes (or Logistic Regression with a lower decision threshold) is the better recall-oriented choice. |

## Streamlit app features

1. **CSV upload** of test data (sidebar). A sample `test_data.csv` can be downloaded from the app.
2. **Model selection dropdown** for the five classifiers.
3. **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC.
4. **Confusion matrix** (heatmap) and a **classification report**.
5. **Compare all models** tab: a side-by-side metric table with the best value in each column highlighted, plus a bar chart.

The app validates that the upload has all 19 feature columns plus `Churn`. It only loads the saved models from `model/` and never retrains on uploaded data.
