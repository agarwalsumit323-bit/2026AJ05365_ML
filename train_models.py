"""
Train and evaluate five classifiers on the IBM Telco Customer Churn dataset.

Outputs:
    model/<name>.joblib   - full sklearn Pipeline (preprocessing + classifier) per model
    model/metrics.json    - Accuracy, AUC, Precision, Recall, F1, MCC on the held-out test split
    test_data.csv         - the held-out 20% test split (raw features + target) for the Streamlit app

Run:
    python train_models.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "model"
TARGET = "Churn"

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_COLS + CATEGORICAL_COLS  # 19 input features


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning shared by training and the app.

    - TotalCharges is stored as text with 11 blank values -> coerce to float (NaN, imputed later).
    - SeniorCitizen is 0/1 but is categorical -> cast to str.
    - Target Yes/No -> 1/0 (if the column is present).
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
    if TARGET in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].astype(str).str.strip().map({"Yes": 1, "No": 0})
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("cat", categorical, CATEGORICAL_COLS),
    ])


MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                            random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": GaussianNB(),
    "svm": SVC(kernel="rbf", random_state=RANDOM_STATE),
}


def positive_scores(model, X):
    """Score for the positive class, used for AUC.

    Uses predict_proba when the model provides it; otherwise falls back to
    decision_function (e.g. SVC without probability calibration). AUC only needs
    a ranking, so raw decision scores are valid.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    raw = pd.read_csv(DATA_PATH)
    df = clean(raw)

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    MODEL_DIR.mkdir(exist_ok=True)
    results = {}
    for name, clf in MODELS.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = positive_scores(pipe, X_test)
        results[name] = evaluate(y_test, y_pred, y_prob)
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")
        print(f"{name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in results[name].items()))

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save the untouched test rows (raw format, Yes/No target) for the Streamlit app.
    test_raw = raw.loc[X_test.index, FEATURES + [TARGET]]
    test_raw.to_csv(ROOT / "test_data.csv", index=False)
    print(f"\nTrain rows: {len(X_train)} | Test rows: {len(X_test)} -> test_data.csv")


if __name__ == "__main__":
    main()
