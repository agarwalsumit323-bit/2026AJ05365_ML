"""
Streamlit app: Telco Customer Churn - classifier evaluation on uploaded test data.

The app only LOADS the pre-trained pipelines from model/ and evaluates them on the
uploaded CSV. No model is retrained here.

Run locally:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from train_models import FEATURES, TARGET, clean, evaluate, positive_scores

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes (Gaussian)": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Telco Churn – Model Evaluator", page_icon="📉", layout="wide")


@st.cache_resource
def load_model(filename: str):
    return joblib.load(MODEL_DIR / filename)


def validate(df: pd.DataFrame) -> list[str]:
    """Return the list of required columns missing from the upload."""
    return [c for c in FEATURES + [TARGET] if c not in df.columns]


def score(model, df: pd.DataFrame):
    X, y = df[FEATURES], df[TARGET]
    y_pred = model.predict(X)
    y_score = positive_scores(model, X)
    return y, y_pred, evaluate(y, y_pred, y_score)


# ---------------- Sidebar ----------------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("CSV with all 19 features + 'Churn' column", type="csv")
if SAMPLE_CSV.exists():
    st.sidebar.download_button("Download sample test_data.csv", SAMPLE_CSV.read_bytes(),
                               file_name="test_data.csv", mime="text/csv")

st.sidebar.header("2. Choose a model")
model_name = st.sidebar.selectbox("Classifier", list(MODEL_FILES.keys()))

# ---------------- Main ----------------
st.title("📉 Telco Customer Churn — Classifier Evaluation")
st.caption("Evaluates five pre-trained scikit-learn pipelines on your uploaded test set. "
           "Positive class = customer churned (Churn = Yes).")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to begin. "
            "You can download the sample `test_data.csv` there.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except Exception as exc:  # malformed file
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

missing = validate(raw)
if missing:
    st.error(f"Uploaded file is missing required columns: {', '.join(missing)}")
    st.stop()

data = clean(raw)
bad_target = data[TARGET].isna().sum()
if bad_target:
    st.warning(f"Dropping {bad_target} row(s) whose 'Churn' value is not Yes/No (or 1/0).")
    data = data.dropna(subset=[TARGET])
data[TARGET] = data[TARGET].astype(int)
if data.empty or data[TARGET].nunique() < 2:
    st.error("The test data needs rows from both classes (Churn = Yes and No) to compute AUC and MCC.")
    st.stop()

with st.expander(f"Preview uploaded data — {len(data)} rows"):
    st.dataframe(raw.head(20), use_container_width=True)
    st.write(f"Churn rate in this file: **{data[TARGET].mean():.1%}**")

tab_single, tab_compare = st.tabs([f"🔍 {model_name}", "📊 Compare all models"])

with tab_single:
    model = load_model(MODEL_FILES[model_name])
    y_true, y_pred, metrics = score(model, data)

    st.subheader("Evaluation metrics")
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, f"{v:.4f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader("Classification report")
        report = classification_report(y_true, y_pred, labels=[0, 1],
                                       target_names=["No churn", "Churn"],
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), use_container_width=True)

with tab_compare:
    st.subheader("All five models on the uploaded data")
    rows = {}
    for name, fname in MODEL_FILES.items():
        _, _, m = score(load_model(fname), data)
        rows[name] = m
    table = pd.DataFrame(rows).T
    st.dataframe(table.style.format("{:.4f}").highlight_max(axis=0, color="#c7e9c0"),
                 use_container_width=True)
    metric_to_plot = st.selectbox("Plot metric", list(table.columns), index=list(table.columns).index("MCC"))
    st.bar_chart(table[metric_to_plot])
    st.caption("Green cells mark the best model for each metric.")
