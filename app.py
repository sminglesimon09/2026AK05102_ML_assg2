import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Steel Plate Fault Lab", page_icon="🔩", layout="wide")
st.title("🔩 Steel Plate Fault Lab")
st.caption("Compare five classifiers on a held-out test set. "
           "Models are pre-trained and are not retrained here.")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "SVM": "svm",
}
TARGET = "fault_type"


@st.cache_resource
def load_artifacts():
    models = {n: joblib.load(f"model/{f}.joblib") for n, f in MODEL_FILES.items()}
    with open("model/features.json") as fh:
        feats = json.load(fh)
    return (models, joblib.load("model/scaler.joblib"),
            joblib.load("model/label_encoder.joblib"), feats)


models, scaler, le, features = load_artifacts()


def evaluate(model, X_s, y):
    pred, proba = model.predict(X_s), model.predict_proba(X_s)
    try:
        auc = roc_auc_score(y, proba, multi_class="ovr", average="weighted",
                            labels=np.arange(len(le.classes_)))
    except ValueError:
        auc = float("nan")
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": auc,
        "Precision": precision_score(y, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y, pred, average="weighted", zero_division=0),
        "F1": f1_score(y, pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


with st.sidebar:
    st.header("Controls")
    up = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Model", list(MODEL_FILES))
    st.markdown(f"CSV must contain the {len(features)} feature columns and a "
                f"`{TARGET}` column.")

if up is None:
    st.info("Upload `test_data.csv` from the repo to begin.")
    st.stop()

df = pd.read_csv(up)
missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

unknown = set(df[TARGET].unique()) - set(le.classes_)
if unknown:
    st.error(f"Unknown labels in `{TARGET}`: {sorted(unknown)}")
    st.stop()

X_s = scaler.transform(df[features])
y = le.transform(df[TARGET])
pred, metrics = evaluate(models[choice], X_s, y)

tab1, tab2, tab3 = st.tabs(["Selected model", "All models", "Data preview"])

with tab1:
    cols = st.columns(6)
    for c, (k, v) in zip(cols, metrics.items()):
        c.metric(k, "n/a" if np.isnan(v) else f"{v:.3f}")
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y, pred, labels=np.arange(len(le.classes_))),
                    annot=True, fmt="d", cmap="mako",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.subheader("Classification report")
        st.text(classification_report(y, pred, labels=np.arange(len(le.classes_)),
                                      target_names=le.classes_, zero_division=0))

with tab2:
    table = pd.DataFrame({n: evaluate(m, X_s, y)[1] for n, m in models.items()}).T.round(4)
    st.dataframe(table, use_container_width=True)
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])

with tab3:
    st.dataframe(df.head(50))
