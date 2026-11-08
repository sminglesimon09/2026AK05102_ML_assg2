import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

SEED = 2678                      # change to your own
DATA_PATH = "data/faults.csv"
FAULT_COLS = ["Pastry", "Z_Scratch", "K_Scatch", "Stains",
              "Dirtiness", "Bumps", "Other_Faults"]
TARGET = "fault_type"

raw = pd.read_csv(DATA_PATH)
missing = [c for c in FAULT_COLS if c not in raw.columns]
if missing:
    raise SystemExit(f"Fault columns not found in CSV: {missing}. "
                     f"Edit FAULT_COLS to match your file's header: {list(raw.columns)}")

raw[TARGET] = raw[FAULT_COLS].idxmax(axis=1)
raw = raw.drop(columns=FAULT_COLS)

le = LabelEncoder()
y = le.fit_transform(raw[TARGET])
X = raw.drop(columns=[TARGET])
feature_names = list(X.columns)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=SEED)

scaler = StandardScaler().fit(X_tr)
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
}

Path("model").mkdir(exist_ok=True)
rows = []
for name, m in models.items():
    m.fit(X_tr_s, y_tr)
    pred, proba = m.predict(X_te_s), m.predict_proba(X_te_s)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_te, pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_te, pred, average="weighted"),
        "F1": f1_score(y_te, pred, average="weighted"),
        "MCC": matthews_corrcoef(y_te, pred),
    })
    joblib.dump(m, f"model/{name.lower().replace(' ', '_')}.joblib")

joblib.dump(scaler, "model/scaler.joblib")
joblib.dump(le, "model/label_encoder.joblib")
with open("model/features.json", "w") as f:
    json.dump(feature_names, f)

test_df = X_te.copy()
test_df[TARGET] = le.inverse_transform(y_te)
test_df.to_csv("test_data.csv", index=False)

res = pd.DataFrame(rows).round(4)
res.to_csv("model/metrics_summary.csv", index=False)
try:
    print(res.to_markdown(index=False))
except ImportError:
    print(res.to_string(index=False))
print(f"\nRows: {len(raw)} | Features: {len(feature_names)} | Classes: {list(le.classes_)}")
