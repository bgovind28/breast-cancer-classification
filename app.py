
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Breast Mass Classification",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "metadata.json", encoding="utf-8") as file:
    metadata = json.load(file)

st.title("Breast Mass Classification")
st.write("Evaluate five trained classifiers on uploaded test data.")
st.caption("Educational project. Not for clinical use.")

model_name = st.sidebar.selectbox(
    "Choose a classifier",
    list(metadata["model_files"])
)

st.sidebar.download_button(
    "Download sample test CSV",
    data=(BASE_DIR / "test_data.csv").read_bytes(),
    file_name="test_data.csv",
    mime="text/csv"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"]
)

features = metadata["feature_columns"]
target = metadata["target_column"]

with st.expander("CSV requirements"):
    st.write("Include these 30 numeric feature columns:")
    st.write(features)
    st.write("Also include diagnosis: 0 = Benign, 1 = Malignant.")
    st.write("Missing values and infinite values are not accepted.")
    st.write("Models are loaded from saved files; no retraining occurs.")

if uploaded_file is None:
    st.info("Download the sample CSV, then upload it using the sidebar.")
    st.stop()

try:
    data = pd.read_csv(uploaded_file)
    required = features + [target]
    missing = [name for name in required if name not in data.columns]

    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        st.stop()

    if data.empty:
        st.error("The CSV has no data rows.")
        st.stop()

    numeric_data = data[required].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(numeric_data.to_numpy(dtype=float)).all():
        st.error("The required columns contain missing or infinite values.")
        st.stop()

    if not numeric_data[target].isin([0, 1]).all():
        st.error("Diagnosis must contain only 0 or 1.")
        st.stop()

    X_uploaded = numeric_data[features]
    y_uploaded = numeric_data[target].astype(int)

except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
    st.error(f"Invalid CSV: {error}")
    st.stop()

model_file = metadata["model_files"][model_name]
model = joblib.load(BASE_DIR / "model" / model_file)

predictions = model.predict(X_uploaded)
positive_index = list(model.classes_).index(1)
probabilities = model.predict_proba(X_uploaded)[:, positive_index]

auc = None
if y_uploaded.nunique() == 2:
    auc = roc_auc_score(y_uploaded, probabilities)

scores = {
    "Accuracy": accuracy_score(y_uploaded, predictions),
    "AUC": auc,
    "Precision": precision_score(
        y_uploaded, predictions, pos_label=1, zero_division=0
    ),
    "Recall": recall_score(
        y_uploaded, predictions, pos_label=1, zero_division=0
    ),
    "F1": f1_score(
        y_uploaded, predictions, pos_label=1, zero_division=0
    ),
    "MCC": matthews_corrcoef(y_uploaded, predictions)
}

st.subheader(model_name)
st.write(f"Test records: {len(data)}")
st.caption("Precision, recall and F1 treat malignant as the positive class.")

columns = st.columns(3)
for index, (name, value) in enumerate(scores.items()):
    columns[index % 3].metric(
        name,
        "N/A" if value is None else f"{value:.4f}"
    )

if auc is None:
    st.warning("AUC cannot be calculated when only one true class is present.")

st.subheader("Confusion matrix")
st.caption("Rows show actual classes; columns show predicted classes.")

matrix = confusion_matrix(y_uploaded, predictions, labels=[0, 1])
st.dataframe(pd.DataFrame(
    matrix,
    index=["Actual Benign", "Actual Malignant"],
    columns=["Predicted Benign", "Predicted Malignant"]
))

st.subheader("Classification report")
report = classification_report(
    y_uploaded,
    predictions,
    labels=[0, 1],
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0
)
st.dataframe(pd.DataFrame(report).transpose().round(4))

with st.expander("Preview uploaded data"):
    st.dataframe(data.head(10))
