# Streamlit_Train11_Rolling_Chunked.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

@st.cache_resource
def load_trained_model(path: str):
    """Load and cache your .h5 LSTM model."""
    return load_model(path)

def build_preprocessor(data_path: str, n_features: int):
    """
    1) Read the CSV header to pick exactly n_features columns (dropping id/date/final_result).
    2) Stream the file in chunks and partial_fit a StandardScaler.
    """
    # read only the header
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    # drop columns we don't use
    candidate = [c for c in header if c not in ("id_student", "date", "final_result")]
    feature_cols = candidate[:n_features]

    scaler = StandardScaler()
    # stream through the file in 200k‐row chunks
    for chunk in pd.read_csv(data_path,
                             usecols=feature_cols,
                             chunksize=200_000):
        scaler.partial_fit(chunk)

    return scaler, feature_cols
   
def rolling_predictions_for_student(
    student_df: pd.DataFrame,
    model,
    scaler: StandardScaler,
    feature_cols: list[str],
    window_length: int,
) -> pd.DataFrame:
    dates = student_df["date"].values
    Xraw  = student_df[feature_cols].reset_index(drop=True)

    rows = []
    for i in range(len(Xraw)):
        # build a length=window_length block ending at day i
        if i+1 < window_length:
            pad_n = window_length - (i+1)
            pad_block = pd.DataFrame(
                np.repeat(Xraw.iloc[[0]].values, pad_n, axis=0),
                columns=feature_cols,
            )
            block = pd.concat([pad_block, Xraw.iloc[:i+1]], ignore_index=True)
        else:
            block = Xraw.iloc[i-window_length+1 : i+1].reset_index(drop=True)

        # scale, reshape, predict
        arr = scaler.transform(block)
        arr = arr.reshape((1, window_length, len(feature_cols)))
        prob = model.predict(arr, verbose=0)[0]
        label = ["Withdrawn","Fail","Pass","Distinction"][np.argmax(prob)]

        rows.append({
            "date": dates[i],
            "prediction": label,
            "prob_Withdrawn": prob[0],
            "prob_Fail":      prob[1],
            "prob_Pass":      prob[2],
            "prob_Distinction": prob[3],
        })

    return pd.DataFrame(rows)

def main():
    st.set_page_config(page_title="Human Digital Twin", layout="wide")
    st.title("Human Digital Twin: Student Performance Preview")

    DATA_PATH  = "data.csv"
    MODEL_PATH = "Train11Model/Train11_best_model.h5"

    # 1) load model & infer its input dims
    model = load_trained_model(MODEL_PATH)
    window_length, n_features = model.input_shape[1], model.input_shape[2]

    # 2) build scaler & feature list (chunked, low RAM)
    scaler, feature_cols = build_preprocessor(DATA_PATH, n_features)

    # 3) load just enough to get IDs & keep df for slicing per student
    df = pd.read_csv(DATA_PATH, usecols=["id_student","date"] + feature_cols)

    st.sidebar.header("Pick a student")
    student_id = st.sidebar.selectbox(
        "Select Student ID",
        sorted(df["id_student"].unique())
    )

    student_df = (
        df[df["id_student"] == student_id]
          .sort_values("date")
          .reset_index(drop=True)
    )
    st.subheader(f"Timeline for Student  {student_id}")
    st.markdown(f"**Total days:** {len(student_df)}")

    if st.button("🔁 Run rolling-window predictions"):
        with st.spinner("Computing predictions…"):
            preds = rolling_predictions_for_student(
                student_df, model, scaler, feature_cols, window_length
            )

        st.subheader("Daily Predictions")
        st.dataframe(preds, use_container_width=True)

        st.subheader("Prediction Probabilities Over Time")
        prob_df = preds.set_index("date")[
            ["prob_Withdrawn","prob_Fail","prob_Pass","prob_Distinction"]
        ]
        st.line_chart(prob_df)

if __name__ == "__main__":
    main()
