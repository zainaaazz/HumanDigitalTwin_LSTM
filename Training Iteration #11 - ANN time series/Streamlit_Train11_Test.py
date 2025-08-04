# app.py
# Streamlit app for testing the trained LSTM model per student
# Run:
#   pip install streamlit tensorflow pandas scikit-learn
#   streamlit run app.py

import os
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")  # same CSV used for training
MODEL_DIR = os.path.join(SCRIPT_DIR, "Train11Model")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "Train11_best_model.h5")

# Map model output indices to labels
LABEL_MAP = {0: 'Withdrawn', 1: 'Fail', 2: 'Pass', 3: 'Distinction'}

@st.cache_data
def load_data(path):
    """
    Load the dataset including id_student, date, features, and final_result.
    """
    return pd.read_csv(path)

@st.cache_resource
def load_trained_model(path):
    """
    Load the saved Keras model.
    """
    return load_model(path)

@st.cache_data
def prepare_scaler(df, feature_cols):
    """
    Fit MinMaxScaler on the full dataset's feature columns.
    """
    return MinMaxScaler().fit(df[feature_cols])


def main():
    st.title("Human Digital Twin: Student Performance Preview")
    df = load_data(DATA_PATH)

    # Determine feature columns by excluding only the label and index
    exclude = {'final_result', 'Unnamed: 0'}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Sidebar: select student ID
    students = df['id_student'].unique().tolist()
    selected = st.sidebar.selectbox("Select Student ID", students)

    # Filter and sort the student's records by date
    student_df = df[df['id_student'] == selected].sort_values('date')
    st.write(f"### Timeline for Student: {selected}")
    st.write(f"Total days: {student_df.shape[0]}")

    # Load model & scaler
    model = load_trained_model(BEST_MODEL_PATH)
    scaler = prepare_scaler(df, feature_cols)

    # Generate predictions day-by-day
    records = []
    for _, row in student_df.iterrows():
        # Extract and scale features
        feats = row[feature_cols].values
        scaled = scaler.transform([feats])  # shape: (1, n_features)
        sample = scaled.reshape(1, 1, len(feature_cols))  # LSTM expects (batch, timesteps, features)

        # Predict and record probabilities
        prob = model.predict(sample, verbose=0)[0]
        idx = np.argmax(prob)
        records.append({
            'date': row['date'],
            'prediction': LABEL_MAP[idx],
            'prob_Withdrawn': prob[0],
            'prob_Fail': prob[1],
            'prob_Pass': prob[2],
            'prob_Distinction': prob[3]
        })

    # Build results DataFrame
    report = pd.DataFrame(records).set_index('date')

    # Display daily predictions
    st.subheader("Daily Predictions")
    st.dataframe(report)

    # Plot probability trends over time
    st.subheader("Prediction Probabilities over Time")
    st.line_chart(report[[
        'prob_Withdrawn',
        'prob_Fail',
        'prob_Pass',
        'prob_Distinction'
    ]])

if __name__ == '__main__':
    main()