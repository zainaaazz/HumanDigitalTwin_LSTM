import os
import streamlit as st
import pandas as pd

# Directories
DATA_DIR = os.path.join(os.getcwd(), 'data')
LOG_DIR = os.path.join(os.getcwd(), 'TrainX_Logs')

@st.cache_data
def load_predictions(model_name: str) -> pd.DataFrame:
    """
    Load test-set predictions for the given model.
    Expected file: TrainX_Logs/{model_name}_all_test_predictions.csv
    Contains columns: student_id, predicted, actual_score, prob_pass
    """
    fname = f"{model_name}_all_test_predictions.csv"
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        st.error(f"Prediction file not found: {path}")
        st.stop()
    return pd.read_csv(path)

# --- Streamlit App ---
st.set_page_config(page_title="All Student Records & Predictions", layout="wide")
st.title("🎓 All Student Records with Predictions")

# Sidebar: models & optional upload
models = ['LSTM_WEEK', 'LSTM_MONTH', 'CNN_WEEK', 'CNN_MONTH']
model = st.sidebar.selectbox("Select model for display", models)

# Load student data, with fallback to uploader
default_students_path = os.path.join(DATA_DIR, 'students.csv')
if os.path.exists(default_students_path):
    students = pd.read_csv(default_students_path)
else:
    uploaded_file = st.sidebar.file_uploader("Upload students CSV", type=['csv'])
    if uploaded_file:
        students = pd.read_csv(uploaded_file)
    else:
        st.sidebar.error(f"No students.csv found at {default_students_path}. Please upload your CSV file.")
        st.stop()

# Load predictions
predictions = load_predictions(model)

# Merge on student_id (right join to show only those with predictions)
df_all = pd.merge(students, predictions, on='student_id', how='right')

# Show table
st.subheader(f"Full Records & Predicted vs Actual for {model}")
st.dataframe(df_all, use_container_width=True)

# Download button
csv = df_all.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download all records as CSV",
    data=csv,
    file_name=f"{model}_all_records_predictions.csv",
    mime="text/csv"
)

# Display counts
st.write(f"**Total students displayed:** {df_all.shape[0]}")

# Example: show basic engagement metrics if present
eng_cols = [c for c in df_all.columns if 'engagement' in c.lower()]
if eng_cols:
    st.subheader("Engagement Metrics Summary")
    st.write(df_all[eng_cols].describe())