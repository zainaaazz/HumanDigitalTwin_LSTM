import os
import glob
import streamlit as st
import pandas as pd

# Directory containing log outputs
LOG_DIR = os.path.join(os.getcwd(), 'TrainX_Logs')

@st.cache_data
def load_logs(model_name: str) -> pd.DataFrame:
    """
    Load and aggregate CSVLogger logs for the selected model.
    Expects files like TrainX_Logs/{model_name}_fold1.csv ... _fold10.csv
    """
    pattern = os.path.join(LOG_DIR, f"{model_name}_fold*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        st.error(f"No log files found for model '{model_name}' in {LOG_DIR}")
        st.stop()
    dfs = []
    for path in files:
        df = pd.read_csv(path)
        try:
            fold = int(os.path.basename(path).split('_fold')[1].split('.')[0])
        except:
            fold = -1
        df['fold'] = fold
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# --- Streamlit UI ---
st.set_page_config(page_title="Model Training Logs & Relationships", layout="wide")
st.title("📊 Model Training Logs & Feature Relationships")

# Sidebar: model selection
models = ['LSTM_WEEK', 'LSTM_MONTH', 'CNN_WEEK', 'CNN_MONTH']
model_name = st.sidebar.selectbox("Select model to inspect", models)

# Load logs
df = load_logs(model_name)
st.subheader(f"Aggregated CSVLogger Metrics for {model_name}")
st.dataframe(df, use_container_width=True)

# Download combined logs
download_bytes = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download combined logs",
    data=download_bytes,
    file_name=f"{model_name}_training_logs.csv",
    mime="text/csv"
)

st.write(f"**Total rows:** {len(df)} (_folds × epochs_)" )

# Relationship mapping: correlate training metrics
st.subheader("🔗 Correlation of Training Metrics to Fold")
metrics_cols = [c for c in df.columns if c not in ['epoch','fold']]
if metrics_cols:
    corr = df[metrics_cols + ['fold']].corr()['fold'].sort_values(ascending=False)
    st.bar_chart(corr.drop('fold'))
else:
    st.info("No numeric metrics columns found for correlation.")

