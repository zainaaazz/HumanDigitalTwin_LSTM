import os
import streamlit as st
import pandas as pd
import numpy as np

# ── Attempt TensorFlow Import ───────────────────────────────────────────────────
try:
    import tensorflow as tf
except ImportError as e:
    tf = None
    TF_IMPORT_ERROR = str(e)
else:
    TF_IMPORT_ERROR = None

# ── Streamlit Page Config ──────────────────────────────────────────────────────
st.set_page_config(page_title="Sequential Pass/Fail Explorer", layout="wide")

# ── Handle Missing TensorFlow ──────────────────────────────────────────────────
if TF_IMPORT_ERROR:
    st.title("Student Pass/Fail Probability Over Time")
    st.error(
        f"""
Failed to load TensorFlow runtime.
Please install the CPU-only TensorFlow package:
    pip install tensorflow-cpu

Error details:
{TF_IMPORT_ERROR}
"""
    )
    st.stop()

# ── Configuration ───────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

WEEKLY_MODEL = os.path.join(MODEL_DIR, 'Train8_Model_week.keras')
MONTHLY_MODEL = os.path.join(MODEL_DIR, 'Train8_Model_month.keras')

# ── Load Clickstream Data ──────────────────────────────────────────────────────
@st.cache_data
def load_clickstream():
    info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
    vle  = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
    meta = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))
    info = info[(info.code_module == 'BBB') & (info.final_result != 'Withdrawn')].copy()
    info['label'] = info.final_result.map(lambda x: 1 if x in ['Pass','Distinction'] else 0)
    data = vle.merge(meta[['id_site']], on='id_site', how='left')
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['id_student','date'])
    data['days_since'] = (data['date'] - data.groupby('id_student')['date'].transform('min')).dt.days
    data['week']  = data['days_since'] // 7
    data['month'] = data['days_since'] // 30
    weekly = data.groupby(['id_student','week'])['id_site'].count().unstack(fill_value=0)
    monthly = data.groupby(['id_student','month'])['id_site'].count().unstack(fill_value=0)
    df_w = info[['id_student','label']].merge(weekly, on='id_student', how='left').fillna(0)
    df_m = info[['id_student','label']].merge(monthly, on='id_student', how='left').fillna(0)
    return df_w, df_m

# ── Load Models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # tf is guaranteed non-None here
    w = tf.keras.models.load_model(WEEKLY_MODEL)
    m = tf.keras.models.load_model(MONTHLY_MODEL)
    return w, m

# ── Main UI ─────────────────────────────────────────────────────────────────────
st.title("Student Pass/Fail Probability Over Time")

df_w, df_m = load_clickstream()
w_model, m_model = load_models()

students = df_w['id_student'].unique()
selected = st.sidebar.selectbox('Select Student ID', students)

ts_w = df_w[df_w.id_student == selected].iloc[0, 2:]
ts_m = df_m[df_m.id_student == selected].iloc[0, 2:]

# ── Prediction Function ─────────────────────────────────────────────────────────
def predict_probs(model, series):
    X = series.values.reshape(1, -1, 1).astype('float32')
    preds = model.predict(X)[0, :, 0]
    return pd.Series(preds, index=series.index)

p_w = predict_probs(w_model, ts_w)
p_m = predict_probs(m_model, ts_m)

# ── Display ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader('Weekly Pass Probability')
    st.line_chart(p_w)
    st.dataframe(
        p_w.to_frame('pass_prob')
           .reset_index()
           .rename(columns={'index':'week'})
    )
with col2:
    st.subheader('Monthly Pass Probability')
    st.line_chart(p_m)
    st.dataframe(
        p_m.to_frame('pass_prob')
           .reset_index()
           .rename(columns={'index':'month'})
    )

st.caption("Run this app with: `streamlit run Sequential_Test.py`")