import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')      # raw data directory
LOG_DIR  = os.path.join(BASE_DIR, 'TrainX_Logs')  # models and logs folder

# ── PAGE SETUP ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Model Tester UI", layout="wide")
st.title("🚀 HumanDigitalTwin Model Testing UI")
st.sidebar.header("Select Model & Run Evaluation")

# ── DATA LOADING ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Load and preprocess student info and VLE panels
    info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
    info = info[(info.code_module == 'BBB') & (info.final_result != 'Withdrawn')]
    info['label_bin'] = info.final_result.map({'Pass': 1, 'Distinction': 1, 'Fail': 0})

    vle  = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
    meta = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))
    vle = vle[vle.code_module == 'BBB'].merge(meta[['id_site','activity_type']], on='id_site')
    clicked = vle.id_student.unique()
    info = info[info.id_student.isin(clicked)].drop_duplicates('id_student')
    students = np.sort(info.id_student.unique())

    # derive calendar units
    for unit, length in [('week',7),('month',30)]:
        vle[unit] = (vle.date // length).astype(int)

    def make_panel(unit):
        df = vle.groupby(['id_student', unit, 'activity_type'])['sum_click'] \
                 .sum().unstack(fill_value=0)
        levels = np.arange(df.index.get_level_values(unit).min(), \
                           df.index.get_level_values(unit).max()+1)
        idx = pd.MultiIndex.from_product([students, levels], names=['id_student', unit])
        df = df.reindex(idx, fill_value=0).reset_index()
        feats = [c for c in df.columns if c not in ('id_student', unit)]
        nstu, nint = len(students), df[unit].nunique()
        X = df[feats].values.reshape(nstu, nint, len(feats))
        sid = df['id_student'].values.reshape(nstu, nint)[:,0]
        y = info.set_index('id_student').loc[sid, 'label_bin'].values
        return X, y, sid

    Xw, yw, sw_full = make_panel('week')
    Xm, ym, sm_full = make_panel('month')

    # Held-out test split
    Xw_tv, Xw_test, sw_tv, sw_test = train_test_split(Xw, sw_full, test_size=0.1, stratify=yw, random_state=42)
    yw_tv, yw_test               = train_test_split(yw,   test_size=0.1, stratify=yw, random_state=42)
    Xm_tv, Xm_test, sm_tv, sm_test = train_test_split(Xm, sm_full, test_size=0.1, stratify=ym, random_state=42)
    ym_tv, ym_test               = train_test_split(ym,   test_size=0.1, stratify=ym, random_state=42)

    info_df = info.set_index('id_student')
    return (Xw_test, yw_test, sw_test, info_df), (Xm_test, ym_test, sm_test, info_df)

(test_week, test_month) = load_data()

# ── MODEL SELECTION ─────────────────────────────────────────────────────────────
models = {
    'LSTM_WEEK':   'LSTM_WEEK_final.keras',
    'LSTM_MONTH':  'LSTM_MONTH_final.keras',
    'CNN_WEEK':    'CNN_WEEK_final.keras',
    'CNN_MONTH':   'CNN_MONTH_final.keras'
}
choice = st.sidebar.selectbox("Model", list(models.keys()))
run_eval = st.sidebar.button("Run Evaluation")

# ── MAIN CONTENT ────────────────────────────────────────────────────────────────
if run_eval:
    model_path = os.path.join(LOG_DIR, models[choice])
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
    else:
        st.subheader(f"🔍 Evaluating **{choice}** on Held-Out Test")
        model = load_model(model_path)
        # select test data
        if 'WEEK' in choice:
            Xtest, ytest, sid_test, info_df = test_week
            unit_name, max_unit = 'Week', Xtest.shape[1]-1
        else:
            Xtest, ytest, sid_test, info_df = test_month
            unit_name, max_unit = 'Month', Xtest.shape[1]-1

        # slider for dynamic horizon
        idx = st.slider(f"Select {unit_name} index (0 = first)", 0, max_unit, max_unit)
        Xdyn = Xtest[:, :idx+1, :]
        st.markdown(f"**Using first {idx+1} {unit_name.lower()}(s) for prediction**")

        # predictions
        probs_dyn = model.predict(Xdyn, verbose=0).ravel()
        preds_dyn = (probs_dyn >= 0.5).astype(int)
        outcome_dyn = ['Fail' if p==0 else 'Pass' for p in preds_dyn]

        # full-horizon metrics
        probs_full = model.predict(Xtest, verbose=0).ravel()
        preds_full = (probs_full >= 0.5).astype(int)
        acc  = accuracy_score(ytest, preds_full)
        f1   = f1_score(ytest, preds_full, average='binary')
        auc  = roc_auc_score(ytest, probs_full)

        cols = st.columns(3)
        cols[0].metric("Accuracy", f"{acc:.4f}")
        cols[1].metric("F1-Score", f"{f1:.4f}")
        cols[2].metric("AUC", f"{auc:.4f}")

        # confusion matrix
        cm = confusion_matrix(ytest, preds_full)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        st.pyplot(fig)

        # build results dataframe
        df = pd.DataFrame({
            'student_id':   sid_test,
            'predicted':    outcome_dyn,
            'prob_pass':    probs_dyn,
            'actual_result':[info_df.loc[s,'final_result'] for s in sid_test]
        })
        # engagement & history
        df['total_clicks'] = Xdyn.sum(axis=(1,2))
        df['n_activities'] = (Xdyn > 0).sum(axis=(1,2))
        df['avg_clicks']   = df['total_clicks'] / (idx+1)
        last_active = (Xdyn.sum(axis=2) > 0).argmax(axis=1)
        df[f'last_active_{unit_name.lower()}'] = last_active

        df_sorted = df.sort_values('prob_pass', ascending=False).head(500)
        st.subheader(f"🏅 Top 500 Students by Predicted Pass Probability ({unit_name} {idx})")
        st.table(
            df_sorted[
                ['student_id','total_clicks','n_activities','avg_clicks',
                 f'last_active_{unit_name.lower()}','predicted','actual_result','prob_pass']
            ]
            .assign(prob_pass=lambda d: d.prob_pass.map(lambda x: f"{x:.3f}"))
            .assign(avg_clicks=lambda d: d.avg_clicks.map(lambda x: f"{x:.1f}"))
        )

        # ── Export to CSV ───────────────────────────────────────────────────
        csv_filename = f"{choice}_{unit_name}_{idx}_predictions.csv"
        csv_path = os.path.join(LOG_DIR, csv_filename)
        df_sorted.to_csv(csv_path, index=False)
        st.success(f"📄 Saved top-500 predictions to `{csv_path}`")
        # download button
        csv_data = df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=csv_filename,
            mime="text/csv"
        )

        st.write("---")
        st.success("✅ Evaluation Complete")
else:
    st.info("Select a model from the sidebar and click **Run Evaluation** to view results.")

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.write("---")
st.caption("Built with Streamlit | HumanDigitalTwin Testing UI")
