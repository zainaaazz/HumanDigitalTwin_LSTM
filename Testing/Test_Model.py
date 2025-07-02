import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tensorflow.keras.models import load_model

# ── CONFIG ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')      # raw data directory
LOG_DIR  = os.path.join(BASE_DIR, 'TrainX_Logs')  # models and logs folder

# ── STEP 1: Load and preprocess student info ────────────────────────────────────
info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
info = info[(info.code_module == 'BBB') & (info.final_result != 'Withdrawn')]
info['label'] = info.final_result.map({'Pass': 1, 'Distinction': 1, 'Fail': 0})

# ── STEP 2: Load and filter click data (only students with activity) ───────────
vle  = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))
vle = vle[vle.code_module == 'BBB']
vle = vle.merge(meta[['id_site','activity_type']], on='id_site', how='left')

clicked_students = vle.id_student.unique()
info = info[info.id_student.isin(clicked_students)].drop_duplicates('id_student')
students = np.sort(info.id_student.unique())

# ── STEP 3: Build WEEK and MONTH panels ─────────────────────────────────────────
for unit, length, col in [('week', 7, 'week'), ('month', 30, 'month')]:
    vle[col] = (vle.date // length).astype(int)

# WEEK panel
dfw = vle.groupby(['id_student','week','activity_type'])['sum_click'] \
         .sum().unstack(fill_value=0)
weeks = np.arange(dfw.index.get_level_values('week').min(),
                  dfw.index.get_level_values('week').max() + 1)
idx_w = pd.MultiIndex.from_product([students, weeks],
                                   names=['id_student','week'])
dfw = dfw.reindex(idx_w, fill_value=0).reset_index()

# MONTH panel
dfm = vle.groupby(['id_student','month','activity_type'])['sum_click'] \
         .sum().unstack(fill_value=0)
months = np.arange(dfm.index.get_level_values('month').min(),
                   dfm.index.get_level_values('month').max() + 1)
idx_m = pd.MultiIndex.from_product([students, months],
                                   names=['id_student','month'])
dfm = dfm.reindex(idx_m, fill_value=0).reset_index()

# ── STEP 4: Prepare arrays ──────────────────────────────────────────────────────
def prepare_panel(df, time_col):
    features = [c for c in df.columns if c not in ('id_student', time_col)]
    n_students = len(students)
    n_intervals = df[time_col].nunique()
    X = df[features].values.reshape(n_students, n_intervals, len(features))
    sid = df['id_student'].values.reshape(n_students, n_intervals)[:, 0]
    y = info.set_index('id_student').loc[sid, 'label'].values
    return X, y

X_week, y_week = prepare_panel(dfw, 'week')
X_mon,  y_mon  = prepare_panel(dfm, 'month')
print(f"Loaded panels: WEEK X={X_week.shape}, y={y_week.shape}; MONTH X={X_mon.shape}, y={y_mon.shape}")

# ── STEP 5: Split out held-out test sets ────────────────────────────────────────
Xw_trainval, Xw_test, yw_trainval, yw_test = train_test_split(
    X_week, y_week, test_size=0.1, stratify=y_week, random_state=42
)
Xm_trainval, Xm_test, ym_trainval, ym_test = train_test_split(
    X_mon,  y_mon,  test_size=0.1, stratify=y_mon,  random_state=42
)
print(f"Hold-out test: WEEK {Xw_test.shape}, MONTH {Xm_test.shape}")

# ── STEP 6: Define final model files ───────────────────────────────────────────
final_models = {
    'LSTM_WEEK':  'LSTM_WEEK_final.keras',
    'LSTM_MONTH': 'LSTM_MONTH_final.keras',
    'CNN_WEEK':   'CNN_WEEK_final.keras',
    'CNN_MONTH':  'CNN_MONTH_final.keras',
}

# ── STEP 7: Load & evaluate each model ─────────────────────────────────────────
results = []
for name, fname in final_models.items():
    path = os.path.join(LOG_DIR, fname)
    if not os.path.exists(path):
        print(f"[ERROR] Model not found: {path}")
        continue
    print(f"\n>> Loading {name} from: {path}")
    model = load_model(path)

    if 'WEEK' in name:
        X_test, y_test = Xw_test, yw_test
    else:
        X_test, y_test = Xm_test, ym_test

    probs = model.predict(X_test, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"{name} → Test Acc: {acc:.4f} | Test F1: {f1:.4f} | Test AUC: {auc:.4f}")

    results.append({
        'Model': name,
        'Test Acc': acc,
        'Test F1': f1,
        'Test AUC': auc
    })

# ── STEP 8: Tabulate final results & save ──────────────────────────────────────
df = pd.DataFrame(results)
print("\nFINAL TEST METRICS")
print(df.to_string(index=False))

out_csv = os.path.join(LOG_DIR, 'final_test_metrics.csv')
df.to_csv(out_csv, index=False)
print(f"Results saved to {out_csv}")