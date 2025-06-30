import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ── 0. Data directory ─────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir   = os.path.join(script_dir, 'data')

# ── 1. Load & filter ───────────────────────────────────────────────────────────────
student_info = pd.read_csv(os.path.join(data_dir, 'studentInfo.csv'))
student_vle  = pd.read_csv(os.path.join(data_dir, 'studentVle.csv'))
vle_meta     = pd.read_csv(os.path.join(data_dir, 'vle.csv'))

student_info = student_info[
    (student_info['code_module']=='BBB') &
    (student_info['final_result']!='Withdrawn')
]
student_vle  = student_vle[student_vle['code_module']=='BBB']

# ── 2. Prepare & label ────────────────────────────────────────────────────────────
student_vle = student_vle.merge(
    vle_meta[['id_site','activity_type']],
    on='id_site', how='left'
)

student_info['label'] = student_info['final_result'].map({
    'Pass':1, 'Distinction':1, 'Fail':0
})

clicked = student_vle['id_student'].unique()
student_info = student_info[student_info['id_student'].isin(clicked)]

# ── 3. Feature extraction ─────────────────────────────────────────────────────────
student_vle['week'] = (student_vle['date'] // 7).astype(int)

df = student_vle.groupby(
    ['id_student','week','activity_type']
)['sum_click'].sum().unstack(fill_value=0)

students = student_info['id_student'].unique()
weeks    = np.arange(df.index.get_level_values('week').min(),
                     df.index.get_level_values('week').max()+1)
idx = pd.MultiIndex.from_product([students, weeks],
                                 names=['id_student','week'])
df = df.reindex(idx, fill_value=0).reset_index()

# ── 4. Reshape for LSTM ────────────────────────────────────────────────────────────
feat_cols = df.columns.drop(['id_student','week'])
n_students = len(students)
n_weeks    = len(weeks)

X = df.pivot(index='id_student', columns='week',
             values=feat_cols).values
X = X.reshape(n_students, n_weeks, len(feat_cols))

label_map = student_info.groupby('id_student')['label'].first()
y         = label_map.loc[students].values

# ── 5. Model builder with explicit Input ──────────────────────────────────────────
def build_lstm(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True, dropout=0.2),
        LSTM(8, dropout=0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(1e-4),
                  loss='binary_crossentropy')
    return model

# ── 6. 10-fold CV training & eval ─────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
accs, f1s, aucs = [], [], []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]

    model = build_lstm((n_weeks, len(feat_cols)))
    print(f"\nTraining fold {fold}...")
    model.fit(X_tr, y_tr,
              epochs=50,           # ← reduced for quick feedback
              batch_size=128,
              verbose=1)           # ← show progress

    probs = model.predict(X_te, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    accs.append( accuracy_score(y_te, preds) )
    f1s.append(  f1_score(y_te, preds) )
    aucs.append( roc_auc_score(y_te, probs) )

print("\nFinal 10-fold CV results:")
print(f"Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"F1-score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
print(f"AUC:       {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")