import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2

# ── 0. Reproducibility & paths ────────────────────────────────────────────────────
SEED     = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── 1. Load & filter data ─────────────────────────────────────────────────────────
info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
vle  = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))

# select course 'BBB' & drop withdrawn
info = info[(info.code_module == 'BBB') & (info.final_result != 'Withdrawn')]
vle  = vle[vle.code_module == 'BBB']

# binarize labels
info['label'] = info.final_result.map({'Pass': 1, 'Distinction': 1, 'Fail': 0})

# keep only students with clicks
clicked = vle.id_student.unique()
info    = info[info.id_student.isin(clicked)]

# ensure one row per student
info = info.drop_duplicates(subset='id_student', keep='last')

# rebuild student list
students = np.sort(info.id_student.unique())

# merge activity types
vle = vle.merge(meta[['id_site', 'activity_type']], on='id_site', how='left')

# ── 2. Build WEEK-panel ─────────────────────────────────────────────────────────
vle['week'] = (vle.date // 7).astype(int)
dfw = (
    vle
    .groupby(['id_student', 'week', 'activity_type'])['sum_click']
    .sum()
    .unstack(fill_value=0)
)
weeks = np.arange(
    dfw.index.get_level_values('week').min(),
    dfw.index.get_level_values('week').max() + 1
)
idxw = pd.MultiIndex.from_product(
    [students, weeks], names=['id_student', 'week']
)
dfw = dfw.reindex(idxw, fill_value=0).reset_index()

# ── 3. Build MONTH-panel ────────────────────────────────────────────────────────
vle['month'] = (vle.date // 30).astype(int)
dfm = (
    vle
    .groupby(['id_student', 'month', 'activity_type'])['sum_click']
    .sum()
    .unstack(fill_value=0)
)
months = np.arange(
    dfm.index.get_level_values('month').min(),
    dfm.index.get_level_values('month').max() + 1
)
idxm = pd.MultiIndex.from_product(
    [students, months], names=['id_student', 'month']
)
dfm = dfm.reindex(idxm, fill_value=0).reset_index()

# ── 4. Prepare arrays with matching y ─────────────────────────────────────────────
def prepare_panel(df, time_col):
    feat_cols   = [c for c in df.columns if c not in ('id_student', time_col)]
    n_students  = len(students)
    n_intervals = df[time_col].nunique()
    X = df[feat_cols].values.reshape(n_students, n_intervals, len(feat_cols))
    sid = df['id_student'].values.reshape(n_students, n_intervals)[:, 0]
    y   = info.set_index('id_student').loc[sid, 'label'].values
    return X, y

X_week, y_week = prepare_panel(dfw, 'week')
X_mon,  y_mon  = prepare_panel(dfm, 'month')

print(f'→ WEEK panel: X={X_week.shape}, y={y_week.shape}')
print(f'→ MONTH panel: X={X_mon.shape}, y={y_mon.shape}')

# ── 5. Model builders (with stronger regularization) ──────────────────────────────
def build_lstm(input_shape, hidden=(32, 8), lr=1e-4, l2_reg=1e-4, dropout=0.3):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(hidden[0], return_sequences=True, dropout=dropout, recurrent_dropout=0.1,
             kernel_regularizer=l2(l2_reg)),
        LSTM(hidden[1], dropout=dropout, recurrent_dropout=0.1,
             kernel_regularizer=l2(l2_reg)),
        Dropout(dropout),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))
    ])
    model.compile(
        optimizer=Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn(input_shape, filters=32, kernel=3, pool=2, dr=0.5, lr=1e-3, l2_reg=1e-4):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters, kernel_size=kernel, activation='relu',
               kernel_regularizer=l2(l2_reg)),
        MaxPooling1D(pool_size=pool),
        Dropout(dr),
        Flatten(),
        Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_reg))
    ])
    model.compile(
        optimizer=Adam(lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── 6. 10-fold CV with tightened early stopping ───────────────────────────────────
def cross_validate(X, y, builder, name, epochs=700, batch_size=128, **kwargs):
    os.makedirs('logs', exist_ok=True)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    accs, f1s, aucs = [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n=== {name} fold {fold} ===")
        model = builder(X.shape[1:], **kwargs)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
            CSVLogger(os.path.join('logs', f'{name}_fold{fold}.csv'), append=False)
        ]

        history = model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[test_idx], y[test_idx]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks
        )

        preds = model.predict(X[test_idx], verbose=0).ravel()
        binary = (preds >= 0.5).astype(int)

        accs.append( accuracy_score(y[test_idx], binary) )
        f1s.append(   f1_score(y[test_idx], binary) )
        aucs.append(  roc_auc_score(y[test_idx], preds)  )

        print(f"-- Fold {fold} | Acc: {accs[-1]:.4f}, F1: {f1s[-1]:.4f}, AUC: {aucs[-1]:.4f}")

    return np.mean(accs), np.mean(f1s), np.mean(aucs)

# ── 7. Run CV over all models ─────────────────────────────────────────────────────
results = {
    'LSTM_WEEK'  : cross_validate(X_week, y_week, build_lstm,  name='LSTM_WEEK',  hidden=(32,  8), lr=1e-4),
    'LSTM_MONTH' : cross_validate(X_mon,  y_mon,  build_lstm,  name='LSTM_MONTH', hidden=(32, 16), lr=1e-4),
    'CNN_WEEK'   : cross_validate(X_week, y_week, build_cnn,   name='CNN_WEEK',   filters=32, kernel=3, pool=2, dr=0.5, lr=1e-3),
    'CNN_MONTH'  : cross_validate(X_mon,  y_mon,  build_cnn,   name='CNN_MONTH',  filters=32, kernel=3, pool=2, dr=0.5, lr=1e-3)
}

# ── 8. Print summary ─────────────────────────────────────────────────────────────
print("\nModel          | Accuracy | F1-score |   AUC")
print("-----------------------------------------------")
for name, (acc, f1, auc) in results.items():
    print(f"{name:<14}|  {acc:.4f}   |  {f1:.4f}   |  {auc:.4f}")
