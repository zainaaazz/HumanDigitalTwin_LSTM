import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    LSTM, Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling1D, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint
)
from tensorflow.keras.regularizers import l2

# ── STEP 0: Set seeds and paths ─────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR  = os.path.join(BASE_DIR, 'TrainX_Logs')

# create log folder
os.makedirs(LOG_DIR, exist_ok=True)

# ── STEP 1: Load and preprocess student info ────────────────────────────────────
info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
info = info[(info.code_module == 'BBB') & (info.final_result != 'Withdrawn')]
info['label'] = info.final_result.map({'Pass': 1, 'Distinction': 1, 'Fail': 0})

# ── STEP 2: Load and filter click data ──────────────────────────────────────────
vle  = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))
vle = vle[vle.code_module == 'BBB']

# merge activity labels
vle = vle.merge(meta[['id_site','activity_type']], on='id_site', how='left')

# keep only students with any clicks
clicked_students = vle.id_student.unique()
info = info[info.id_student.isin(clicked_students)].drop_duplicates('id_student')

# sorted list of students
students = np.sort(info.id_student.unique())

# ── STEP 3: Build WEEK-based panel ──────────────────────────────────────────────
vle['week'] = (vle.date // 7).astype(int)
dfw = (
    vle
    .groupby(['id_student','week','activity_type'])['sum_click']
    .sum()
    .unstack(fill_value=0)
)
weeks = np.arange(dfw.index.get_level_values('week').min(), dfw.index.get_level_values('week').max()+1)
index_w = pd.MultiIndex.from_product([students, weeks], names=['id_student','week'])
dfw = dfw.reindex(index_w, fill_value=0).reset_index()

# ── STEP 4: Build MONTH-based panel ─────────────────────────────────────────────
vle['month'] = (vle.date // 30).astype(int)
dfm = (
    vle
    .groupby(['id_student','month','activity_type'])['sum_click']
    .sum()
    .unstack(fill_value=0)
)
months = np.arange(dfm.index.get_level_values('month').min(), dfm.index.get_level_values('month').max()+1)
index_m = pd.MultiIndex.from_product([students, months], names=['id_student','month'])
dfm = dfm.reindex(index_m, fill_value=0).reset_index()

# ── STEP 5: Prepare input arrays ─────────────────────────────────────────────────
def prepare_panel(df, time_col):
    features = [c for c in df.columns if c not in ('id_student', time_col)]
    n_students  = len(students)
    n_intervals = df[time_col].nunique()
    X = df[features].values.reshape(n_students, n_intervals, len(features))
    sid = df['id_student'].values.reshape(n_students, n_intervals)[:,0]
    y   = info.set_index('id_student').loc[sid, 'label'].values
    return X, y

X_week, y_week = prepare_panel(dfw, 'week')
X_mon,  y_mon  = prepare_panel(dfm, 'month')
print(f"WEEK X={X_week.shape}, y={y_week.shape}")
print(f"MONTH X={X_mon.shape}, y={y_mon.shape}")

# ── STEP 6: Split out a held-out test set ───────────────────────────────────────
Xw_trainval, Xw_test, yw_trainval, yw_test = train_test_split(
    X_week, y_week,
    test_size=0.1,
    stratify=y_week,
    random_state=SEED
)
Xm_trainval, Xm_test, ym_trainval, ym_test = train_test_split(
    X_mon, y_mon,
    test_size=0.1,
    stratify=y_mon,
    random_state=SEED
)

# ── STEP 7: Define model builders ────────────────────────────────────────────────
def build_lstm(input_shape, hidden=(32,8), lr=1e-4, l2_reg=1e-3, dropout=0.4):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(hidden[0], return_sequences=True, kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(LSTM(hidden[1], kernel_regularizer=l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn(input_shape, filters=32, kernel=5, pool=2, dr=0.5, lr=5e-4, l2_reg=5e-4):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Conv1D(filters, kernel_size=kernel, padding='same', kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool))
    model.add(Dropout(dr))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ── STEP 8: Cross-validation function ────────────────────────────────────────────
def cross_validate(X, y, builder, name, folds=10, epochs=700, batch_size=128, **kwargs):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    metrics = []
    fold_no = 1
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        model = builder(X.shape[1:], **kwargs)
        cp_path = os.path.join(LOG_DIR, f"{name}_fold{fold_no}.h5")
        csv_path = os.path.join(LOG_DIR, f"{name}_fold{fold_no}.csv")
        callbacks = [
            EarlyStopping('val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=1e-6),
            CSVLogger(csv_path),
            ModelCheckpoint(cp_path, monitor='val_loss', save_best_only=True)
        ]
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size,
            callbacks=callbacks, verbose=0
        )
        best = load_model(cp_path)
        p_val = best.predict(X_val).ravel()
        b_val = (p_val >= 0.5).astype(int)
        acc = accuracy_score(y_val, b_val)
        f1  = f1_score(y_val, b_val)
        auc = roc_auc_score(y_val, p_val)
        print(f"--- Fold {fold_no} | Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
        metrics.append((acc, f1, auc))
        fold_no += 1
    accs, f1s, aucs = zip(*metrics)
    print(f"\n{name} CV Mean | Acc: {np.mean(accs):.4f}, F1: {np.mean(f1s):.4f}, AUC: {np.mean(aucs):.4f}")
    return metrics

# ── STEP 9: Run CV across train+val sets ─────────────────────────────────────────
models = {
    'LSTM_WEEK':  dict(X_trainval=Xw_trainval, y_trainval=yw_trainval,  X_test=Xw_test, y_test=yw_test, builder=build_lstm, kwargs={'hidden':(32,8)}),
    'LSTM_MONTH': dict(X_trainval=Xm_trainval, y_trainval=ym_trainval,  X_test=Xm_test, y_test=ym_test, builder=build_lstm, kwargs={'hidden':(32,16)}),
    'CNN_WEEK':   dict(X_trainval=Xw_trainval, y_trainval=yw_trainval,  X_test=Xw_test, y_test=yw_test, builder=build_cnn, kwargs={}),
    'CNN_MONTH':  dict(X_trainval=Xm_trainval, y_trainval=ym_trainval,  X_test=Xm_test, y_test=ym_test, builder=build_cnn, kwargs={})
}

reports = {}
for name, cfg in models.items():
    print(f"\n*** {name} ***")
    reports[name] = cross_validate(
        cfg['X_trainval'], cfg['y_trainval'],
        cfg['builder'], name=name, **cfg['kwargs']
    )

# ── STEP 10: Final model training & held-out test evaluation ────────────────────
print("\nHELD-OUT TEST EVALUATION")
eval_results = []
for name, cfg in models.items():
    print(f"Training final {name} on full train+val set...")
    model = cfg['builder'](cfg['X_trainval'].shape[1:], **cfg.get('kwargs', {}))
    cp_path = os.path.join(LOG_DIR, f"{name}_final.h5")
    callbacks = [
        EarlyStopping('val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(cp_path, monitor='val_loss', save_best_only=True)
    ]
    model.fit(
        cfg['X_trainval'], cfg['y_trainval'],
        validation_split=0.1,
        epochs=700, batch_size=128,
        callbacks=callbacks, verbose=1
    )
    best = load_model(cp_path)
    p_test = best.predict(cfg['X_test']).ravel()
    b_test = (p_test >= 0.5).astype(int)
    acc = accuracy_score(cfg['y_test'], b_test)
    f1  = f1_score(cfg['y_test'], b_test)
    auc = roc_auc_score(cfg['y_test'], p_test)
    print(f"{name} → Test Acc: {acc:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    eval_results.append({'model':name,'accuracy':acc,'f1_score':f1,'auc':auc})

# ── STEP 11: Summary of fold and test metrics ──────────────────────────────────
print("\nFINAL TEST METRICS")
print("Model     | Test Accuracy | Test F1 | Test AUC")
print("-------------------------------------------")
for r in eval_results:
    print(f"{r['model']:<10} | {r['accuracy']:.4f}        | {r['f1_score']:.4f}  | {r['auc']:.4f}")