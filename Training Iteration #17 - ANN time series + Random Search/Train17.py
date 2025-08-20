# -*- coding: utf-8 -*-
"""
Training Iteration #17 — Random Architecture Search for LSTM (Many‑to‑Many)
---------------------------------------------------------------------------
Goal
 - Split data by student into 70% DEV (train+val) and 30% TEST (hold‑out).
 - For each of N_TRIALS random architectures:
     • Re-split the 70% DEV into 80% train / 20% val (new seed per trial)
     • Train with EarlyStopping/ReduceLROnPlateau
     • Log metrics and keep the best by validation categorical_accuracy
 - Retrain the best architecture on the FULL 70% DEV (with a tiny val split
   for early stopping), then evaluate ONCE on the 30% TEST.

This script keeps Train15/16 conventions:
  • Saves scaler/feature_cols/label_encoder
  • CSV logs per trial + global trials_summary.csv
  • Plots for the best model (train/val curves, per‑day test accuracy)
  • Saves best_config.json, best_model_dev.h5 (trained on 70%), and
    best_model_tested.h5 (after optional fine‑tune), plus predictions.
  • (NEW) Optional per‑student probability CSV dumps for the TEST split.

Directory layout (created automatically):
  Train17_RandomSearch/
    ├─ artifacts/
    │   ├─ scaler.pkl, feature_cols.pkl, label_encoder.pkl
    │   └─ best_config.json
    ├─ trials/
    │   ├─ trial_000/
    │   │   ├─ history.csv
    │   │   └─ model.h5
    │   ├─ trial_001/ ...
    │   └─ ...
    ├─ per_student_probs/  ← per-student CSV dumps (TEST set)
    ├─ trials_summary.csv
    ├─ best_trainval_curve.png
    ├─ best_per_day_accuracy.png
    ├─ predictions_test.npy
    ├─ best_model_dev.h5
    ├─ best_model_tested.h5
    └─ test_report.json

Notes
 - Many‑to‑many like Train16: input [N, 270, F], output [N, 270, C].
 - Uses sample_weight masks to ignore padded days in loss/metrics.
 - Random search space is defined in `sample_random_config()`.
 - Adjust N_TRIALS for time (default 100).
"""

import os
import json
import math
import time
import pickle
import random
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Masking, LSTM, Dropout, Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback

from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------------------
# Version prints (Train15/16 style)
# --------------------------------------------------------------------------------------
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print('##############################################################################################################################')
print('stage 1 import necessary packages completed successfully')
print('##############################################################################################################################')

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
SEQ_LEN = 270
ROOT_DIR = "Train17_RandomSearch"
ARTIF_DIR = os.path.join(ROOT_DIR, "artifacts")
TRIALS_DIR = os.path.join(ROOT_DIR, "trials")
PER_STUDENT_DIR = os.path.join(ROOT_DIR, "per_student_probs")

os.makedirs(ROOT_DIR, exist_ok=True)
os.makedirs(ARTIF_DIR, exist_ok=True)
os.makedirs(TRIALS_DIR, exist_ok=True)
os.makedirs(PER_STUDENT_DIR, exist_ok=True)

# Random search params
N_TRIALS = 5
BASE_EPOCHS = 60
BATCH_SIZE = 100   # mirrors Train15/16 style
PATIENCE_ES = 7
PATIENCE_RLR = 2
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------------------------------------------
# Search Space Definition
# --------------------------------------------------------------------------------------
@dataclass
class LSTMConfig:
    bidirectional: bool
    n_lstm_layers: int
    lstm_units: List[int]
    lstm_recurrent_dropout: float
    lstm_dropout: float
    dense_units: int
    dense_activation: str
    final_dropout: float
    optimizer: str
    learning_rate: float


# Custom training progress callback
class TrainingProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"Epoch {epoch + 1}: "
            f"loss={logs.get('loss'):.4f}, "
            f"accuracy={logs.get('accuracy'):.4f}, "
            f"val_loss={logs.get('val_loss'):.4f}, "
            f"val_accuracy={logs.get('val_accuracy'):.4f}"
        )

# --------------------------------------------------------------------------------------
# Custom succinct per-epoch printer (terminal updates)
# --------------------------------------------------------------------------------------
class EpochPrinter(Callback):
    def __init__(self, header: str = ""):
        super().__init__()
        self.header = header
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # metric names used in this script
        loss = logs.get('loss')
        acc = logs.get('categorical_accuracy') or logs.get('accuracy')
        vloss = logs.get('val_loss')
        vacc = logs.get('val_categorical_accuracy') or logs.get('val_accuracy')

        def fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) else "-"
        print(
            f"{self.header} Epoch {epoch+1}: "
            f"loss={fmt(loss)} acc={fmt(acc)} "
            f"val_loss={fmt(vloss)} val_acc={fmt(vacc)}",
            flush=True
        )



def sample_random_config() -> LSTMConfig:
    n_layers = random.choice([1, 2, 3])
    units_choices = [64, 96, 128, 160, 200, 256, 320]
    lstm_units = [random.choice(units_choices) for _ in range(n_layers)]
    return LSTMConfig(
        bidirectional=random.choice([False, True]),
        n_lstm_layers=n_layers,
        lstm_units=lstm_units,
        lstm_recurrent_dropout=random.choice([0.0, 0.1, 0.2, 0.3]),
        lstm_dropout=random.choice([0.2, 0.3, 0.4, 0.5]),
        dense_units=random.choice([64, 96, 128, 160, 200, 256]),
        dense_activation=random.choice(["relu", "tanh"]),
        final_dropout=random.choice([0.0, 0.2, 0.3, 0.4, 0.5]),
        optimizer=random.choice(["adam", "adamw", "nadam"]),
        learning_rate=random.choice([1e-3, 5e-4, 2e-4, 1e-4])
    )


def make_optimizer(name: str, lr: float):
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=lr)
    if name == "adamw":
        return tf.keras.optimizers.AdamW(learning_rate=lr)
    if name == "nadam":
        return tf.keras.optimizers.Nadam(learning_rate=lr)
    return tf.keras.optimizers.Adam(learning_rate=lr)


def build_model(config: LSTMConfig, n_features: int, n_classes: int) -> Model:
    inp = Input(shape=(SEQ_LEN, n_features))
    x = Masking(mask_value=0.0, name='masking')(inp)
    for i, units in enumerate(config.lstm_units, start=1):
        if config.bidirectional:
            x = Bidirectional(LSTM(units, return_sequences=True,
                                   dropout=config.lstm_dropout,
                                   recurrent_dropout=config.lstm_recurrent_dropout),
                              name=f"BiLSTM_{i}")(x)
        else:
            x = LSTM(units, return_sequences=True,
                     dropout=config.lstm_dropout,
                     recurrent_dropout=config.lstm_recurrent_dropout,
                     name=f"LSTM_{i}")(x)
    if config.final_dropout > 0:
        x = Dropout(config.final_dropout, name='final_dropout')(x)
    x = Dense(config.dense_units, activation=config.dense_activation, name='dense')(x)
    out = Dense(n_classes, activation='softmax', name='out')(x)

    model = Model(inp, out, name='ANN_LSTM_RandomSearch')
    opt = make_optimizer(config.optimizer, config.learning_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# --------------------------------------------------------------------------------------
# Data Loading
# --------------------------------------------------------------------------------------
print('[Load] data.csv')
df = pd.read_csv('data.csv', sep=',')
print(df.head())
print(df.shape)
print('\n')

# --- check for duplicate (student, day) combos ---
dups = df.duplicated(subset=['id_student', 'date']).sum()
print('Duplicate (id_student, date) rows:', dups)
print('\n')

print('Preparing fixed-length sequences per student (vectorized)...')

# 1) Identify numeric columns to sum per (id_student, date)
num_cols = df.select_dtypes(include=['number']).columns.tolist()
exclude_from_sum = {'id_student', 'date'}
num_sum_cols = [c for c in num_cols if c not in exclude_from_sum]

# (Optional) reduce memory before heavy ops
for c in num_sum_cols:
    if pd.api.types.is_float_dtype(df[c]):
        df[c] = df[c].astype('float32')
    elif pd.api.types.is_integer_dtype(df[c]):
        df[c] = pd.to_numeric(df[c], downcast='integer')

# 2) Collapse duplicates ONCE with a single groupby (fast, all numeric at once)
#    This gives one row per (student, day) with summed counts.
g_num = df.groupby(['id_student', 'date'], sort=False, observed=True)[num_sum_cols].sum()

# 3) Build the full student×day index and reindex (pads missing days)
all_students = df['id_student'].dropna().unique()
full_index = pd.MultiIndex.from_product([all_students, range(SEQ_LEN)],
                                        names=['id_student', 'date'])
g_num = g_num.reindex(full_index, fill_value=0)

# 4) Get each student's final_result (mode or first) once per student
def mode_or_first(s: pd.Series):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else s.iloc[0]

final_per_student = df.groupby('id_student', observed=True)['final_result'].agg(mode_or_first)

# 5) Assemble final dataframe: reset_index and attach final_result
df = g_num.reset_index()
df['final_result'] = df['id_student'].map(final_per_student)

print('[OK] Sequences padded to 270 timesteps each (vectorized).')


# --------------------------------------------------------------------------------------
# Feature scaling, label encoding, reshape to sequences, and 70/30 DEV/TEST split
# --------------------------------------------------------------------------------------
# Keep 'date' as a feature; exclude only final_result and id_student
feature_cols = [c for c in df.columns if c not in ['final_result', 'id_student']]
# Optional: drop index-artifact feature if present
feature_cols = [c for c in feature_cols if c != 'Unnamed: 0']

# Build per-timestep mask BEFORE fillna (1 where original features were present)
row_mask = (~df[feature_cols].isna()).any(axis=1).astype(float)

# Fill NaNs then scale (MinMaxScaler cannot handle NaN)
scaler = MinMaxScaler()
X_flat = df[feature_cols].copy()
X_flat[feature_cols] = X_flat[feature_cols].fillna(0)
X_flat[feature_cols] = scaler.fit_transform(X_flat[feature_cols])

# Encode labels
le = LabelEncoder()
df['final_result_enc'] = le.fit_transform(df['final_result'].astype(str))
num_classes = len(le.classes_)

# Persist preprocessing artifacts
with open(os.path.join(ARTIF_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(ARTIF_DIR, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
with open(os.path.join(ARTIF_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
print('[Saved] scaler.pkl, feature_cols.pkl, label_encoder.pkl')

# Reshape into [N_students, 270, F] and targets [N_students, 270, C]
students = df['id_student'].unique()
N = len(students)
F = len(feature_cols)

X = X_flat.to_numpy().reshape(N, SEQ_LEN, F)
y_int = df['final_result_enc'].to_numpy().reshape(N, SEQ_LEN)
y = to_categorical(y_int, num_classes=num_classes)
mask = row_mask.to_numpy().reshape(N, SEQ_LEN)

print(f'X shape: {X.shape}  y shape: {y.shape}  mask shape: {mask.shape}')

# DEV/TEST split by student: 70% / 30%
all_idx = np.arange(N)
idx_dev, idx_test = train_test_split(all_idx, test_size=0.30, random_state=SEED)

X_dev, X_test = X[idx_dev], X[idx_test]
y_dev, y_test = y[idx_dev], y[idx_test]
mask_dev, mask_test = mask[idx_dev], mask[idx_test]
students_dev, students_test = students[idx_dev], students[idx_test]

print('##############################################################################################################################')
print(f'DEV size: {X_dev.shape[0]} students  |  TEST size: {X_test.shape[0]} students')
print('##############################################################################################################################')

# --------------------------------------------------------------------------------------
# Helper: single trial train/val split, train model, return best val metric
# --------------------------------------------------------------------------------------

def run_single_trial(trial_id: int, config: LSTMConfig) -> Tuple[float, str]:
    """Train one random architecture on a fresh (80/20) split of DEV.
    Returns (best_val_acc, trial_dir).
    """
    trial_name = f"trial_{trial_id:03d}"
    tdir = os.path.join(TRIALS_DIR, trial_name)
    os.makedirs(tdir, exist_ok=True)

    # fresh 80/20 split
    seed = SEED + trial_id * 100 + 7
    n_dev = X_dev.shape[0]
    dev_idx = np.arange(n_dev)
    idx_tr, idx_va = train_test_split(dev_idx, test_size=0.20, random_state=seed)

    X_tr, X_va = X_dev[idx_tr], X_dev[idx_va]
    y_tr, y_va = y_dev[idx_tr], y_dev[idx_va]
    m_tr, m_va = mask_dev[idx_tr], mask_dev[idx_va]

    model = build_model(config, n_features=F, n_classes=num_classes)

    # callbacks
    csv_log_path = os.path.join(tdir, 'history.csv')
    csv_logger = CSVLogger(csv_log_path, append=False)
    es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=PATIENCE_ES, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max', patience=PATIENCE_RLR, factor=0.5, min_lr=1e-6)

    hist = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va, m_va),
    epochs=BASE_EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,  # keep Keras quiet; we print via EpochPrinter
    callbacks=[csv_logger, es, rlr, EpochPrinter(header=f"[Trial {trial_id:03d}]")],
    sample_weight=m_tr
    )


    # best val acc
    best_val_acc = max(hist.history.get('val_categorical_accuracy', [0.0]))

    # save model for this trial
    model.save(os.path.join(tdir, 'model.h5'))

    # save config
    with open(os.path.join(tdir, 'config.json'), 'w') as f:
        json.dump(asdict(config), f, indent=2)

    return float(best_val_acc), tdir

# --------------------------------------------------------------------------------------
# Random Search Loop
# --------------------------------------------------------------------------------------
results = []
start_time = time.time()
for t in range(N_TRIALS):
    cfg = sample_random_config()
    val_acc, tdir = run_single_trial(t, cfg)
    results.append({
        'trial': t,
        'val_categorical_accuracy': val_acc,
        **asdict(cfg)
    })
    print(f"Trial {t:03d} — val_cat_acc={val_acc:.4f} — saved in {tdir}")

elapsed = time.time() - start_time
print(f"Random search finished in {elapsed/60.0:.1f} min across {N_TRIALS} trials.")

# save summary
summary_df = pd.DataFrame(results).sort_values('val_categorical_accuracy', ascending=False)
sum_path = os.path.join(ROOT_DIR, 'trials_summary.csv')
summary_df.to_csv(sum_path, index=False)
print(f"[Saved] trials_summary.csv → {sum_path}")

best_row = summary_df.iloc[0]
best_cfg = LSTMConfig(
    bidirectional=bool(best_row['bidirectional']),
    n_lstm_layers=int(best_row['n_lstm_layers']),
    lstm_units=[int(u) for u in eval(str(best_row['lstm_units']))] if isinstance(best_row['lstm_units'], str) else list(best_row['lstm_units']),
    lstm_recurrent_dropout=float(best_row['lstm_recurrent_dropout']),
    lstm_dropout=float(best_row['lstm_dropout']),
    dense_units=int(best_row['dense_units']),
    dense_activation=str(best_row['dense_activation']),
    final_dropout=float(best_row['final_dropout']),
    optimizer=str(best_row['optimizer']),
    learning_rate=float(best_row['learning_rate'])
)

with open(os.path.join(ARTIF_DIR, 'best_config.json'), 'w') as f:
    json.dump(asdict(best_cfg), f, indent=2)
print('[Saved] best_config.json')

# --------------------------------------------------------------------------------------
# Retrain best config on FULL DEV (70%) with small val split for ES
# --------------------------------------------------------------------------------------
seed_final = SEED + 999
n_dev = X_dev.shape[0]
dev_idx = np.arange(n_dev)
idx_tr_final, idx_va_final = train_test_split(dev_idx, test_size=0.10, random_state=seed_final)

X_trf, X_vaf = X_dev[idx_tr_final], X_dev[idx_va_final]
y_trf, y_vaf = y_dev[idx_tr_final], y_dev[idx_va_final]
m_trf, m_vaf = mask_dev[idx_tr_final], mask_dev[idx_va_final]

best_model = build_model(best_cfg, n_features=F, n_classes=num_classes)

csv_logger = CSVLogger(os.path.join(ROOT_DIR, 'best_history.csv'), append=False)
es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=PATIENCE_ES, restore_best_weights=True)
rlr = ReduceLROnPlateau(monitor='val_categorical_accuracy', mode='max', patience=PATIENCE_RLR, factor=0.5, min_lr=1e-6)

hist = best_model.fit(
    X_trf, y_trf,
    validation_data=(X_vaf, y_vaf, m_vaf),
    epochs=BASE_EPOCHS*2,
    batch_size=BATCH_SIZE,
    verbose=0,  # use our own concise printer
    callbacks=[csv_logger, es, rlr, EpochPrinter(header='[Best]')],
    sample_weight=m_trf
)


best_model.save(os.path.join(ROOT_DIR, 'best_model_dev.h5'))

# plot best train/val curve
plt.title('Best model — categorical accuracy')
plt.plot(hist.history['categorical_accuracy'], label='train')
plt.plot(hist.history['val_categorical_accuracy'], label='val')
plt.ylabel('categorical accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(os.path.join(ROOT_DIR, 'best_trainval_curve.png'))
plt.close()
print('[Saved] best_trainval_curve.png')

# --------------------------------------------------------------------------------------
# Final evaluation on TEST (30%)
# --------------------------------------------------------------------------------------
print('[Eval] testing best model on 30% hold‑out test set (masked) ...')
results_test = best_model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1, sample_weight=mask_test)
metrics_dict = dict(zip(best_model.metrics_names, [float(v) for v in results_test]))
print('[Test] ', metrics_dict)

# predictions for analysis
probs_test = best_model.predict(X_test, batch_size=BATCH_SIZE)
np.save(os.path.join(ROOT_DIR, 'predictions_test.npy'), probs_test)

# per‑day accuracy on test
preds_test = probs_test.argmax(axis=-1)
true_test = y_test.argmax(axis=-1)
m = (mask_test > 0.5).astype(bool)

per_day_acc = []
for t in range(SEQ_LEN):
    valid = m[:, t]
    if valid.any():
        acc_t = (preds_test[valid, t] == true_test[valid, t]).mean()
    else:
        acc_t = np.nan
    per_day_acc.append(float(acc_t) if not math.isnan(acc_t) else None)

plt.title('Best model — Per‑day categorical accuracy (test)')
plt.plot([a if a is not None else np.nan for a in per_day_acc], label='test')
plt.ylabel('categorical accuracy')
plt.xlabel('days (0..269)')
plt.legend(['test'], loc='upper left')
plt.savefig(os.path.join(ROOT_DIR, 'best_per_day_accuracy.png'))
plt.close()
print('[Saved] best_per_day_accuracy.png')

# micro accuracy across all valid positions
masked_micro_acc = float((preds_test[m] == true_test[m]).mean())

report = {
    'test_metrics': metrics_dict,
    'masked_micro_accuracy': masked_micro_acc,
    'per_day_accuracy': per_day_acc
}
with open(os.path.join(ROOT_DIR, 'test_report.json'), 'w') as f:
    json.dump(report, f, indent=2)
print('[Saved] test_report.json')

# --------------------------------------------------------------------------------------
# (NEW) Optional per‑student probability CSV dumps for easy inspection
# --------------------------------------------------------------------------------------
print('[Dump] Writing per‑student TEST probability CSVs ...')
class_cols = [f"prob_{cls}" for cls in le.classes_]
for i in range(X_test.shape[0]):
    sid = students_test[i]
    df_out = pd.DataFrame(probs_test[i], columns=class_cols)
    df_out.insert(0, 'day', np.arange(SEQ_LEN))
    df_out['pred_label_idx'] = preds_test[i]
    df_out['true_label_idx'] = true_test[i]
    df_out['pred_label'] = [le.classes_[k] for k in preds_test[i]]
    df_out['true_label'] = [le.classes_[k] for k in true_test[i]]
    out_path = os.path.join(PER_STUDENT_DIR, f'student_{sid}.csv')
    df_out.to_csv(out_path, index=False)
print(f"[Saved] {X_test.shape[0]} CSVs under {os.path.abspath(PER_STUDENT_DIR)}")

# Also save a final tested model snapshot
best_model.save(os.path.join(ROOT_DIR, 'best_model_tested.h5'))

print('\nDone. All artifacts saved under:', os.path.abspath(ROOT_DIR))
