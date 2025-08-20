# -*- coding: utf-8 -*-
"""
Training Iteration #16 — ANN‑LSTM (Single Sequence Model, Many‑to‑Many)
---------------------------------------------------------------------
This script replaces the per‑day training loop with ONE sequence model that
ingests the full 270‑day sequence per student and outputs a prediction for
EVERY day (many‑to‑many). It preserves the Train15 logging/plot conventions:
  • CSV history logging
  • Saved scaler / feature columns
  • Saved model (.h5)
  • Saved training/validation curves plot
  • Saved per‑day accuracy plot (analogous to Train15's all‑days summary)

Foldering mirrors prior iterations, with a new directory name for clarity:
  MODEL_DIR = "Train16_Sequence_Model"

Inputs
------
  data.csv — same flattened OULAD‑style table with one row per (student, day).

Outputs (in MODEL_DIR)
----------------------
  scaler.pkl, feature_cols.pkl, label_encoder.pkl
  history_all_days.csv
  trainval_all_days.png
  per_day_accuracy.png
  model_all_days.h5
  predictions_test.npy (softmax probabilities)

Notes
-----
  • We keep 'date' as a feature (to remain consistent with Train15, which
    dropped only ['final_result','id_student']).
  • We create a per‑time‑step mask so padded days do not affect the loss.
  • If a student's sequence is shorter than 270 days, we pad zeros and mask.
"""

import os
import pickle
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.layers import Input, Masking, LSTM, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# --------------------------------------------------------------------------------------
# Version prints (kept from Train15 style)
# --------------------------------------------------------------------------------------
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print('##############################################################################################################################')
print('stage 1 import necessary packages completed successfully')
print('\n##############################################################################################################################')

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
SEQ_LEN = 270
MODEL_DIR = "Train16_Sequence_Model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Reproducibility
SEED = 42
np.random.seed(SEED)
 tf_random = tf.random.set_seed(SEED)

# --------------------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------------------
print('[Load] data.csv')
df = pd.read_csv('data.csv', sep=',')
print(df.head())
print(df.shape)
print('\n\n')

# --------------------------------------------------------------------------------------
# Build fixed‑length sequences per student (0..269), pad missing days
# --------------------------------------------------------------------------------------
print('Preparing fixed-length sequences per student...')

def to_fixed_sequence(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values('date')
    g = g.set_index('date').reindex(range(SEQ_LEN))  # pad to 0..269
    # keep id_student value for padded rows
    g['id_student'] = g['id_student'].iloc[0]
    return g.reset_index().rename(columns={'index': 'date'})

df = df.groupby('id_student', group_keys=False).apply(to_fixed_sequence)
print('[OK] Sequences padded to 270 timesteps each.')

# --------------------------------------------------------------------------------------
# Feature scaling & label encoding (consistent with Train15 conventions)
# --------------------------------------------------------------------------------------
feature_cols = [c for c in df.columns if c not in ['final_result', 'id_student']]

# scaler
scaler = MinMaxScaler()
X_flat = df[feature_cols].copy()
X_flat[feature_cols] = scaler.fit_transform(X_flat[feature_cols])

# label encoder
le = LabelEncoder()
# Final result per row (same as Train15). If your dataset only has final labels,
# they will repeat across days — that's fine for many-to-many supervision.
df['final_result_enc'] = le.fit_transform(df['final_result'].astype(str))
num_classes = len(le.classes_)

# mask: mark rows where any feature was originally present
row_mask = (~df[feature_cols].isna()).any(axis=1).astype(float)

# fill NaNs in features/labels after padding
X_flat[feature_cols] = X_flat[feature_cols].fillna(0)
df['final_result_enc'] = df['final_result_enc'].fillna(0).astype(int)

# save preprocessing artifacts
with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, 'feature_cols.pkl'), 'wb') as f:
    pickle.dump(feature_cols, f)
with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)

print('[Saved] scaler.pkl, feature_cols.pkl, label_encoder.pkl')

# --------------------------------------------------------------------------------------
# Reshape into sequences: [n_students, 270, n_features] and targets: [n_students, 270, C]
# --------------------------------------------------------------------------------------
students = df['id_student'].unique()
N = len(students)
F = len(feature_cols)

X = X_flat.to_numpy().reshape(N, SEQ_LEN, F)
y_int = df['final_result_enc'].to_numpy().reshape(N, SEQ_LEN)
y = to_categorical(y_int, num_classes=num_classes)  # [N, 270, C]
mask = row_mask.to_numpy().reshape(N, SEQ_LEN)      # [N, 270]

print(f'X shape: {X.shape}  y shape: {y.shape}  mask shape: {mask.shape}')

# --------------------------------------------------------------------------------------
# Train/test split by student (avoid leakage)
# --------------------------------------------------------------------------------------
idx = np.arange(N)
train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED)
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
mask_train, mask_test = mask[train_idx], mask[test_idx]

print('##############################################################################################################################')
print('stage 9 splitting dataset X,y into train/test by student completed successfully')
print('##############################################################################################################################')

# --------------------------------------------------------------------------------------
# Build ONE sequence model (many‑to‑many) — architecture mirrors Train15
# --------------------------------------------------------------------------------------
inputs = Input(shape=(SEQ_LEN, F))
# Mask timesteps that are all zeros in features
x = Masking(mask_value=0.0, name='masking')(inputs)
x = LSTM(200, return_sequences=True, recurrent_dropout=0.2, name='LSTM_Layer')(x)
x = Dropout(0.5, name='Dropout_layer')(x)
x = Dense(100, activation='relu', name='ANN_Hidden_Layer')(x)
outputs = Dense(num_classes, activation='softmax', name='ANN_Output_Layer')(x)

model = Model(inputs, outputs, name='ANN_LSTM_Sequence_AllDays')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())

# --------------------------------------------------------------------------------------
# Train with CSVLogger; use sample_weight (mask) to ignore padded timesteps
# --------------------------------------------------------------------------------------
log_path = os.path.join(MODEL_DIR, 'history_all_days.csv')
csv_logger = CSVLogger(log_path, append=False)

epochs = 100
batch_size = 100   # to echo Train15 defaults

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[csv_logger],
    sample_weight=mask_train
)

print(f"[Saved] training history → {log_path}")

# --------------------------------------------------------------------------------------
# Evaluate (masked)
# --------------------------------------------------------------------------------------
print('[Eval] masked evaluation on test set')
results = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1, sample_weight=mask_test)
print(f"Test results: {dict(zip(model.metrics_names, results))}")

# --------------------------------------------------------------------------------------
# Save training/validation curves (categorical_accuracy)
# --------------------------------------------------------------------------------------
plt.title('categorical accuracy')
plt.plot(history.history['categorical_accuracy'], label='train')
plt.plot(history.history['val_categorical_accuracy'], label='val')
plt.ylabel('categorical accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
trainval_path = os.path.join(MODEL_DIR, 'trainval_all_days.png')
plt.savefig(trainval_path)
plt.close()
print(f"[Saved] train/val curves → {trainval_path}")

# --------------------------------------------------------------------------------------
# Per‑day accuracy curve on TEST set (like Train15's all_days_accuracy)
# --------------------------------------------------------------------------------------
print('[Per‑day metrics] computing per‑timestep accuracy on test set...')
probs_test = model.predict(X_test, batch_size=batch_size)
preds_test = probs_test.argmax(axis=-1)   # [n_test, 270]
true_test = y_test.argmax(axis=-1)        # [n_test, 270]

# mask
m = (mask_test > 0.5).astype(bool)        # [n_test, 270]

per_day_acc = []
for t in range(SEQ_LEN):
    valid = m[:, t]
    if valid.any():
        acc_t = (preds_test[valid, t] == true_test[valid, t]).mean()
    else:
        acc_t = np.nan
    per_day_acc.append(acc_t)

# plot accuracy over days
plt.title('Per‑day categorical accuracy (test)')
plt.plot(per_day_acc, label='test')
plt.ylabel('categorical accuracy')
plt.xlabel('days (0..269)')
plt.legend(['test'], loc='upper left')
perday_path = os.path.join(MODEL_DIR, 'per_day_accuracy.png')
plt.savefig(perday_path)
plt.close()
print(f"[Saved] per‑day accuracy plot → {perday_path}")

# Overall micro accuracy on masked positions
overall_masked_acc = (preds_test[m] == true_test[m]).mean()
print(f"[Per‑day metrics] overall masked micro accuracy: {overall_masked_acc:.4f}")

# --------------------------------------------------------------------------------------
# Save model + predictions for analysis/Streamlit
# --------------------------------------------------------------------------------------
model_path = os.path.join(MODEL_DIR, 'model_all_days.h5')
model.save(model_path)
np.save(os.path.join(MODEL_DIR, 'predictions_test.npy'), probs_test)
print(f"[Saved] model → {model_path}")
print(f"[Saved] predictions_test.npy (softmax probs)")

print('\nDone. All artifacts saved under:', os.path.abspath(MODEL_DIR))
