#conda activate hdt-lstm
#cd C:\Users\USER-PC\Desktop\HumanDigitalTwin_LSTM
#cd Training Iteration #10 - ANN time series

import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ── 0) Configuration ─────────────────────────────────────────────────────────────
DATA_PATH        = r"C:\Users\USER-PC\Desktop\HumanDigitalTwin_LSTM\Training Iteration #10 - ANN time series\data.csv"
LABEL_COL        = 'final_result'
LABEL_MAP        = {'Withdrawn':0, 'Fail':1, 'Pass':2, 'Distinction':3}
MODEL_SAVE_DIR   = os.path.dirname(DATA_PATH)
MODEL_SAVE_PATH  = os.path.join(MODEL_SAVE_DIR, "Train10Model")

# ── 1) Auto-detect feature columns ────────────────────────────────────────────────
all_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
exclude  = {'Unnamed: 0', 'student_id', 'time_step', LABEL_COL}
feature_cols = [c for c in all_cols if c not in exclude]
print(f"Detected {len(feature_cols)} feature columns.")

# ── 2) Pre-fit scaler on a small sample ────────────────────────────────────────────
sample = pd.read_csv(DATA_PATH, usecols=feature_cols, nrows=5000)
scaler = MinMaxScaler().fit(sample)

# ── 3) Build ANN-LSTM ─────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(200, input_shape=(1, len(feature_cols))),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(len(LABEL_MAP), activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-3),
    metrics=['accuracy']
)

# ── 4) Generator for streaming CSV ────────────────────────────────────────────────
def csv_generator(path, chunksize=5000):
    usecols = feature_cols + [LABEL_COL]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        X = scaler.transform(chunk[feature_cols]).reshape(-1,1,len(feature_cols))
        y = to_categorical(chunk[LABEL_COL].map(LABEL_MAP).values,
                           num_classes=len(LABEL_MAP))
        yield X, y

# ── 5) Train by streaming ─────────────────────────────────────────────────────────

chunksize = 5000
steps_per_epoch = 100   # <<< add this

model.fit(
    csv_generator(DATA_PATH),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)],
    verbose=2
)

# ── 6) Notify and progress‐bar the evaluation ─────────────────────────────────────
print("\n=== Starting evaluation over the entire dataset… ===")
y_true = []
y_prob = []

usecols     = feature_cols + [LABEL_COL]
chunksize   = 5000
# count total batches so tqdm can display progress
total_batches = sum(1 for _ in pd.read_csv(DATA_PATH, usecols=usecols, chunksize=chunksize))

for chunk in tqdm(
    pd.read_csv(DATA_PATH, usecols=usecols, chunksize=chunksize),
    total=total_batches,
    desc="Evaluating"
):
    X_batch = scaler.transform(chunk[feature_cols])\
                     .reshape(-1,1,len(feature_cols))
    y_batch = to_categorical(
        chunk[LABEL_COL].map(LABEL_MAP).values,
        num_classes=len(LABEL_MAP)
    )
    preds = model.predict(X_batch, verbose=0)
    y_prob.extend(preds)
    y_true.extend(np.argmax(y_batch, axis=1))

y_true = np.array(y_true)
y_prob = np.array(y_prob)
y_pred = np.argmax(y_prob, axis=1)

# ── 7) Compute metrics (unchanged) ────────────────────────────────────────────────
acc    = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=list(LABEL_MAP.keys()))
cm     = confusion_matrix(y_true, y_pred)
try:
    auc = roc_auc_score(pd.get_dummies(y_true), y_prob, multi_class='ovr')
except ValueError:
    auc = None

# ── 8) Print detailed report ──────────────────────────────────────────────────────
print("\n======== Model Evaluation Report ========")
print(f"Accuracy       : {acc:.4f}")
if auc is not None:
    print(f"ROC AUC (ovr)  : {auc:.4f}")
print("\nClassification Report:\n", report)
print("Confusion Matrix:\n", cm)

# ── 9) Save the model ──────────────────────────────────────────────────────────────
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved to: {MODEL_SAVE_PATH}")