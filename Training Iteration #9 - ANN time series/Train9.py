import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import random

# ── 0) Configuration ─────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Users\USER-PC\Desktop\HumanDigitalTwin_LSTM\Training Iteration #9 - ANN time series\data.csv"
label_col = 'final_result'   # adjust if your label column is named differently
label_map = {'Withdrawn':0, 'Fail':1, 'Pass':2, 'Distinction':3}

# ── 1) Auto-detect feature columns ────────────────────────────────────────────────
#    Read only the header
cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()

#    Exclude ID, date, and label columns
exclude = {'Unnamed: 0', 'id_student', 'date', label_col}
feature_cols = [c for c in cols if c not in exclude]

print(f"Detected {len(feature_cols)} feature columns:", feature_cols[:10], "…")

# ── 2) Pre-fit MinMaxScaler on a small sample ─────────────────────────────────────
sample = pd.read_csv(DATA_PATH, usecols=feature_cols, nrows=5000)
scaler = MinMaxScaler().fit(sample)

# ── 3) Build ANN-LSTM model ───────────────────────────────────────────────────────
model = Sequential([
    LSTM(200, input_shape=(1, len(feature_cols))),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(1e-3),
    metrics=['accuracy']
)

# ── 4) CSV-streaming generator ─────────────────────────────────────────────────────
def csv_generator(path, chunksize=5000):
    for chunk in pd.read_csv(
        path,
        usecols=feature_cols + [label_col],
        chunksize=chunksize
    ):
        X = scaler.transform(chunk[feature_cols]).reshape(-1,1,len(feature_cols))
        y = to_categorical(
            chunk[label_col].map(label_map).values,
            num_classes=4
        )
        yield X, y

# ── 5) Train by streaming ─────────────────────────────────────────────────────────
steps_per_epoch = 100  # ≈ total_rows / chunksize
model.fit(
    csv_generator(DATA_PATH),
    steps_per_epoch=steps_per_epoch,
    epochs=50,
    callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3)],
    verbose=2
)

# ── 6) Predict demo for one student & one day ──────────────────────────────────────
# Streaming a single row:
row = pd.read_csv(
    DATA_PATH,
    usecols=['id_student', 'date'] + feature_cols,
    nrows=1
)
sid, day = int(row['id_student']), int(row['date'])
Xrow = scaler.transform(row[feature_cols].values).reshape(1,1,len(feature_cols))
probs = model.predict(Xrow, verbose=0)[0]
print(f"\nStudent {sid} at day {day} →",
      dict(zip(label_map.keys(), probs)))

# grab the unique (student_id, time_step) pairs in your test set
test_pairs = df_test[['student_id','time_step']].drop_duplicates().values.tolist()

# choose 3 students at random
sampled_students = random.sample(
    list({sid for sid, _ in test_pairs}), 
    k=3
)

# final day in the test split
final_day = df_test.time_step.max()

for sid in sampled_students:
    print(f"\n--- Student {sid} ---")
    for day in (30, 90, final_day):
        # find the row for this student/day
        row = df_test[
            (df_test.student_id==sid) & (df_test.time_step==day)
        ]
        if row.empty:
            print(f" Day {day}: no data")
            continue

        # extract & scale features
        X_raw = row[feature_cols].values
        X_scaled = scaler.transform(X_raw).reshape(1,1,len(feature_cols))

        # predict
        probs = model.predict(X_scaled, verbose=0)[0]
        print(f" Day {day:>3} →", 
              dict(zip(label_map.keys(), probs)))
