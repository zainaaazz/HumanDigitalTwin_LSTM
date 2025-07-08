import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, LSTM, Dropout, TimeDistributed, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

# ── Reproducibility & Directories ──────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR   = os.path.join(BASE_DIR, 'logs')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ── Load & Preprocess Data ──────────────────────────────────────────────────────
info = pd.read_csv(os.path.join(DATA_DIR, 'studentInfo.csv'))
vle  = pd.read_csv(os.path.join(DATA_DIR, 'studentVle.csv'))
meta = pd.read_csv(os.path.join(DATA_DIR, 'vle.csv'))

# Filter and create label
info = info[(info.code_module == 'BBB') & (info.final_result != 'Withdrawn')].copy()
info['label'] = info.final_result.map(lambda x: 1 if x in ['Pass','Distinction'] else 0)

# Merge and compute time features
data = vle.merge(meta[['id_site']], on='id_site', how='left')
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values(['id_student','date'])
data['days_since'] = (data['date'] - data.groupby('id_student')['date'].transform('min')).dt.days

# Define weekly/monthly bins
data['week']  = data['days_since'] // 7
data['month'] = data['days_since'] // 30

# Aggregate click counts
weekly_counts  = data.groupby(['id_student','week'])['id_site'].count().unstack(fill_value=0)
monthly_counts = data.groupby(['id_student','month'])['id_site'].count().unstack(fill_value=0)

# Merge with labels
weekly_df  = info[['id_student','label']].merge(weekly_counts,  on='id_student', how='left').fillna(0)
monthly_df = info[['id_student','label']].merge(monthly_counts, on='id_student', how='left').fillna(0)

# Prepare arrays and cast to float32
Xw = weekly_df.drop(['id_student','label'], axis=1).values[...,None].astype('float32')
Xm = monthly_df.drop(['id_student','label'], axis=1).values[...,None].astype('float32')
y  = weekly_df['label'].values.astype('float32')

t_w, t_m = Xw.shape[1], Xm.shape[1]
# Create sequence labels
yw = np.repeat(y[:,None], t_w, axis=1)[...,None]
ym = np.repeat(y[:,None], t_m, axis=1)[...,None]

# Train/test split
Xw_tr, Xw_te, yw_tr, yw_te = train_test_split(
    Xw, yw, test_size=0.1, random_state=SEED, stratify=y)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split(
    Xm, ym, test_size=0.1, random_state=SEED, stratify=y)

# ── Model Definitions ───────────────────────────────────────────────────────────
def cnn_lstm_model(T, F):
    inp = Input(shape=(T,F))
    x = Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(inp)
    x = BatchNormalization()(x)
    if T >= 2:
        x = MaxPooling1D(2)(x)
    x = Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(1e-3))(x)
    x = BatchNormalization()(x)
    if T >= 4:
        x = MaxPooling1D(2)(x)
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.4)(x)
    x = LSTM(32, return_sequences=True, kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.4)(x)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def lstm_only_model(T, F):
    inp = Input(shape=(T,F))
    x = LSTM(64, return_sequences=True, kernel_regularizer=l2(1e-3))(inp)
    x = Dropout(0.4)(x)
    x = LSTM(32, return_sequences=True, kernel_regularizer=l2(1e-3))(x)
    x = Dropout(0.4)(x)
    out = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ── Callbacks ──────────────────────────────────────────────────────────────────
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
csv_w = CSVLogger(os.path.join(LOG_DIR, 'Train8_week_history.csv'))
csv_m = CSVLogger(os.path.join(LOG_DIR, 'Train8_month_history.csv'))
ckpt_w = ModelCheckpoint(os.path.join(MODEL_DIR, 'Train8_Model_week.keras'), save_best_only=True)
ckpt_m = ModelCheckpoint(os.path.join(MODEL_DIR, 'Train8_Model_month.keras'), save_best_only=True)

# ── Training ───────────────────────────────────────────────────────────────────
print("Weekly input dtype:", Xw_tr.dtype, "labels dtype:", yw_tr.dtype)
print("Monthly input dtype:", Xm_tr.dtype, "labels dtype:", ym_tr.dtype)

# Manual validation split (10% of training)
y_w_labels = yw_tr[:,0,0]
Xw_train2, Xw_val, yw_train2, yw_val = train_test_split(
    Xw_tr, yw_tr, test_size=0.1, random_state=SEED, stratify=y_w_labels)
y_m_labels = ym_tr[:,0,0]
Xm_train2, Xm_val, ym_train2, ym_val = train_test_split(
    Xm_tr, ym_tr, test_size=0.1, random_state=SEED, stratify=y_m_labels)

# Weekly model training
model_w = cnn_lstm_model(t_w, 1) if t_w >= 4 else lstm_only_model(t_w, 1)
print("\n=== Weekly Training ===")
hist_w = model_w.fit(
    Xw_train2, yw_train2,
    validation_data=(Xw_val, yw_val),
    epochs=30,
    batch_size=32,
    callbacks=[es, csv_w, ckpt_w],
    verbose=2
)
print("\nEpoch-by-epoch Results (Weekly):")
for i in range(len(hist_w.history['loss'])):
    print(f"Epoch {i+1:02d} | loss={hist_w.history['loss'][i]:.4f} | val_loss={hist_w.history['val_loss'][i]:.4f} | acc={hist_w.history['accuracy'][i]:.4f} | val_acc={hist_w.history['val_accuracy'][i]:.4f}")
loss_w, acc_w = model_w.evaluate(Xw_te, yw_te, verbose=2)
print(f"Weekly Test >>> loss={loss_w:.4f}, acc={acc_w:.4f}\n")

# Monthly model training
model_m = cnn_lstm_model(t_m, 1) if t_m >= 4 else lstm_only_model(t_m, 1)
print("=== Monthly Training ===")
hist_m = model_m.fit(
    Xm_train2, ym_train2,
    validation_data=(Xm_val, ym_val),
    epochs=30,
    batch_size=32,
    callbacks=[es, csv_m, ckpt_m],
    verbose=2
)
print("\nEpoch-by-epoch Results (Monthly):")
for i in range(len(hist_m.history['loss'])):
    print(f"Epoch {i+1:02d} | loss={hist_m.history['loss'][i]:.4f} | val_loss={hist_m.history['val_loss'][i]:.4f} | acc={hist_m.history['accuracy'][i]:.4f} | val_acc={hist_m.history['val_accuracy'][i]:.4f}")
loss_m, acc_m = model_m.evaluate(Xm_te, ym_te, verbose=2)
print(f"Monthly Test >>> loss={loss_m:.4f}, acc={acc_m:.4f}\n")

# ── Inference Utility ────────────────────────────────────────────────────────────
def get_student_probs(model_path, series):
    model = tf.keras.models.load_model(model_path)
    X = series.values.reshape(1, -1, 1).astype('float32')
    preds = model.predict(X)[0, ..., 0]
    return pd.Series(preds, index=series.index)

print("Models saved as 'Train8_Model_week.keras' & 'Train8_Model_month.keras'. Logs in 'logs/'.")