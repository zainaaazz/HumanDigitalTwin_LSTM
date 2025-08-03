#cd C:\Users\S_CSIS-PostGrad\Desktop\HumanDigitalTwin_LSTM
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#. .\venv\Scripts\Activate.ps1
#python ".\Training Iteration #11 - ANN time series\Train11.py"


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# --- Configuration ---
DATA_FILE = "data.csv"  # rename if different
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, DATA_FILE)
MODEL_SAVE_DIR = SCRIPT_DIR
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "Train11Model")

# Hyperparameters
TEST_SPLIT = 0.30
VAL_SPLIT = 0.20
EPOCHS = 100
BATCH_SIZE = 100
RANDOM_SEED = 42
LABEL_COL = 'final_result'
LABEL_MAP = {'Withdrawn':0, 'Fail':1, 'Pass':2, 'Distinction':3}

# --- Load & Prepare Data ---
cols_all = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
exclude = {'Unnamed: 0', 'student_id', 'time_step', LABEL_COL}
feature_cols = [c for c in cols_all if c not in exclude]
print(f"Detected {len(feature_cols)} feature columns: {feature_cols}")

df = pd.read_csv(DATA_PATH, usecols=feature_cols + [LABEL_COL])
X_all = df[feature_cols].values
y_all = df[LABEL_COL].map(LABEL_MAP).values
y_all_cat = to_categorical(y_all, num_classes=len(LABEL_MAP))

# Split into train+val and test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_all, y_all_cat,
    test_size=TEST_SPLIT,
    stratify=y_all,
    random_state=RANDOM_SEED
)
# Further split train+val into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=VAL_SPLIT,
    stratify=np.argmax(y_train_val, axis=1),
    random_state=RANDOM_SEED
)
print(f"Samples -> Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Scale features
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# Reshape for LSTM: [samples, timesteps=1, features]
n_features = len(feature_cols)
X_train = X_train.reshape(-1, 1, n_features)
X_val   = X_val.reshape(-1, 1, n_features)
X_test  = X_test.reshape(-1, 1, n_features)

# --- Build & Compile Model ---
model = Sequential([
    Input(shape=(1, n_features)),
    LSTM(200),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dropout(0.2),
    Dense(len(LABEL_MAP), activation='softmax')
])
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)
model.summary()

# --- Callbacks ---
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(MODEL_SAVE_PATH, 'Train11_best_model.h5'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)
callbacks = [early_stop, checkpoint]

# --- Train ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2
)

# --- Evaluate on Test Set ---
y_pred_prob = model.predict(X_test, batch_size=BATCH_SIZE)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Basic metrics
acc = accuracy_score(y_true, y_pred)
print("\nBasic Evaluation:")
print(f"Test Accuracy     : {acc:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(LABEL_MAP.keys())))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Additional aggregated metrics
prec_macro = precision_score(y_true, y_pred, average='macro')
prec_micro = precision_score(y_true, y_pred, average='micro')
prec_weighted = precision_score(y_true, y_pred, average='weighted')
rec_macro = recall_score(y_true, y_pred, average='macro')
rec_micro = recall_score(y_true, y_pred, average='micro')
rec_weighted = recall_score(y_true, y_pred, average='weighted')
f1_macro = f1_score(y_true, y_pred, average='macro')
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_weighted = f1_score(y_true, y_pred, average='weighted')

print("\nAggregated Metrics:")
print(f"Precision (Macro):    {prec_macro:.4f}")
print(f"Precision (Micro):    {prec_micro:.4f}")
print(f"Precision (Weighted): {prec_weighted:.4f}")
print(f"Recall    (Macro):    {rec_macro:.4f}")
print(f"Recall    (Micro):    {rec_micro:.4f}")
print(f"Recall    (Weighted): {rec_weighted:.4f}")
print(f"F1-score  (Macro):    {f1_macro:.4f}")
print(f"F1-score  (Micro):    {f1_micro:.4f}")
print(f"F1-score  (Weighted): {f1_weighted:.4f}")

# ROC AUC OVR and per-class AUC
try:
    auc_ovr = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')
    print(f"\nOverall ROC AUC (OVR): {auc_ovr:.4f}")
    print("Class-wise ROC AUC:")
    for idx, label in enumerate(LABEL_MAP.keys()):
        auc_i = roc_auc_score(y_test[:, idx], y_pred_prob[:, idx])
        print(f"  {label:<10}: {auc_i:.4f}")
except ValueError:
    print("\nROC AUC could not be computed for one or more classes.")

# --- Save Final Model ---
final_path = os.path.join(MODEL_SAVE_PATH, 'Train11_final_model')
model.save(final_path)
print(f"\nFinal model saved at: {final_path}")
