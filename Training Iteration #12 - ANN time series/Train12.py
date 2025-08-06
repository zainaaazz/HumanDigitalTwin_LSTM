#cd C:\Users\S_CSIS-PostGrad\Desktop\HumanDigitalTwin_LSTM
#Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#. .\venv\Scripts\Activate.ps1
#python ".\Training Iteration #12 - ANN time series\Train12.py"
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description='Train ANN-LSTM models day-by-day as in Al-Azazi & Ghurab')
    parser.add_argument('-d', '--data_path', type=str, default='data.csv',
                        help='Path to preprocessed CSV')
    parser.add_argument('-o', '--output_dir', type=str, default='models_daywise',
                        help='Directory to save daily models and results')
    parser.add_argument('--id_col', type=str, default='id_student',
                        help='Name of the student ID column')
    parser.add_argument('--time_col', type=str, default='date',
                        help='Name of the time-step column')
    parser.add_argument('--label_col', type=str, default='final_result',
                        help='Name of the final label column')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    OUTPUT_DIR = args.output_dir
    ID_COL = args.id_col
    TIME_COL = args.time_col
    LABEL_COL = args.label_col
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Resolve data file
    if os.path.isfile(DATA_PATH):
        data_file = DATA_PATH
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt = os.path.join(script_dir, DATA_PATH)
        if os.path.isfile(alt):
            data_file = alt
        else:
            raise FileNotFoundError(f"Data file not found at '{DATA_PATH}' or '{alt}'")
    print(f"Using data file: {data_file}")

    # Load data
    df = pd.read_csv(data_file)
    print(f"Loaded data: shape={df.shape}")

    # Check required columns
    missing = [c for c in [ID_COL, TIME_COL, LABEL_COL] if c not in df.columns]
    if missing:
        print(f"Error: Missing required column(s): {missing}")
        print("Available columns:", df.columns.tolist())
        return
    print(f"Using columns -> ID: '{ID_COL}', Time: '{TIME_COL}', Label: '{LABEL_COL}'")

    # Identify feature columns
    feature_cols = [c for c in df.columns if c not in [ID_COL, TIME_COL, LABEL_COL]]
    print(f"Number of feature columns: {len(feature_cols)}")

    # Ensure all features are numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Encode labels
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df[LABEL_COL])
    num_classes = len(le.classes_)

    # Determine time range
    time_min = df[TIME_COL].min()
    time_max = df[TIME_COL].max()
    total_days = int(time_max - time_min + 1)
    print(f"Time steps from {time_min} to {time_max} -> {total_days} days")

    results = []
    # Loop day by day
    for k in range(1, total_days + 1):
        day_threshold = time_min + k - 1
        print(f"\n--- Training day {k}/{total_days} (threshold={day_threshold}) ---")
        sub = df[df[TIME_COL] <= day_threshold]

        # Prepare arrays
        students = df[ID_COL].unique()
        n_students = len(students)
        n_feats = len(feature_cols)
        X = np.zeros((n_students, k, n_feats), dtype=np.float32)
        y = np.zeros(n_students, dtype=int)
        id2idx = {sid: idx for idx, sid in enumerate(students)}

        # Fill sequences
        for _, row in sub.iterrows():
            idx = id2idx[row[ID_COL]]
            t = int(row[TIME_COL] - time_min)
            if 0 <= t < k:
                X[idx, t, :] = row[feature_cols].values.astype(np.float32)
        # Assign labels
        for sid, idx in id2idx.items():
            y[idx] = df.loc[df[ID_COL] == sid, 'label_enc'].iloc[0]

        # Stratified splits
        X_trval, X_test, y_trval, y_test = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trval, y_trval, test_size=0.20, stratify=y_trval, random_state=42)

        # Scale features
        scaler = MinMaxScaler()
        flat_train = X_train.reshape(-1, n_feats)
        scaler.fit(flat_train)
        X_train = scaler.transform(flat_train).reshape(X_train.shape).astype(np.float32)
        X_val = scaler.transform(X_val.reshape(-1, n_feats)).reshape(X_val.shape).astype(np.float32)
        X_test = scaler.transform(X_test.reshape(-1, n_feats)).reshape(X_test.shape).astype(np.float32)

        # One-hot encode targets
        y_train_cat = to_categorical(y_train, num_classes).astype(np.float32)
        y_val_cat = to_categorical(y_val, num_classes).astype(np.float32)
        y_test_cat = to_categorical(y_test, num_classes).astype(np.float32)

        # Build model
        model = Sequential()
        model.add(LSTM(200, input_shape=(k, n_feats)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

        # Callbacks
        es = EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)
        ckpt_path = os.path.join(OUTPUT_DIR, f'model_day{k:03d}.h5')
        cp = ModelCheckpoint(ckpt_path, monitor='val_categorical_accuracy', save_best_only=True)

        # Train
        model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=100, batch_size=100,
            callbacks=[es, cp], verbose=1
        )

        # Evaluate
        loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"Day {k} test_accuracy: {acc:.4f}")
        results.append({'day': k, 'test_accuracy': float(acc)})

    # Save results
    out_csv = os.path.join(OUTPUT_DIR, 'daily_results.csv')
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\nDone. Results saved to {out_csv}")

if __name__ == '__main__':
    main()
