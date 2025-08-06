import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split


def build_generator(df, feature_cols, tmin, total_days, W, scaler, id_col, time_col):
    n_feats = len(feature_cols)
    def gen(ids):
        for sid in ids:
            stud = df[df[id_col] == sid]
            # build full sequence for student
            seq_full = np.zeros((total_days, n_feats), dtype=np.float32)
            for _, row in stud.iterrows():
                t = int(row[time_col]) - tmin
                if 0 <= t < total_days:
                    seq_full[t] = row[feature_cols].values
            # scale
            seq_full = scaler.transform(seq_full)
            label = stud['label_enc'].iloc[0]
            for k in range(W, total_days + 1):
                # mask future
                seq = seq_full.copy()
                seq[k:] = 0
                day = np.array([k / total_days], dtype=np.float32)
                yield (seq, day), np.int32(label)
    return gen


def main():
    parser = argparse.ArgumentParser(description='Train via tf.data generator to avoid RAM blowup')
    parser.add_argument('-d', '--data_path', type=str, default='data.csv')
    parser.add_argument('-o', '--output_model', type=str, default='Train13_model.keras')
    parser.add_argument('-r', '--results_csv', type=str, default='daily_results.csv')
    parser.add_argument('-p', '--report_txt', type=str, default='classification_report.txt')
    parser.add_argument('--id_col', type=str, default='id_student')
    parser.add_argument('--time_col', type=str, default='date')
    parser.add_argument('--label_col', type=str, default='final_result')
    parser.add_argument('--warmup_days', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    # Load
    if os.path.isfile(args.data_path):
        data_file = args.data_path
    else:
        alt = os.path.join(os.path.dirname(__file__), args.data_path)
        if os.path.isfile(alt): data_file = alt
        else: raise FileNotFoundError(f"Data file not found: {args.data_path}")
    df = pd.read_csv(data_file)
    df = df.loc[:, ~df.columns.str.match(r'Unnamed')]
    # Validate
    for col in [args.id_col, args.time_col, args.label_col]:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col}")
    # Features
    feature_cols = [c for c in df.columns if c not in [args.id_col, args.time_col, args.label_col]]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    # Labels
    le = LabelEncoder()
    df['label_enc'] = le.fit_transform(df[args.label_col])
    class_labels = list(le.classes_)
    num_classes = len(class_labels)
    # Time range
    tmin = int(df[args.time_col].min()); tmax = int(df[args.time_col].max())
    total_days = tmax - tmin + 1; W = args.warmup_days
    if W >= total_days: raise ValueError("warmup_days >= total_days")
    # Split students
    student_labels = df.groupby(args.id_col)['label_enc'].first()
    student_ids = student_labels.index.to_numpy()
    labels_arr = student_labels.values
    train_ids, test_ids = train_test_split(
        student_ids, test_size=0.3, stratify=labels_arr, random_state=42)
    train_ids, val_ids = train_test_split(
        train_ids,
        test_size=0.2,
        stratify=student_labels.loc[train_ids].values,
        random_state=42)
    # Scaler
    scaler = MinMaxScaler().fit(df[feature_cols].values)
    # Build datasets
    gen_train = build_generator(df, feature_cols, tmin, total_days, W,
                                scaler, args.id_col, args.time_col)
    gen_val   = build_generator(df, feature_cols, tmin, total_days, W,
                                scaler, args.id_col, args.time_col)
    gen_test  = build_generator(df, feature_cols, tmin, total_days, W,
                                scaler, args.id_col, args.time_col)
    # tf.data.Datasets
    output_sig = (
      (tf.TensorSpec((total_days, len(feature_cols)), tf.float32),
       tf.TensorSpec((1,), tf.float32)),
      tf.TensorSpec((), tf.int32)
    )
    train_ds = tf.data.Dataset.from_generator(lambda: gen_train(train_ids), output_signature=output_sig)
    val_ds   = tf.data.Dataset.from_generator(lambda: gen_val(val_ids), output_signature=output_sig)
    test_ds  = tf.data.Dataset.from_generator(lambda: gen_test(test_ids), output_signature=output_sig)
    train_ds = train_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.batch(args.batch_size)

    # Model
    n_feats = len(feature_cols)
    seq_in = Input(shape=(total_days, n_feats), name='sequence')
    day_in = Input(shape=(1,), name='day_index')
    x = LSTM(200)(seq_in)
    x = Dropout(0.5)(x)
    x = Concatenate()([x, day_in])
    x = Dense(100, activation='relu')(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model([seq_in, day_in], out)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )

    # Callbacks
    es = EarlyStopping(monitor='val_categorical_accuracy', patience=10, restore_best_weights=True)
    mc = ModelCheckpoint(args.output_model, monitor='val_categorical_accuracy', save_best_only=True)

    # Train
    model.fit(train_ds, validation_data=val_ds,
              epochs=100, callbacks=[es, mc], verbose=1)

    # Evaluate and metrics
    print("\nEvaluating on test set...")
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Overall test accuracy: {acc:.4f}\n")

    # Collect preds for reports
    y_true, y_pred, day_norms = [], [], []
    for (seq_b, day_b), y_b in test_ds:
        preds = model.predict((seq_b, day_b), verbose=0)
        y_pred.extend(np.argmax(preds, axis=1).tolist())
        y_true.extend(y_b.numpy().tolist())
        day_norms.extend(day_b.numpy().flatten().tolist())

    # Convert to int days
    days_int = [int(round(n * total_days)) for n in day_norms]

    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_labels)
    cm = confusion_matrix(y_true, y_pred)
    print(report)
    print(cm)
    with open(args.report_txt, 'w') as f:
        f.write("Classification Report\n" + report + "\nConfusion Matrix\n" + np.array2string(cm))

    # Per-day metrics
    rows = []
    for k in sorted(set(days_int)):
        idxs = [i for i, d in enumerate(days_int) if d == k]
        y_t = [y_true[i] for i in idxs]
        y_p = [y_pred[i] for i in idxs]
        acc_k = np.mean([pt==tt for pt,tt in zip(y_p, y_t)])
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_p, average='macro', zero_division=0)
        rows.append({'day': k, 'accuracy': acc_k, 'precision': prec, 'recall': rec, 'f1_score': f1})
    df_days = pd.DataFrame(rows)
    print(df_days.to_string(index=False))
    df_days.to_csv(args.results_csv, index=False)
    print(f"Per-day metrics saved to {args.results_csv}")

if __name__ == '__main__':
    main()
