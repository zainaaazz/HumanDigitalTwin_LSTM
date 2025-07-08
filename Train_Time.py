import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# --- 1) Data loading (assume your existing loader functions) ---
def load_panels():
    # Replace these with your actual data-loading logic
    # Should return X_week_full, X_month_full, y_week_full, y_month_full
    # Shapes:
    #   X_week_full: (n_students, T_weeks, n_features)
    #   y_week_full: (n_students,)
    #   X_month_full: (n_students, T_months, n_features)
    #   y_month_full: (n_students,)
    panels = np.loadz(np.load)  # placeholder
    return X_week_full, X_month_full, y_week_full, y_month_full

# --- 2) Build variable-length LSTM ---
def build_lstm_variable(n_features,
                        hidden=(32,8), lr=1e-4,
                        l2_reg=1e-3, dropout=0.4):
    inp = Input(shape=(None, n_features), name="time_series_input")
    x = LSTM(hidden[0], return_sequences=True,
             kernel_regularizer=l2(l2_reg))(inp)
    x = Dropout(dropout)(x)
    x = LSTM(hidden[1], kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inp, out)
    model.compile(optimizer=Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"] )
    return model

# --- 3) Incremental predictions function ---
def incremental_predictions(model, X_panel):
    n_students, T_full, _ = X_panel.shape
    records = []
    for i in range(n_students):
        for t in range(1, T_full+1):
            x_in = X_panel[i:i+1, :t, :]
            p = float(model.predict(x_in, verbose=0)[0,0])
            pred = int(p >= 0.5)
            records.append({
                'student_idx': i,
                'time_step':   t,
                'prob_pass':   p,
                'pred_label':  pred
            })
    return pd.DataFrame(records)

# --- 4) Main script ---
if __name__ == '__main__':
    # Load your full panels
    Xw_full, Xm_full, yw_full, ym_full = load_panels()

    # Split into train+val vs test
    Xw_trainval, Xw_test, yw_trainval, yw_test = \
        train_test_split(Xw_full, yw_full, test_size=0.1, random_state=42)
    Xm_trainval, Xm_test, ym_trainval, ym_test = \
        train_test_split(Xm_full, ym_full, test_size=0.1, random_state=42)

    # Build & train
    n_feats_w = Xw_trainval.shape[-1]
    model_w = build_lstm_variable(n_features=n_feats_w)
    model_w.fit(Xw_trainval, yw_trainval,
                validation_split=0.1,
                epochs=50, batch_size=128,
                callbacks=[] )

    n_feats_m = Xm_trainval.shape[-1]
    model_m = build_lstm_variable(n_features=n_feats_m)
    model_m.fit(Xm_trainval, ym_trainval,
                validation_split=0.1,
                epochs=50, batch_size=128 )

    # Generate incremental predictions
    df_weekly_preds  = incremental_predictions(model_w, Xw_test)
    df_monthly_preds = incremental_predictions(model_m, Xm_test)

    # Save to CSV
    os.makedirs('output', exist_ok=True)
    df_weekly_preds.to_csv('output/weekly_time_series_predictions.csv', index=False)
    df_monthly_preds.to_csv('output/monthly_time_series_predictions.csv', index=False)
    print("Incremental prediction files written to output/")
