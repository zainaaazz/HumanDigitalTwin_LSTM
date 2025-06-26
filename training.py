#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_local.py

Train an LSTM model locally using the OULAD dataset with no manual insertions needed.
Ensure the CSV files vle.csv, studentVle.csv, and studentInfo.csv are located in the data directory specified below.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Masking
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import itertools

def load_data(data_dir):
    vle_meta = pd.read_csv(os.path.join(data_dir, "vle.csv"))
    stVLE = pd.read_csv(os.path.join(data_dir, "studentVle.csv"))
    stInfo = pd.read_csv(os.path.join(data_dir, "studentInfo.csv"))
    return vle_meta, stVLE, stInfo

def build_mappings(vle_meta):
    dic_site_activity = dict(zip(vle_meta['id_site'], vle_meta['activity_type']))
    activity_types = sorted(vle_meta['activity_type'].unique())
    dic_activity_idx = {act: i for i, act in enumerate(activity_types)}
    actions_len = len(dic_activity_idx)
    print(f"Mapped {len(dic_site_activity)} sites to {actions_len} activity types.")
    return dic_site_activity, dic_activity_idx, actions_len

def extract_features_for_course(course, vle_meta, stVLE, stInfo):
    from sklearn.preprocessing import LabelEncoder

    # 1) Set up label encoder for the three classes we care about
    labelencoder = LabelEncoder()
    labelencoder.fit(["Distinction", "Fail", "Pass"])

    # 2) Subset to just this module
    subsetVLE = stVLE[stVLE['code_module'] == course].copy()
    subsetInfo = (
        stInfo[stInfo['code_module'] == course]
        .drop_duplicates(subset=['id_student'], keep=False)
        .copy()
    )

    # 3) Re-map Withdrawn → Fail so our encoder knows all labels
    subsetInfo['final_result'] = subsetInfo['final_result'].replace('Withdrawn', 'Fail')

    # 4) Keep only students that appear in both tables
    subsetVLE  = subsetVLE[subsetVLE['id_student'].isin(subsetInfo['id_student'])]
    subsetInfo = subsetInfo[subsetInfo['id_student'].isin(subsetVLE['id_student'])]

    # 5) Split into train/test by presentation code
    train_pres = ['2013B', '2013J']
    test_pres  = ['2014B', '2014J']

    trainVLE = subsetVLE[subsetVLE['code_presentation'].isin(train_pres)].copy()
    testVLE  = subsetVLE[subsetVLE['code_presentation'].isin(test_pres)].copy()
    trainInfo = subsetInfo[subsetInfo['code_presentation'].isin(train_pres)].copy()
    testInfo  = subsetInfo[subsetInfo['code_presentation'].isin(test_pres)].copy()

    # 6) Assign array indices for each student
    n_train = trainInfo['id_student'].nunique()
    trainInfo['arrayIdx'] = np.arange(n_train)
    print(f"Train students: {n_train}")

    # 7) Build activity‐to‐index mapping from the training set
    trainVLE['activity_type'] = trainVLE['id_site']
    activity_list = sorted(trainVLE['activity_type'].unique())
    dic_act_idx = {act: idx for idx, act in enumerate(activity_list)}
    n_actions = len(dic_act_idx)
    print(f"Detected {n_actions} activity types.")

    trainVLE['activity_idx'] = trainVLE['activity_type'].map(dic_act_idx)
    trainVLE = trainVLE.merge(
        trainInfo[['arrayIdx', 'id_student']],
        on='id_student', validate='m:1'
    )

    # 8) Build X_train, y_train arrays
    days = int(trainVLE['date'].max()) + 1
    X_train = np.zeros((n_train, days, n_actions))
    y_train = labelencoder.transform(trainInfo['final_result'])

    for d in range(days):
        day_data = trainVLE[trainVLE['date'] == d]
        mat = (
            day_data
            .groupby(['arrayIdx', 'activity_idx'])['sum_click']
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=range(n_actions), fill_value=0)
            .reindex(index=range(n_train), fill_value=0)
        )
        X_train[:, d, :] = mat.values

    # 9) Repeat for test set
    n_test = testInfo['id_student'].nunique()
    testInfo['arrayIdx'] = np.arange(n_test)
    print(f"Test students: {n_test}")

    testVLE['activity_type'] = testVLE['id_site']
    testVLE['activity_idx'] = testVLE['activity_type'].map(dic_act_idx)
    testVLE = testVLE.merge(
        testInfo[['arrayIdx', 'id_student']],
        on='id_student', validate='m:1'
    )

    days_test = int(testVLE['date'].max()) + 1
    X_test = np.zeros((n_test, days_test, n_actions))
    y_test = labelencoder.transform(testInfo['final_result'])

    for d in range(days_test):
        day_data = testVLE[testVLE['date'] == d]
        mat = (
            day_data
            .groupby(['arrayIdx', 'activity_idx'])['sum_click']
            .sum()
            .unstack(fill_value=0)
            .reindex(columns=range(n_actions), fill_value=0)
            .reindex(index=range(n_test), fill_value=0)
        )
        X_test[:, d, :] = mat.values

    return X_train, y_train, X_test, y_test, n_actions


def aggregate_weeks(X, total_weeks=37):
    weeks = []
    for i in range(total_weeks):
        weeks.append(np.sum(X[:,7*i:7*(i+1),:], axis=1))
    return np.swapaxes(np.array(weeks), 0, 1)

def scale_data(X_train, X_test):
    ns, nt, nf = X_train.shape
    scaler = MinMaxScaler(feature_range=(0,1))
    flat = np.log1p(X_train.reshape((-1,nf)))
    scaler.fit(flat)
    train_scaled = scaler.transform(flat).reshape((ns,nt,nf))
    nt2, = X_test.shape[:1]
    flat_t = np.log1p(X_test.reshape((-1,nf)))
    test_scaled = scaler.transform(flat_t).reshape(X_test.shape)
    return train_scaled, test_scaled

def build_model(n_steps, n_features):
    model = Sequential([
        Masking(mask_value=0., input_shape=(n_steps,n_features)),
        LSTM(300, dropout=0.1, recurrent_dropout=0.1),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        metrics=[tf.keras.metrics.AUC(name='accuracy'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.Precision(name='precision')]
    )
    model.summary()
    return model

def plot_history(history):
    # only plot metrics that actually exist in history.history
    metrics = [m for m in history.history.keys() if not m.startswith("val_")]
    n = len(metrics)

    plt.figure(figsize=(12, 5))
    for i, m in enumerate(metrics):
        plt.subplot(1, n, i + 1)
        plt.plot(history.history[m], label="train")
        val_m = f"val_{m}"
        if val_m in history.history:
            plt.plot(history.history[val_m], "--", label="val")
        plt.title(m)
        plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion(cm, classes, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title); plt.colorbar()
    ticks = range(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i,cm[i,j],ha='center',
                 color='white' if cm[i,j]>thresh else 'black')
    plt.ylabel('True'); plt.xlabel('Pred'); plt.tight_layout()

def main():
    # Update this path if your data directory is different
    data_dir = r"C:\Users\USER-PC\Desktop\4. Implementation\data"

    vle_meta, stVLE, stInfo = load_data(data_dir)
    build_mappings(vle_meta)

    X_train, y_train, X_test, y_test, n_actions = extract_features_for_course('FFF', vle_meta, stVLE, stInfo)
    # Merge Distinction (2) into Pass (0)
    y_train = np.where(y_train==2, 0, y_train)
    y_test  = np.where(y_test==2, 0, y_test)

    weeks_train = aggregate_weeks(X_train)
    weeks_test  = aggregate_weeks(X_test)
    used_weeks = 10
    weeks_train = weeks_train[:,:used_weeks,:]
    weeks_test  = weeks_test[:,:used_weeks,:]

    print("Before scaling:", weeks_train.shape)
    weeks_train, weeks_test = scale_data(weeks_train, weeks_test)
    print("After scaling:", weeks_train.shape, weeks_test.shape)

    # Class weights
    neg = (y_train==0).sum(); pos = (y_train==1).sum()
    total = neg+pos
    cw = {0:(1/neg)*(total/2), 1:(1/pos)*(total/2)}

    model = build_model(weeks_train.shape[1], weeks_train.shape[2])
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        weeks_train, y_train, epochs=50, batch_size=100,
        validation_data=(weeks_test,y_test),
        class_weight=cw, callbacks=[es], verbose=2, shuffle=True
    )

    model.evaluate(weeks_test, y_test, verbose=2)
    plot_history(history)

    cm_train = confusion_matrix(y_train, (model.predict(weeks_train)>0.5).astype(int))
    plot_confusion(cm_train, ["safe","risky"], "Train Confusion")
    cm_test = confusion_matrix(y_test, (model.predict(weeks_test)>0.5).astype(int))
    plot_confusion(cm_test, ["safe","risky"], "Test Confusion")

if __name__ == "__main__":
    main()
