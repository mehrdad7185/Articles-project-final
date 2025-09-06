# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib # Import joblib to save the scaler

# --- Configuration ---
DATASET_PATH = 'dataset.csv'
MODEL_SAVE_PATH = 'digital_twin_model.h5'
SCALER_SAVE_PATH = 'scaler.joblib' # Path to save the scaler object
SEQUENCE_LENGTH = 10
EPOCHS = 50
BATCH_SIZE = 32

def create_sequences(data, labels, sequence_length):
    # This function remains unchanged
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(labels[i + sequence_length - 1])
    return np.array(X), np.array(y)

def build_model(input_shape):
    # This function remains unchanged
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] Dataset file '{DATASET_PATH}' not found. Please run data_generator.py first.")
    else:
        print(f"[INFO] Loading data from '{DATASET_PATH}'...")
        df = pd.read_csv(DATASET_PATH)
        
        features = ['cpu', 'memory', 'latency']
        target = 'is_failure_imminent'
        
        # --- KEY CHANGE: Save the fitted scaler ---
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])
        joblib.dump(scaler, SCALER_SAVE_PATH) # Save the scaler to a file
        print(f"[INFO] Scaler has been saved to '{SCALER_SAVE_PATH}'")
        
        print(f"[INFO] Creating sequences with length {SEQUENCE_LENGTH}...")
        X, y = create_sequences(df[features].values, df[target].values, SEQUENCE_LENGTH)
        
        if len(X) > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"[INFO] Data prepared: {len(X_train)} sequences for training, {len(X_test)} sequences for testing.")

            model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            print("\n[INFO] Starting model training...")
            model.fit(
                X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                validation_split=0.2, callbacks=[early_stopping], verbose=1
            )
            
            print("\n[INFO] Evaluating model on test data...")
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"  -> Final Model Accuracy: {accuracy * 100:.2f}%")
            
            model.save(MODEL_SAVE_PATH)
            print(f"\n[SUCCESS] Model was trained successfully and saved to '{MODEL_SAVE_PATH}'")

