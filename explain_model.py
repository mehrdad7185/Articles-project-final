# explain_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import shap
import os
import joblib
import warnings
import matplotlib.pyplot as plt

# Suppress UserWarnings from Keras/SHAP to keep the output clean
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
DATASET_PATH = 'dataset.csv'
MODEL_PATH = 'digital_twin_model.h5'
SCALER_PATH = 'scaler.joblib'
SEQUENCE_LENGTH = 10  # Must match the training script
OUTPUT_PLOT_PATH = 'shap_summary_plot.png'

def create_sequences(data, labels, sequence_length):
    """
    Helper function to create sequences, identical to the one in train_model.py.
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(labels[i + sequence_length - 1])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    print("[INFO] Loading model, scaler, and dataset...")
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, DATASET_PATH]):
        print("[ERROR] Model, scaler, or dataset file not found.")
    else:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)
        df = pd.read_csv(DATASET_PATH)

        features = ['cpu', 'memory', 'latency']
        target = 'is_failure_imminent'
        
        df_scaled = df.copy()
        df_scaled[features] = scaler.transform(df_scaled[features])

        X, y = create_sequences(df_scaled[features].values, df_scaled[target].values, SEQUENCE_LENGTH)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        if len(X_test) < 50:
            print("[ERROR] Not enough test data to explain. Need at least 50 test samples.")
        else:
            print("[INFO] Explaining model predictions on a subset of test samples...")
            
            def predict_fn(x_2d):
                num_samples = x_2d.shape[0]
                num_features = len(features)
                x_3d = x_2d.reshape(num_samples, SEQUENCE_LENGTH, num_features)
                return model.predict(x_3d, verbose=0)

            background_data_3d = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]
            background_data_2d = background_data_3d.reshape(background_data_3d.shape[0], -1)

            explainer = shap.KernelExplainer(predict_fn, background_data_2d)
            
            samples_to_explain_3d = X_test[np.random.choice(X_test.shape[0], 50, replace=False)]
            samples_to_explain_2d = samples_to_explain_3d.reshape(samples_to_explain_3d.shape[0], -1)
            
            print("[INFO] Calculating SHAP values. This may take a moment...")
            shap_values_2d = explainer.shap_values(samples_to_explain_2d)
            
            print("[INFO] SHAP values calculated successfully.")

            print(f"[INFO] Generating and saving SHAP summary plot to '{OUTPUT_PLOT_PATH}'...")
            
            shap_values_3d = shap_values_2d.reshape(samples_to_explain_3d.shape)
            
            # --- FINAL FIX: Remove np.abs to show positive/negative impacts ---
            # This will create a plot just like the example you sent.
            aggregated_shap_values = shap_values_3d.mean(axis=1)
            
            last_timestep_features = samples_to_explain_3d[:, -1, :]
            
            shap.summary_plot(
                aggregated_shap_values,
                last_timestep_features,
                feature_names=features,
                show=False
            )
            
            plt.savefig(OUTPUT_PLOT_PATH, bbox_inches='tight')
            plt.close()
            
            print(f"\n[SUCCESS] SHAP analysis complete. Plot saved to '{OUTPUT_PLOT_PATH}'")

