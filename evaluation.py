import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import joblib
import pandas as pd
import warnings
import logging
from sklearn.metrics import accuracy_score
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf

def evaluate():
    print("FULL MODEL EVALUATION")
    
    X_test = np.load(os.path.join(Config.PROCESSED_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(Config.PROCESSED_DIR, "y_test.npy"))
    
    results = []

    base_models = ['rf', 'xgb', 'gb', 'svm'] 
    print(f"\nBase Models Performance")
    
    for name in base_models:
        path = os.path.join(Config.MODELS_DIR, "base", f"{name}.pkl")
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            print(f"{name.upper():<5} Accuracy: {acc:.2%}")
            
            results.append({"Model": name.upper(), "Accuracy": acc})
        else:
            print(f"{name.upper():<5} model file not found at {path}")

    print(f"\nEnsemble Layer")
    stack_path = os.path.join(Config.MODELS_DIR, "ensemble", "stacking_model.pkl")
    if os.path.exists(stack_path):
        stack = joblib.load(stack_path)
        y_pred = stack.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"STACKING ENSEMBLE Accuracy: {acc:.2%}")
        results.append({"Model": "STACKING ENSEMBLE", "Accuracy": acc})

    lstm_path = os.path.join(Config.MODELS_DIR, "base", "lstm_model.h5")
    if os.path.exists(lstm_path):
        print(f"Deep Learning Layer")
        lstm = tf.keras.models.load_model(lstm_path)
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        y_pred_prob = lstm.predict(X_test_lstm, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"LSTM Accuracy: {acc:.2%}")
        results.append({"Model": "LSTM (DL)", "Accuracy": acc})

    metrics_df = pd.DataFrame(results)
    save_dir = os.path.join(Config.REPORTS_DIR, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, "model_performance.csv")
    metrics_df.to_csv(save_path, index=False)
    
    print(f"Full performance saved")

if __name__ == "__main__":
    evaluate()