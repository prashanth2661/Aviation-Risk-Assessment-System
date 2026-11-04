import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config

def train():
    print("Training Models")
    X_train = np.load(os.path.join(Config.PROCESSED_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(Config.PROCESSED_DIR, "y_train.npy"))

    models = {
        'rf': RandomForestClassifier(n_estimators=100),
        'xgb': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'gb': GradientBoostingClassifier(),
        'svm': SVC(probability=True)
    }

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(Config.MODELS_DIR, "base", f"{name}.pkl"))

    print("Building Stacking Ensemble...")
    stack = StackingClassifier(
        estimators=[(n, models[n]) for n in models],
        final_estimator=LogisticRegression()
    )
    stack.fit(X_train, y_train)
    joblib.dump(stack, os.path.join(Config.MODELS_DIR, "ensemble", "stacking_model.pkl"))

    print("Training LSTM...")
    X_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    lstm = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(1, X_train.shape[1])),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    lstm.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    lstm.fit(X_lstm, y_train, epochs=5, batch_size=64, verbose=0)
    lstm.save(os.path.join(Config.MODELS_DIR, "base", "lstm_model.h5"))
    print("Training completed. Models saved...")

if __name__ == "__main__":
    train()