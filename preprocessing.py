import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from src.config import Config

def preprocess():
    print("Preprocessing")
    df = pd.read_csv(Config.RAW_DATA)
    
    df['tcas_alert'] = df['tcas_alert'].fillna('None')
    df['gpws_alert'] = df['gpws_alert'].fillna('None')
    
    X = df.drop(columns=['risk_level'])
    y = df['risk_level']
    
    num_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')), 
                         ('oh', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    joblib.dump(le, os.path.join(Config.MODELS_DIR, "label_encoder.pkl"))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, stratify=y_enc)

    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    joblib.dump(preprocessor, os.path.join(Config.MODELS_DIR, "preprocessor.pkl"))

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    np.save(os.path.join(Config.PROCESSED_DIR, "X_train.npy"), X_train_res)
    np.save(os.path.join(Config.PROCESSED_DIR, "y_train.npy"), y_train_res)
    np.save(os.path.join(Config.PROCESSED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(Config.PROCESSED_DIR, "y_test.npy"), y_test)
    print("Preprocessing completed. Data saved")

if __name__ == "__main__":
    preprocess()