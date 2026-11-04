import os

class Config:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    RAW_DATA = os.path.join(PROJECT_ROOT, "telemetry.csv")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
    MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
    REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
    
    for path in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, 
                 os.path.join(MODELS_DIR, "base"), 
                 os.path.join(MODELS_DIR, "ensemble")]:
        os.makedirs(path, exist_ok=True)