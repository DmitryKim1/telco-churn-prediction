import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
RAW_DATA_PATH = os.path.join(BASE_DIR, 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Model paths
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.pkl')

# Reports
REPORT_DIR = os.path.join(BASE_DIR, 'reports')
REPORT_PATH = os.path.join(REPORT_DIR, 'metrics.txt')
FEATURE_IMPORTANCE_PATH = os.path.join(REPORT_DIR, 'feature_importance.png')
CONFUSION_MATRIX_PATH = os.path.join(REPORT_DIR, 'confusion_matrix.png')

# Create directories
for path in [MODEL_DIR, REPORT_DIR]:
    os.makedirs(path, exist_ok=True)