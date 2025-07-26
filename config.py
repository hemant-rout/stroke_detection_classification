# Configuration file for paths and constants

DATA_RAW_PATH = "../data/raw"
DATA_PROCESSED_PATH = "../data/processed"
DATA_PREDICTED_PATH = "../reports"
MODEL_DIR = "../models"
SAMPLE_SUBMISSION_FILE = "sample_solution.csv"
TRAIN_FILE = "train.csv"
TRAIN_PREPROCESSED_FILE = "train_preprocessed.csv"
TEST_PREPROCESSED_FILE = "test_preprocessed.csv"
TEST_FILE = "test.csv"
RANDOM_STATE = 42
FEATURES = [
    'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
    'avg_glucose_level', 'bmi',
    'gender_Male', 'gender_Other',
    'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
    'work_type_children',
    'smoking_status_formerly smoked', 'smoking_status_never smoked',
    'smoking_status_smokes'
]
SCALING_FILE_NAME = "scaler"
MODEL_MAPPING = {
    "XGBoost": "xgb_model.pkl",
    "Random Forest": "rf_model.pkl",
    "LightGBM": "lgbm_model.pkl",
    "Stacking Model": "stacking_model.pkl",
    "Decision Tree": "decision_tree_model.pkl"
}

MODEL_NAMES=["XGBoost", "Random Forest", "LightGBM", "Stacking Model", "Decision Tree"]