import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import logging
from sklearn.base import BaseEstimator
from config import MODEL_DIR, MODEL_MAPPING, SCALING_FILE_NAME, MODEL_NAMES

# ------------------------- Logging Setup ------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stroke_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- Model & Scaler Loader --------------------- #
@st.cache_resource(show_spinner=False)
def load_models_and_scaler() -> tuple[dict[str, BaseEstimator], object]:
    """Load pre-trained models and scaler from the specified directory."""
    logger.info("Loading models and scaler...")
    models = {}
    for model_name, model_file in MODEL_MAPPING.items():
        try:
            model = joblib.load(f"models/{model_file}")
            models[model_name] = model
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
    try:
        scaler = joblib.load(f"models/{SCALING_FILE_NAME}.pkl")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
        scaler = None
    return models, scaler

# ------------------------ Data Preprocessing ---------------------- #
def preprocess_input(input_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input DataFrame to match the model's expected format."""
    logger.info(f"Input before preprocessing:\n{input_df.to_string()}")
    
    binary_map = {'Yes': 1, 'No': 0, 'Urban': 1, 'Rural': 0}
    input_df['ever_married'] = input_df['ever_married'].map(binary_map)
    input_df['Residence_type'] = input_df['Residence_type'].map(binary_map)

    input_df = pd.get_dummies(input_df, columns=['gender', 'work_type', 'smoking_status'], drop_first=False)

    required_features = [
        'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
        'avg_glucose_level', 'bmi',
        'gender_Male', 'gender_Other',
        'work_type_Never_worked', 'work_type_Private', 'work_type_Self-employed',
        'work_type_children',
        'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes'
    ]

    # Add missing columns
    for col in required_features:
        if col not in input_df.columns:
            input_df[col] = 0

    logger.info(f"Input after preprocessing:\n{input_df[required_features].to_string()}")
    return input_df[required_features]

# ------------------------ Sidebar Form UI ------------------------ #
def get_user_input() -> pd.DataFrame:
    """Collect patient input from sidebar."""
    st.sidebar.header("📝 Patient Information")

    data = {
        'gender': [st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))],
        'age': [st.sidebar.slider('Age', 0, 100, 50)],
        'hypertension': [st.sidebar.selectbox('Hypertension', (0, 1))],
        'heart_disease': [st.sidebar.selectbox('Heart Disease', (0, 1))],
        'ever_married': [st.sidebar.selectbox('Ever Married', ('Yes', 'No'))],
        'work_type': [st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'))],
        'Residence_type': [st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))],
        'avg_glucose_level': [st.sidebar.number_input('Average Glucose Level', min_value=0.0)],
        'bmi': [st.sidebar.number_input('BMI', min_value=0.0)],
        'smoking_status': [st.sidebar.selectbox('Smoking Status', ('never smoked', 'formerly smoked', 'smokes', 'Unknown'))]
    }
    return pd.DataFrame(data)

# --------------------- Prediction & Display ---------------------- #
def make_prediction(model: BaseEstimator, input_df: pd.DataFrame):
    """Make prediction and display results."""
    prediction = model.predict(input_df)[0]
    result_label = 'Yes' if prediction == 1 else 'No'
    st.success(f"🔍 Predicted Stroke Risk: {'🟥 YES' if prediction == 1 else '🟩 NO'}")
    logger.info(f"Prediction: {result_label}")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1] * 100
        st.info(f"📊 Stroke Risk Probability: {prob:.2f}%")
        logger.info(f"Probability: {prob:.2f}%")

# ------------------------ Main Function -------------------------- #
def main():
    st.title("🧠 Stroke Risk Prediction App")
    
    models, scaler = load_models_and_scaler()
    input_df = get_user_input()
    processed_input = preprocess_input(input_df)

    st.subheader("📈 Select Prediction Model")
    model_choice = st.selectbox("Model", MODEL_NAMES, index=4)

    if st.button("🚀 Predict"):
        with st.spinner(f"Analyzing with {model_choice} model..."):
            try:
                model = models.get(model_choice)
                if model is None:
                    st.error("Model not loaded properly.")
                    return
                time.sleep(1.5)
                make_prediction(model, processed_input)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                logger.exception("Prediction Error")

# ----------------------- Entry Point ----------------------------- #
if __name__ == "__main__":
    main()
