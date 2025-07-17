import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import time
from config import MODEL_DIR,MODELS_DICT,SCALING_FILE_NAME
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stroke_app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=False)
def load_models_and_scaler():
    models = {}
    for model_name, model_file in MODELS_DICT.items():
        model_path = f"models/{model_file}"
        model = joblib.load(model_path)
        models[model_name] = model
    scaler = joblib.load(f"models/{SCALING_FILE_NAME}.pkl")
    return models, scaler

models, scaler = load_models_and_scaler()

def preprocess_input(input_df):
    input_df['bmi'] = input_df['bmi'].fillna(input_df['bmi'].median())
    input_df['smoking_status'] = input_df['smoking_status'].fillna('Unknown')

    binary_map = {'Yes': 1, 'No': 0, 'Urban': 1, 'Rural': 0}
    input_df['ever_married'] = input_df['ever_married'].map(binary_map)
    input_df['Residence_type'] = input_df['Residence_type'].map(binary_map)

    input_df = pd.get_dummies(input_df, columns=['gender', 'work_type', 'smoking_status'], drop_first=False)
    features = ['age',
            'hypertension',
            'heart_disease',
            'ever_married',
            'Residence_type',
            'avg_glucose_level',
            'bmi',
            'gender_Male',
            'gender_Other',
            'work_type_Never_worked',
            'work_type_Private',
            'work_type_Self-employed',
            'work_type_children',
            'smoking_status_formerly smoked',
            'smoking_status_never smoked',
            'smoking_status_smokes']

    for col in features:
        if col not in input_df.columns:
            input_df[col] = 0
    logger.info(f"Input DataFrame after preprocessing:\n {input_df.to_string()}")
    return input_df[features]

# Streamlit UI
st.title("🩺 Stroke Risk Prediction App")

st.sidebar.header("Patient Details")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
    age = st.sidebar.slider('Age', 0, 100, 30)
    hypertension = st.sidebar.selectbox('Hypertension', (0, 1))
    heart_disease = st.sidebar.selectbox('Heart Disease', (0, 1))
    ever_married = st.sidebar.selectbox('Ever Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'Children', 'Never_worked'))
    residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    avg_glucose_level = st.sidebar.number_input(label='Average Glucose Level')#.slider('Average Glucose Level', 50.0, 300.0, 100.0)
    bmi = st.sidebar.number_input(label='BMI')#.slider('BMI', 10.0, 60.0, 25.0)
    smoking_status = st.sidebar.selectbox('Smoking Status', ('never smoked', 'formerly smoked', 'smokes', 'Unknown'))

    data = {
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    }
    features = pd.DataFrame(data)
    return features

input_df = user_input_features()
processed_input = preprocess_input(input_df)

st.subheader("🧠 Prediction Results")

model_choice = st.selectbox(
    "Select Model for Prediction",
    ["XGBoost", "Random Forest", "Logistic Regression", "LightGBM", "Stacking Model"],
    index=4
)

model_mapping = {
    "XGBoost": 'xgb_model',
    "Random Forest": 'rf_model',
    "Logistic Regression": 'lr_model',
    "LightGBM": 'lgm_model',
    "Stacking Model": 'stacking_model'
}

if st.button("Predict"):
    with st.spinner(f"Analyzing with {model_choice} model..."):
        model_key = model_mapping[model_choice]
        model = models[model_key]
        time.sleep(2)
        prediction = model.predict(processed_input)
        result_label = 'Yes' if prediction[0] == 1 else 'No'
        st.success(f"Predicted Stroke Risk: {'Yes 🟥' if prediction[0] == 1 else 'No 🟩'}")
        logger.info(f"Model: {model_choice}, Input: {input_df.to_dict(orient='records')[0]}, Prediction: {result_label}")
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(processed_input)[0][1] * 100
            st.info(f"Stroke Risk Probability: {probability:.2f}%")
            logger.info(f"Probability: {probability:.2f}%")
