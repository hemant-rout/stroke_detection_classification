# 🧠 Stroke Classification - Spring 2025 Kaggle Competition

This repository contains my solution for the [Spring 2025 Classification Competition](https://www.kaggle.com/competitions/spring-2025-classification-competition) hosted on Kaggle. The objective is to predict the risk of stroke in patients based on clinical and demographic data using robust machine learning models.

---

## 🏆 Competition Goal

Predict whether a patient is likely to experience a stroke using provided health-related features. The classification task is evaluated based on **F1 Score** and **ROC-AUC**.

---

## 📁 Dataset Overview

- **Train Set**: Labeled clinical and demographic features.
- **Test Set**: Unlabeled data for predictions.
- **Target Variable**: `stroke` (1 = stroke, 0 = no stroke)
- **Features**: 
  - `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`,  
    `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`

---

## 🧪 Evaluation Metric

- **Primary Metric**: F1 Score
- **Secondary Metric**: ROC-AUC Score

---

## 📊 Model Performance (Cross-Validation)

| Model                | F1 Score | ROC-AUC Score |
|----------------------|----------|----------------|
| 🥇 Stacking          | **0.973** | **0.997**       |
| 🥈 XGBoost           | 0.970     | 0.997          |
| 🥉 LightGBM          | 0.968     | 0.996          |
| Random Forest        | 0.967     | 0.996          |
| Decision Tree        | 0.903     | 0.961          |
| K-Nearest Neighbors  | 0.734     | 0.813          |
| SVM                  | 0.664     | 0.665          |
| Logistic Regression  | 0.530     | 0.634          |
| Naive Bayes          | 0.324     | 0.666          |

---

## 📂 Repository Structure

stroke_detection_classification/
│
├── data/
│   ├── raw/          # Original raw datasets
│   └── processed/    # Cleaned & engineered datasets
│
├── notebooks/        # Jupyter notebooks (EDA, modeling)
├── reports/          # Model predictions, evaluation reports
├── src/              # Core Python code for modeling & preprocessing
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/           # Trained models (.pkl)
├── app.py            # Streamlit web application
├── requirements.in   # Dependency list (editable)
├── requirements.txt  # Locked dependencies
└── README.md

---

## 🚀 How to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/hemant-rout/stroke_detection_classification.git
   cd stroke_detection_classification

2. **🔧 Setup Environment**

   2.1 Install pip-tools if not already installed
   pip install pip-tools

   2.2 Compile and install dependencies
   pip-compile requirements.in
   pip install -r requirements.txt

3. **▶️ Run the Streamlit App**
   a. Local Machine:
      streamlit run app.py
         or
   b. Streamlit hosted app
      https://strokedetectionclassification-yt5ojuyg6orf2nsxs85ab8.streamlit.app/
      

🧪 Try It Yourself (Sample Inputs)
Use the following examples in the app's sidebar to see prediction results:

gender	age	hypertension	heart_disease	ever_married	work_type	Residence_type	avg_glucose_level	bmi	smoking_status
Male	68	0	0	Yes	Private	Rural	95.82	27.5	never smoked
Female	42	0	0	Yes	Private	Rural	84.18	26.1	smokes

   🧠 The app will predict whether the patient is at risk of stroke and show the probability score.


📌 Future Improvements
🔧 Advanced Hyperparameter Tuning

🧬 Feature Selection & Dimensionality Reduction
   * 🧠 Model Interpretation with SHAP
   * 🔁 Automated Retraining Pipeline
   * 📈 Live Model Monitoring Dashboard

📚 References
   * Kaggle Competition Page
   * Streamlit Documentation
   * scikit-learn Documentation
   * [XGBoost & LightGBM Docs](https://xgboost.readthedocs.io/, https://lightgbm.readthedocs.io/)

🧑‍💻 Author
Hemant Kumar Rout
🔗 GitHub([text](https://github.com/hemant-rout))


