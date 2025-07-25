import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class ModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def balance(self, X, y):
        smote = SMOTE(random_state=self.random_state,sampling_strategy='minority')
        return smote.fit_resample(X, y)

    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model