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
        smote = SMOTE(random_state=self.random_state)
        return smote.fit_resample(X, y)

    def get_model(self, model_type='lightgbm', **kwargs):
        if model_type == 'lightgbm':
            return lgb.LGBMClassifier(random_state=self.random_state, **kwargs)
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=self.random_state, **kwargs)
        elif model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=self.random_state, **kwargs)
        else:
            raise ValueError("Unsupported model_type")

    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    def stacking(self, estimators, final_estimator=None, cv=5):
        if final_estimator is None:
            final_estimator = lgb.LGBMClassifier()
        return StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=cv)