import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

class Evaluator:
    def evaluate(self, model, X_val, y_val):
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)
        return f1, report, cm

    def plot_confusion_matrix(self, cm, title='Confusion Matrix', cmap='Blues'):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=cmap)
        plt.title(title)
        plt.show()