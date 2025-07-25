from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE
class DataPreprocessor:
    # def __init__(self):
    #     self.encoder = LabelEncoder()

    # def fill_missing(self, df):
    #     df['bmi'].fillna(df['bmi'].median(), inplace=True)
    #     df['smoking_status'].fillna('Unknown', inplace=True)
    #     return df

    # def encode_categorical(self, df, cat_features):
    #     for col in cat_features:
    #         df[col] = self.encoder.fit_transform(df[col])
    #     return df

    def split_data(X, y, test_size=0.2,stratify=None, random_state=RANDOM_STATE):
        """
        Split the data into training and validation sets.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
