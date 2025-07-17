from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:
    def __init__(self):
        self.encoder = LabelEncoder()

    def fill_missing(self, df):
        df['bmi'].fillna(df['bmi'].median(), inplace=True)
        df['smoking_status'].fillna('Unknown', inplace=True)
        return df

    def encode_categorical(self, df, cat_features):
        for col in cat_features:
            df[col] = self.encoder.fit_transform(df[col])
        return df