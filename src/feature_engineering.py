class FeatureSelector:
    def __init__(self, exclude=None):
        self.exclude = exclude if exclude else ['id', 'stroke']

    def select(self, df):
        return [col for col in df.columns if col not in self.exclude]