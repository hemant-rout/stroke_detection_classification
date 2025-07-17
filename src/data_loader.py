import pandas as pd
class DataLoader:

    def __init__(self, base_path="../data/raw"):
        self.base_path = base_path

    def load(self):
        train = pd.read_csv(f'{self.base_path}/train.csv')
        test = pd.read_csv(f'{self.base_path}/test.csv')
        return train, test