import pandas as pd
import joblib
import os
from config import MODEL_DIR

class FileHandler:

    @staticmethod
    def read_data(base_path, file_name):
        """Reads a CSV file from the specified base path and returns a DataFrame."""
        try:
            df = pd.read_csv(f'{base_path}/{file_name}')
            return df
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            return pd.DataFrame()
        
    @staticmethod    
    def save_data(df, base_path, file_name):
        """Writes a DataFrame to a CSV file in the specified base path."""
        try:
            df.to_csv(f'{base_path}/{file_name}', index=False)
            print(f"Data written to {file_name} successfully.")
        except Exception as e:
            print(f"Error writing {file_name}: {e}")

    @staticmethod
    def save_pickle_file(model, model_name,model_dir=MODEL_DIR):
        try:
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model, f'{model_dir}/{model_name}.pkl')
            print(f"Pickle File saved as {model_name}.pkl")
        except Exception as e:
            print(f"Error saving model {model_name}: {e}")
    @staticmethod
    def load_pickle_file(model_name,model_dir=MODEL_DIR):
        try:
            model = joblib.load(f'{model_dir}/{model_name}.pkl')
            print(f"Model {model_name} loaded successfully.")
            return model
        except FileNotFoundError:
            print(f"Model {model_name} not found in {model_dir}.")
            return None