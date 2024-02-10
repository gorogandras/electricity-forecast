import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/training/model.joblib'))

    def create_datetime_df(self, moment_to_pred):
        df = pd.DataFrame(moment_to_pred, columns=['datetime'])
        df['datetime'] = pd.to_datetime(df['datetime'], format="%Y-%m-%d %H:%M:%S")
        datetime_df = pd.DataFrame()

        datetime_df['hour'] = df['datetime'].dt.hour
        datetime_df['dayofweek'] = df['datetime'].dt.dayofweek
        datetime_df['quarter']  = df['datetime'].dt.quarter
        datetime_df['month']  = df['datetime'].dt.month
        datetime_df['year']  = df['datetime'].dt.year
        datetime_df['dayofyear']  = df['datetime'].dt.dayofyear
        datetime_df['minute']  = df['datetime'].dt.minute
        return datetime_df

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction 