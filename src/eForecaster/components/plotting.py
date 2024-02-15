import os
import urllib.request as request
from zipfile import ZipFile
import time
import pandas as pd
import joblib
from src.eForecaster.entity.config_entity import PlotConfig
import xgboost as xgb
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime

class Plotting:
    def __init__(self, config: PlotConfig):
        self.config = config


    def create_datetime_df(self, start_date: str, end_date: str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
        end_date = datetime.strptime(end_date,"%Y-%m-%d %H:%M:%S")
        datetime_df = pd.date_range(start_date, end_date, freq='15min')
        datetime_df = datetime_df.to_frame(index=False, name="datetime")
        return datetime_df




    def get_prediction(self, df):
        model = joblib.load(self.config.model_path)
        df_to_predict = df[[item for item in self.config.columns.keys()]]
        df_predicted = model.predict(df_to_predict)
        return df_predicted

    def get_scatterplot(self, model, X_test, y_test):
        start_date = "2023-04-15 00:00:00"
        end_date = "2023-04-15 00:00:00"
        #df_predicted = model.predict(df_to_predict)
