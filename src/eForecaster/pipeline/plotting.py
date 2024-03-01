import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.eForecaster.config.configuration import ConfigurationManager
from src.eForecaster.entity.config_entity import PlotConfig
from src.eForecaster import logger
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

class PlottingPipeline:
    def __init__(self, config: PlotConfig):
        self.config = config

    def create_datetime_df(self, start_moment: str, end_moment: str):
        print(start_moment)
        start_moment = datetime.strptime(start_moment, "%Y-%m-%d %H:%M:%S")
        end_moment = datetime.strptime(end_moment,"%Y-%m-%d %H:%M:%S")
        datetime_df = pd.date_range(start_moment, end_moment, freq='15min')
        datetime_df = datetime_df.to_frame(index=False, name="datetime")
        return datetime_df

    def create_prediction_df(self, df):
        df['hour'] = df.datetime.dt.hour
        df['dayofweek'] = df.datetime.dt.dayofweek
        df['quarter'] = df.datetime.dt.quarter
        df['month'] = df.datetime.dt.month
        df['year'] = df.datetime.dt.year
        df['dayofyear'] = df.datetime.dt.dayofyear
        df['minute'] = df.datetime.dt.minute
        return df    

    def get_prediction(self, df):
        df_copy = df.copy()
        model = joblib.load(self.config.model_path)
        df_to_predict = df[[item for item in self.config.columns.keys()]]
        prediction = model.predict(df_to_predict)
        df_copy['prediction'] = prediction
        return df_copy
    
    def get_lineplot(self, df):
        ax = sns.lineplot(data=df, x='datetime', y='prediction', marker='o')
        plt.title('Forecast')
        plt.xticks(rotation=45)
        ax.set_xlabel('Date')
        ax.set_ylabel('MW')
        plt.savefig(Path(self.config.scatterplot_path), bbox_inches='tight')
        plt.clf() 
