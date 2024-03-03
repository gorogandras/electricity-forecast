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
        test_data = pd.read_csv(self.config.test_data_path)
        train_data = pd.read_csv(self.config.train_data_path)
        test_data["datetime"] = pd.to_datetime(test_data["datetime"], format="%Y-%m-%d %H:%M:%S")
        train_data["datetime"] = pd.to_datetime(train_data["datetime"], format="%Y-%m-%d %H:%M:%S")
        train_test = pd.concat([test_data, train_data])
        joined_df = df.merge(train_test, how='left', on='datetime')
        joined_df['set'] = joined_df['set'].fillna('N/A')
        colors = ["#3287AA", "#46A0D2"]
        custom_palette = sns.set_palette(sns.color_palette(colors))
        ax = sns.lineplot(data=joined_df, x='datetime', y='power', marker='o', markersize=4,
                          hue="set",
                          palette=custom_palette)
        ax = sns.lineplot(data=joined_df, x='datetime', y='prediction', 
                          label='prediction', color="#7dce74", alpha=0.8, marker='o', markersize=4)
        plt.title('Forecast')
        plt.xticks(rotation=45)
        ax.set_xlabel('Date')
        ax.set_ylabel('MW')
        plt.savefig(Path(self.config.lineplot_path), bbox_inches='tight')
        plt.show()
        plt.clf() 