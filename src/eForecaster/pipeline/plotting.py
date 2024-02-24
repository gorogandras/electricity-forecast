import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from src.eForecaster.config.configuration import ConfigurationManager
from src.eForecaster.components.plotting import Plotting
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
    

    def get_scatterplot(self, df):
        ax = sns.lineplot(data=df, x='datetime', y='prediction', marker='o')
        plt.title('Forecast')
        plt.xticks(rotation=45)
        ax.set_xlabel('Date')
        ax.set_ylabel('MW')
        #g = sns.scatterplot(data=df, x="datetime", y="prediction")
        #g.set_title("placeholder")
        #g.xticks 

        plt.savefig(Path(self.config.scatterplot_path), bbox_inches='tight')
        #plt.show()
        plt.clf() 



    """#def main(self):
        #config = ConfigurationManager()
        plot_config = config.get_plot_config()
        plotting = Plotting(config=plot_config)
        datetime_df = plotting.create_datetime_df("2023-02-13 10:15:00", "2023-02-15 10:15:00")
        prediction_df = plotting.create_prediction_df(datetime_df)
        predicted_df = plotting.get_prediction(prediction_df)
        plotting.get_scatterplot(predicted_df)
        #print(datetime_df)
        #print(predicted_df)"""