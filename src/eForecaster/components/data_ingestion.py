import os
import urllib.request as request
import zipfile
from eForecaster import logger
from eForecaster.utils.common import get_size
from eForecaster.entity.config_entity import DataIngestionConfig
from pathlib import Path
from datetime import date, timedelta
import pandas as pd

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")  

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
    
    def split_dataset(self):
        path_to_csv = os.path.join(self.config.unzip_dir, self.config.csv_name)
        df = pd.read_csv(path_to_csv, sep=';', decimal=',')
        before_symbol = df['datetime'].str.split('+').str[0]
        df["datetime"] = pd.to_datetime(before_symbol, format="%Y.%m.%d %H:%M:%S ")
        
        df.set_index('datetime', inplace=True)

        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['dayofyear'] = df.index.dayofyear
        df['minute'] = df.index.minute
        d = timedelta(days=365)
        start_date = df.index[-1] - d #One year before the last date in the dataset
        start_date = start_date.strftime("%Y-%m-%d")
        train=df[(df.index<start_date)] 
        test=df[(df.index>=start_date)]

        unzip_path = self.config.unzip_dir
        train_csv_name = Path("train.csv")
        test_csv_name = Path("test.csv")
        train.to_csv(path_or_buf=os.path.join(unzip_path,train_csv_name))
        test.to_csv(path_or_buf=os.path.join(unzip_path,test_csv_name))