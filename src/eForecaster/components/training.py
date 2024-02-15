import os
import urllib.request as request
from zipfile import ZipFile
import time
import pandas as pd
import joblib
from src.eForecaster.entity.config_entity import TrainingConfig
import xgboost as xgb


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def get_dataset(self):
        self.training_data = self.config.training_data
        self.testing_data = self.config.testing_data
        df_train = pd.read_csv(self.training_data)
        df_test = pd.read_csv(self.testing_data)
        return df_train, df_test

    def get_features_target(self, df_train, df_test):

        
        FEATURES = [item for item in self.config.columns.keys()]
        TARGET = self.config.target_column
        X_train = df_train[FEATURES]
        y_train = df_train[TARGET]
        X_test = df_test[FEATURES]
        y_test = df_test[TARGET]
        return X_train, y_train, X_test, y_test
    
    @staticmethod
    def save_model(path, model):
        joblib.dump(model, path)
    def train(self, X_train, y_train, X_test, y_test):
        self.model = xgb.XGBRegressor(base_score=self.config.params_base_score,
                                      booster=self.config.params_booster,
                                      n_estimators=self.config.params_n_estimators,
                                      objective=self.config.params_objective,
                                      max_depth=self.config.params_max_depth,
                                      learning_rate=self.config.params_learning_rate)
        self.model.fit(
            X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)]
        )
        self.save_model(
            path=self.config.trained_model_path,
            model = self.model
        )
        pass
