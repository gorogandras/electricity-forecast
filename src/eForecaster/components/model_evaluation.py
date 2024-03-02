import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib
import seaborn as sns
from matplotlib import pyplot as plt
from eForecaster.entity.config_entity import ModelEvaluationConfig
from eForecaster.utils.common import save_json
from pathlib import Path
from datetime import datetime

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    


    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        test_x = test_data[[item for item in self.config.columns.keys()]]
        test_y = test_data[[self.config.target_column]]

        experiment_name = "eforecaster-xgbr" ## !!
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        mlflow.set_experiment(experiment_name=experiment_name) #experiment_name
        with mlflow.start_run(run_name=run_name):

            #run_name=run_name
        
            predicted_consumption = model.predict(test_x)

            (rmse, mae, r2) = self.eval_metrics(test_y, predicted_consumption)
            
            # Saving metrics as local
            scores = {"rmse": rmse, "mae": mae, "r2": r2}
            save_json(path=Path(self.config.metric_file_name), data=scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)


            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="XGRBoostModel")
            else:
                mlflow.sklearn.log_model(model, "model")
    
    
