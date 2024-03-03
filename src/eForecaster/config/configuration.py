from eForecaster.constants import *
from eForecaster.utils.common import read_yaml, create_directories
from eForecaster.entity.config_entity import DataIngestionConfig, TrainingConfig, ModelEvaluationConfig, PlotConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            csv_name=config.csv_name 
        )

        return data_ingestion_config
    
    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        params = self.params.XGRBoost
        schema =  self.schema
        

        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(training.train_dataset_path),
            testing_data=Path(training.test_dataset_path),
            params_base_score= params.BASE_SCORE,
            params_booster=params.BOOSTER,
            params_n_estimators=params.N_ESTIMATORS,
            params_early_stopping_rounds=params.EARLY_STOPPING_ROUNDS,
            params_objective=params.OBJECTIVE,
            params_max_depth=params.MAX_DEPTH,
            params_learning_rate= params.LEARNING_RATE,
            target_column = schema.TARGET_COLUMN.name,
            columns = schema.COLUMNS,
        )
        return training_config
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.XGRBoost
        schema =  self.schema

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            train_data_path=config.train_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.TARGET_COLUMN.name,
            columns = schema.COLUMNS,
            mlflow_uri="https://dagshub.com/gorogandras/electricity-forecast.mlflow"
           
        )
            
        return model_evaluation_config
    
    def get_plot_config(self) -> PlotConfig:
        config = self.config.plotting

        schema = self.schema
        create_directories([
            Path(config.root_dir)
        ])
        plot_config = PlotConfig(
            root_dir=config.root_dir,
            model_path = config.model_path,
            test_data_path = config.test_data_path,
            train_data_path= config.train_data_path,
            lineplot_path = config.lineplot_path,
            target_column = schema.TARGET_COLUMN.name,
            columns = schema.COLUMNS
        )
        return plot_config