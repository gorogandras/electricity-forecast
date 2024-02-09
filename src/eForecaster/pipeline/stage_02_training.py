from src.eForecaster.config.configuration import ConfigurationManager
from src.eForecaster.components.training import Training
from src.eForecaster import logger

STAGE_NAME = "Model training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        df_train, df_test = training.get_dataset()
        X_train, y_train, X_test, y_test = training.get_features_target(df_train, df_test)
        training.train(X_train, y_train, X_test, y_test)

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e