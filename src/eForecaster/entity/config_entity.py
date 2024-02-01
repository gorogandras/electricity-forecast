from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    csv_name: Path

@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    training_data: Path
    testing_data: Path
    params_base_score: float
    params_booster: str
    params_n_estimators: int
    params_early_stopping_rounds: int
    params_objective: str
    params_max_depth: int
    params_learning_rate: float
    target_column: str
    columns: str


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    train_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str
    columns: str
    mlflow_uri: str