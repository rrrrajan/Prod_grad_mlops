from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Configuration for the Data Ingestion component.
    """

    root_dir: Path

    raw_data_path: Path

    local_data_path: Path

    train_data_path: Path

    test_data_path: Path

    target_column: str

    random_state: int

    test_size: float

@dataclass(frozen=True)
class DataValidationConfig:
    """
    Configuration for the Data Validation component.
    """

    root_dir: Path
    STATUS_FILE: Path
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    """
    Configuration required for the Data Transformation stage.
    """

    # Artifact directory
    root_dir: Path

    # Input datasets
    train_data_path: Path
    test_data_path: Path

    # Output artifacts
    preprocessor_object_path: Path
    transformed_train_path: Path
    transformed_test_path: Path

    # Dataset schema
    target_column: str
    numerical_columns: list[str]
    categorical_columns: list[str]
    numeric_conversion_columns: list[str]

@dataclass(frozen=True)
class ModelTrainerConfig:
    """
    Configuration for the Model Trainer stage.

    Attributes
    ----------
    root_dir : Path
        Directory where all model training artifacts are stored.

    train_data_path : Path
        Path to the transformed training NumPy array.

    test_data_path : Path
        Path to the transformed testing NumPy array.

    trained_model_path : Path
        Path where the best trained model will be serialized.

    metrics_file_name : Path
        Path to the JSON file containing metrics of the best model.

    model_report_file_name : Path
        Path to the JSON file containing evaluation metrics
        for all candidate models.
    """
    root_dir: Path

    transformed_train_path: Path
    transformed_test_path: Path

    trained_model_path: Path
    metrics_file_name: Path
    model_report_file_name: Path

    evaluation_metric: str
    model_params: dict[str, Any]