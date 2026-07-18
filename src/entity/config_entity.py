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


@dataclass(frozen=True)
class ModelEvaluationConfig:
    """
    Configuration for the Model Evaluation stage.

    Attributes
    ----------
    root_dir : Path
        Directory where evaluation artifacts will be stored.

    test_array_path : Path
        Path to the transformed testing dataset.

    model_path : Path
        Path to the trained model.

    metric_file_name : Path
        Path where evaluation metrics will be saved.

    model_report_file_name : Path
        Path where detailed evaluation report will be saved.

    target_column : str
        Name of the target column.

    evaluation_metric : str
        Primary metric used for selecting the best model.

    random_state : int
        Random seed for reproducibility.
    """

    root_dir: Path

    trained_model_path: Path
    test_array_path: Path

    metrics_file_name: Path
    metadata_file_name: Path
    classification_report_file_name: Path

    confusion_matrix_json_file_name: Path

    roc_curve_file_name: Path
    confusion_matrix_file_name: Path

    target_column: str

    evaluation_metric: str
    random_state: int

    experiment_name: str
    run_name: str

    log_model: bool
    register_model: bool
    registered_model_name: str


@dataclass(frozen=True)
class ModelPusherConfig:
    """
    Configuration for the Model Pusher component.

    Attributes
    ----------
    root_dir : Path
        Directory where the pushed artifacts will be stored.
    source_model_path : Path
        Path to the trained model produced by the Model Trainer.
    source_preprocessor_path : Path
        Path to the fitted preprocessor produced by the Data Transformation stage.
    pushed_model_path : Path
        Destination path for the deployed model.
    pushed_preprocessor_path : Path
        Destination path for the deployed preprocessor.
    """

    root_dir: Path
    source_model_path: Path
    source_preprocessor_path: Path
    pushed_model_path: Path
    pushed_preprocessor_path: Path


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    preprocessor_path: Path


@dataclass(frozen=True)
class MLflowConfig:
    tracking_uri: str
    registry_uri: str
    experiment_name: str
    enabled: bool


@dataclass(frozen=True)
class DeploymentConfig:
    """
    Configuration for the Deployment component.
    """

    root_dir: Path
    registered_model_name: str
    downloaded_model_dir: Path


@dataclass(frozen=True)
class DockerBuilderConfig:
    """
    Configuration for the Docker Builder component.
    """

    root_dir: Path

    image_name: str

    image_tag: str

    dockerfile_path: Path

    context_path: Path