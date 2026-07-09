from dataclasses import dataclass
from pathlib import Path


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