from dataclasses import dataclass
from pathlib import Path


# ==========================================================
# Data Ingestion Artifact
# ==========================================================

@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact generated after the Data Ingestion stage.
    """

    train_file_path: Path
    test_file_path: Path


# ==========================================================
# Data Validation Artifact
# ==========================================================

@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact generated after the Data Validation stage.
    """

    validation_status: bool
    validation_report_file_path: Path
    validated_train_file_path: Path
    validated_test_file_path: Path


# ==========================================================
# Data Transformation Artifact
# ==========================================================

@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact generated after the Data Transformation stage.
    """

    transformed_train_path: Path
    transformed_test_path: Path
    preprocessor_object_path: Path


# ==========================================================
# Model Trainer Artifact
# ==========================================================

@dataclass(frozen=True)
class ModelTrainerArtifact:
    """
    Artifact generated after the Model Trainer stage.
    """

    trained_model_path: Path
    metrics_file_path: Path
    model_report_file_name: Path
    best_model_name: str