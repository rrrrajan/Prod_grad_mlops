import pytest
from dataclasses import FrozenInstanceError
from pathlib import Path

from src.entity.config_entity import (
    DataIngestionConfig,
    PredictionConfig,
    MLflowConfig,
    DeploymentConfig,
)

pytestmark = pytest.mark.unit


def test_data_ingestion_config_creation():
    """Test DataIngestionConfig is created correctly."""

    config = DataIngestionConfig(
        root_dir=Path("artifacts/data_ingestion"),
        raw_data_path=Path("raw.csv"),
        local_data_path=Path("data.csv"),
        train_data_path=Path("train.csv"),
        test_data_path=Path("test.csv"),
        target_column="Churn",
        random_state=42,
        test_size=0.2,
    )

    assert config.target_column == "Churn"
    assert config.random_state == 42
    assert config.test_size == 0.2


def test_prediction_config_creation():
    """Test PredictionConfig."""

    config = PredictionConfig(
        model_path=Path("model.pkl"),
        preprocessor_path=Path("preprocessor.pkl"),
    )

    assert config.model_path.name == "model.pkl"
    assert config.preprocessor_path.name == "preprocessor.pkl"


def test_mlflow_config_creation():
    """Test MLflowConfig."""

    config = MLflowConfig(
        tracking_uri="http://localhost:5000",
        registry_uri="http://localhost:5000",
        experiment_name="customer-churn",
        enabled=True,
    )

    assert config.enabled is True
    assert config.experiment_name == "customer-churn"


def test_deployment_config_default_value():
    """DeploymentConfig should use default values."""

    config = DeploymentConfig(
        root_dir=Path("deployment"),
        registered_model_name="CustomerChurnModel",
        downloaded_model_dir=Path("downloaded_model"),
        image_name="customer-churn-api",
        container_name="customer-churn-api",
        host_port=8000,
        container_port=8000,
        model_mount_path="/app/model",
        model_env_variable="MODEL_DIR",
        health_endpoint="/health",
        startup_timeout=60,
    )

    assert config.remove_existing_container is True


def test_dataclasses_are_frozen():
    """Frozen dataclasses should be immutable."""

    config = PredictionConfig(
        model_path=Path("model.pkl"),
        preprocessor_path=Path("preprocessor.pkl"),
    )

    with pytest.raises(FrozenInstanceError):
        config.model_path = Path("new_model.pkl")