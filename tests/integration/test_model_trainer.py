from pathlib import Path

import pytest

from src.components.model_trainer import ModelTrainer
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import ModelTrainerArtifact

pytestmark = pytest.mark.integration


def test_model_trainer_pipeline():
    """
    Integration test for the complete Model Trainer stage.
    """

    # Arrange
    config = ConfigurationManager()
    trainer_config = config.get_model_trainer_config()

    trainer = ModelTrainer(config=trainer_config)

    # Act
    artifact = trainer.initiate_model_trainer()

    # Assert
    assert isinstance(artifact, ModelTrainerArtifact)

    assert artifact.trained_model_path.exists()
    assert artifact.metrics_file_path.exists()
    assert artifact.model_report_file_path.exists()

    assert artifact.best_model_name is not None
    assert artifact.best_model_name != ""

    assert artifact.trained_model_path.is_file()
    assert artifact.metrics_file_path.is_file()
    assert artifact.model_report_file_path.is_file()