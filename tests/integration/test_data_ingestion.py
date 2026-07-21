import pytest

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import DataIngestionArtifact

pytestmark = pytest.mark.integration


def test_data_ingestion_pipeline():
    """
    Integration test for the complete Data Ingestion stage.
    """

    config = ConfigurationManager()
    ingestion_config = config.get_data_ingestion_config()

    ingestion = DataIngestion(config=ingestion_config)

    artifact = ingestion.initiate_data_ingestion()

    assert isinstance(artifact, DataIngestionArtifact)

    assert artifact.train_file_path.exists()
    assert artifact.test_file_path.exists()

    assert artifact.train_file_path.is_file()
    assert artifact.test_file_path.is_file()

    assert artifact.train_file_path.stat().st_size > 0
    assert artifact.test_file_path.stat().st_size > 0
