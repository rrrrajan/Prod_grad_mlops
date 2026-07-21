import pytest

from src.components.model_pusher import ModelPusher
from src.config.configuration import ConfigurationManager
from src.entity.artifact_entity import ModelPusherArtifact

pytestmark = pytest.mark.integration


def test_model_pusher_pipeline():
    """
    Integration test for the complete Model Pusher stage.
    """

    config = ConfigurationManager()
    pusher_config = config.get_model_pusher_config()

    pusher = ModelPusher(config=pusher_config)

    artifact = pusher.initiate_model_pusher()

    assert isinstance(artifact, ModelPusherArtifact)

    assert artifact.pushed_model_path.exists()
    assert artifact.pushed_preprocessor_path.exists()

    assert artifact.pushed_model_path.is_file()
    assert artifact.pushed_preprocessor_path.is_file()
