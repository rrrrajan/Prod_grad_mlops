import pytest

from src.config.configuration import ConfigurationManager

pytestmark = pytest.mark.unit


def test_configuration_manager_creation():
    """Verify ConfigurationManager can be instantiated."""

    config = ConfigurationManager()

    assert config is not None


def test_configuration_files_loaded():
    """Verify configuration, params, and schema are loaded."""

    config = ConfigurationManager()

    assert config.config is not None
    assert config.params is not None
    assert config.schema is not None


def test_data_validation_config_creation():
    """Verify DataValidation configuration is created."""

    config = ConfigurationManager()
    validation_config = config.get_data_validation_config()

    assert validation_config is not None
    assert validation_config.root_dir is not None