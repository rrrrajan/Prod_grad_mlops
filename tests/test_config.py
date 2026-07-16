from src.config.configuration import ConfigurationManager


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
