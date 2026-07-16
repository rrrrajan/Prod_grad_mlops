from pathlib import Path

from src.config.configuration import ConfigurationManager
from src.components.data_validation import DataValidation


def test_data_validation_config_creation():
    """Verify DataValidation configuration is created."""

    config = ConfigurationManager()
    validation_config = config.get_data_validation_config()

    assert validation_config is not None
    assert validation_config.root_dir is not None


def test_data_validation_runs_successfully():
    """Run the validation stage and verify status file is created."""

    config = ConfigurationManager()
    validation_config = config.get_data_validation_config()

    validation = DataValidation(validation_config)

    status = validation.validate_all_columns()

    assert status is True

    status_file = Path(validation_config.STATUS_FILE)

    assert status_file.exists()
