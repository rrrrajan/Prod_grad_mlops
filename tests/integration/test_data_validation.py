from pathlib import Path

import pytest

from src.components.data_validation import DataValidation
from src.config.configuration import ConfigurationManager

pytestmark = pytest.mark.integration


def test_data_validation_runs_successfully():
    """Run the validation stage and verify status file is created."""

    config = ConfigurationManager()
    validation_config = config.get_data_validation_config()

    validation = DataValidation(validation_config)

    status = validation.validate_all_columns()

    assert status is True

    status_file = Path(validation_config.STATUS_FILE)

    assert status_file.exists()
