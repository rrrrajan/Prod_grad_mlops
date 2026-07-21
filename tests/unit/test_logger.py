import logging

import pytest

from src.logger import logger

pytestmark = pytest.mark.unit


def test_logger_instance():
    """Logger should be a logging.Logger instance."""
    assert isinstance(logger, logging.Logger)


def test_logger_name():
    """Logger should have the expected name."""
    assert logger.name == "customer_churn_mlops"


def test_logger_level():
    """Logger should be configured with INFO level."""
    assert logger.getEffectiveLevel() == logging.INFO


def test_logger_has_handler():
    """Logger should have at least one configured handler."""
    assert logger.hasHandlers()


def test_logger_can_log(caplog):
    """Logger should emit log messages."""
    with caplog.at_level(logging.INFO):
        logger.info("Unit test log message")

    assert "Unit test log message" in caplog.text