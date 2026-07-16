import pytest

from src.pipeline.prediction_pipeline import PredictionPipeline


@pytest.fixture(scope="session")
def prediction_pipeline():
    """
    Load prediction artifacts once for the entire test session.
    """
    return PredictionPipeline()
