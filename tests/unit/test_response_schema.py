import pytest
from pydantic import ValidationError

from src.schema.response import PredictionResponse

pytestmark = pytest.mark.unit


def test_prediction_response_creation():
    """Verify PredictionResponse can be created with valid data."""

    response = PredictionResponse(
        prediction="Yes",
        probability=0.91,
        model_version="1.0.0",
    )

    assert response.prediction == "Yes"
    assert response.probability == 0.91
    assert response.model_version == "1.0.0"


def test_prediction_response_without_probability():
    """Verify probability is optional."""

    response = PredictionResponse(
        prediction="No",
        model_version="1.0.0",
    )

    assert response.prediction == "No"
    assert response.probability is None
    assert response.model_version == "1.0.0"


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_probability_must_be_between_0_and_1(probability):
    """Verify probability must be within [0, 1]."""

    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction="Yes",
            probability=probability,
            model_version="1.0.0",
        )


def test_prediction_is_required():
    """Verify prediction is a required field."""

    with pytest.raises(ValidationError):
        PredictionResponse(
            probability=0.91,
            model_version="1.0.0",
        )


def test_model_version_is_required():
    """Verify model_version is a required field."""

    with pytest.raises(ValidationError):
        PredictionResponse(
            prediction="Yes",
            probability=0.91,
        )
