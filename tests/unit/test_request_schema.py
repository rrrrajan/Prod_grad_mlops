import pytest
from pydantic import ValidationError

from src.schema.request import CustomerRequest

pytestmark = pytest.mark.unit


@pytest.fixture
def valid_request_data():
    """Return valid request data."""

    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.15,
        "TotalCharges": 2135.75,
    }


def test_customer_request_creation(valid_request_data):
    """Verify CustomerRequest can be created with valid data."""

    request = CustomerRequest(**valid_request_data)

    assert request.gender == "Female"
    assert request.SeniorCitizen == 0
    assert request.tenure == 24
    assert request.MonthlyCharges == 89.15
    assert request.TotalCharges == 2135.75


@pytest.mark.parametrize(
    "field,value",
    [
        ("SeniorCitizen", -1),
        ("SeniorCitizen", 2),
        ("tenure", -1),
        ("MonthlyCharges", -0.01),
        ("TotalCharges", -10.0),
    ],
)
def test_numeric_field_validation(valid_request_data, field, value):
    """Verify numeric fields enforce their constraints."""

    valid_request_data[field] = value

    with pytest.raises(ValidationError):
        CustomerRequest(**valid_request_data)


@pytest.mark.parametrize(
    "missing_field",
    [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
    ],
)
def test_required_fields(valid_request_data, missing_field):
    """Verify all required fields are enforced."""

    valid_request_data.pop(missing_field)

    with pytest.raises(ValidationError):
        CustomerRequest(**valid_request_data)
