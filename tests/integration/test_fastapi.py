import pytest
from fastapi.testclient import TestClient

from api import app

pytestmark = pytest.mark.integration


@pytest.fixture
def prediction_payload():
    """
    Valid payload for prediction endpoint.
    """

    return {
        "gender": "Male",
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
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 1688.65,
    }


def test_root_endpoint():
    """
    Verify the root endpoint.
    """

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "running"
    assert "message" in body
    assert "version" in body


def test_health_endpoint():
    """
    Verify the health endpoint.
    """

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ready_endpoint():
    """
    Verify the readiness endpoint.
    """

    with TestClient(app) as client:
        response = client.get("/api/v1/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict_endpoint(prediction_payload):
    """
    Verify prediction endpoint.
    """

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json=prediction_payload,
        )

    assert response.status_code == 200

    body = response.json()

    assert set(body.keys()) == {
        "prediction",
        "probability",
        "model_version",
    }

    assert isinstance(body["prediction"], str)
    assert isinstance(body["model_version"], str)

    assert body["prediction"] in {"Churn", "No Churn"}

    if body["probability"] is not None:
        assert isinstance(body["probability"], float)
        assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_payload():
    """
    Verify request validation for missing required fields.
    """

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json={},
        )

    assert response.status_code == 422


@pytest.mark.parametrize(
    "field,value",
    [
        ("SeniorCitizen", 2),
        ("SeniorCitizen", -1),
        ("tenure", -5),
        ("MonthlyCharges", -10.0),
        ("TotalCharges", -100.0),
    ],
)
def test_predict_invalid_numeric_values(prediction_payload, field, value):
    """
    Verify request validation for invalid numeric values.
    """

    prediction_payload[field] = value

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json=prediction_payload,
        )

    assert response.status_code == 422
