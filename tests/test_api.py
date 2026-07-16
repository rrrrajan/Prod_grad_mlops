from fastapi.testclient import TestClient

from api import app


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


def test_predict_endpoint():
    """
    Verify prediction endpoint.
    """

    payload = {
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

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json=payload,
        )

        assert response.status_code == 200

        body = response.json()

        assert "prediction" in body
        assert "probability" in body
        assert "model_version" in body

        assert body["prediction"] in [
            "Churn",
            "No Churn",
        ]

        if body["probability"] is not None:
            assert 0.0 <= body["probability"] <= 1.0


def test_predict_invalid_payload():
    """
    Verify request validation.
    """

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            json={},
        )

        assert response.status_code == 422
