import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.pipeline.prediction_pipeline import CustomData

pytestmark = pytest.mark.integration


@pytest.fixture
def customer():
    """
    Return a valid customer instance for prediction tests.
    """

    return CustomData(
        gender="Male",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=24,
        PhoneService="Yes",
        MultipleLines="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=70.35,
        TotalCharges=1688.65,
    )


def test_custom_data_dataframe(customer):
    """
    Verify CustomData creates a valid DataFrame.
    """

    df = customer.get_data_as_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 19)

    expected_columns = [
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
    ]

    assert list(df.columns) == expected_columns


def test_prediction_pipeline_loads(prediction_pipeline):
    """
    Verify prediction pipeline loads the MLflow inference pipeline.
    """

    assert prediction_pipeline.model is not None

    # The loaded object should be a fitted sklearn Pipeline
    assert isinstance(prediction_pipeline.model, Pipeline)

    # It should contain both preprocessing and classification steps
    assert "preprocessor" in prediction_pipeline.model.named_steps
    assert "classifier" in prediction_pipeline.model.named_steps


def test_prediction_pipeline_predict(prediction_pipeline, customer):
    """
    Verify the prediction pipeline produces a valid prediction.
    """

    df = customer.get_data_as_dataframe()

    result = prediction_pipeline.predict(df)

    assert isinstance(result, dict)

    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Churn", "No Churn")

    if result["probability"] is not None:
        assert 0.0 <= result["probability"] <= 1.0
