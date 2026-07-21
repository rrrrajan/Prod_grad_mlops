import pandas as pd
import pytest

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
    Verify prediction pipeline loads required artifacts.
    """

    assert prediction_pipeline.model is not None
    assert prediction_pipeline.preprocessor is not None


def test_prediction_pipeline_predict(customer, prediction_pipeline):
    """
    Verify prediction pipeline returns a valid prediction.
    """

    result = prediction_pipeline.predict(
        customer.get_data_as_dataframe()
    )

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "prediction",
        "label",
        "probability",
    }

    assert result["prediction"] in {0, 1}
    assert result["label"] in {"Churn", "No Churn"}

    if result["probability"] is not None:
        assert isinstance(result["probability"], float)
        assert 0.0 <= result["probability"] <= 1.0