import pandas as pd

from src.pipeline.prediction_pipeline import (
    CustomData,
)


def test_custom_data_dataframe():
    """
    Verify CustomData creates a valid DataFrame.
    """

    customer = CustomData(
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

    df = customer.get_data_as_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (1, 19)


def test_prediction_pipeline_loads(prediction_pipeline):
    assert prediction_pipeline.model is not None
    assert prediction_pipeline.preprocessor is not None


def test_prediction_pipeline_predict(prediction_pipeline):
    """
    Verify prediction pipeline returns a valid response.
    """

    customer = CustomData(
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

    result = prediction_pipeline.predict(customer.get_data_as_dataframe())

    assert isinstance(result, dict)

    assert "prediction" in result
    assert "label" in result
    assert "probability" in result

    assert result["prediction"] in [0, 1]
    assert result["label"] in ["Churn", "No Churn"]

    if result["probability"] is not None:
        assert 0.0 <= result["probability"] <= 1.0
