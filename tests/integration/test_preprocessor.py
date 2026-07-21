import joblib
import pandas as pd
import pytest

from src.config.configuration import ConfigurationManager

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def transformation_config():
    """
    Return the data transformation configuration.
    """

    config = ConfigurationManager()
    return config.get_data_transformation_config()


@pytest.fixture(scope="session")
def preprocessor(transformation_config):
    """
    Load the saved preprocessor once per test session.
    """

    return joblib.load(transformation_config.preprocessor_object_path)


@pytest.fixture
def sample_customer():
    """
    Return a valid sample customer record.
    """

    return pd.DataFrame(
        [
            {
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
        ]
    )


def test_preprocessor_artifact_exists(transformation_config):
    """
    Verify that the preprocessor artifact exists.
    """

    assert transformation_config.preprocessor_object_path.exists()


def test_preprocessor_can_transform_sample(
    preprocessor,
    sample_customer,
):
    """
    Verify that the saved preprocessor can transform
    a sample customer record.
    """

    transformed = preprocessor.transform(sample_customer)

    assert transformed is not None
    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
