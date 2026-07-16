import joblib
import pandas as pd

from src.config.configuration import ConfigurationManager


def test_preprocessor_artifact_exists():
    """
    Verify that the preprocessor artifact exists.
    """

    config = ConfigurationManager()
    transformation_config = config.get_data_transformation_config()

    assert transformation_config.preprocessor_object_path.exists()


def test_preprocessor_can_transform_sample():
    """
    Verify that the saved preprocessor can transform
    a sample customer record.
    """

    config = ConfigurationManager()
    transformation_config = config.get_data_transformation_config()

    preprocessor = joblib.load(transformation_config.preprocessor_object_path)

    sample = pd.DataFrame(
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

    transformed = preprocessor.transform(sample)

    assert transformed.shape[0] == 1
    assert transformed.shape[1] > 0
