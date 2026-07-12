import sys

from src.exception import CustomException
from src.logger import logger
from src.pipeline.prediction_pipeline import (
    CustomData,
    PredictionPipeline,
)


def main():
    """
    Test the prediction pipeline using sample customer data.
    """

    try:
        logger.info("=" * 60)
        logger.info("Testing Prediction Pipeline")
        logger.info("=" * 60)

        pipeline = PredictionPipeline()

        customer = CustomData(
            gender="Female",
            SeniorCitizen=0,
            Partner="Yes",
            Dependents="No",
            tenure=1,
            PhoneService="No",
            MultipleLines="No phone service",
            InternetService="DSL",
            OnlineSecurity="No",
            OnlineBackup="Yes",
            DeviceProtection="No",
            TechSupport="No",
            StreamingTV="No",
            StreamingMovies="No",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=29.85,
            TotalCharges=29.85,
        )

        features = customer.get_data_as_dataframe()

        logger.info("Input DataFrame:")
        print(features)

        result = pipeline.predict(features)

        logger.info("Prediction Result:")
        print(result)

        logger.info("=" * 60)
        logger.info("Prediction Pipeline Test Completed Successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Prediction Pipeline Test Failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()