import sys
from pathlib import Path

from src.exception import CustomException
from src.logger import logger
from src.pipeline.prediction_pipeline import (
    CustomData,
    PredictionPipeline,
)


def main():
    """
    Test the PredictionPipeline using the downloaded MLflow model.
    """

    try:
        logger.info("=" * 60)
        logger.info("Testing Prediction Pipeline")
        logger.info("=" * 60)

        model_dir = Path("artifacts/deployment/downloaded_model")

        logger.info("Using model directory: %s", model_dir.resolve())

        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {model_dir.resolve()}"
            )

        mlmodel_file = model_dir / "MLmodel"

        if not mlmodel_file.exists():
            raise FileNotFoundError(
                f"MLmodel file not found: {mlmodel_file.resolve()}"
            )

        logger.info("MLmodel file found.")

        pipeline = PredictionPipeline(model_dir=model_dir)

        logger.info("PredictionPipeline loaded successfully.")
        logger.info("Loaded model type: %s", type(pipeline.model).__name__)

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

        logger.info("Running prediction...")

        result = pipeline.predict(features)

        logger.info("Prediction completed successfully.")

        print("\nPrediction Result")
        print("-" * 40)
        print(f"Prediction : {result['prediction']}")
        print(f"Label      : {result['label']}")
        print(f"Probability: {result['probability']}")

        logger.info("=" * 60)
        logger.info("Prediction Pipeline Test Completed Successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception("Prediction Pipeline Test Failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":
    main()