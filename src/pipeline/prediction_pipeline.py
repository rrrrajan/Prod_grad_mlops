from pathlib import Path
from typing import Any
import os
import sys

import mlflow.sklearn
import pandas as pd

from src.logger import logger
from src.exception import CustomException

# Base directory containing the downloaded MLflow model.
DEFAULT_MODEL_DIR = Path(
    os.getenv(
        "MODEL_DIR",
        "artifacts/deployment/downloaded_model",
    )
)


class CustomData:
    """
    Represents a single customer record for prediction.
    """

    def __init__(
        self,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        tenure: int,
        PhoneService: str,
        MultipleLines: str,
        InternetService: str,
        OnlineSecurity: str,
        OnlineBackup: str,
        DeviceProtection: str,
        TechSupport: str,
        StreamingTV: str,
        StreamingMovies: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str,
        MonthlyCharges: float,
        TotalCharges: float,
    ) -> None:

        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_data_as_dataframe(self) -> pd.DataFrame:
        """
        Convert customer data into a pandas DataFrame.
        """

        try:
            logger.info("Creating prediction dataframe.")

            input_data = {
                "gender": [self.gender],
                "SeniorCitizen": [self.SeniorCitizen],
                "Partner": [self.Partner],
                "Dependents": [self.Dependents],
                "tenure": [self.tenure],
                "PhoneService": [self.PhoneService],
                "MultipleLines": [self.MultipleLines],
                "InternetService": [self.InternetService],
                "OnlineSecurity": [self.OnlineSecurity],
                "OnlineBackup": [self.OnlineBackup],
                "DeviceProtection": [self.DeviceProtection],
                "TechSupport": [self.TechSupport],
                "StreamingTV": [self.StreamingTV],
                "StreamingMovies": [self.StreamingMovies],
                "Contract": [self.Contract],
                "PaperlessBilling": [self.PaperlessBilling],
                "PaymentMethod": [self.PaymentMethod],
                "MonthlyCharges": [self.MonthlyCharges],
                "TotalCharges": [self.TotalCharges],
            }

            df = pd.DataFrame(input_data)

            logger.info("Prediction dataframe created successfully.")

            return df

        except Exception as e:
            raise CustomException(e, sys)


class PredictionPipeline:
    """
    Prediction pipeline for customer churn inference using
    a registered MLflow sklearn Pipeline.
    """

    def __init__(
        self,
        model_dir: Path = DEFAULT_MODEL_DIR,
    ) -> None:
        """
        Load the registered MLflow sklearn Pipeline.
        """

        try:
            logger.info(
                "Loading MLflow model from '%s'.",
                model_dir,
            )

            mlmodel_file = model_dir / "MLmodel"

            if not mlmodel_file.exists():
                raise FileNotFoundError(
                    f"MLflow model not found. Expected '{mlmodel_file}'."
                )

            self.model = mlflow.sklearn.load_model(
                model_uri=str(model_dir)
            )

            logger.info("MLflow model loaded successfully.")

        except Exception as e:
            logger.exception("Failed to load MLflow model.")
            raise CustomException(e, sys)

    def predict(
        self,
        features: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Predict customer churn.
        """

        try:
            logger.info("Generating prediction.")

            prediction = int(
                self.model.predict(features)[0]
            )

            probability = None

            if hasattr(self.model, "predict_proba"):
                probability = float(
                    self.model.predict_proba(features)[0][1]
                )

            result = {
                "prediction": prediction,
                "label": "Churn" if prediction else "No Churn",
                "probability": probability,
            }

            logger.info(
                "Prediction successful. "
                "Prediction=%s, Label=%s, Probability=%s",
                result["prediction"],
                result["label"],
                result["probability"],
            )

            return result

        except Exception as e:
            logger.exception("Prediction pipeline failed.")
            raise CustomException(e, sys)