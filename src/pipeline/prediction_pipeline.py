from pathlib import Path
from typing import Any

import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.utils.common import load_object

DEFAULT_MODEL_PATH = Path("artifacts/model_pusher/model.pkl")
DEFAULT_PREPROCESSOR_PATH = Path("artifacts/model_pusher/preprocessor.pkl")


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
    Prediction pipeline for customer churn inference.
    """

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        preprocessor_path: Path = DEFAULT_PREPROCESSOR_PATH,
    ) -> None:
        """
        Initialize prediction pipeline by loading model artifacts.

        Parameters
        ----------
        model_path : Path
            Path to the trained model.

        preprocessor_path : Path
            Path to the fitted preprocessor.
        """

        try:
            logger.info("Loading prediction artifacts.")

            self.model = load_object(model_path)
            self.preprocessor = load_object(preprocessor_path)

            logger.info("Prediction artifacts loaded successfully.")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame) -> dict[str, Any]:
        """
        Predict customer churn.

        Parameters
        ----------
        features : pd.DataFrame
            Customer data.

        Returns
        -------
        dict[str, Any]
            Prediction results.
        """

        try:
            logger.info("Applying preprocessing.")

            transformed_features = self.preprocessor.transform(features)

            logger.info("Generating prediction.")

            prediction = int(self.model.predict(transformed_features)[0])

            probability = None

            if hasattr(self.model, "predict_proba"):
                probability = float(
                    self.model.predict_proba(transformed_features)[0][1]
                )

            result = {
                "prediction": prediction,
                "label": "Churn" if prediction == 1 else "No Churn",
                "probability": probability,
            }

            logger.info(
                "Prediction successful. " "Prediction=%s, Label=%s, Probability=%s",
                result["prediction"],
                result["label"],
                result["probability"],
            )

            return result

        except Exception as e:
            logger.exception("Prediction pipeline failed.")
            raise CustomException(e, sys)
