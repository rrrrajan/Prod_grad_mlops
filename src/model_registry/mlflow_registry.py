import sys

import mlflow
from mlflow import MlflowClient

from src.exception import CustomException
from src.logger import logger

from .model_registry import ModelRegistry


class MLflowModelRegistry(ModelRegistry):
    """
    MLflow implementation of the Model Registry interface.
    """

    def __init__(self):
        self.client = MlflowClient()

    def register_model(
        self,
        model_uri: str,
        model_name: str,
    ):
        try:
            logger.info(
                "Registering model '%s' from URI '%s'.",
                model_name,
                model_uri,
            )

            return mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def set_alias(
        self,
        model_name: str,
        alias: str,
        version: str,
    ) -> None:
        try:
            logger.info(
                "Assigning alias '%s' to version %s.",
                alias,
                version,
            )

            self.client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=version,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def get_model_version(
        self,
        model_name: str,
        alias: str,
    ):
        try:
            return self.client.get_model_version_by_alias(
                name=model_name,
                alias=alias,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def download_model(
        self,
        model_uri: str,
        dst_path: str | None = None,
    ) -> str:
        try:
            logger.info("Downloading model %s", model_uri)

            return mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=dst_path,
            )

        except Exception as e:
            raise CustomException(e, sys)