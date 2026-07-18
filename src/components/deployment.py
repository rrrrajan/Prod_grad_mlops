import sys

import mlflow
from mlflow.tracking import MlflowClient

from src.entity.config_entity import DeploymentConfig
from src.exception import CustomException
from src.logger import logger


class Deployment:
    """
    Handles deployment-related interactions with MLflow.

    Current responsibilities:
        - Connect to the MLflow Tracking Server.
        - Verify connectivity.
        - Prepare for Model Registry integration.

    Future responsibilities:
        - Register trained models.
        - Promote model versions.
        - Retrieve production models.
    """

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client = None

    def connect_to_mlflow(self) -> None:
        """
        Connect to the configured MLflow Tracking Server.
        """
        try:
            tracking_uri = mlflow.get_tracking_uri()

            logger.info(f"Connecting to MLflow Tracking Server: {tracking_uri}")

            self.client = MlflowClient(
              tracking_uri=self.config.tracking_uri,
              registry_uri=self.config.registry_uri,
            )


            # Simple API call to verify connectivity.
            self.client.search_experiments()

            logger.info("Successfully connected to MLflow.")

        except Exception as e:
            logger.exception("Unable to connect to MLflow.")
            raise CustomException(e, sys)

    def deploy(self) -> None:
        """
        Executes the deployment workflow.
        """
        logger.info("Starting deployment process.")

        self.connect_to_mlflow()

        logger.info("Deployment component completed successfully.")


    def get_latest_model_version(self):

        if self.client is None:
            raise RuntimeError("MLflow client not initialized.")

        versions = self.client.search_model_versions(
            f"name='{self.config.registered_model_name}'"
        )

        if not versions:
            raise ValueError(
                f"No versions found for model {self.config.registered_model_name}"
            )

        latest = max(
            versions,
            key=lambda mv: int(mv.version)
        )

        logger.info(
            "Latest registered model version: %s",
            latest.version,
        )

        return latest