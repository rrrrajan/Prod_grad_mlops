import sys
from pathlib import Path
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from src.entity.config_entity import DeploymentConfig, MLflowConfig
from src.exception import CustomException
from src.logger import logger


class Deployment:
    """
    Handles deployment-related interactions with MLflow.

    Current responsibilities:
        - Connect to the MLflow Tracking Server.
        - Verify connectivity.
        - Retrieve the latest registered model.

    Future responsibilities:
        - Download registered model artifacts.
        - Build deployment image.
        - Push image.
        - Deploy.
    """

    def __init__(
        self,
        deployment_config: DeploymentConfig,
        mlflow_config: MLflowConfig,
    ):
        self.config = deployment_config
        self.mlflow_config = mlflow_config

        self.client = None

    def connect_to_mlflow(self) -> None:
        """
        Connect to the configured MLflow Tracking Server.
        """
        try:
            logger.info(
                f"Connecting to MLflow Tracking Server: "
                f"{self.mlflow_config.tracking_uri}"
            )

            mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
            mlflow.set_registry_uri(self.mlflow_config.registry_uri)

            self.client = MlflowClient(
                tracking_uri=self.mlflow_config.tracking_uri,
                registry_uri=self.mlflow_config.registry_uri,
            )

            # Verify connectivity
            self.client.search_experiments()

            logger.info("Successfully connected to MLflow.")

        except Exception as e:
            logger.exception("Unable to connect to MLflow.")
            raise CustomException(e, sys)

    def deploy(self) -> None:
        logger.info("Starting deployment process.")

        self.connect_to_mlflow()

        self.download_registered_model()

        logger.info("Deployment component completed successfully.")

    def get_latest_model_version(self):
        """
        Returns the latest registered version of the configured model.
        """
        if self.client is None:
            raise RuntimeError("MLflow client not initialized.")

        versions = self.client.search_model_versions(
            f"name='{self.config.registered_model_name}'"
        )

        if not versions:
            raise ValueError(
                f"No versions found for model "
                f"{self.config.registered_model_name}"
            )

        latest = max(
            versions,
            key=lambda mv: int(mv.version),
        )

        logger.info(
            "Latest registered model version: %s",
            latest.version,
        )

        return latest
    

    def download_registered_model(self) -> Path:
        """
        Download the latest registered model artifacts from MLflow.

        Returns:
            Path: Local directory containing the downloaded model artifacts.
        """
        try:
            if self.client is None:
                raise RuntimeError("MLflow client not initialized.")

            latest = self.get_latest_model_version()

            logger.info(
                "Downloading registered model '%s' (version %s).",
                self.config.registered_model_name,
                latest.version,
            )

            model_uri = (
                f"models:/{self.config.registered_model_name}/{latest.version}"
            )

            download_dir = self.config.downloaded_model_dir

            # Remove previous download if it exists
            if download_dir.exists():
                shutil.rmtree(download_dir)

            download_dir.mkdir(parents=True, exist_ok=True)

            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=model_uri,
                dst_path=str(download_dir),
            )

            logger.info(
                "Model artifacts downloaded successfully to: %s",
                local_path,
            )

            return {
                   "version": latest.version,
                   "path": Path(local_path),
                   }

        except Exception as e:
            logger.exception("Failed to download registered model.")
            raise CustomException(e, sys)