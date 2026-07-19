import sys
import shutil
import subprocess
from pathlib import Path

import time

import http.client

from urllib.request import urlopen
from urllib.error import URLError, HTTPError

import mlflow
from mlflow.tracking import MlflowClient

from src.entity.config_entity import DeploymentConfig, MLflowConfig
from src.exception import CustomException
from src.logger import logger


class Deployment:
    """
    Handles deployment of the latest registered ML model.

    Current responsibilities:
        - Connect to the MLflow Tracking Server.
        - Retrieve the latest registered model.
        - Download model artifacts.
        - Verify Docker installation and daemon.

    Future responsibilities:
        - Verify Docker image.
        - Stop existing container.
        - Remove existing container.
        - Run new container.
        - Verify application health.
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
                "Connecting to MLflow Tracking Server: %s",
                self.mlflow_config.tracking_uri,
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

    def verify_docker(self) -> None:
        """
        Verify that Docker is installed and the Docker daemon is running.
        """
        try:
            logger.info("Verifying Docker installation...")

            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(
                "Docker detected: %s",
                result.stdout.strip(),
            )

            logger.info("Verifying Docker daemon...")

            subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info("Docker daemon is running.")

        except FileNotFoundError:
            logger.exception("Docker executable not found.")
            raise CustomException(
                "Docker is not installed or is not available in PATH.",
                sys,
            )

        except subprocess.CalledProcessError as e:
            logger.exception("Docker verification failed.")
            raise CustomException(
                e.stderr or "Docker daemon is not running.",
                sys,
            )

        except Exception as e:
            logger.exception("Failed while verifying Docker.")
            raise CustomException(e, sys)
        

    def verify_image(self) -> None:
        """
        Verify that the configured Docker image exists locally.
        """
        try:
            logger.info(
                "Verifying Docker image '%s'...",
                self.config.image_name,
            )

            subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    self.config.image_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(
                "Docker image '%s' found.",
                self.config.image_name,
            )

        except subprocess.CalledProcessError:
            logger.exception(
                "Docker image '%s' does not exist.",
                self.config.image_name,
            )

            raise CustomException(
                (
                    f"Docker image '{self.config.image_name}' "
                    "does not exist.\n"
                    "Run the Docker Builder stage first."
                ),
                sys,
            )

        except Exception as e:
            logger.exception("Failed while verifying Docker image.")
            raise CustomException(e, sys)


    def stop_existing_container(self) -> None:
        """
        Stop the existing Docker container if it is currently running.

        If the container does not exist or is already stopped,
        deployment continues without error.
        """
        try:
            logger.info(
                "Checking if container '%s' is running...",
                self.config.container_name,
            )

            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-q",
                    "-f",
                    f"name={self.config.container_name}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            container_id = result.stdout.strip()

            if not container_id:
                logger.info(
                    "No running container found with name '%s'.",
                    self.config.container_name,
                )
                return

            logger.info(
                "Stopping container '%s'...",
                self.config.container_name,
            )

            subprocess.run(
                [
                    "docker",
                    "stop",
                    self.config.container_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(
                "Container '%s' stopped successfully.",
                self.config.container_name,
            )

        except subprocess.CalledProcessError as e:
            logger.exception(
                "Failed to stop container '%s'.",
                self.config.container_name,
            )

            raise CustomException(
                e.stderr or "Unable to stop Docker container.",
                sys,
            )

        except Exception as e:
            logger.exception("Error while stopping Docker container.")
            raise CustomException(e, sys)


   
    def remove_existing_container(self) -> None:
        """
        Remove the existing Docker container if it exists.

        If container removal is disabled in the configuration or
        the container does not exist, deployment continues.
        """
        try:
            if not self.config.remove_existing_container:
                logger.info(
                    "Container removal is disabled. Skipping removal step."
                )
                return

            logger.info(
                "Checking if container '%s' exists...",
                self.config.container_name,
            )

            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-aq",
                    "-f",
                    f"name={self.config.container_name}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            container_id = result.stdout.strip()

            if not container_id:
                logger.info(
                    "No existing container found with name '%s'.",
                    self.config.container_name,
                )
                return

            logger.info(
                "Removing container '%s'...",
                self.config.container_name,
            )

            subprocess.run(
                [
                    "docker",
                    "rm",
                    self.config.container_name,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(
                "Container '%s' removed successfully.",
                self.config.container_name,
            )

        except subprocess.CalledProcessError as e:
            logger.exception(
                "Failed to remove container '%s'.",
                self.config.container_name,
            )

            raise CustomException(
                e.stderr or "Unable to remove Docker container.",
                sys,
            )

        except Exception as e:
            logger.exception(
                "Error while removing Docker container."
            )
            raise CustomException(e, sys)
        

    def run_container(self) -> None:
        """
        Run a new Docker container from the configured image.
        """
        try:
            logger.info(
                "Starting container '%s' from image '%s'.",
                self.config.container_name,
                self.config.image_name,
            )

            command = [
                "docker",
                "run",
                "-d",
                "--name",
                self.config.container_name,
                "-p",
                f"{self.config.host_port}:{self.config.container_port}",
                "-v",
                (
                    f"{self.config.downloaded_model_dir.resolve()}:"
                    f"{self.config.model_mount_path}"
                ),
                "-e",
                (
                    f"{self.config.model_env_variable}="
                    f"{self.config.model_mount_path}"
                ),
                "-e",
                "LOG_DIR=/tmp/customer-churn-logs",
                self.config.image_name,
            ]
            
            logger.info(
                "Running command: %s",
                " ".join(command),
            )

            subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True,
            )

            logger.info(
                "Container '%s' started successfully.",
                self.config.container_name,
            )

        except subprocess.CalledProcessError as e:
            logger.exception(
                "Failed to start Docker container."
            )

            raise CustomException(
                e.stderr or "Unable to start Docker container.",
                sys,
            )

        except Exception as e:
            logger.exception(
                "Error while starting Docker container."
            )

            raise CustomException(e, sys)
        

    def wait_until_ready(self) -> None:
        """
        Wait until the deployed API becomes responsive.
        """
        url = (
            f"http://localhost:{self.config.host_port}"
            f"{self.config.health_endpoint}"
        )

        logger.info("Waiting for API to become ready: %s", url)

        start = time.monotonic()
        deadline = start + self.config.startup_timeout

        while time.monotonic() < deadline:
            try:
                with urlopen(url, timeout=5) as response:
                    if response.status == 200:
                        logger.info(
                            "API became ready in %.2f seconds.",
                            time.monotonic() - start,
                        )
                        return

                    logger.info(
                        "API returned status %s. Retrying...",
                        response.status,
                    )

            except (
                URLError,
                HTTPError,
                http.client.RemoteDisconnected,
                ConnectionResetError,
            ) as ex:
                logger.info(
                    "API not ready yet (%s). Retrying...",
                    ex.__class__.__name__,
                )

            time.sleep(1)

        raise CustomException(
            TimeoutError(
                f"API at '{url}' did not become ready within "
                f"{self.config.startup_timeout} seconds."
            ),
            sys,
        )


    def health_check(self) -> None:
        """
        Perform a final health check on the deployed API.
        """
        try:
            url = (
                f"http://localhost:{self.config.host_port}"
                f"{self.config.health_endpoint}"
            )

            logger.info(
                "Performing health check: %s",
                url,
            )

            with urlopen(url, timeout=5) as response:
                if response.status != 200:
                    raise RuntimeError(
                        "Health check failed."
                    )

            logger.info(
                "Deployment health check passed."
            )

        except Exception as e:
            logger.exception(
                "Deployment health check failed."
            )
            raise CustomException(e, sys)
                    

        

    def deploy(self) -> None:
        """
        Execute the deployment workflow.
        """
        logger.info("Starting deployment process.")

        self.connect_to_mlflow()

        model_info = self.download_registered_model()

        logger.info(
            "Downloaded model version: %s",
            model_info["version"],
        )

        self.verify_docker()

        self.verify_image()

        self.stop_existing_container()

        self.remove_existing_container()

        self.run_container()

        self.wait_until_ready()

        self.health_check()

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

    def download_registered_model(self) -> dict:
        """
        Download the latest registered model artifacts from MLflow.

        Returns:
            dict: Contains downloaded model version and local path.
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