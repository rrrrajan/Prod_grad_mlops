import subprocess
import sys

from src.entity.config_entity import DockerBuilderConfig
from src.exception import CustomException
from src.logger import logger


class DockerBuilder:
    """
    Component responsible for building the Docker image
    for the FastAPI application.
    """

    def __init__(self, config: DockerBuilderConfig):
        self.config = config

    def _build_command(self) -> list[str]:
        """
        Creates and returns the Docker build command.

        Returns
        -------
        list[str]
            Docker build command.
        """

        image = f"{self.config.image_name}:{self.config.image_tag}"

        return [
            "docker",
            "build",
            "-f",
            str(self.config.dockerfile_path),
            "-t",
            image,
            str(self.config.context_path),
        ]

    def build_image(self) -> None:
        """
        Builds the Docker image using the configured Dockerfile.
        """

        try:
            logger.info("=" * 80)
            logger.info("Starting Docker image build.")
            logger.info("=" * 80)

            image = f"{self.config.image_name}:{self.config.image_tag}"

            logger.info(f"Dockerfile : {self.config.dockerfile_path}")
            logger.info(f"Context    : {self.config.context_path}")
            logger.info(f"Image      : {image}")

            result = subprocess.run(
                self._build_command(),
                capture_output=True,
                text=True,
                check=True,
            )

            if result.stdout:
                logger.info(result.stdout)

            
            logger.info("=" * 80)
            logger.info("Docker image built successfully.")
            logger.info("=" * 80)

        except subprocess.CalledProcessError as e:
            logger.error("Docker build failed.")

            if e.stderr:
                logger.error(e.stderr)

            raise CustomException(e, sys)

        except Exception as e:
            raise CustomException(e, sys)